"""CPU-only checks for the Native layer benchmark's explicit configuration."""

from contextlib import redirect_stderr
import fnmatch
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from benchmarks.bench_qwen35_native import (
    comparison_tolerance, count_grouped_gemm_calls, parse_args, precision_evidence,
    select_grouped_tensor, shared_bf16_quant_recipe,
)


class NativeCLI(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual((args.num_local_tokens, args.hidden_dim, args.intermediate_dim, args.num_experts, args.topk, args.ep_size), (4096, 4096, 1024, 64, 10, 4))
        self.assertEqual(args.gate, "both")
        self.assertEqual(args.precision, "bf16")
        self.assertEqual(args.gemm_backend, "auto")
        self.assertTrue(args.bias_activation_fusion)
        self.assertFalse(args.hybridep_fuse_permute)

    def test_mxfp8_shape_and_explicit_backend(self):
        args = parse_args(["--precision", "mxfp8", "--num-experts", "32", "--gemm-backend", "device-init"])
        self.assertEqual((args.precision, args.num_experts, args.ep_size), ("mxfp8", 32, 4))
        self.assertEqual(select_grouped_tensor(args.gemm_backend, True), (True, None))

    def test_shared_override_does_not_match_routed(self):
        recipe = shared_bf16_quant_recipe()
        matcher = recipe["matchers"]["shared"]
        for fc in ("linear_fc1", "linear_fc2"):
            self.assertTrue(fnmatch.fnmatch(f"qwen35_native_baseline.shared_experts.{fc}", matcher["pattern"]))
            self.assertFalse(fnmatch.fnmatch(f"qwen35_native_baseline.experts.{fc}", matcher["pattern"]))
        config = recipe["configs"][matcher["config"]]
        self.assertEqual(config["transformer_engine_config_type"], "TEQuantizationParams")
        self.assertIsNone(config["training_recipe"]["fp8_quantization_recipe"])
        self.assertTrue(config["training_recipe"]["override_quantized_autocast"])

    def test_shared_tolerances_stay_bf16(self):
        bf16, mxfp8 = (0.5, 0.01), (1.0, 0.1)
        for name in ("output", "d_x", "d_router_weights", "d_w_routed_gate"):
            self.assertEqual(comparison_tolerance(name, "mxfp8", bf16, mxfp8), mxfp8)
            self.assertEqual(comparison_tolerance(name, "bf16", bf16, mxfp8), bf16)
        for name in ("d_w_shared_gate", "d_w_shared_up", "d_w_shared_down", "d_w_shared_output_gate"):
            self.assertEqual(comparison_tolerance(name, "mxfp8", bf16, mxfp8), bf16)

    @staticmethod
    def fake_layer():
        def linear(fp8):
            return SimpleNamespace(
                fp8=fp8, fp8_meta={"recipe": SimpleNamespace(mxfp8=lambda: fp8)},
                is_first_microbatch=False, disable_parameter_transpose_cache=False,
                parameters=lambda: (SimpleNamespace(dtype="torch.bfloat16"),),
            )
        return SimpleNamespace(
            experts=SimpleNamespace(linear_fc1=linear(True), linear_fc2=linear(True)),
            shared_experts=SimpleNamespace(linear_fc1=linear(False), linear_fc2=linear(False), gate_weight=None),
        )

    def test_precision_probe_rejects_fake_mxfp8_and_quantized_shared(self):
        evidence = precision_evidence(self.fake_layer(), "mxfp8")
        self.assertTrue(evidence["routed_fc1"]["fp8"])
        self.assertFalse(evidence["shared_fc1"]["fp8"])
        for target, value in (("routed_fc1", False), ("shared_fc1", True)):
            layer = self.fake_layer()
            module = layer.experts if target.startswith("routed") else layer.shared_experts
            module.linear_fc1.fp8 = value
            with self.assertRaisesRegex(RuntimeError, "expected fp8"):
                precision_evidence(layer, "mxfp8")
        layer = self.fake_layer()
        layer.experts.linear_fc1.fp8_meta["recipe"].mxfp8 = lambda: False
        with self.assertRaisesRegex(RuntimeError, "expected actual MXFP8"):
            precision_evidence(layer, "mxfp8")

    def test_precision_probe_rejects_uncached_weights(self):
        for field in ("is_first_microbatch", "disable_parameter_transpose_cache"):
            layer = self.fake_layer()
            setattr(layer.experts.linear_fc1, field, True)
            with self.assertRaises(RuntimeError):
                precision_evidence(layer, "mxfp8")

    def test_help_does_not_import_torch(self):
        script = (
            "import runpy, sys; "
            "sys.modules['torch'] = None; "
            "sys.argv = ['bench_qwen35_native', '--help']; "
            "runpy.run_module('benchmarks.bench_qwen35_native', run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], check=True,
        )
        self.assertIn("--precision {bf16,mxfp8}", result.stdout)

    def test_explicit_fallback_and_short_run(self):
        args = parse_args(["--gemm-backend", "multistream", "--gate", "on", "--warmup-iters", "3", "--timed-iters", "5"])
        self.assertEqual((args.warmup_iters, args.timed_iters), (3, 5))
        self.assertEqual(args.gate, "on")
        self.assertEqual(select_grouped_tensor(args.gemm_backend, True), (False, None))

    def test_auto_fallback_is_recorded(self):
        enabled, reason = select_grouped_tensor("auto", False)
        self.assertFalse(enabled)
        self.assertIn("fallback", reason)
        self.assertEqual(select_grouped_tensor("auto", True), (True, None))

    def test_required_device_init_does_not_silently_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "Device-init requested"):
            select_grouped_tensor("device-init", False)

    def test_te_internal_grouped_does_not_enable_mcore_grouped_tensor(self):
        args = parse_args(["--gemm-backend", "te-cublas-grouped"])
        self.assertEqual(select_grouped_tensor(args.gemm_backend, False), (False, None))
        self.assertEqual(select_grouped_tensor(args.gemm_backend, True), (False, None))

    def test_call_probe_counts_and_restores_on_exception(self):
        grouped = lambda value: value + 1
        legacy = lambda value: value - 1
        module = SimpleNamespace(
            general_grouped_gemm_for_grouped_tensor=grouped,
            general_grouped_gemm=legacy,
        )
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with count_grouped_gemm_calls(module) as counts:
                self.assertEqual(module.general_grouped_gemm_for_grouped_tensor(4), 5)
                self.assertEqual(module.general_grouped_gemm(4), 3)
                self.assertEqual(counts, {"grouped_tensor": 1, "legacy": 1})
                raise RuntimeError("probe failure")
        self.assertIs(module.general_grouped_gemm_for_grouped_tensor, grouped)
        self.assertIs(module.general_grouped_gemm, legacy)

    def test_invalid_arguments(self):
        for argv in (["--timed-iters", "0"], ["--warmup-iters", "-1"], ["--num-experts", "63"], ["--rank-capacity-factor", "nan"], ["--gemm-backend", "op-fuser"], ["--precision", "fp8"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
