"""CPU-only checks for the Native layer benchmark's explicit configuration."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest

from benchmarks.bench_qwen35_native import count_grouped_gemm_calls, parse_args, select_grouped_tensor


class NativeCLI(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])
        self.assertEqual((args.num_local_tokens, args.hidden_dim, args.intermediate_dim, args.num_experts, args.topk, args.ep_size), (4096, 4096, 1024, 64, 10, 4))
        self.assertEqual(args.gate, "both")
        self.assertEqual(args.gemm_backend, "auto")
        self.assertTrue(args.bias_activation_fusion)
        self.assertFalse(args.hybridep_fuse_permute)

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
        for argv in (["--timed-iters", "0"], ["--warmup-iters", "-1"], ["--num-experts", "63"], ["--rank-capacity-factor", "nan"], ["--gemm-backend", "op-fuser"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
