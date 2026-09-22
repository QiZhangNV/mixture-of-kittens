"""Standard-library CLI/dispatch checks; no Torch or CUDA is imported."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock

from benchmarks.bench_qwen35_mxfp8_gate import MXFP8GateBenchmark, parse_args


class MXFP8GateCLI(unittest.TestCase):
    def test_defaults_hold_397b_runtime_parameters_and_local_eight(self):
        args = parse_args([])
        self.assertEqual(
            (args.num_local_tokens, args.hidden_dim, args.intermediate_dim,
             args.num_experts, args.topk, args.ep_size),
            (4096, 4096, 1024, 32, 10, 4),
        )
        self.assertEqual(
            (args.fwd_comm_sms, args.bwd_comm_sms, args.minibatch_size,
             args.macrobatch_size, args.schedule_capacity_multiplier),
            (48, 56, 4096, 65536, 0.5),
        )
        self.assertEqual((args.warmup_iters, args.timed_iters), (500, 100))
        self.assertEqual((args.gate, args.label, args.grad_mode), ("off", "MOK-new-U", "fresh"))
        self.assertFalse(args.skip_correctness)

    def test_gated_main_grad_and_custom_labels(self):
        args = parse_args(["--gate", "on", "--grad-mode", "fp32-main-grad"])
        self.assertEqual((args.label, args.grad_mode), ("MOK-new-G", "fp32-main-grad"))
        args = parse_args(["--label", "MOK-old-U", "--warmup-iters", "0", "--timed-iters", "2"])
        self.assertEqual((args.label, args.warmup_iters, args.timed_iters), ("MOK-old-U", 0, 2))

    def test_invalid_counts_and_modes(self):
        for argv in (
            ["--timed-iters", "0"], ["--warmup-iters", "-1"],
            ["--num-experts", "31"], ["--topk", "33"],
            ["--hidden-dim", "300"], ["--intermediate-dim", "300"],
            ["--num-local-tokens", "300"], ["--minibatch-size", "300"],
            ["--macrobatch-size", "5000"], ["--ep-size", "0"],
            ["--schedule-capacity-multiplier", "nan"],
            ["--schedule-capacity-multiplier", "inf"],
            ["--schedule-capacity-multiplier", "0"],
            ["--grad-mode", "split"], ["--gate", "true"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_import_and_help_do_not_load_gpu_libraries(self):
        code = (
            "import sys; from benchmarks.bench_qwen35_mxfp8_gate import parse_args; "
            "parse_args([]); assert 'torch' not in sys.modules; assert 'mok' not in sys.modules"
        )
        repo_root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, "-c", code], check=True, cwd=repo_root)
        help_result = subprocess.run(
            [sys.executable, "-m", "benchmarks.bench_qwen35_mxfp8_gate", "--help"],
            check=True, capture_output=True, text=True, cwd=repo_root,
        )
        self.assertIn("--grad-mode", help_result.stdout)


class MXFP8Dispatch(unittest.TestCase):
    def make_benchmark(self, gated):
        benchmark = MXFP8GateBenchmark.__new__(MXFP8GateBenchmark)
        benchmark.functional = Mock()
        benchmark.functional.build_schedule.return_value = "schedule"
        benchmark.functional.forward.return_value = ("output", "context")
        benchmark.functional.backward.return_value = tuple(range(8)) + (("dw_gate" if gated else None),)
        benchmark.config = "config"
        benchmark.workspace = "workspace"
        benchmark.topk_experts = "routes"
        benchmark.num_local_experts = 8
        benchmark.x = "x"
        benchmark.router_weights = "p"
        benchmark.weights = ("shared_gate", "shared_up", "shared_down", "bf16_a", "bf16_u", "bf16_d")
        benchmark.quantized_routed = tuple(tuple(f"{name}{i}" for i in range(4)) for name in ("a", "u", "d"))
        benchmark.d_output = "dy"
        benchmark.main_grads = "main_grads"
        benchmark.gate_weight = "gate_weight" if gated else None
        benchmark.gate_main_grad = "gate_main_grad" if gated else None
        return benchmark

    def test_forward_normal_orientation_and_backward_mixed_orientation(self):
        benchmark = self.make_benchmark(True)
        self.assertEqual(benchmark.run_fwd(), ("output", ("schedule", "context")))
        call = benchmark.functional.forward.call_args
        self.assertEqual(call.args[5:8], benchmark.weights[:3])
        self.assertEqual(call.args[8:], (("a0", "a1"), ("u0", "u1"), ("d0", "d1")))
        self.assertEqual(call.kwargs, {"shared_output_gate_weight": "gate_weight"})
        benchmark.run_bwd(("schedule", "context"))
        call = benchmark.functional.backward.call_args
        self.assertEqual(call.args[7:10], benchmark.weights[:3])
        self.assertEqual(call.args[10:], (("a0", "a1", "a2", "a3"), ("u0", "u1", "u2", "u3"), ("d2", "d3")))
        self.assertEqual(call.kwargs, {
            "main_grads": "main_grads", "shared_output_gate_weight": "gate_weight",
            "shared_output_gate_main_grad": "gate_main_grad",
        })

    def test_ungated_omits_gate_keywords_and_accepts_old_eight_return_api(self):
        benchmark = self.make_benchmark(False)
        benchmark.run_fwd()
        self.assertEqual(benchmark.functional.forward.call_args.kwargs, {})
        for count in (8, 9):
            benchmark.functional.backward.return_value = tuple(range(8)) + (() if count == 8 else (None,))
            self.assertEqual(len(benchmark.run_bwd(("schedule", "context"))), count)
            self.assertEqual(benchmark.functional.backward.call_args.kwargs, {"main_grads": "main_grads"})

    def test_gated_rejects_missing_gradient(self):
        benchmark = self.make_benchmark(True)
        benchmark.functional.backward.return_value = tuple(range(8)) + (None,)
        with self.assertRaises(AssertionError):
            benchmark.run_bwd(("schedule", "context"))


if __name__ == "__main__":
    unittest.main()
