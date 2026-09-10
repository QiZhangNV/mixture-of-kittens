"""Standard-library-only checks; safe to run on a login node with unittest."""

from contextlib import redirect_stderr
import io
import unittest

from benchmarks.bench_qwen35_baseline import parse_args


class BaselineCLI(unittest.TestCase):
    def test_defaults_match_qwen_proxy_dense_fresh(self):
        args = parse_args([])
        self.assertEqual(
            (args.num_local_tokens, args.hidden_dim, args.intermediate_dim,
             args.num_experts, args.topk, args.ep_size),
            (4096, 4096, 1024, 64, 10, 4),
        )
        self.assertEqual(args.grad_mode, "fresh")
        self.assertEqual(
            (args.fwd_comm_sms, args.bwd_comm_sms, args.minibatch_size,
             args.macrobatch_size, args.schedule_capacity_multiplier),
            (48, 56, 4096, 65536, 0.0625),
        )
        self.assertFalse(args.skip_correctness)

    def test_short_run_and_accumulation_are_explicit(self):
        args = parse_args([
            "--warmup-iters", "3", "--timed-iters", "5",
            "--grad-mode", "fp32-main-grad", "--label", "MOK-old-U-accum",
        ])
        self.assertEqual((args.warmup_iters, args.timed_iters), (3, 5))
        self.assertEqual(args.grad_mode, "fp32-main-grad")
        self.assertEqual(args.label, "MOK-old-U-accum")

    def test_invalid_counts_and_modes_are_rejected(self):
        invalid = (
            ["--timed-iters", "0"], ["--warmup-iters", "-1"],
            ["--num-experts", "63"], ["--topk", "65"],
            ["--macrobatch-size", "5000"],
            ["--schedule-capacity-multiplier", "nan"],
            ["--grad-mode", "split"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(argv)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
