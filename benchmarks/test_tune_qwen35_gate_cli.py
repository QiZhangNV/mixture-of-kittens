"""Standard-library-only tests for candidate parsing and schedule accounting."""

from contextlib import redirect_stderr
import io
import unittest

from benchmarks.tune_qwen35_gate import normalize_configs, parse_args, schedule_counts


class GateTuneCLI(unittest.TestCase):
    def test_screen_defaults(self):
        args = parse_args(["--configs-json", "stage.json"])
        self.assertEqual((args.warmup_iters, args.timed_iters, args.phases), (100, 50, "bwd"))
        config = normalize_configs([{}])[0]
        self.assertEqual(config["name"], "candidate-000")
        self.assertEqual(config["config"], {
            "fwd_num_comm_sms": 48, "bwd_num_comm_sms": 56,
            "minibatch_size": 4096, "macrobatch_size": 65536,
            "schedule_capacity_multiplier": 0.0625,
        })

    def test_user_bf16_seeds(self):
        seeds = ((32, 40, 4096, 131072), (44, 52, 4096, 131072),
                 (48, 56, 4096, 65536), (48, 60, 8192, 131072))
        payload = [dict(zip(("fwd_num_comm_sms", "bwd_num_comm_sms", "minibatch_size", "macrobatch_size"), seed)) for seed in seeds]
        self.assertEqual(len(normalize_configs(payload)), 4)

    def test_invalid_candidates(self):
        for payload in ([], {}, ["bad"], [{"macro": 65536}],
                        [{"bwd_num_comm_sms": 41}], [{"minibatch_size": 300}],
                        [{"macrobatch_size": 70000}], [{"schedule_capacity_multiplier": 0.5}],
                        [{"name": "x"}, {"name": "x"}]):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                normalize_configs(payload)

    def test_actual_replay_counts_not_capacity_counts(self):
        single = schedule_counts(44544, 4096, 65536)
        self.assertEqual((single["num_macrobatches"], single["replay_minibatches"]), (1, 0))
        replay = schedule_counts(81920, 4096, 65536)
        self.assertEqual((replay["num_macrobatches"], replay["replay_macrobatches"], replay["replay_minibatches"]), (2, 1, 4))

    def test_invalid_iterations(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--configs-json", "stage.json", "--timed-iters", "0"])


if __name__ == "__main__":
    unittest.main()
