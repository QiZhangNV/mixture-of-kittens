"""Screen explicit gated-BF16 MOK configs; no production changes or auto-sweep.

configs.json is a nonempty list of objects, for example::

    [{"name": "control", "fwd_num_comm_sms": 48, "bwd_num_comm_sms": 56,
      "minibatch_size": 4096, "macrobatch_size": 65536}]

Within the existing OCI allocation, run from this MOK checkout::

    torchrun --standalone --nproc-per-node=4 -m benchmarks.tune_qwen35_gate \
        --configs-json configs.json --phases bwd

Every candidate must pass the full reference before it is timed. Defaults are
100 warmup / 50 timed iterations for screening, not the final 500/100 paired
measurement. The GPU reference remains live: do not treat this as a memory
benchmark. This fixed-route harness excludes router GEMM/top-k and CUDA graphs.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket

from benchmarks.bench_qwen35_baseline import BF16Benchmark, git_metadata


DEFAULT_CONFIG = {
    "fwd_num_comm_sms": 48,
    "bwd_num_comm_sms": 56,
    "minibatch_size": 4096,
    "macrobatch_size": 65536,
    "schedule_capacity_multiplier": 0.0625,
}
SHAPE = {"num_local_tokens": 4096, "hidden_dim": 4096, "intermediate_dim": 1024,
         "num_experts": 64, "topk": 10, "ep_size": 4}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-json", type=Path, required=True)
    parser.add_argument("--phases", choices=("bwd", "fwd", "both"), default="bwd")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--timed-iters", type=int, default=50)
    parser.add_argument("--stage", default="screen")
    args = parser.parse_args(argv)
    if args.warmup_iters < 0 or args.timed_iters <= 0:
        parser.error("warmup iterations must be nonnegative; timed iterations must be positive")
    return args


def normalize_configs(payload):
    if not isinstance(payload, list) or not payload:
        raise ValueError("configs JSON must be a nonempty list")
    configs = []
    names = set()
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"candidate {index} must be an object")
        unknown = set(entry) - set(DEFAULT_CONFIG) - {"name"}
        if unknown:
            raise ValueError(f"candidate {index}: unknown config keys {sorted(unknown)}")
        name = entry.get("name", f"candidate-{index:03d}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("candidate names must be nonempty, unique strings")
        names.add(name)
        config = {**DEFAULT_CONFIG, **{key: value for key, value in entry.items() if key != "name"}}
        for key in ("fwd_num_comm_sms", "bwd_num_comm_sms", "minibatch_size", "macrobatch_size"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"{name}: {key} must be a positive integer")
        if config["fwd_num_comm_sms"] % 2 or config["bwd_num_comm_sms"] % 2:
            raise ValueError(f"{name}: communication SM counts must be even")
        if config["minibatch_size"] % 256 or config["macrobatch_size"] % config["minibatch_size"]:
            raise ValueError(f"{name}: minibatch must align to 256 and divide macrobatch")
        if config["schedule_capacity_multiplier"] != 0.0625:
            raise ValueError(f"{name}: this sweep keeps schedule_capacity_multiplier=0.0625")
        configs.append({"name": name, "config": config})
    return configs


def schedule_counts(num_tokens, mini, macro):
    """Match backward.cuh's actual local schedule, not workspace capacity."""
    num_macrobatches = math.ceil(num_tokens / macro)
    num_minibatches = math.ceil(num_tokens / mini)
    saved_minibatches = math.ceil(min(num_tokens, macro) / mini)
    return {
        "num_tokens": num_tokens,
        "num_macrobatches": num_macrobatches,
        "num_minibatches": num_minibatches,
        "replay_macrobatches": max(num_macrobatches - 1, 0),
        "replay_minibatches": num_minibatches - saved_minibatches,
        "saved_macrobatch_tokens": min(num_tokens, macro),
    }


def main(argv=None):
    args = parse_args(argv)
    candidates = normalize_configs(json.loads(args.configs_json.read_text()))
    import torch
    import torch.distributed as dist
    from benchmarks import utils as benchmark_utils
    from mok import _C, functional
    from tests.utils import BF16_TOLERANCE, generate_inputs, run_reference_bf16

    repo = Path(__file__).resolve().parents[1]
    if Path(functional.__file__).resolve() != repo / "mok" / "functional.py":
        raise RuntimeError(f"Loaded a different MOK source: {functional.__file__}")
    benchmark_utils.WARMUP_ITERS = args.warmup_iters
    benchmark_utils.TIMED_ITERS = args.timed_iters
    rank, world_size, device = benchmark_utils.init_distributed()
    try:
        if world_size != SHAPE["ep_size"]:
            raise ValueError("This Qwen parameter screen requires exactly four EP ranks")
        local_hardware = {
            "rank": rank, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(device),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
        hardware = [None] * world_size
        dist.all_gather_object(hardware, local_hardware)
        if rank == 0:
            print(json.dumps({
                "event": "qwen35_gate_tune_config", "stage": args.stage, "shape": SHAPE,
                "phases": args.phases, "warmup_iters": args.warmup_iters, "timed_iters": args.timed_iters,
                "configs_file": str(args.configs_json.resolve()), "candidates": candidates,
                "dtype": "bfloat16", "routed_layout": "dense", "grad_mode": "fresh", "output_gate": True,
                "seed": "1234 + EP rank", "output_gate_weight_seed": "1919 + EP rank",
                "git": git_metadata(repo), "functional_source": functional.__file__,
                "extension_source": getattr(_C, "__file__", None),
                "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                "hardware": hardware,
                "environment": {key: os.environ.get(key) for key in (
                    "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "OMP_NUM_THREADS",
                    "PYTORCH_CUDA_ALLOC_CONF", "MOK_BENCHMARK_NUMA_BINDING",
                )},
                "correctness_tolerance": BF16_TOLERANCE,
                "timing": "CUDA events: median of per-iteration EP-rank maximum; fwd includes build_schedule; bwd excludes its setup forward; no router GEMM/top-k or CUDA graph",
                "memory": "not reported: one full GPU reference is retained throughout screening",
                "baseline_comparison": "screening only; final winner must be independently rerun with controls on the same nodes",
            }, sort_keys=True), flush=True)
        inputs = generate_inputs(
            rank, device, SHAPE["num_experts"], SHAPE["num_experts"] // world_size,
            SHAPE["topk"], SHAPE["num_local_tokens"], SHAPE["hidden_dim"], SHAPE["intermediate_dim"],
        )
        generator = torch.Generator(device=device).manual_seed(1919 + rank)
        gate_weight = torch.randn(
            1, SHAPE["hidden_dim"], generator=generator, device=device, dtype=torch.bfloat16,
        ) * SHAPE["hidden_dim"] ** -0.5
        reference = tuple(value.detach() for value in run_reference_bf16(
            *inputs, shared_output_gate_weight=gate_weight,
        ))
        phases = ("fwd", "bwd") if args.phases == "both" else (args.phases,)
        summary = []
        for index, candidate in enumerate(candidates):
            name, config_kwargs = candidate["name"], candidate["config"]
            if rank == 0:
                print(json.dumps({"event": "qwen35_gate_tune_candidate_start", "stage": args.stage,
                                  "index": index, "name": name, "config": config_kwargs}, sort_keys=True), flush=True)
            config = functional.MoKConfig(**config_kwargs)
            benchmark = BF16Benchmark(inputs, config, dist.group.WORLD, functional, "fresh", gate_weight)
            checked_schedule = []

            def correctness_forward():
                output, saved = benchmark.run_fwd()
                checked_schedule.append(saved[0])
                return output, saved

            try:
                benchmark_utils.check_benchmark_correctness(
                    f"{args.stage}/{name}", correctness_forward, benchmark.run_bwd,
                    reference, BF16_TOLERANCE, rank,
                )
            except Exception as error:
                if rank == 0:
                    print(json.dumps({"event": "qwen35_gate_tune_candidate_rejected", "name": name,
                                      "eligible": False, "error": str(error)}, sort_keys=True), flush=True)
                raise
            schedule = checked_schedule.pop()
            local_schedule = schedule_counts(
                int(schedule.num_tokens.item()), config.minibatch_size, config.macrobatch_size,
            )
            local_schedule.update({
                "rank": rank, "effective_schedule_capacity": benchmark.workspace.schedule_capacity,
                "tokens_per_expert": schedule.tokens_per_expert.tolist(),
            })
            rank_schedules = [None] * world_size
            dist.all_gather_object(rank_schedules, local_schedule)
            del schedule
            gc.collect()
            torch.cuda.synchronize(device)
            latencies = {}
            for phase in phases:
                if phase == "fwd":
                    ms = benchmark_utils.benchmark_fwd(benchmark.run_fwd, device)
                else:
                    ms = benchmark_utils.benchmark_bwd(benchmark.run_fwd, benchmark.run_bwd, device)
                latencies[phase + "_median_rank_max_ms"] = ms
            result = {
                "index": index, "name": name, "config": config_kwargs, "eligible": True,
                "correctness_passed": True, "schedule_by_rank": rank_schedules, **latencies,
            }
            summary.append(result)
            if rank == 0:
                print(json.dumps({"event": "qwen35_gate_tune_candidate_result", "stage": args.stage, **result}, sort_keys=True), flush=True)
            del benchmark
            gc.collect()
            torch.cuda.synchronize(device)
        if rank == 0:
            print(json.dumps({"event": "qwen35_gate_tune_summary", "stage": args.stage,
                              "results": summary}, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        functional.clear_workspace_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
