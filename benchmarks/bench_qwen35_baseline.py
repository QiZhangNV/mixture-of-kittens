"""Ungated BF16 Qwen shape baseline; does not require the output-gate feature.

Run from the MOK checkout, on four GPUs in the existing OCI allocation::

    torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_qwen35_baseline

The default is the existing dense/fresh-gradient path. Use a separate invocation
with ``--grad-mode fp32-main-grad`` for accumulation; do not mix those results.
CLI parsing and ``--help`` only use the standard library (no CUDA imports).
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import socket
import subprocess


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="MOK-old-U")
    parser.add_argument("--num-local-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--intermediate-dim", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--fwd-comm-sms", type=int, default=48)
    parser.add_argument("--bwd-comm-sms", type=int, default=56)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--macrobatch-size", type=int, default=65536)
    parser.add_argument("--schedule-capacity-multiplier", type=float, default=0.0625)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--grad-mode", choices=("fresh", "fp32-main-grad"), default="fresh")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "num_local_tokens", "hidden_dim", "intermediate_dim", "num_experts",
        "topk", "ep_size", "fwd_comm_sms", "bwd_comm_sms", "minibatch_size",
        "macrobatch_size", "timed_iters",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be nonnegative")
    if args.topk > args.num_experts:
        parser.error("--topk cannot exceed --num-experts")
    if args.num_experts % args.ep_size:
        parser.error("--num-experts must be divisible by --ep-size")
    if args.macrobatch_size % args.minibatch_size:
        parser.error("--macrobatch-size must be divisible by --minibatch-size")
    if not math.isfinite(args.schedule_capacity_multiplier) or args.schedule_capacity_multiplier <= 0:
        parser.error("--schedule-capacity-multiplier must be positive and finite")
    return args


def git_metadata(repo):
    def read(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *arguments],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "root": str(repo),
        "commit": read("rev-parse", "HEAD"),
        "branch": read("branch", "--show-current"),
        "worktree_status": read("status", "--short"),
        "submodules": read("submodule", "status", "--recursive"),
    }


class UngatedBF16Benchmark:
    def __init__(self, inputs, config, group, functional, grad_mode):
        import torch

        self.functional = functional
        self.config = config
        self.x, self.topk_experts, self.router_weights = inputs[:3]
        self.weights = inputs[3:9]  # shared gate/up/down, routed gate/up/down
        self.d_output = inputs[9]
        self.num_local_experts = self.weights[3].shape[0]
        self.workspace = functional.get_workspace(
            config, group, device=self.x.device,
            num_local_tokens=self.x.shape[0], hidden_size=self.x.shape[1],
            topk=self.topk_experts.shape[1],
        )
        self.main_grads = None
        if grad_mode == "fp32-main-grad":
            # Existing MOK ABI: shared gate, routed gate, shared up, routed up,
            # shared down, routed down. This is not the return-value order.
            self.main_grads = tuple(
                torch.zeros_like(self.weights[i], dtype=torch.float32)
                for i in (0, 3, 1, 4, 2, 5)
            )

    def zero_main_grads(self):
        if self.main_grads is not None:
            for gradient in self.main_grads:
                gradient.zero_()

    def run_fwd(self):
        schedule = self.functional.build_schedule(
            self.workspace, self.config, self.topk_experts,
            num_local_experts=self.num_local_experts,
        )
        output, context = self.functional.forward(
            self.config, self.workspace, schedule,
            self.x, self.router_weights, *self.weights,
        )
        return output, (schedule, context)

    def run_bwd(self, saved):
        schedule, context = saved
        return self.functional.backward(
            self.config, self.workspace, schedule, context,
            self.d_output, self.x, self.router_weights, *self.weights,
            main_grads=self.main_grads,
        )


def measure_phase(phase, benchmark, benchmark_utils, torch, dist, device):
    # Reference tensors/cached allocator blocks must not inflate either phase.
    # Main-grad zeroing is outside the timed region; iterations then accumulate.
    benchmark.zero_main_grads()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.synchronize(device)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    if phase == "fwd":
        latency_ms = benchmark_utils.benchmark_fwd(benchmark.run_fwd, device)
    else:
        latency_ms = benchmark_utils.benchmark_bwd(benchmark.run_fwd, benchmark.run_bwd, device)
    local_memory = {
        "rank": dist.get_rank(),
        "baseline_allocated_bytes": before_allocated,
        "baseline_reserved_bytes": before_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    local_memory["extra_peak_allocated_bytes"] = local_memory["peak_allocated_bytes"] - before_allocated
    rank_memory = [None] * dist.get_world_size()
    dist.all_gather_object(rank_memory, local_memory)
    return {
        "median_rank_max_ms": latency_ms,
        "memory_by_rank": rank_memory,
        "rank_max_peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in rank_memory),
        "rank_max_peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in rank_memory),
        "rank_max_extra_peak_allocated_bytes": max(m["extra_peak_allocated_bytes"] for m in rank_memory),
    }


def main(argv=None):
    args = parse_args(argv)
    # Keep the CLI usable without importing CUDA libraries on a login node.
    import torch
    import torch.distributed as dist

    from benchmarks import utils as benchmark_utils
    from mok import _C, functional
    from tests.utils import BF16_TOLERANCE, generate_inputs, run_reference_bf16

    repo = Path(__file__).resolve().parents[1]
    if Path(functional.__file__).resolve() != repo / "mok" / "functional.py":
        raise RuntimeError(f"Expected this checkout's MOK source, loaded {functional.__file__}")
    benchmark_utils.WARMUP_ITERS = args.warmup_iters
    benchmark_utils.TIMED_ITERS = args.timed_iters
    rank, world_size, device = benchmark_utils.init_distributed()
    try:
        if world_size != args.ep_size:
            raise ValueError(f"WORLD_SIZE={world_size} does not match --ep-size={args.ep_size}")
        local_experts = benchmark_utils.get_num_local_experts(args.num_experts, world_size)
        config = functional.MoKConfig(
            fwd_num_comm_sms=args.fwd_comm_sms, bwd_num_comm_sms=args.bwd_comm_sms,
            minibatch_size=args.minibatch_size, macrobatch_size=args.macrobatch_size,
            schedule_capacity_multiplier=args.schedule_capacity_multiplier,
        )
        local_hardware = {
            "rank": rank, "host": socket.gethostname(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
        hardware = [None] * world_size
        dist.all_gather_object(hardware, local_hardware)
        if rank == 0:
            print(json.dumps({
                "event": "qwen35_baseline_config", "config": vars(args),
                "dtype": "bfloat16", "routed_layout": "dense", "output_gate": False,
                "local_experts": local_experts, "seed": "1234 + EP rank (generate_inputs)",
                "git": git_metadata(repo), "functional_source": functional.__file__,
                "extension_source": getattr(_C, "__file__", None),
                "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "OMP_NUM_THREADS",
                        "PYTORCH_CUDA_ALLOC_CONF", "MOK_BENCHMARK_NUMA_BINDING",
                    )
                },
                "hardware": hardware,
                "timing": {
                    "method": "CUDA events, median of per-iteration maximum across EP ranks",
                    "fwd_scope": "build_schedule + functional.forward, fixed input routes (no router)",
                    "bwd_scope": "functional.backward; fresh forward/context outside bwd events",
                    "main_grad_zeroing": "once per phase outside timing; warmup and timed calls accumulate",
                    "cuda_graph": False,
                },
                "memory_scope": "PyTorch allocator peak during each whole phase, including warmup and bwd setup forward; excludes external CUDA allocations",
            }, sort_keys=True), flush=True)

        inputs = generate_inputs(
            rank, device, args.num_experts, local_experts, args.topk,
            args.num_local_tokens, args.hidden_dim, args.intermediate_dim,
        )
        benchmark = UngatedBF16Benchmark(inputs, config, dist.group.WORLD, functional, args.grad_mode)
        if not args.skip_correctness:
            reference = run_reference_bf16(*inputs)
            benchmark.zero_main_grads()
            benchmark_utils.check_benchmark_correctness(
                f"{args.label}/{args.grad_mode}", benchmark.run_fwd, benchmark.run_bwd,
                reference, BF16_TOLERANCE, rank,
            )
            del reference
            if rank == 0:
                print(json.dumps({"event": "qwen35_baseline_correctness", "status": "passed"}), flush=True)

        results = {}
        for phase in ("fwd", "bwd"):
            results[phase] = measure_phase(phase, benchmark, benchmark_utils, torch, dist, device)
            results[phase]["tflops_per_gpu"] = benchmark_utils.get_tflops(
                results[phase]["median_rank_max_ms"], args.num_local_tokens,
                args.topk, args.hidden_dim, args.intermediate_dim, backward=phase == "bwd",
            )
            if rank == 0:
                print(json.dumps({
                    "event": "qwen35_baseline_phase", "label": args.label,
                    "grad_mode": args.grad_mode, "phase": phase, **results[phase],
                }, sort_keys=True), flush=True)
        if rank == 0:
            print(json.dumps({
                "event": "qwen35_baseline_result", "label": args.label,
                "grad_mode": args.grad_mode, "correctness_checked": not args.skip_correctness,
                "results": results,
            }, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        functional.clear_workspace_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
