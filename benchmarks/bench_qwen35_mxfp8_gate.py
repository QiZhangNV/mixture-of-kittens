"""Dense MXFP8-routed/BF16-shared Qwen gate benchmark, without MCore.

Run each source checkout in its own process, on the same allocated GPUs::

    torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_qwen35_mxfp8_gate \
        --gate off --label MOK-old-U
    torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_qwen35_mxfp8_gate \
        --gate on --label MOK-new-G

The default EP4/E32 retains eight local experts and the original 397B runtime
parameters, but does NOT represent EP64 performance. Quantized routed weights
are cached before timing; activation quantization remains inside functional.
Fresh and FP32-main-grad results must be compared separately. No CUDA/Torch
imports occur until main() or benchmark construction; --help is CPU-safe.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket

from benchmarks.bench_qwen35_baseline import BF16Benchmark, git_metadata, measure_phase


RESULT_NAMES = (
    "output", "d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up",
    "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down",
    "d_w_shared_output_gate",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--gate", choices=("off", "on"), default="off")
    parser.add_argument("--num-local-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--intermediate-dim", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--fwd-comm-sms", type=int, default=48)
    parser.add_argument("--bwd-comm-sms", type=int, default=56)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--macrobatch-size", type=int, default=65536)
    parser.add_argument("--schedule-capacity-multiplier", type=float, default=0.5)
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
    for name in ("num_local_tokens", "hidden_dim", "intermediate_dim", "minibatch_size"):
        if getattr(args, name) % 256:
            parser.error(f"--{name.replace('_', '-')} must be divisible by 256")
    if not math.isfinite(args.schedule_capacity_multiplier) or args.schedule_capacity_multiplier <= 0:
        parser.error("--schedule-capacity-multiplier must be positive and finite")
    if args.label is None:
        args.label = "MOK-new-G" if args.gate == "on" else "MOK-new-U"
    return args


def sha256_file(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MXFP8GateBenchmark(BF16Benchmark):
    def __init__(self, inputs, config, group, functional, ops, grad_mode, gate_weight=None):
        super().__init__(inputs, config, group, functional, grad_mode, gate_weight)
        # Dense layout matches bench_mok.py: both quantization orientations are
        # required by backward. Shared weights and their gradients stay BF16
        # (or FP32 main-grad); quantization is exclusively for routed weights.
        self.quantized_routed = tuple(
            ops.mxfp8_quantize(weight, True, True) for weight in self.weights[3:]
        )

    def run_fwd(self):
        schedule = self.functional.build_schedule(
            self.workspace, self.config, self.topk_experts,
            num_local_experts=self.num_local_experts,
        )
        # Omit the new keyword entirely in U so the same harness can run an
        # older ungated checkout. Do not emulate U with an all-ones gate.
        kwargs = {} if self.gate_weight is None else {"shared_output_gate_weight": self.gate_weight}
        output, context = self.functional.forward(
            self.config, self.workspace, schedule, self.x, self.router_weights,
            *self.weights[:3], *(weight[:2] for weight in self.quantized_routed),
            **kwargs,
        )
        return output, (schedule, context)

    def run_bwd(self, saved):
        schedule, context = saved
        kwargs = {"main_grads": self.main_grads}
        if self.gate_weight is not None:
            kwargs.update(
                shared_output_gate_weight=self.gate_weight,
                shared_output_gate_main_grad=self.gate_main_grad,
            )
        gate, up, down = self.quantized_routed
        gradients = self.functional.backward(
            self.config, self.workspace, schedule, context,
            self.d_output, self.x, self.router_weights,
            *self.weights[:3], gate, up, down[2:], **kwargs,
        )
        if self.gate_weight is None:
            assert len(gradients) in (8, 9)
            if len(gradients) == 9:
                assert gradients[-1] is None
        else:
            assert len(gradients) == 9 and gradients[-1] is not None
        return gradients


def check_reference(benchmark, inputs, rank):
    """Full BF16 anchor plus independent, strict BF16 gate checks; untimed."""
    import torch

    from tests.utils import (
        BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, get_error_stats,
        run_reference_bf16, run_shared_output_gate_reference, run_swiglu_reference,
    )

    kwargs = {} if benchmark.gate_weight is None else {
        "shared_output_gate_weight": benchmark.gate_weight,
    }
    reference = run_reference_bf16(*inputs, **kwargs)
    benchmark.zero_main_grads()
    output, saved = benchmark.run_fwd()
    gradients = benchmark.run_bwd(saved)
    actual = (output, *gradients)
    metrics = {}
    for index, expected in enumerate(reference):
        # Shared has not changed precision. Do not give shared/gate wgrads
        # the looser MXFP8 tolerance used for routed-dependent quantities.
        tolerance = BF16_TOLERANCE if index >= 6 else MXFP8_TOLERANCE
        name = RESULT_NAMES[index]
        check_correctness(name, expected, actual[index], tolerance, print_stats=rank == 0)
        metrics[name] = dict(zip(
            ("mean_abs_error", "max_abs_error", "relative_l1_error"),
            get_error_stats(expected, actual[index]),
        ), tolerance=list(tolerance))
        assert torch.isfinite(actual[index]).all()

    if benchmark.main_grads is not None:
        # Functional return order differs from the six-buffer input ABI.
        for result, buffer_index in zip(gradients[2:8], (1, 3, 5, 0, 2, 4), strict=True):
            assert result is benchmark.main_grads[buffer_index]
            assert result.dtype == torch.float32

    gate_metrics = None
    if benchmark.gate_weight is not None:
        context = saved[1]
        x, _, _, w_a, w_u, w_d = inputs[:6]
        shared_reference = run_swiglu_reference(x @ w_a.T, x @ w_u.T) @ w_d.T
        check_correctness(
            "saved_shared_output", shared_reference, context.shared_output,
            BF16_TOLERANCE, print_stats=rank == 0,
        )
        # Isolate the gate derivative given the saved BF16 S (already checked
        # against an independent shared MLP above). The reference constructs
        # its own autograd graph and dZ; no DUT dG/dZ is used as a golden.
        gate, _, _, dz_reference, _, golden_dw = run_shared_output_gate_reference(
            x, context.shared_output, benchmark.gate_weight, benchmark.d_output,
        )
        assert context.shared_output.dtype == torch.bfloat16
        assert context.shared_output_gate.dtype == torch.bfloat16
        torch.testing.assert_close(context.shared_output_gate, gate, rtol=0, atol=0)
        gate_gradient = gradients[-1]
        assert gate_gradient.dtype == torch.float32
        if benchmark.gate_main_grad is not None:
            assert gate_gradient is benchmark.gate_main_grad
        # Same gamma_T bound as the existing production gate-wgrad test.
        # BF16 products here are exactly representable before FP32 summation.
        unit_roundoff = torch.finfo(torch.float32).eps / 2
        gamma_t = x.shape[0] * unit_roundoff / (1 - x.shape[0] * unit_roundoff)
        bound = gamma_t * (dz_reference.double().abs().T @ x.double().abs())
        error = (gate_gradient.double() - golden_dw).abs()
        assert torch.isfinite(bound).all() and torch.all(error <= bound)
        gate_metrics = {
            "oracle": "independent autograd BF16 gate derivative on verified saved S; FP64 dZ_ref.T @ X",
            "bound": "gamma_T * sum(abs(dZ_ref * X)), elementwise",
            "max_abs_error": error.max().item(),
            "max_bound": bound.max().item(),
        }

    # Stable bytes permit exact old-U/new-U checks outside this invocation.
    # All copies and hashing occur before either timed phase.
    fingerprints = {}
    for name, tensor in zip(RESULT_NAMES, actual):
        fingerprints[name] = None if tensor is None else {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest(),
        }
    return {
        "rank": rank, "metrics": metrics,
        "metrics_scope": "EP-global reductions; gate_fp64 and fingerprints are rank-local",
        "gate_fp64": gate_metrics, "fingerprints": fingerprints,
    }


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torch.distributed as dist

    from benchmarks import utils as benchmark_utils
    from mok import _C, functional, ops
    from tests.utils import generate_inputs

    repo = Path(__file__).resolve().parents[1]
    if Path(functional.__file__).resolve() != repo / "mok" / "functional.py":
        raise RuntimeError(f"Expected this checkout's MOK source, loaded {functional.__file__}")
    benchmark_utils.WARMUP_ITERS = args.warmup_iters
    benchmark_utils.TIMED_ITERS = args.timed_iters
    rank, world_size, device = benchmark_utils.init_distributed()
    try:
        if world_size != args.ep_size:
            raise ValueError(f"WORLD_SIZE={world_size} does not match --ep-size={args.ep_size}")
        config = functional.MoKConfig(
            fwd_num_comm_sms=args.fwd_comm_sms, bwd_num_comm_sms=args.bwd_comm_sms,
            minibatch_size=args.minibatch_size, macrobatch_size=args.macrobatch_size,
            schedule_capacity_multiplier=args.schedule_capacity_multiplier,
        )
        hardware = [None] * world_size
        dist.all_gather_object(hardware, {
            "rank": rank, "host": socket.gethostname(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        })
        local_experts = args.num_experts // world_size
        if rank == 0:
            print(json.dumps({
                "event": "qwen35_mxfp8_gate_config", "config": vars(args),
                "routed_precision": "mxfp8", "shared_and_output_gate_precision": "bfloat16",
                "routed_layout": "dense", "local_experts": local_experts,
                "scope": "standalone MOK, actual EP shown in config; not MCore/E2E or EP64 emulation",
                "seed": "1234 + EP rank (generate_inputs)",
                "output_gate_weight_seed": "1919 + EP rank" if args.gate == "on" else None,
                "git": git_metadata(repo), "functional_source": functional.__file__,
                "extension_source": getattr(_C, "__file__", None),
                "extension_sha256": sha256_file(getattr(_C, "__file__", None)),
                "functional_sha256": sha256_file(functional.__file__),
                "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                "hardware": hardware,
                "environment": {name: os.environ.get(name) for name in (
                    "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "OMP_NUM_THREADS",
                    "PYTORCH_CUDA_ALLOC_CONF", "MOK_BENCHMARK_NUMA_BINDING",
                )},
                "reference": "BF16 anchor with existing MXFP8 tolerance; shared/gate BF16 tolerance; separate strict gate FP64 oracle",
                "timing": {
                    "method": "CUDA events, median of per-iteration maximum across EP ranks",
                    "fwd_scope": "build_schedule + functional.forward, including activation quantization, no router",
                    "bwd_scope": "functional.backward; fresh forward/context outside bwd events",
                    "routed_weight_quantization": "both orientations cached once before timing; optimizer/weight update excluded",
                    "main_grad_zeroing": "once per phase outside timing; warmup and timed calls accumulate",
                    "cuda_graph": False,
                },
                "memory_scope": "PyTorch peak over whole phase including warmup and bwd setup forward; excludes external CUDA allocations",
            }, sort_keys=True), flush=True)

        inputs = generate_inputs(
            rank, device, args.num_experts, local_experts, args.topk,
            args.num_local_tokens, args.hidden_dim, args.intermediate_dim,
        )
        gate_weight = None
        if args.gate == "on":
            generator = torch.Generator(device=device).manual_seed(1919 + rank)
            gate_weight = torch.randn(
                1, args.hidden_dim, generator=generator, device=device, dtype=torch.bfloat16,
            ) * args.hidden_dim ** -0.5
        benchmark = MXFP8GateBenchmark(
            inputs, config, dist.group.WORLD, functional, ops, args.grad_mode, gate_weight,
        )
        if not args.skip_correctness:
            local_checks = check_reference(benchmark, inputs, rank)
            checks = [None] * world_size
            dist.all_gather_object(checks, local_checks)
            if rank == 0:
                print(json.dumps({
                    "event": "qwen35_mxfp8_gate_correctness", "label": args.label,
                    "grad_mode": args.grad_mode, "status": "passed", "ranks": checks,
                }, sort_keys=True), flush=True)

        results = {}
        for phase in ("fwd", "bwd"):
            results[phase] = measure_phase(phase, benchmark, benchmark_utils, torch, dist, device)
            if rank == 0:
                print(json.dumps({
                    "event": "qwen35_mxfp8_gate_phase", "label": args.label,
                    "grad_mode": args.grad_mode, "phase": phase, **results[phase],
                }, sort_keys=True), flush=True)
        if rank == 0:
            print(json.dumps({
                "event": "qwen35_mxfp8_gate_result", "label": args.label,
                "grad_mode": args.grad_mode, "correctness_checked": not args.skip_correctness,
                "results": results,
            }, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        functional.clear_workspace_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
