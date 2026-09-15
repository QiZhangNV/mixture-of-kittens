"""Native-U/Native-G: actual MCore MoELayer with fixed routes and no op fuser.

Run in the OCI allocation, from this MOK checkout::

    torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_qwen35_native \
        --mcore-repo /path/to/Megatron-LM --warmup-iters 3 --timed-iters 5

This is a fixed-route, fresh-gradient layer baseline, not a router-inclusive
training benchmark. Native and MOK references are checked separately because
their probability application and BF16 rounding boundaries differ.
Use --precision mxfp8 for MXFP8 routed GEMMs with BF16 shared/gate. Its BF16
reference is an accuracy anchor, not a quantization-exact MXFP8 oracle.
"""

import argparse
from contextlib import contextmanager
import gc
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import types

from benchmarks.bench_qwen35_baseline import git_metadata, measure_phase


RESULT_NAMES = (
    "output", "d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up",
    "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down",
    "d_w_shared_output_gate",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcore-repo", type=Path, default=Path(__file__).resolve().parents[3] / "vendor" / "Megatron-LM")
    parser.add_argument("--gate", choices=("both", "off", "on"), default="both")
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    parser.add_argument(
        "--gemm-backend", choices=("auto", "device-init", "te-cublas-grouped", "multistream"), default="auto",
        help="device-init: MCore public GroupedTensor API (requires TE use_grouped_tensor); "
             "te-cublas-grouped: ordinary GroupedLinear with TE internal cuBLAS grouped switch; "
             "multistream: explicit legacy path; auto: public API when available, otherwise multistream",
    )
    parser.add_argument("--num-local-tokens", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--intermediate-dim", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--hybridep-num-sms", type=int, default=32)
    parser.add_argument("--hybridep-preprocess-sms", type=int, default=108)
    parser.add_argument("--rank-capacity-factor", type=float, default=1.25)
    parser.add_argument("--hybridep-fuse-permute", action="store_true", default=False)
    parser.add_argument("--no-hybridep-fuse-permute", action="store_false", dest="hybridep_fuse_permute")
    parser.add_argument("--bias-activation-fusion", action="store_true", default=True)
    parser.add_argument("--no-bias-activation-fusion", action="store_false", dest="bias_activation_fusion")
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--profile-backend", action="store_true", help="outside timing, record one warmed-up forward/backward CUDA kernel-name list")
    args = parser.parse_args(argv)
    for name in (
        "num_local_tokens", "hidden_dim", "intermediate_dim", "num_experts", "topk",
        "ep_size", "hybridep_num_sms", "hybridep_preprocess_sms", "timed_iters",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be nonnegative")
    if args.num_experts % args.ep_size or args.topk > args.num_experts:
        parser.error("experts must divide EP and top-k cannot exceed experts")
    if not math.isfinite(args.rank_capacity_factor) or args.rank_capacity_factor <= 0:
        parser.error("--rank-capacity-factor must be positive and finite")
    return args


def shared_bf16_quant_recipe():
    """MCore's public per-module override; no monkey-patched shared forward."""
    return {
        "configs": {
            "shared_bf16": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {
                    "fp8_quantization_recipe": None,
                    "fp4_quantization_recipe": None,
                    "override_quantized_autocast": True,
                },
            },
        },
        "matchers": {
            "shared": {
                "config": "shared_bf16", "type": "glob",
                "pattern": "*.shared_experts.*", "enabled": True,
            },
        },
    }


def comparison_tolerance(tensor_name, precision, bf16_tolerance, mxfp8_tolerance):
    # Shared/gate were never quantized. Do not relax their checks to 10%.
    return (
        mxfp8_tolerance
        if precision == "mxfp8" and not tensor_name.startswith("d_w_shared_")
        else bf16_tolerance
    )


def select_grouped_tensor(requested, supports_grouped_tensor):
    if requested == "device-init" and not supports_grouped_tensor:
        raise RuntimeError(
            "Device-init requested for the MCore public GroupedTensor API, but installed TE "
            "GroupedLinear has no use_grouped_tensor constructor argument. This does not rule "
            "out TE's internal cuBLAS grouped path: try --gemm-backend te-cublas-grouped, "
            "or explicitly choose --gemm-backend multistream."
        )
    enabled = requested not in ("multistream", "te-cublas-grouped") and supports_grouped_tensor
    fallback = None
    if requested == "auto" and not enabled:
        fallback = "TE GroupedLinear lacks use_grouped_tensor; explicitly recorded multistream API fallback"
    return enabled, fallback


@contextmanager
def count_grouped_gemm_calls(grouped_linear):
    """Temporary Python-entrypoint probe, restored before any timed calls."""
    counts = {"grouped_tensor": 0, "legacy": 0}
    originals = {}
    try:
        for name, key in (
            ("general_grouped_gemm_for_grouped_tensor", "grouped_tensor"),
            ("general_grouped_gemm", "legacy"),
        ):
            function = getattr(grouped_linear, name, None)
            if callable(function):
                originals[name] = function

                def wrapper(*args, _key=key, _function=function, **kwargs):
                    counts[_key] += 1
                    return _function(*args, **kwargs)

                setattr(grouped_linear, name, wrapper)
        yield counts
    finally:
        for name, function in originals.items():
            setattr(grouped_linear, name, function)


def build_layer(args, world_size, gated, use_grouped_tensor):
    import torch
    import torch.nn.functional as F
    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_submodules
    from megatron.core.quantization.quant_config import RecipeConfig
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Existing real-DDP harnesses reuse this builder with a BF16-only namespace.
    precision = getattr(args, "precision", "bf16")
    config = TransformerConfig(
        num_layers=1, hidden_size=args.hidden_dim, num_attention_heads=8,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=1, expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1, sequence_parallel=False,
        num_moe_experts=args.num_experts, moe_ffn_hidden_size=args.intermediate_dim,
        moe_router_topk=args.topk, moe_router_score_function="softmax",
        moe_router_dtype="fp32", moe_router_load_balancing_type="none", moe_aux_loss_coeff=0.0,
        moe_router_fusion=False, moe_router_padding_for_quantization=False,
        moe_grouped_gemm=True, moe_use_grouped_tensor=use_grouped_tensor,
        moe_single_grouped_weight=False,
        moe_shared_expert_intermediate_size=args.intermediate_dim,
        moe_shared_expert_overlap=False, moe_shared_expert_gate=gated,
        use_grouped_gemm_for_shared_expert=False,
        gated_linear_unit=True, activation_func=F.silu, add_bias_linear=False,
        bias_activation_fusion=args.bias_activation_fusion, use_te_activation_func=False,
        params_dtype=torch.bfloat16, bf16=True,
        fp8="e4m3" if precision == "mxfp8" else None,
        fp8_recipe="mxfp8", fp8_param=False, fp8_wgrad=True,
        quant_recipe=(
            RecipeConfig.from_config_dict(shared_bf16_quant_recipe())
            if precision == "mxfp8" else None
        ),
        # Native initializes/caches quantized routed weights on its first untimed
        # correctness forward. No weight update happens during this benchmark.
        disable_parameter_transpose_cache=False,
        gradient_accumulation_fusion=False, use_cpu_initialization=False,
        use_transformer_engine_op_fuser=False,
        moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep",
        moe_flex_dispatcher_num_sms=args.hybridep_num_sms,
        moe_hybridep_num_sms_preprocessing=args.hybridep_preprocess_sms,
        moe_permute_fusion=not args.hybridep_fuse_permute,
        moe_permute_fusion_into_hybridep=args.hybridep_fuse_permute,
        # Non-grouped-tensor/non-fuser HybridEP only supports dynamic capacity.
        moe_expert_rank_capacity_factor=args.rank_capacity_factor if use_grouped_tensor else None,
        moe_hybridep_routing_map_mode="bool",
    )
    submodules = get_submodules(get_gpt_layer_with_transformer_engine_submodules(
        num_experts=args.num_experts, moe_grouped_gemm=True,
    ).mlp)
    with get_fp8_context(config, is_init=True):
        layer = MoELayer(config=config, submodules=submodules, name="qwen35_native_baseline")
    layer = layer.cuda().to(dtype=torch.bfloat16).train()
    if config.use_transformer_engine_op_fuser or getattr(layer.experts, "_with_fused_impl", False):
        raise RuntimeError("Native baseline must not activate the TE op-fuser path")
    if bool(getattr(layer.experts, "_use_grouped_tensor", False)) != use_grouped_tensor:
        raise RuntimeError("Constructed expert module did not select the requested grouped-tensor path")
    return layer


class NativeBenchmark:
    def __init__(self, args, inputs, world_size, gate_weight, use_grouped_tensor):
        import torch

        x, self.experts, probs, *rest = inputs
        shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down, self.dy = rest
        self.args = args
        self.gated = gate_weight is not None
        self.layer = build_layer(args, world_size, self.gated, use_grouped_tensor)
        self.x = x.detach().unsqueeze(1).requires_grad_()
        self.probs = probs.detach().requires_grad_()
        if self.probs.dtype != torch.float32:
            raise TypeError("Fixed routing probabilities must remain FP32 for HybridEP")
        self.local_experts = routed_gate.shape[0]
        self.routed_fc1 = tuple(getattr(self.layer.experts.linear_fc1, f"weight{i}") for i in range(self.local_experts))
        self.routed_fc2 = tuple(getattr(self.layer.experts.linear_fc2, f"weight{i}") for i in range(self.local_experts))
        self.shared_fc1 = self.layer.shared_experts.linear_fc1.weight
        self.shared_fc2 = self.layer.shared_experts.linear_fc2.weight
        self.gate_parameter = self.layer.shared_experts.gate_weight if self.gated else None
        self.grad_inputs = (
            self.x, self.probs, *self.routed_fc1, *self.routed_fc2,
            self.shared_fc1, self.shared_fc2,
        ) + ((self.gate_parameter,) if self.gated else ())
        with torch.no_grad():
            self.shared_fc1.copy_(torch.cat((shared_gate, shared_up), dim=0))
            self.shared_fc2.copy_(shared_down)
            for i, (fc1, fc2) in enumerate(zip(self.routed_fc1, self.routed_fc2, strict=True)):
                fc1.copy_(torch.cat((routed_gate[i], routed_up[i]), dim=0))
                fc2.copy_(routed_down[i])
            if self.gated:
                self.gate_parameter.copy_(gate_weight)
        self.routing_map = torch.zeros(
            args.num_local_tokens, args.num_experts, device=x.device, dtype=torch.bool,
        ).scatter_(1, self.experts, True)

        def fixed_route(_layer, _hidden_states, *unused_args, **unused_kwargs):
            # Rebuild this differentiable scatter each forward. Do not retain a
            # stale graph between iterations; no router GEMM/top-k is timed.
            dense_probs = self.probs.new_zeros(args.num_local_tokens, args.num_experts)
            return dense_probs.scatter(1, self.experts, self.probs), self.routing_map

        self.layer.route = types.MethodType(fixed_route, self.layer)

    def zero_main_grads(self):
        # Shared timing helper interface; all gradients are fresh autograd outputs.
        pass

    def run_fwd(self):
        from megatron.core.fp8_utils import get_fp8_context

        with get_fp8_context(self.layer.config):
            output, bias = self.layer(self.x)
        if bias is not None:
            raise RuntimeError("The bias-free Native baseline unexpectedly returned a bias")
        output = output.squeeze(1)
        return output, output

    def run_bwd(self, output):
        import torch
        from megatron.core.fp8_utils import get_fp8_context

        with get_fp8_context(self.layer.config):
            return torch.autograd.grad(output, self.grad_inputs, self.dy)

    def format_gradients(self, gradients):
        import torch

        count = self.local_experts
        d_fc1 = torch.stack(gradients[2:2 + count])
        d_gate, d_up = d_fc1.split(self.args.intermediate_dim, dim=1)
        shared_index = 2 + 2 * count
        d_shared_gate, d_shared_up = gradients[shared_index].split(self.args.intermediate_dim, dim=0)
        result = (
            gradients[0].squeeze(1), gradients[1], d_gate, d_up,
            torch.stack(gradients[2 + count:shared_index]),
            d_shared_gate, d_shared_up, gradients[shared_index + 1],
        )
        return result + ((gradients[shared_index + 2],) if self.gated else ())

    def check_overflow(self):
        import torch
        import torch.distributed as dist

        overflow = self.layer.token_dispatcher.check_over_budget().to(dtype=torch.int32)
        dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
        if overflow.item():
            raise RuntimeError("HybridEP capacity overflow dropped tokens; result is invalid")


def module_evidence(layer, te, selected):
    modules = {}
    for name, module in (("experts", layer.experts), ("fc1", layer.experts.linear_fc1), ("fc2", layer.experts.linear_fc2)):
        modules[name] = {
            "class": f"{type(module).__module__}.{type(module).__name__}",
            "source": inspect.getsourcefile(type(module)),
            "flags": {
                attr: getattr(module, attr)
                for attr in ("_use_grouped_tensor", "use_grouped_tensor", "_with_fused_impl", "single_grouped_weight")
                if isinstance(getattr(module, attr, None), (bool, int, str))
            },
        }
    return {
        "selected_api_path": "MCore explicit grouped-tensor API" if selected else "ordinary TE GroupedLinear; MCore grouped-tensor flag off",
        "te_internal_grouped_switch": os.environ.get("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"),
        "op_fuser": False,
        "kernel_backend": "unverified: API selection alone does not establish cuBLAS vs another kernel backend",
        "te_version": te.__version__, "te_source": te.__file__,
        "te_grouped_linear_source": inspect.getsourcefile(te.pytorch.GroupedLinear),
        "te_grouped_linear_signature": str(inspect.signature(te.pytorch.GroupedLinear.__init__)),
        "modules": modules,
    }


def precision_evidence(layer, precision):
    """Inspect after real forward/backward, before publishing any timing."""
    result = {}
    for name, module in (
        ("routed_fc1", layer.experts.linear_fc1),
        ("routed_fc2", layer.experts.linear_fc2),
        ("shared_fc1", layer.shared_experts.linear_fc1),
        ("shared_fc2", layer.shared_experts.linear_fc2),
    ):
        routed = name.startswith("routed_")
        fp8 = getattr(module, "fp8", None)
        recipe = getattr(module, "fp8_meta", {}).get("recipe")
        is_mxfp8 = callable(getattr(recipe, "mxfp8", None)) and recipe.mxfp8()
        expected_fp8 = precision == "mxfp8" and routed
        if fp8 is None or bool(fp8) != expected_fp8:
            raise RuntimeError(f"{name}: expected fp8={expected_fp8}, got {fp8}")
        if expected_fp8 and not is_mxfp8:
            raise RuntimeError(f"{name}: expected actual MXFP8 recipe, got {recipe}")
        if getattr(module, "disable_parameter_transpose_cache", True):
            raise RuntimeError(f"{name}: weight-cache policy unexpectedly disabled")
        if getattr(module, "is_first_microbatch", True):
            raise RuntimeError(f"{name}: first untimed forward did not initialize weight cache")
        result[name] = {
            "fp8": bool(fp8), "recipe": str(recipe), "recipe_is_mxfp8": bool(is_mxfp8),
            "parameter_dtypes": sorted({str(p.dtype) for p in module.parameters()}),
            "is_first_microbatch": module.is_first_microbatch,
            "disable_parameter_transpose_cache": module.disable_parameter_transpose_cache,
            "module_quant_override": repr(getattr(module, "te_quant_params", None)),
        }
    gate = layer.shared_experts.gate_weight
    result["shared_output_gate"] = {
        "implementation": "MCore Native Torch linear/sigmoid/multiply; no TE quantization",
        "parameter_dtype": str(gate.dtype) if gate is not None else None,
    }
    return result


def compare_results(actual, reference, name, rank, precision="bf16"):
    from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, get_error_stats

    stats = {}
    passed = True
    for tensor_name, value, golden in zip(RESULT_NAMES, actual, reference, strict=False):
        if value.shape != golden.shape:
            raise AssertionError(f"{tensor_name}: shape mismatch {value.shape} vs {golden.shape}")
        mean, maximum, relative = get_error_stats(golden, value)
        tolerance = comparison_tolerance(tensor_name, precision, BF16_TOLERANCE, MXFP8_TOLERANCE)
        ok = all(math.isfinite(x) for x in (mean, maximum, relative)) and maximum <= tolerance[0] and relative <= tolerance[1]
        stats[tensor_name] = {"absolute_mean": mean, "absolute_max": maximum, "relative_l1": relative, "passed": ok, "actual_dtype": str(value.dtype), "reference_dtype": str(golden.dtype), "tolerance": tolerance}
        passed &= ok
    if len(actual) != len(reference):
        raise AssertionError("Native/reference output lengths differ")
    if rank == 0:
        print(json.dumps({"event": "qwen35_native_correctness", "comparison": name, "precision": precision, "reference_scope": "BF16 semantic anchor; not quantization-exact MXFP8 oracle", "passed": passed, "tensors": stats}, sort_keys=True), flush=True)
    return passed


def profile_backend(benchmark, torch):
    output, context = benchmark.run_fwd()
    grads = benchmark.run_bwd(context)
    del output, context, grads
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        output, context = benchmark.run_fwd()
        grads = benchmark.run_bwd(context)
        torch.cuda.synchronize()
    events = profile.events()
    names = sorted({event.name for event in events if str(event.device_type).endswith("CUDA")})
    cpu_names = sorted({
        event.name for event in events
        if str(event.device_type).endswith("CPU")
        and any(token in event.name.lower() for token in ("grouped", "gemm", "cublas", "linear"))
    })
    del output, context, grads
    return {"cuda_kernel_names": names, "cpu_grouped_op_names": cpu_names}


def main(argv=None):
    args = parse_args(argv)
    te_internal_grouped = args.gemm_backend == "te-cublas-grouped"
    # This internal TE path is distinct from MCore's grouped-tensor API flag.
    # Set explicitly before TE import; multistream must not inherit env=1.
    os.environ["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "1" if te_internal_grouped else "0"
    mcore_repo = args.mcore_repo.resolve()
    if not (mcore_repo / "megatron" / "core").is_dir():
        raise FileNotFoundError(f"Not a MCore checkout: {mcore_repo}")
    mok_repo = Path(__file__).resolve().parents[1]
    # Both repos have a tests package; resolve our reference helpers first.
    sys.path[:0] = [str(mok_repo), str(mcore_repo)]
    import torch
    import torch.distributed as dist
    import transformer_engine as te
    import transformer_engine.pytorch
    import transformer_engine.pytorch.module.grouped_linear as grouped_linear
    from megatron.core import parallel_state
    from megatron.core.config import set_experimental_flag
    from megatron.core.transformer.moe import moe_layer, fused_a2a
    from benchmarks import utils as benchmark_utils
    from tests import utils as reference_utils
    from tests.utils import generate_inputs, run_reference_bf16, run_reference_native_bf16

    expected_layer_source = mcore_repo / "megatron" / "core" / "transformer" / "moe" / "moe_layer.py"
    if Path(moe_layer.__file__).resolve() != expected_layer_source:
        raise RuntimeError(f"Loaded the wrong MCore source: {moe_layer.__file__}")
    if Path(reference_utils.__file__).resolve() != mok_repo / "tests" / "utils.py":
        raise RuntimeError(f"Loaded the wrong reference helpers: {reference_utils.__file__}")
    supports_grouped_tensor = "use_grouped_tensor" in inspect.signature(te.pytorch.GroupedLinear.__init__).parameters
    selected, fallback = select_grouped_tensor(args.gemm_backend, supports_grouped_tensor)
    effective_backend = "te-cublas-grouped" if te_internal_grouped else ("device-init" if selected else "multistream")
    benchmark_utils.WARMUP_ITERS = args.warmup_iters
    benchmark_utils.TIMED_ITERS = args.timed_iters
    rank, world_size, device = benchmark_utils.init_distributed()
    try:
        if world_size != args.ep_size:
            raise ValueError(f"WORLD_SIZE={world_size} does not match --ep-size={args.ep_size}")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, expert_model_parallel_size=world_size,
            expert_tensor_parallel_size=1,
        )
        set_experimental_flag(True)
        local_experts = benchmark_utils.get_num_local_experts(args.num_experts, world_size)
        hardware = [None] * world_size
        dist.all_gather_object(hardware, {
            "rank": rank, "host": socket.gethostname(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        })
        if rank == 0:
            config = vars(args).copy()
            config["mcore_repo"] = str(mcore_repo)
            print(json.dumps({
                "event": "qwen35_native_config", "config": config,
                "mcore_git": git_metadata(mcore_repo), "mok_git": git_metadata(Path(__file__).resolve().parents[1]),
                "mcore_layer_source": moe_layer.__file__, "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda, "te_version": te.__version__, "hardware": hardware,
                "fallback_reason": fallback, "grad_mode": "fresh", "routed_weights": "non-single per-expert Parameters",
                "precision": args.precision, "shared_precision": "bfloat16",
                "shared_quant_override": shared_bf16_quant_recipe() if args.precision == "mxfp8" else None,
                "effective_backend_candidate": effective_backend,
                "mcore_grouped_tensor_api": selected, "te_internal_grouped_candidate": te_internal_grouped,
                "shared_expert_overlap": False, "op_fuser": False,
                "effective_rank_capacity_factor": args.rank_capacity_factor if selected else None,
                "environment": {key: os.environ.get(key) for key in (
                    "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "OMP_NUM_THREADS",
                    "PYTORCH_CUDA_ALLOC_CONF", "MOK_BENCHMARK_NUMA_BINDING", "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
                    "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM",
                )},
                "timing": {
                    "method": "CUDA events, median of per-iteration maximum across EP ranks; no CUDA graph",
                    "fwd_scope": "actual MoELayer.forward with fixed route; includes differentiable FP32 probability scatter; bool route-map prebuilt; excludes router GEMM/top-k",
                    "bwd_scope": "autograd.grad on actual MoELayer graph; setup forward and result gradient stacking excluded",
                    "mok_comparison_scope": "MOK baseline includes build_schedule + functional.forward; both exclude router and use same compact FP32 P/ids",
                    "weight_quantization": "Native caches routed quantization after untimed correctness forward; MOK prequantizes routed weights outside timing; no parameter updates",
                    "activation_quantization": "included in fwd/bwd when precision=mxfp8",
                    "grad_comparison": "fresh-to-fresh only; Native leaf weight gradients are BF16, MOK output-gate wgrad is FP32; neither includes DDP/main-grad accumulation",
                },
                "memory_scope": "PyTorch allocator peaks for each whole phase incl warmup/setup forward; excludes external CUDA allocations",
            }, sort_keys=True), flush=True)
        inputs = generate_inputs(rank, device, args.num_experts, local_experts, args.topk, args.num_local_tokens, args.hidden_dim, args.intermediate_dim)
        generator = torch.Generator(device=device).manual_seed(1919 + rank)
        gate_weight = torch.randn(1, args.hidden_dim, device=device, dtype=torch.bfloat16, generator=generator) * args.hidden_dim ** -0.5
        variants = (False, True) if args.gate == "both" else (args.gate == "on",)
        for gated in variants:
            label = "Native-G" if gated else "Native-U"
            weight = gate_weight if gated else None
            benchmark = NativeBenchmark(args, inputs, world_size, weight, selected)
            if rank == 0:
                print(json.dumps({"event": "qwen35_native_backend", "label": label, **module_evidence(benchmark.layer, te, selected)}, sort_keys=True), flush=True)
            native_reference = run_reference_native_bf16(*inputs, shared_output_gate_weight=weight, bias_activation_fusion=args.bias_activation_fusion)
            with count_grouped_gemm_calls(grouped_linear) as call_counts:
                output, context = benchmark.run_fwd()
                raw_gradients = benchmark.run_bwd(context)
            gradients = benchmark.format_gradients(raw_gradients)
            del raw_gradients
            precision_by_rank = [None] * world_size
            dist.all_gather_object(precision_by_rank, precision_evidence(benchmark.layer, args.precision))
            if rank == 0:
                print(json.dumps({"event": "qwen35_native_precision", "label": label, "precision": args.precision, "ranks": precision_by_rank}, sort_keys=True), flush=True)
            rank_counts = [None] * world_size
            dist.all_gather_object(rank_counts, call_counts)
            expected_calls = {"grouped_tensor": 6, "legacy": 0} if (selected or te_internal_grouped) else {"grouped_tensor": 0, "legacy": 6}
            backend_verified = all(counts == expected_calls for counts in rank_counts)
            if rank == 0:
                print(json.dumps({
                    "event": "qwen35_native_backend_calls", "label": label,
                    "backend_candidate": effective_backend, "grouped_linear_source": grouped_linear.__file__,
                    "expected_calls": expected_calls, "calls_by_rank": rank_counts,
                    "backend_entrypoint_verified": backend_verified,
                    "probe_scope": "one correctness fwd+bwd; two routed FCs times fwd/dgrad/wgrad; wrappers removed before timing",
                }, sort_keys=True), flush=True)
            if not backend_verified:
                raise RuntimeError(f"Requested {effective_backend}, but actual TE entrypoints disagree: {rank_counts}")
            actual = (output, *gradients)
            benchmark.check_overflow()
            native_ok = compare_results(actual, native_reference, f"{label} vs Native BF16 semantic anchor", rank, args.precision)
            del native_reference
            mok_reference = run_reference_bf16(*inputs, shared_output_gate_weight=weight)
            mok_ok = compare_results(actual, mok_reference, f"{label} vs MOK BF16 semantic anchor (cross-semantics)", rank, args.precision)
            del mok_reference, actual, output, context, gradients
            if not native_ok:
                raise AssertionError(f"{label} failed Native-semantic oracle; not publishing performance as valid")
            if args.profile_backend:
                profile_evidence = profile_backend(benchmark, torch)
                if rank == 0:
                    print(json.dumps({"event": "qwen35_native_backend_kernel_names", "label": label, **profile_evidence}, sort_keys=True), flush=True)
            results = {}
            for phase in ("fwd", "bwd"):
                results[phase] = measure_phase(phase, benchmark, benchmark_utils, torch, dist, device)
                results[phase]["tflops_per_gpu_mlp_only"] = benchmark_utils.get_tflops(
                    results[phase]["median_rank_max_ms"], args.num_local_tokens, args.topk,
                    args.hidden_dim, args.intermediate_dim, backward=phase == "bwd",
                )
            benchmark.check_overflow()
            if rank == 0:
                print(json.dumps({
                    "event": "qwen35_native_result", "label": label, "grad_mode": "fresh", "precision": args.precision,
                    "native_reference_passed": native_ok, "mok_semantic_reference_passed": mok_ok,
                    "selected_grouped_tensor": selected, "effective_backend": effective_backend,
                    "backend_entrypoint_verified": backend_verified, "backend_calls_by_rank": rank_counts,
                    "results": results,
                }, sort_keys=True), flush=True)
            del benchmark
            gc.collect()
            torch.cuda.empty_cache()
        dist.barrier()
    finally:
        fused_a2a.reset_hybrid_ep_buffer()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
