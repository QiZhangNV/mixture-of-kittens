"""Tests for the eager fast path of MoK custom ops (mok/_eager_fast_path.py)."""

import contextlib
from collections.abc import Callable, Iterator

import pytest
import torch
import torch.distributed as dist
from torch._library.custom_ops import CustomOpDef

from mok import _eager_fast_path, functional
from mok.functional import get_workspace
from mok.ops import mxfp8_quantize

from .utils import generate_inputs


@_eager_fast_path.eager_fast_path_custom_op("moktest::scale_into", mutates_args=("out",))
def scale_into(x: torch.Tensor, out: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    out.copy_(x * factor)
    return (x * factor).sum().reshape(1)


@torch.library.register_fake("moktest::scale_into")
def _scale_into_fake(x: torch.Tensor, out: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    return x.new_empty(1)


@_eager_fast_path.eager_fast_path_custom_op("moktest::sum_with_ptrs", mutates_args=())
def sum_with_ptrs(x: torch.Tensor, ptrs: list[int], flag: bool) -> torch.Tensor:
    # Mirrors the strict type checks in mok/ops.py bodies.
    if not isinstance(ptrs, list) or any(type(p) is not int for p in ptrs) or type(flag) is not bool:
        raise TypeError("ptrs must be a list of int and flag a bool")
    return x.sum().reshape(1) + len(ptrs)


class _CountingOpOverload:
    """Stands in for CustomOpDef._opoverload: counts calls, forwards everything else."""

    def __init__(self, original, calls: list[int]) -> None:
        self._original = original
        self._calls = calls

    def __call__(self, *args, **kwargs):
        self._calls.append(1)
        return self._original(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextlib.contextmanager
def dispatcher_calls(op: CustomOpDef) -> Iterator[list[int]]:
    """Count calls that went through the dispatcher instead of the direct path."""
    calls: list[int] = []
    original = op._opoverload
    op._opoverload = _CountingOpOverload(original, calls)
    try:
        yield calls
    finally:
        op._opoverload = original


def run_both_paths(op: CustomOpDef, make_args: Callable[[], tuple]) -> tuple[tuple, tuple, list[int]]:
    """Call op once on the direct path and once through the dispatcher; return both and the counts."""
    counts = []
    results = []
    for enabled in (True, False):
        args = make_args()
        previous = _eager_fast_path.ENABLED
        _eager_fast_path.ENABLED = enabled
        try:
            with dispatcher_calls(op) as calls:
                results.append((op(*args), args))
            counts.append(len(calls))
        finally:
            _eager_fast_path.ENABLED = previous
    return results[0], results[1], counts


def test_ops_are_custom_op_defs() -> None:
    assert isinstance(scale_into, CustomOpDef)
    assert isinstance(mxfp8_quantize, CustomOpDef)
    assert torch.library._maybe_get_opdef("moktest::scale_into") is scale_into


def test_direct_call_matches_dispatcher_without_grad() -> None:
    x = torch.randn(8)

    def make_args():
        return x, torch.zeros(8)

    with torch.no_grad():
        (direct, direct_args), (dispatched, dispatched_args), counts = run_both_paths(scale_into, make_args)
    assert counts == [0, 1]
    assert torch.equal(direct, dispatched)
    assert torch.equal(direct_args[1], dispatched_args[1])
    assert direct_args[1]._version == dispatched_args[1]._version


def test_direct_call_with_grad_enabled_but_no_requires_grad() -> None:
    x = torch.randn(8)
    out = torch.zeros(8)
    with dispatcher_calls(scale_into) as calls:
        scale_into(x, out)
    assert calls == []
    assert torch.equal(out, x * 2)


def test_dispatches_when_an_argument_requires_grad() -> None:
    x = torch.randn(8, requires_grad=True)
    out = torch.zeros(8)
    with dispatcher_calls(scale_into) as calls:
        with pytest.raises(RuntimeError):
            # custom_op without an autograd formula refuses to differentiate; the direct path
            # must not hide that by silently running the body.
            scale_into(x, out).backward()
    assert calls == [1]


def test_parameter_arguments_take_the_direct_path() -> None:
    weight = torch.nn.Parameter(torch.randn(8), requires_grad=False)
    with dispatcher_calls(scale_into) as calls:
        scale_into(weight, torch.zeros(8))
    assert calls == []


def test_tensor_subclass_dispatches() -> None:
    class Marked(torch.Tensor):
        pass

    x = torch.randn(8).as_subclass(Marked)
    with torch.no_grad(), dispatcher_calls(scale_into) as calls:
        scale_into(x, torch.zeros(8))
    assert calls == [1]


def test_meta_tensors_dispatch_to_fake_impl() -> None:
    with torch.no_grad(), dispatcher_calls(scale_into) as calls:
        result = scale_into(torch.empty(8, device="meta"), torch.empty(8, device="meta"))
    assert calls == [1]
    assert result.is_meta


def test_fake_tensor_mode_dispatches_to_fake_impl() -> None:
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode(), torch.no_grad(), dispatcher_calls(scale_into) as calls:
        result = scale_into(torch.randn(8), torch.zeros(8))
    assert calls == [1]
    assert isinstance(result, FakeTensor)


def test_functorch_transform_dispatches() -> None:
    with dispatcher_calls(sum_with_ptrs) as calls:
        with contextlib.suppress(Exception):  # whether vmap supports the op does not matter here
            torch.vmap(lambda x: sum_with_ptrs(x, [1, 2], True))(torch.randn(3, 8))
    assert calls


def test_argument_types_the_dispatcher_would_convert_dispatch() -> None:
    x = torch.randn(8)
    with dispatcher_calls(sum_with_ptrs) as calls:
        direct = sum_with_ptrs(x, [1, 2, 3], True)
        converted = sum_with_ptrs(x, (1, 2, 3), True)  # tuple for int[]: the dispatcher makes it a list
    assert calls == [1]
    assert torch.equal(direct, converted)


def test_kill_switch_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_eager_fast_path, "ENABLED", False)
    with torch.no_grad(), dispatcher_calls(scale_into) as calls:
        scale_into(torch.randn(8), torch.zeros(8))
    assert calls == [1]


def test_mutated_keyword_argument_bumps_version() -> None:
    out = torch.zeros(8)
    before = out._version
    with torch.no_grad(), dispatcher_calls(scale_into) as calls:
        scale_into(torch.randn(8), out=out, factor=3.0)
    assert calls == []
    assert out._version > before


def test_fallback_reasons() -> None:
    x = torch.randn(8)
    out = torch.zeros(8)
    assert scale_into._mok_fallback_reason((x, out), {}) is None
    assert scale_into._mok_fallback_reason((x, out, 3), {}) == "argument type"  # int for float
    assert scale_into._mok_fallback_reason((x.requires_grad_(), out), {}) == (
        "tensor subclass, meta tensor or requires grad"
    )


def test_profiler_records_the_op_on_the_direct_path() -> None:
    with torch.no_grad(), dispatcher_calls(scale_into) as calls:
        with torch.profiler.profile() as profile:
            scale_into(torch.randn(8), torch.zeros(8))
    assert calls == []
    assert "moktest::scale_into" in {event.name for event in profile.events()}


def test_compile_records_the_custom_op() -> None:
    graphs = []

    def backend(graph_module, example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def run(x, out):
        return scale_into(x, out) + 1

    torch._dynamo.reset()
    compiled = torch.compile(run, backend=backend, fullgraph=True)
    x = torch.randn(8)
    out = torch.zeros(8)
    with torch.no_grad():
        result = compiled(x, out)
    assert torch.equal(out, x * 2)
    assert torch.allclose(result, (x * 2).sum().reshape(1) + 1)
    assert len(graphs) == 1
    assert "moktest.scale_into" in graphs[0].code


def test_stats_do_not_cause_recompiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_eager_fast_path, "_STATS_ENABLED", True)
    graphs = []

    def backend(graph_module, example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(lambda x, out: scale_into(x, out), backend=backend, fullgraph=True)
    with torch.no_grad():
        for _ in range(4):
            compiled(torch.randn(8), torch.zeros(8))
    assert len(graphs) == 1


def test_kernels_registered_later_disable_the_direct_path() -> None:
    @_eager_fast_path.eager_fast_path_custom_op("moktest::negate_into", mutates_args=("out",))
    def negate_into(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(-x)

    @_eager_fast_path.eager_fast_path_custom_op("moktest::negate_into_by_name", mutates_args=("out",))
    def negate_into_by_name(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(-x)

    @negate_into.register_kernel("cpu")
    def _negate_into_cpu(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(-2 * x)

    @torch.library.register_kernel("moktest::negate_into_by_name", "cpu")
    def _negate_into_by_name_cpu(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(-3 * x)

    x = torch.randn(8)
    for op, factor in ((negate_into, -2), (negate_into_by_name, -3)):
        out = torch.zeros(8)
        with torch.no_grad(), dispatcher_calls(op) as calls:
            op(x, out)
        assert calls == [1]
        assert torch.equal(out, factor * x)


def test_mxfp8_quantize_matches_dispatcher(context: tuple[int, int, torch.device]) -> None:
    _, _, device = context
    x = torch.randn(512, 1024, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        (direct, _), (dispatched, _), counts = run_both_paths(mxfp8_quantize, lambda: (x, True, True))
    assert counts == [0, 1]
    for direct_output, dispatched_output in zip(direct, dispatched, strict=True):
        assert torch.equal(direct_output, dispatched_output)


def _forward_backward(rank: int, device: torch.device, *, enabled: bool) -> list[torch.Tensor]:
    world_size = dist.get_world_size()
    num_local_experts = 8
    num_experts = world_size * num_local_experts
    hidden_dim, intermediate_dim, topk, num_local_tokens = 1024, 512, 6, 1024
    config = functional.MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=1024,
        schedule_capacity_multiplier=1.5,
    )
    workspace = get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    (
        x,
        topk_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    ) = generate_inputs(
        rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim
    )
    previous = _eager_fast_path.ENABLED
    _eager_fast_path.ENABLED = enabled
    try:
        with torch.no_grad():
            schedule = functional.build_schedule(
                workspace, config, topk_experts, num_local_experts=num_local_experts
            )
            gate = mxfp8_quantize(w_routed_gate, True, True)
            up = mxfp8_quantize(w_routed_up, True, True)
            down = mxfp8_quantize(w_routed_down, True, True)
            output, forward_context = functional.forward(
                config, workspace, schedule, x, router_weights,
                w_shared_gate, w_shared_up, w_shared_down,
                gate[:2], up[:2], down[:2], None,
            )
            gradients = functional.backward(
                config, workspace, schedule, forward_context, d_output, x, router_weights,
                w_shared_gate, w_shared_up, w_shared_down,
                gate, up, down[2:], None,
            )
    finally:
        _eager_fast_path.ENABLED = previous
    torch.cuda.synchronize(device)
    return [output.clone(), *(gradient.clone() for gradient in gradients if gradient is not None)]


def test_forward_backward_match_dispatcher(context: tuple[int, int, torch.device]) -> None:
    rank, _, device = context
    dispatched = _forward_backward(rank, device, enabled=False)
    dispatched_again = _forward_backward(rank, device, enabled=False)
    direct = _forward_backward(rank, device, enabled=True)
    deterministic = all(torch.equal(a, b) for a, b in zip(dispatched, dispatched_again, strict=True))
    for index, (expected, actual, repeat) in enumerate(zip(dispatched, direct, dispatched_again, strict=True)):
        if deterministic:
            assert torch.equal(expected, actual), f"result {index} differs between direct and dispatcher paths"
        else:
            # Same bound as two dispatcher runs: the kernels themselves are not bitwise reproducible.
            bound = (expected.float() - repeat.float()).abs().max()
            assert (expected.float() - actual.float()).abs().max() <= bound, f"result {index}"
