"""Routed-only uses no shared storage; the existing zero-shared oracle is equivalent."""

import pytest
import torch
import torch.distributed as dist

from mok import functional, ops
from .utils import BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16


@pytest.mark.parametrize("precision,grad_dtype,situ", [
    ("bf16", None, False), ("mxfp8", None, False),
    ("bf16", torch.float32, True), ("mxfp8", torch.float32, True),
    ("bf16", torch.bfloat16, True), ("mxfp8", torch.bfloat16, True),
])
def test_routed_only_reference_and_accumulation(context, precision, grad_dtype, situ):
    rank, world, device = context
    inputs = generate_inputs(rank, device, 2 * world, 2, 2, 512, 512, 256)
    x, routes, probs, *rest = inputs
    shared, weights, dy = rest[:3], rest[3:6], rest[6]
    for weight in shared:
        weight.zero_()
    if precision == "bf16" and grad_dtype is None:
        routes[:, 0], routes[:, 1] = 0, 1  # Other EP ranks own only empty experts.
    routes[0].fill_(-1)
    probs[0].zero_()
    if precision == "mxfp8":
        gate, up, down = [ops.mxfp8_quantize(w, True, True) for w in weights]
        fwd_weights = (gate[:2], up[:2], down[:2])
        bwd_weights = (gate, up, down[2:])
    else:
        fwd_weights = bwd_weights = weights
    config = functional.MoKConfig(2, 2, 256, 512 if grad_dtype else 4096, 1.5)
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device, num_local_tokens=512, hidden_size=512, topk=2,
    )
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=2)
    if grad_dtype:
        assert schedule.num_tokens.item() > config.macrobatch_size
    buffers = [torch.zeros_like(w, dtype=grad_dtype) for w in weights] if grad_dtype else None
    main_grads = (None, buffers[0], None, buffers[1], None, buffers[2]) if buffers else None
    expected = [torch.zeros_like(w, dtype=grad_dtype) for w in weights] if grad_dtype else None
    tolerance = MXFP8_TOLERANCE if precision == "mxfp8" else BF16_TOLERANCE
    for step in range(2 if grad_dtype else 1):
        xx, ddy = (x * -0.5, dy * 2) if step == 1 else (x, dy)
        beta = ((2.0, 8.0) if step == 1 else (4.0, 25.0)) if situ else None
        y, ctx = functional.forward(config, workspace, schedule, xx, probs, None, None, None, *fwd_weights,
                                    situ_beta=beta)
        assert ctx.gate_shared is ctx.up_shared is ctx.hidden_shared is None
        assert ctx.shared_output is ctx.shared_output_gate is None
        if not grad_dtype or (situ and step == 0):
            ctx = functional.recompute_forward_context(
                config, workspace, schedule, xx, None, None, *fwd_weights[:2],
                situ_beta=beta,
            )
            assert ctx.gate_shared is ctx.up_shared is ctx.hidden_shared is None
        assert ctx.situ_beta == beta
        grads = functional.backward(
            config, workspace, schedule, ctx, ddy, xx, probs, None, None, None, *bwd_weights,
            main_grads=main_grads,
        )
        assert len(grads) == 9 and all(g is None for g in grads[5:])
        def situ_reference(a, u):
            a, u = a.float(), u.float()
            return (beta[0] * torch.tanh(a / beta[0]) * torch.sigmoid(a)
                    * beta[1] * torch.tanh(u / beta[1])).bfloat16()

        golden = list(run_reference_bf16(
            xx, routes, probs, *shared, *weights, ddy,
            routed_activation=situ_reference if situ else None,
        )[:6])
        if grad_dtype:
            for actual, buffer, total, contribution in zip(grads[2:5], buffers, expected, golden[3:]):
                assert actual is buffer
                total.add_(contribution)
            golden[3:] = expected
        for name, actual, reference in zip(("Y", "dX", "dP", "dWa", "dWu", "dWd"), (y, *grads[:5]), golden):
            check_correctness(name, reference, actual, tolerance, print_stats=rank == 0)
        torch.testing.assert_close(y[0], torch.zeros_like(y[0]), rtol=0, atol=0)
        torch.testing.assert_close(grads[0][0], torch.zeros_like(grads[0][0]), rtol=0, atol=0)
