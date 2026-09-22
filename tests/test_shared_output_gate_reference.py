"""Independent gate references and real BF16 MOK output-gate API tests.

Run through the repository's torchrun/pytest CUDA fixture. Full Qwen shape
checks are opt-in; missing production APIs fail rather than silently skip.
"""

import os

import pytest
import torch
import torch.distributed as dist

from mok import functional, ops

from .utils import (
    BF16_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_forward_reference_bf16,
    run_reference_bf16,
    run_shared_output_gate_reference,
)


GATED_RESULT_NAMES = (
    "output", "d_x", "d_router_weights",
    "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down",
    "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down",
    "d_w_shared_output_gate",
)


def _gate_inputs(context):
    rank, _, device = context
    generator = torch.Generator(device=device).manual_seed(617 + rank)
    x = torch.randn(32, 64, generator=generator, device=device, dtype=torch.bfloat16)
    shared = torch.randn(x.shape, generator=generator, device=device, dtype=torch.bfloat16)
    weight = torch.randn(1, 64, generator=generator, device=device, dtype=torch.bfloat16) / 8
    dy = torch.randn(x.shape, generator=generator, device=device, dtype=torch.bfloat16) / 8
    return x, shared, weight, dy


def _dense_inputs(context):
    rank, world_size, device = context
    inputs = generate_inputs(rank, device, 2 * world_size, 2, 2, 512, 256, 256)
    generator = torch.Generator(device=device).manual_seed(919 + rank)
    weight = torch.randn(1, 256, generator=generator, device=device, dtype=torch.bfloat16) / 16
    return inputs, weight


def test_gate_reference_autograd_matches_manual_bf16_boundaries(context):
    x, shared, weight, dy = _gate_inputs(context)
    gate, ds, dg, dz, dx_gate, dw_fp64 = run_shared_output_gate_reference(
        x, shared, weight, dy
    )
    expected_gate = torch.sigmoid(torch.nn.functional.linear(x, weight))
    expected_ds = dy * expected_gate
    expected_dg = (dy * shared).sum(-1, keepdim=True)
    expected_dz = torch.ops.aten.sigmoid_backward.default(expected_dg, expected_gate)
    for actual, expected in (
        (gate, expected_gate), (ds, expected_ds), (dg, expected_dg),
        (dz, expected_dz), (dx_gate, expected_dz @ weight),
    ):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dg.shape == (x.shape[0], 1)  # Reduce H, not T.
    assert dw_fp64.dtype == torch.float64
    torch.testing.assert_close(
        dw_fp64, expected_dz.double().T @ x.double(), rtol=0, atol=0
    )


def _bf16_product_rounding_witness(device, rows, hidden):
    # Both products are near +/-1. BF16 rounds the positive product down,
    # yielding exact cancellation. FP32 retains its 1/16384 residual.
    x = torch.zeros(rows, hidden, device=device, dtype=torch.bfloat16)
    x[:, :2] = 1
    shared = torch.zeros_like(x)
    shared[:, 0] = 1 + 1 / 128
    shared[:, 1] = 1
    dy = torch.zeros_like(x)
    dy[:, 0] = 1 + 1 / 128
    dy[:, 1] = -(1 + 1 / 64)
    weight = torch.zeros(1, hidden, device=device, dtype=torch.bfloat16)
    weight[0, 0], weight[0, 1] = 1, -1  # Z=0 and G=0.5, but Wg is nonzero.
    return x, shared, weight, dy


def test_gate_reference_rounds_bf16_product_before_hidden_reduction(context):
    x, shared, weight, dy = _bf16_product_rounding_witness(context[2], 32, 64)
    gate, ds, dg, dz, dx_gate, dw_fp64 = run_shared_output_gate_reference(
        x, shared, weight, dy
    )
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5), rtol=0, atol=0)
    torch.testing.assert_close(ds, dy * 0.5, rtol=0, atol=0)
    for gradient in (dg, dz, dx_gate, dw_fp64):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
    old_dg = (dy.float() * shared.float()).sum(-1, keepdim=True).to(torch.bfloat16)
    torch.testing.assert_close(old_dg, torch.full_like(dg, 1 / 16384), rtol=0, atol=0)
    assert not torch.equal(dg, old_dg)


@pytest.mark.parametrize("zero_input", ["weight", "shared", "dy"])
def test_gate_reference_zero_boundaries(context, zero_input):
    x, shared, weight, dy = _gate_inputs(context)
    {"weight": weight, "shared": shared, "dy": dy}[zero_input].zero_()
    gate, ds, dg, dz, dx_gate, dw_fp64 = run_shared_output_gate_reference(
        x, shared, weight, dy
    )
    if zero_input == "weight":
        # Wg=0 is G=0.5, NOT the ungated/None path. Gate wgrad can be nonzero.
        torch.testing.assert_close(gate, torch.full_like(gate, 0.5), rtol=0, atol=0)
        torch.testing.assert_close(ds, dy * 0.5, rtol=0, atol=0)
        assert torch.count_nonzero(dx_gate) == 0
        assert torch.count_nonzero(dw_fp64) > 0
    else:
        for gradient in (dg, dz, dx_gate, dw_fp64):
            assert torch.count_nonzero(gradient) == 0
        if zero_input == "dy":
            assert torch.count_nonzero(ds) == 0


def test_gated_moe_reference_preserves_ungated_and_fp32_accum_contract(context):
    inputs, weight = _dense_inputs(context)
    legacy = run_reference_bf16(*inputs)
    explicit_none = run_reference_bf16(*inputs, shared_output_gate_weight=None)
    assert len(legacy) == len(explicit_none) == 9  # Includes Y; old helper unchanged.
    for actual, expected in zip(explicit_none, legacy, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    fresh = run_reference_bf16(*inputs, shared_output_gate_weight=weight)
    assert len(fresh) == 10
    assert fresh[-1].dtype == torch.float32
    fresh_snapshot = fresh[-1].clone()

    main_grad = torch.full_like(weight, 0.25, dtype=torch.float32)
    expected = main_grad.clone()
    for _ in range(2):
        actual = run_reference_bf16(
            *inputs,
            shared_output_gate_weight=weight,
            shared_output_gate_main_grad=main_grad,
        )
        expected.add_(fresh_snapshot)
        assert actual[-1] is main_grad
        torch.testing.assert_close(actual[-1], expected, rtol=0, atol=0)
    torch.testing.assert_close(fresh[-1], fresh_snapshot, rtol=0, atol=0)


def test_gated_moe_reference_b_epilogue_order(context):
    inputs, weight = _dense_inputs(context)
    x, experts, probs, shared_gate, shared_up, shared_down, *tail = inputs
    routed_gate, routed_up, routed_down, _ = tail
    combine, _, _, _, shared = run_forward_reference_bf16(
        x, experts, shared_gate, shared_up, shared_down,
        routed_gate, routed_up, routed_down,
    )
    gate = torch.sigmoid(torch.nn.functional.linear(x, weight))
    accumulator = shared.float() * gate.float()
    routed = combine.view(x.shape[0], probs.shape[1], x.shape[1])
    for k in range(probs.shape[1]):
        accumulator = accumulator + routed[:, k].float() * probs[:, k, None]
    actual = run_reference_bf16(*inputs, shared_output_gate_weight=weight)
    torch.testing.assert_close(actual[0], accumulator.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["bf16_buffer", "wrong_shape", "missing_weight"])
def test_reference_rejects_invalid_gate_accum_contract(context, invalid):
    inputs, weight = _dense_inputs(context)
    buffer = torch.zeros_like(weight, dtype=torch.float32)
    if invalid == "bf16_buffer":
        buffer = buffer.to(torch.bfloat16)
    elif invalid == "wrong_shape":
        buffer = buffer.flatten()
    else:
        weight = None
    with pytest.raises(ValueError, match="gate main-grad"):
        run_reference_bf16(
            *inputs,
            shared_output_gate_weight=weight,
            shared_output_gate_main_grad=buffer,
        )


def _check_saved_gate(saved, x, weight):
    assert saved.shared_output.shape == x.shape
    assert saved.shared_output.dtype == torch.bfloat16
    assert saved.shared_output_gate.shape == (x.shape[0], 1)
    assert saved.shared_output_gate.dtype == torch.bfloat16
    assert not saved.shared_output_gate.requires_grad
    assert saved.shared_output_gate.grad_fn is None
    expected_gate = torch.sigmoid(torch.nn.functional.linear(x, weight))
    torch.testing.assert_close(saved.shared_output_gate, expected_gate, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [
    (32, 64),
    pytest.param(
        (4096, 4096),
        marks=pytest.mark.skipif(
            os.getenv("MOK_TEST_QWEN_GATE_SHAPE") != "1",
            reason="opt in to Qwen wgrad shape with MOK_TEST_QWEN_GATE_SHAPE=1",
        ),
        id="qwen-target",
    ),
])
def test_production_gate_wgrad_fp32_without_bf16_intermediate(context, shape):
    rank, _, device = context
    rows, hidden = shape
    generator = torch.Generator(device=device).manual_seed(2819 + rank)
    x = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16)
    shared = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16)
    weight = torch.randn(1, hidden, generator=generator, device=device, dtype=torch.bfloat16) * hidden ** -0.5
    dy = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16) * hidden ** -0.5
    _, _, _, dz_ref, _, dw_fp64 = run_shared_output_gate_reference(x, shared, weight, dy)

    # Exercise the ACTUAL production helper, using an independently derived dZ.
    actual = functional._shared_output_gate_wgrad(dz_ref, x)
    assert actual.dtype == torch.float32 and actual.shape == weight.shape
    assert torch.isfinite(actual).all()
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    gamma_k = rows * unit_roundoff / (1 - rows * unit_roundoff)
    bound = gamma_k * (dz_ref.double().abs().T @ x.double().abs())
    assert torch.isfinite(bound).all()
    assert torch.all((actual.double() - dw_fp64).abs() <= bound)

    # Same dispatch shape; exact FP32 dot distinguishes genuine FP32 output
    # from a BF16 matmul result that was subsequently converted to FP32.
    exact_dz = torch.ones_like(dz_ref)
    exact_x = torch.ones_like(x)
    exact_x[-1].fill_(1 / 64)
    expected = torch.full_like(weight, rows - 1 + 1 / 64, dtype=torch.float32)
    assert not torch.equal(expected, expected.to(torch.bfloat16).float())
    fresh = functional._shared_output_gate_wgrad(exact_dz, exact_x)
    assert fresh.dtype == torch.float32
    torch.testing.assert_close(fresh, expected, rtol=0, atol=0)
    fresh_snapshot = fresh.clone()
    main_grad = torch.full_like(expected, 0.25)
    accumulated = main_grad.clone()
    for _ in range(2):
        returned = functional._shared_output_gate_wgrad(exact_dz, exact_x, main_grad)
        accumulated.add_(expected)
        assert returned is main_grad
        torch.testing.assert_close(returned, accumulated, rtol=0, atol=0)
    torch.testing.assert_close(fresh, fresh_snapshot, rtol=0, atol=0)


def _mok_schedule(context, inputs, macrobatch_size=4096):
    _, _, device = context
    x, experts, probs, *_ = inputs
    config = functional.MoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2,
        minibatch_size=256, macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device,
        num_local_tokens=x.shape[0], hidden_size=x.shape[1], topk=probs.shape[1],
    )
    schedule = functional.build_schedule(workspace, config, experts, num_local_experts=2)
    return config, workspace, schedule


def test_mok_explicit_ungated_nine_item_contract(context):
    inputs, _ = _dense_inputs(context)
    x, _, probs, *rest = inputs
    weights, dy = rest[:-1], rest[-1]
    config, workspace, schedule = _mok_schedule(context, inputs)
    output, saved = functional.forward(
        config, workspace, schedule, x, probs, *weights,
        shared_output_gate_weight=None,
    )
    assert saved.shared_output is None and saved.shared_output_gate is None
    gradients = functional.backward(
        config, workspace, schedule, saved, dy, x, probs, *weights,
        shared_output_gate_weight=None,
    )
    assert len(gradients) == 9
    assert gradients[-1] is None
    reference = run_reference_bf16(*inputs)
    for name, actual, golden in zip(
        GATED_RESULT_NAMES[:-1], (output, *gradients[:8]), reference, strict=True
    ):
        check_correctness(name, golden, actual, BF16_TOLERANCE, print_stats=context[0] == 0)


def test_mok_gate_backward_uses_bf16_product_rounding(context):
    """A real MOK context whose gate wgrad detects the old formula exactly."""
    inputs, _ = _dense_inputs(context)
    x, experts, probs, *rest = inputs
    weights = rest[:-1]
    x, shared, weight, dy = _bf16_product_rounding_witness(
        context[2], x.shape[0], x.shape[1]
    )
    for matrix in weights:
        matrix.zero_()
    # Only one shared intermediate channel is active. BF16(silu(16))=16,
    # so the down GEMM produces our exact S; routed experts remain zero.
    weights[0][0, 0] = 16
    weights[1][0, 0] = 1
    weights[2][0, 0] = (1 + 1 / 128) / 16
    weights[2][1, 0] = 1 / 16
    inputs = (x, experts, probs, *weights, dy)
    config, workspace, schedule = _mok_schedule(context, inputs)
    output, saved = functional.forward(
        config, workspace, schedule, x, probs, *weights,
        shared_output_gate_weight=weight,
    )
    torch.testing.assert_close(saved.shared_output, shared, rtol=0, atol=0)
    torch.testing.assert_close(output, shared * 0.5, rtol=0, atol=0)
    _, _, _, _, _, golden_dw = run_shared_output_gate_reference(x, shared, weight, dy)
    gradients = functional.backward(
        config, workspace, schedule, saved, dy, x, probs, *weights,
        shared_output_gate_weight=weight,
    )
    assert len(gradients) == 9 and gradients[-1].dtype == torch.float32
    torch.testing.assert_close(golden_dw, torch.zeros_like(golden_dw), rtol=0, atol=0)
    torch.testing.assert_close(gradients[-1].double(), golden_dw, rtol=0, atol=0)
    # With the old FP32-product dG, dZ would be 1/65536 per token and the
    # first two gate-weight gradients would be 512/65536, not zero.
    assert x.shape[0] == 512


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("macrobatch_size", [4096, 512])
def test_mok_gated_dense_matches_reference(context, accumulate, macrobatch_size):
    inputs, weight = _dense_inputs(context)
    x, _, probs, *rest = inputs
    weights, dy = rest[:-1], rest[-1]
    if not accumulate:
        # Even trainable inputs/weights must not retain an inner gate/Z graph.
        x.requires_grad_(True)
        weight.requires_grad_(True)
    config, workspace, schedule = _mok_schedule(context, inputs, macrobatch_size)
    main_grads = None
    gate_main_grad = None
    if accumulate:
        main_grads = tuple(
            torch.zeros_like(w, dtype=torch.float32)
            for w in (weights[0], weights[3], weights[1], weights[4], weights[2], weights[5])
        )
        gate_main_grad = torch.full_like(weight, 0.25, dtype=torch.float32)

    reference = run_reference_bf16(*inputs, shared_output_gate_weight=weight)
    expected_gate = torch.zeros_like(weight, dtype=torch.float32) if not accumulate else gate_main_grad.clone()
    for iteration in range(2 if accumulate else 1):
        output, saved = functional.forward(
            config, workspace, schedule, x, probs, *weights,
            shared_output_gate_weight=weight,
        )
        _check_saved_gate(saved, x, weight)
        gradients = functional.backward(
            config, workspace, schedule, saved, dy, x, probs, *weights,
            main_grads=main_grads,
            shared_output_gate_weight=weight,
            shared_output_gate_main_grad=gate_main_grad,
        )
        assert len(gradients) == 9
        assert gradients[-1].dtype == torch.float32
        expected_gate.add_(reference[-1])
        if accumulate:
            assert gradients[-1] is gate_main_grad
        expected = list(reference)
        if accumulate:
            expected[3:9] = [gradient.float() * (iteration + 1) for gradient in reference[3:9]]
        expected[-1] = expected_gate
        for name, actual, golden in zip(GATED_RESULT_NAMES, (output, *gradients), expected, strict=True):
            check_correctness(name, golden, actual, BF16_TOLERANCE, print_stats=context[0] == 0)


def test_mok_gated_multiple_live_contexts(context):
    inputs, weight = _dense_inputs(context)
    second = list(inputs)
    second[0] = inputs[0] * -0.5
    second[-1] = inputs[-1] * 2
    batches = (inputs, tuple(second))
    pending = []
    for batch in batches:
        x, _, probs, *rest = batch
        weights = rest[:-1]
        config, workspace, schedule = _mok_schedule(context, batch)
        output, saved = functional.forward(
            config, workspace, schedule, x, probs, *weights,
            shared_output_gate_weight=weight,
        )
        _check_saved_gate(saved, x, weight)
        snapshots = (saved.shared_output.clone(), saved.shared_output_gate.clone())
        pending.append((batch, config, workspace, schedule, output, saved, snapshots))

    # Both contexts must survive a subsequent forward using the same workspace.
    for batch, config, workspace, schedule, output, saved, snapshots in reversed(pending):
        torch.testing.assert_close(saved.shared_output, snapshots[0], rtol=0, atol=0)
        torch.testing.assert_close(saved.shared_output_gate, snapshots[1], rtol=0, atol=0)
        x, _, probs, *rest = batch
        weights, dy = rest[:-1], rest[-1]
        gradients = functional.backward(
            config, workspace, schedule, saved, dy, x, probs, *weights,
            shared_output_gate_weight=weight,
        )
        reference = run_reference_bf16(*batch, shared_output_gate_weight=weight)
        assert len(gradients) == 9
        for name, actual, golden in zip(GATED_RESULT_NAMES, (output, *gradients), reference, strict=True):
            check_correctness(name, golden, actual, BF16_TOLERANCE, print_stats=context[0] == 0)


@pytest.mark.parametrize("invalid", ["weight_dtype", "weight_shape", "mxfp8"])
def test_mok_gated_forward_rejects_unsupported_inputs(context, invalid):
    inputs, weight = _dense_inputs(context)
    x, _, probs, *rest = inputs
    weights = list(rest[:-1])
    config, workspace, schedule = _mok_schedule(context, inputs)
    if invalid == "weight_dtype":
        weight = weight.float()
    elif invalid == "weight_shape":
        weight = weight.flatten()
    else:
        weights[3:] = [ops.mxfp8_quantize(w, True, False)[:2] for w in weights[3:]]
    error = NotImplementedError if invalid == "mxfp8" else ValueError
    with pytest.raises(error, match="BF16"):
        functional.forward(
            config, workspace, schedule, x, probs, *weights,
            shared_output_gate_weight=weight,
        )


@pytest.mark.parametrize("invalid", ["missing_saved_state", "missing_weight", "bf16_main_grad"])
def test_mok_gated_backward_rejects_inconsistent_state(context, invalid):
    inputs, weight = _dense_inputs(context)
    x, _, probs, *rest = inputs
    weights, dy = rest[:-1], rest[-1]
    config, workspace, schedule = _mok_schedule(context, inputs)
    _, saved = functional.forward(
        config, workspace, schedule, x, probs, *weights,
        shared_output_gate_weight=None if invalid == "missing_saved_state" else weight,
    )
    with pytest.raises(ValueError):
        functional.backward(
            config, workspace, schedule, saved, dy, x, probs, *weights,
            shared_output_gate_weight=None if invalid == "missing_weight" else weight,
            shared_output_gate_main_grad=torch.zeros_like(weight) if invalid == "bf16_main_grad" else None,
        )
