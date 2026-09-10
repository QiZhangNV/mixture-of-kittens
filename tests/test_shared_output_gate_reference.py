"""Preparation tests; MOK-gated checks skip until the feature API exists.

Run through the repository's torchrun/pytest CUDA fixture. Reference tests
do not claim that gated MOK has been implemented or validated.
"""

import inspect
import os

import pytest
import torch
import torch.distributed as dist

from mok import functional

from .utils import (
    BF16_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_forward_reference_bf16,
    run_reference_bf16,
    run_shared_output_gate_reference,
    run_swiglu_reference,
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
    expected_ds = (dy.float() * expected_gate.float()).to(torch.bfloat16)
    expected_dg = (dy.float() * shared.float()).sum(-1, keepdim=True).to(torch.bfloat16)
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


def test_gate_fp32_mm_against_independent_reference_dz(context):
    x, shared, weight, dy = _gate_inputs(context)
    *_, dz, _, dw_fp64 = run_shared_output_gate_reference(x, shared, weight, dy)
    # Deliberately fail if the selected runtime lacks this required interface;
    # do not silently replace it with a BF16 result followed by .float().
    actual = torch.mm(dz.T, x, out_dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), dw_fp64, rtol=1e-5, atol=1e-6)
    assert torch.any(dw_fp64.float() != dw_fp64.to(torch.bfloat16).float())


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


@pytest.mark.skipif(
    os.getenv("MOK_TEST_QWEN_GATE_SHAPE") != "1",
    reason="opt in to the single Qwen target shape with MOK_TEST_QWEN_GATE_SHAPE=1",
)
def test_qwen_shape_gated_reference_and_fp32_wgrad(context):
    """Target-shape reference/runtime smoke, NOT a gated-MOK validation."""
    rank, world_size, device = context
    assert world_size == 4, "the Qwen reference smoke requires EP4 / four ranks"
    inputs = generate_inputs(rank, device, 64, 16, 10, 4096, 4096, 1024)
    x, _, probs, shared_gate, shared_up, shared_down, *tail = inputs
    routed_gate, routed_up, routed_down, dy = tail
    generator = torch.Generator(device=device).manual_seed(1919 + rank)
    weight = torch.randn(
        1, 4096, generator=generator, device=device, dtype=torch.bfloat16
    ) / 64

    reference = run_reference_bf16(*inputs, shared_output_gate_weight=weight)
    shapes = (
        x.shape, x.shape, probs.shape,
        routed_gate.shape, routed_up.shape, routed_down.shape,
        shared_gate.shape, shared_up.shape, shared_down.shape, weight.shape,
    )
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float32) + (
        torch.bfloat16,
    ) * 6 + (torch.float32,)
    for name, value, shape, dtype in zip(
        GATED_RESULT_NAMES, reference, shapes, dtypes, strict=True
    ):
        assert value.shape == shape, (name, value.shape, shape)
        assert value.dtype == dtype, (name, value.dtype, dtype)
        assert torch.isfinite(value).all(), f"non-finite Qwen reference {name}"

    # Recompute only the local shared branch with the same reference helpers.
    # The oracle's dZ is derived from this graph, not supplied by MOK/DUT.
    shared = run_swiglu_reference(x @ shared_gate.T, x @ shared_up.T) @ shared_down.T
    _, _, _, dz_ref, _, dw_fp64 = run_shared_output_gate_reference(x, shared, weight, dy)
    actual = torch.mm(dz_ref.T, x, out_dtype=torch.float32)
    assert actual.dtype == torch.float32
    assert actual.shape == weight.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(dw_fp64).all()

    error = actual.double() - dw_fp64
    reference_norm = torch.linalg.vector_norm(dw_fp64).item()
    assert reference_norm > 0, "the random target-shape oracle must be nonzero"
    reference_rms = reference_norm / dw_fp64.numel() ** 0.5
    relative_l2 = torch.linalg.vector_norm(error).item() / reference_norm
    max_abs_error = error.abs().max().item()
    max_error_over_rms = max_abs_error / reference_rms

    # BF16 products in this non-underflowing test range are exact in FP32.
    # For K=4096 additions, use the FP32 rounding envelope gamma_K * sum|a*b|,
    # where u=2^-24 and gamma_K=K*u/(1-K*u). A K-independent relative-error
    # cutoff is not a general FP32 guarantee, especially under cancellation.
    reduction_length = x.shape[0]
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    gamma_k = reduction_length * unit_roundoff / (1 - reduction_length * unit_roundoff)
    sum_abs_products = dz_ref.double().abs().T @ x.double().abs()
    error_bound = gamma_k * sum_abs_products
    max_error_over_bound = (
        error.abs() / error_bound.clamp_min(torch.finfo(torch.float64).tiny)
    ).max().item()
    print(
        f"rank={rank} Qwen reference / BF16-input FP32-output gate wgrad: "
        f"relative_l2={relative_l2:.6e}, max_abs_error={max_abs_error:.6e}, "
        f"reference_rms={reference_rms:.6e}, max_error_over_rms={max_error_over_rms:.6e}, "
        f"gamma_k={gamma_k:.6e}, max_error_over_bound={max_error_over_bound:.6e}"
    )
    assert torch.isfinite(error_bound).all()
    assert torch.all(error.abs() <= error_bound)

    # The general envelope alone is too loose to exclude a BF16 result.
    # Exercise the SAME [1,4096] @ [4096,4096] dispatch shape with an exact
    # dot: 4095 ones plus 1/64. Every FP32 partial sum is representable, but
    # 4095.015625 itself is not representable in BF16.
    exact_dz = torch.ones_like(dz_ref)
    exact_x = torch.ones_like(x)
    exact_x[-1].fill_(1 / 64)
    exact_actual = torch.mm(exact_dz.T, exact_x, out_dtype=torch.float32)
    assert exact_actual.dtype == torch.float32
    exact_expected = torch.full_like(
        exact_actual, reduction_length - 1 + 1 / 64, dtype=torch.float32
    )
    torch.testing.assert_close(exact_actual, exact_expected, rtol=0, atol=0)
    assert not torch.equal(exact_expected, exact_expected.to(torch.bfloat16).float())
    print(f"rank={rank} same-shape exact FP32 gate-wgrad dot: 4095.015625 passed")


def _require_mok_gate_api():
    expected = {
        functional.forward: ("shared_output_gate_weight",),
        functional.backward: ("shared_output_gate_weight", "shared_output_gate_main_grad"),
    }
    for function, arguments in expected.items():
        if not all(name in inspect.signature(function).parameters for name in arguments):
            pytest.skip("MOK shared output-gate API is not implemented; reference-only preparation")


def _mok_schedule(context, inputs):
    _, _, device = context
    x, experts, probs, *_ = inputs
    config = functional.MoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2,
        minibatch_size=256, macrobatch_size=4096,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device,
        num_local_tokens=x.shape[0], hidden_size=x.shape[1], topk=probs.shape[1],
    )
    schedule = functional.build_schedule(workspace, config, experts, num_local_experts=2)
    return config, workspace, schedule


def test_mok_explicit_ungated_nine_item_contract_when_implemented(context):
    _require_mok_gate_api()
    inputs, _ = _dense_inputs(context)
    x, _, probs, *rest = inputs
    weights, dy = rest[:-1], rest[-1]
    config, workspace, schedule = _mok_schedule(context, inputs)
    output, saved = functional.forward(
        config, workspace, schedule, x, probs, *weights,
        shared_output_gate_weight=None,
    )
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


@pytest.mark.parametrize("accumulate", [False, True])
def test_mok_gated_dense_matches_reference_when_implemented(context, accumulate):
    _require_mok_gate_api()
    inputs, weight = _dense_inputs(context)
    x, _, probs, *rest = inputs
    weights, dy = rest[:-1], rest[-1]
    config, workspace, schedule = _mok_schedule(context, inputs)
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


def test_mok_gated_multiple_live_contexts_when_implemented(context):
    _require_mok_gate_api()
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
        pending.append((batch, config, workspace, schedule, output, saved))

    # Both contexts must survive a subsequent forward using the same workspace.
    for batch, config, workspace, schedule, output, saved in reversed(pending):
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
