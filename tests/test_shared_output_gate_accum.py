"""BF16 gate main-grad must round after adding the FP32 contribution.

Run on four CUDA ranks with the repository's torchrun/pytest fixture. These
exact cancellation witnesses supplement the random MoE reference comparisons:
they distinguish direct FP32-to-BF16 accumulation from pre-rounded wgrads.
"""

import os

import pytest
import torch

from mok import functional, ops

from .test_shared_output_gate_mxfp8 import (
    _backward, _forward, _inputs, _main_grads, _routed, _schedule,
)
from .utils import run_shared_output_gate_reference


@pytest.mark.parametrize("shape", [
    pytest.param((512, 256), id="small"),
    pytest.param((4096, 4096), id="qwen-target", marks=pytest.mark.skipif(
        os.getenv("MOK_TEST_QWEN_GATE_SHAPE") != "1",
        reason="set MOK_TEST_QWEN_GATE_SHAPE=1 for the Qwen producer shape",
    )),
])
def test_gate_bf16_main_grad_rounds_after_fp32_add(context, shape):
    rows, hidden = shape
    x = torch.ones(rows, hidden, device=context[2], dtype=torch.bfloat16)
    x[-1].fill_(1 / 64)
    main_grad = torch.full((1, hidden), -rows, device=context[2], dtype=torch.bfloat16)
    for scale in (1, 1 / rows):
        dz = torch.full((rows, 1), scale, device=context[2], dtype=torch.bfloat16)
        # All products and sums in this fixture are exact in FP32.
        contribution = (dz.double().T @ x.double()).float()
        before = main_grad.clone()
        expected = (before.float() + contribution).to(torch.bfloat16)
        pre_rounded = (before.float() + contribution.bfloat16().float()).bfloat16()
        assert not torch.equal(expected, pre_rounded)
        returned = functional._shared_output_gate_wgrad(dz, x, main_grad)
        assert returned is main_grad and returned.dtype == torch.bfloat16
        torch.testing.assert_close(returned, expected, rtol=0, atol=0)

    pointer = main_grad.data_ptr()
    main_grad.zero_()
    fresh = functional._shared_output_gate_wgrad(dz, x)
    assert fresh.dtype == torch.float32
    returned = functional._shared_output_gate_wgrad(dz, x, main_grad)
    assert returned is main_grad and returned.data_ptr() == pointer
    torch.testing.assert_close(returned, fresh.bfloat16(), rtol=0, atol=0)


def _bf16_routed(inputs, layout):
    if layout == "legacy":
        weights = inputs[6:9]
        return {"forward": weights, "backward": weights}
    fc1 = torch.cat(inputs[6:8], dim=1)
    if layout == "native":
        weights = (fc1, fc1, inputs[8])
        return {"forward": weights, "backward": weights}
    fc1_experts = [weight.clone() for weight in fc1]
    down_experts = [weight.clone() for weight in inputs[8]]
    fc1_view = functional.SplitRoutedWeight(
        fc1_experts[0], ops.make_routed_weight_storage_table_bf16(fc1_experts))
    down_view = functional.SplitRoutedWeight(
        down_experts[0], ops.make_routed_weight_storage_table_bf16(down_experts))
    weights = (fc1_view, fc1_view, down_view)
    return {"forward": weights, "backward": weights,
            "owners": (fc1_experts, down_experts)}


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("layout", ["legacy", "native", "split_native"])
@pytest.mark.parametrize("macro", [4096, 512])
def test_gated_bf16_main_grad_exact_accumulation_and_reset(context, precision, layout, macro):
    inputs, gate = _inputs(context)
    x = torch.zeros_like(inputs[0])
    x[:, :2] = 1
    x[-1, 1] = 1 / 64
    gate.zero_()  # G=0.5; the gated branch must still run.
    weights = inputs[3:9]
    for weight in weights:
        weight.zero_()
    weights[0][0, 0], weights[1][0, 0], weights[2][0, 0] = 16, 1, 1 / 16
    shared = torch.zeros_like(x)
    shared[:, 0] = 1
    inputs = (x, inputs[1], inputs[2], *weights, inputs[-1])
    routed = _routed(inputs, layout) if precision == "mxfp8" else _bf16_routed(inputs, layout)
    state = _schedule(context, inputs, macro)
    main = _main_grads(inputs, gate, layout, torch.bfloat16)
    main["gate"].fill_(-512)
    pointer = main["gate"].data_ptr()

    for iteration, scale in enumerate((1, 1 / 512, -1 / 256)):
        if iteration == 2:
            main["gate"].zero_()  # Reuse the same allocation in a new accumulation window.
        dy = torch.zeros_like(x)
        dy[:, 0] = 4 * scale
        batch = (*inputs[:-1], dy)
        output, saved = _forward(batch, gate, routed, state)
        torch.testing.assert_close(saved.shared_output, shared, rtol=0, atol=0)
        torch.testing.assert_close(output, shared * 0.5, rtol=0, atol=0)
        *_, dw64 = run_shared_output_gate_reference(x, shared, gate, dy)
        before = main["gate"].clone()
        expected = (before.float() + dw64.float()).bfloat16()
        if iteration < 2:
            pre_rounded = (before.float() + dw64.bfloat16().float()).bfloat16()
            assert expected[0, 1] != pre_rounded[0, 1]
        gradients = _backward(batch, gate, routed, state, saved, main)
        assert len(gradients) == 9 and gradients[-1] is main["gate"]
        assert gradients[-1].data_ptr() == pointer
        assert all(value.dtype == torch.bfloat16 for value in gradients[2:])
        torch.testing.assert_close(gradients[-1], expected, rtol=0, atol=0)
