"""Fused shared output-gate backward kernels against the eager BF16 operations they replace."""

import pytest
import torch

from mok import _gate_fused


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape
    differing = (actual.view(torch.int16) != expected.view(torch.int16)).sum().item()
    assert differing == 0, f"{differing} of {actual.numel()} elements differ bitwise"


def _sign_mantissa(rows, hidden, generator, device):
    # +/-(1 + j/128) is exact in BF16. A BF16-rounded product of two such values is a multiple of 2^-7 in [1, 4), so
    # every FP32 summation order of up to 2^14 products is exact, while the products themselves still round.
    mantissa = torch.randint(0, 128, (rows, hidden), generator=generator, device=device)
    sign = torch.randint(0, 2, (rows, hidden), generator=generator, device=device) * 2 - 1
    return (sign * (1 + mantissa / 128)).to(torch.bfloat16)


def _gate(rows, generator, device):
    logits = torch.randn(rows, 1, generator=generator, device=device) * 3
    return torch.sigmoid(logits.to(torch.bfloat16))


@pytest.mark.parametrize("rows, hidden", [(512, 2048), (256, 4096), (384, 2304)])
def test_gate_bwd_pre_matches_eager_bf16_boundaries(context, rows, hidden):
    """With an order-independent dG sum, dS and dZ equal the eager BF16 operations bitwise."""
    rank, _, device = context
    generator = torch.Generator(device=device).manual_seed(2718 + rank)
    dy = _sign_mantissa(rows, hidden, generator, device)
    shared = _sign_mantissa(rows, hidden, generator, device)
    gate = _gate(rows, generator, device)

    products = dy * shared
    d_gate = products.sum(-1, keepdim=True)
    torch.testing.assert_close(d_gate, products.double().sum(-1, keepdim=True).to(torch.bfloat16), rtol=0, atol=0)
    # FP32 products would change dG, so the bitwise dZ check below also covers the product rounding.
    assert not torch.equal(d_gate, (dy.float() * shared.float()).sum(-1, keepdim=True).to(torch.bfloat16))

    d_shared, d_z = _gate_fused.gate_bwd_pre(dy, shared, gate)
    _assert_bitwise(d_shared, dy * gate)
    _assert_bitwise(d_z, torch.ops.aten.sigmoid_backward.default(d_gate, gate))


def test_gate_bwd_pre_rounds_products_to_bf16(context):
    """Exact cancellation: BF16 products give dZ = 0, FP32 products would leave dZ = 1/65536."""
    _, _, device = context
    dy = torch.zeros(64, 256, device=device, dtype=torch.bfloat16)
    shared = torch.zeros_like(dy)
    dy[:, 0], shared[:, 0] = 1 + 1 / 128, 1 + 1 / 128
    dy[:, 1], shared[:, 1] = -(1 + 1 / 64), 1
    gate = torch.full((dy.shape[0], 1), 0.5, device=device, dtype=torch.bfloat16)
    d_shared, d_z = _gate_fused.gate_bwd_pre(dy, shared, gate)
    _assert_bitwise(d_shared, dy * gate)
    _assert_bitwise(d_z, torch.zeros_like(d_z))


@pytest.mark.parametrize("rows, hidden", [(2048, 2048), (1024, 4096)])
def test_gate_bwd_pre_random_inputs(context, rows, hidden):
    """On unstructured data only the FP32 summation order of dG differs from the PyTorch reduction."""
    rank, _, device = context
    generator = torch.Generator(device=device).manual_seed(3141 + rank)
    dy = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16) * hidden**-0.5
    shared = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16)
    gate = _gate(rows, generator, device)
    d_shared, d_z = _gate_fused.gate_bwd_pre(dy, shared, gate)
    _assert_bitwise(d_shared, dy * gate)
    eager = torch.ops.aten.sigmoid_backward.default((dy * shared).sum(-1, keepdim=True), gate)
    assert (d_z != eager).sum().item() <= rows // 100
    torch.testing.assert_close(d_z, eager, rtol=2**-6, atol=1e-5)


@pytest.mark.parametrize("rows, hidden", [(512, 2048), (256, 4096), (384, 2304)])
def test_gate_bwd_post_matches_eager_merge(context, rows, hidden):
    """Given the same dZ, the in-place merge equals eager d_x + dZ @ Wg bitwise."""
    rank, _, device = context
    generator = torch.Generator(device=device).manual_seed(1618 + rank)
    d_x = torch.randn(rows, hidden, generator=generator, device=device, dtype=torch.bfloat16)
    d_z = (torch.randn(rows, 1, generator=generator, device=device) / 4).to(torch.bfloat16)
    weight = (torch.randn(1, hidden, generator=generator, device=device) * 1.5 * hidden**-0.5).to(torch.bfloat16)
    expected = d_x + d_z @ weight
    actual = d_x.clone()
    assert _gate_fused.gate_bwd_post_(actual, d_z, weight) is actual
    _assert_bitwise(actual, expected)


def test_gate_bwd_post_rejects_non_contiguous(context):
    _, _, device = context
    d_x = torch.zeros(512, 256, device=device, dtype=torch.bfloat16)
    d_z = torch.zeros(512, 1, device=device, dtype=torch.bfloat16)
    weight = torch.ones(1, 256, device=device, dtype=torch.bfloat16)
    for operands in (
        (torch.zeros_like(d_x).t().contiguous().t(), d_z, weight),
        (d_x, d_z, torch.ones(256, 2, device=device, dtype=torch.bfloat16)[:, :1].t()),
    ):
        with pytest.raises(ValueError, match="contiguous"):
            _gate_fused.gate_bwd_post_(*operands)


def test_gate_bwd_empty_rows(context):
    _, _, device = context
    dy = torch.empty(0, 256, device=device, dtype=torch.bfloat16)
    gate = torch.empty(0, 1, device=device, dtype=torch.bfloat16)
    weight = torch.ones(1, 256, device=device, dtype=torch.bfloat16)
    d_shared, d_z = _gate_fused.gate_bwd_pre(dy, dy, gate)
    assert d_shared.shape == (0, 256) and d_z.shape == (0, 1) and d_z.dtype == torch.bfloat16
    assert _gate_fused.gate_bwd_post_(d_shared, d_z, weight) is d_shared
