import torch

from mok import functional


def _call_backward(*, main_grads=None):
    return functional.backward(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        main_grads=main_grads,
    )


def test_backward_combines_physical_gate_up_grads(monkeypatch):
    raw = (
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.full((2, 3, 4), 3.0),
        torch.full((2, 3, 4), 4.0),
        torch.full((2, 4, 3), 5.0),
        torch.full((3, 4), 6.0),
        torch.full((3, 4), 7.0),
        torch.full((4, 3), 8.0),
    )
    monkeypatch.setattr(functional, "_backward_impl", lambda *args, **kwargs: raw)

    result = _call_backward()

    assert result[2].shape == (2, 6, 4)
    assert torch.equal(result[2][:, :3], raw[2])
    assert torch.equal(result[2][:, 3:], raw[3])
    assert result[4].shape == (6, 4)
    assert torch.equal(result[4][:3], raw[5])
    assert torch.equal(result[4][3:], raw[6])


def test_backward_returns_canonical_main_grad_buffers(monkeypatch):
    raw = tuple(torch.tensor([float(index)]) for index in range(8))
    monkeypatch.setattr(functional, "_backward_impl", lambda *args, **kwargs: raw)
    main_grads = tuple(torch.tensor([float(index)]) for index in range(4))

    result = _call_backward(main_grads=main_grads)

    assert result[2] is main_grads[1]
    assert result[3] is main_grads[3]
    assert result[4] is main_grads[0]
    assert result[5] is main_grads[2]
