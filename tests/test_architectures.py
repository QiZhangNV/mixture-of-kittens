"""Run these checks on each target GPU using the same ALL wheel."""

import pytest
import torch
import torch.distributed as dist

from mok import _C, functional

from .test_misc import _run_e2e_case


def _validate(device: torch.device) -> None:
    functional.validate_workspace_args(
        functional.MoKConfig(), dist.group.WORLD, device=device,
        num_local_tokens=512, hidden_size=256, topk=1,
    )


def test_installed_extension_contains_current_device(context) -> None:
    _, _, device = context
    assert torch.cuda.get_device_capability(device) in _C.compiled_architectures
    _validate(device)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
def test_supported_workspace_architectures(context, monkeypatch, capability) -> None:
    _, _, device = context
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(_C, "compiled_architectures", (capability,))
    _validate(device)


def test_workspace_rejects_missing_cubin(context, monkeypatch) -> None:
    _, _, device = context
    monkeypatch.setattr(_C, "compiled_architectures", ())
    with pytest.raises(NotImplementedError, match="MoK was not built for"):
        _validate(device)


@pytest.mark.parametrize("capability", [(9, 0), (11, 0), (12, 0)])
def test_workspace_rejects_unsupported_device(context, monkeypatch, capability) -> None:
    _, _, device = context
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(NotImplementedError, match="requires an SM100, SM103 or SM107"):
        _validate(device)


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("hidden_size,intermediate_size", [(256, 256), (640, 640)])
def test_architecture_forward_backward(context, precision, hidden_size, intermediate_size) -> None:
    """Check routed/shared forward and all gradients against the PyTorch reference."""
    config = functional.MoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2,
        minibatch_size=256, macrobatch_size=1024,
        schedule_capacity_multiplier=1.5,
    )
    _run_e2e_case(
        context, name=f"architecture/{hidden_size}/{intermediate_size}",
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_local_experts=2, topk=2, config=config, precisions=(precision,),
    )
