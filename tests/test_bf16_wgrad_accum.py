import pytest
import torch

from mok import functional
from mok.ops import mxfp8_quantize
from .utils import (
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_reference_bf16,
)


@pytest.mark.parametrize("leave_one_expert_empty", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_wgrad_accumulates_directly_into_fp32_main_grad(
    context: tuple[int, int, torch.device],
    leave_one_expert_empty: bool,
    precision: str,
) -> None:
    rank, world_size, device = context
    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        torch.distributed.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    inputs = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
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
    ) = inputs
    if precision == "mxfp8":
        routed_gate_quantized = mxfp8_quantize(w_routed_gate, True, True)
        routed_up_quantized = mxfp8_quantize(w_routed_up, True, True)
        routed_down_quantized = mxfp8_quantize(w_routed_down, True, True)
        forward_routed_gate = routed_gate_quantized[:2]
        forward_routed_up = routed_up_quantized[:2]
        forward_routed_down = routed_down_quantized[:2]
        backward_routed_gate = routed_gate_quantized
        backward_routed_up = routed_up_quantized
        backward_routed_down = routed_down_quantized[2:]
        tolerance = MXFP8_TOLERANCE
    else:
        forward_routed_gate = w_routed_gate
        forward_routed_up = w_routed_up
        forward_routed_down = w_routed_down
        backward_routed_gate = w_routed_gate
        backward_routed_up = w_routed_up
        backward_routed_down = w_routed_down
        tolerance = BF16_TOLERANCE
    if leave_one_expert_empty:
        topk_experts[:, 0] = (
            torch.arange(num_local_tokens, device=device) + rank
        ) % (num_experts - 1)
    schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    _, forward_context = functional.forward(
        config,
        workspace,
        schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        forward_routed_gate,
        forward_routed_up,
        forward_routed_down,
    )

    main_grads = (
        torch.full_like(w_shared_gate, 0.25, dtype=torch.float32),
        torch.full_like(w_routed_gate, 0.25, dtype=torch.float32),
        torch.full_like(w_shared_up, 0.25, dtype=torch.float32),
        torch.full_like(w_routed_up, 0.25, dtype=torch.float32),
        torch.full_like(w_shared_down, 0.25, dtype=torch.float32),
        torch.full_like(w_routed_down, 0.25, dtype=torch.float32),
    )
    d_x, d_router_weights, *_ = functional.backward(
        config,
        workspace,
        schedule,
        forward_context,
        d_output,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        backward_routed_gate,
        backward_routed_up,
        backward_routed_down,
        main_grads=main_grads,
    )

    reference = run_reference_bf16(*inputs)[1:]
    check_correctness("d_x", reference[0], d_x, tolerance, rank == 0)
    check_correctness(
        "d_router_weights", reference[1], d_router_weights, tolerance, rank == 0
    )
    reference_wgrads = (
        reference[5],
        reference[2],
        reference[6],
        reference[3],
        reference[7],
        reference[4],
    )
    for name, expected, actual in zip(
        (
            "shared_gate", "routed_gate", "shared_up",
            "routed_up", "shared_down", "routed_down",
        ),
        reference_wgrads,
        main_grads,
        strict=True,
    ):
        check_correctness(
            f"main_grad/{name}",
            expected.float() + 0.25,
            actual,
            tolerance,
            rank == 0,
        )
