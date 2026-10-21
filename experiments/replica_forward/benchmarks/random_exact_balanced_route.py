#!/usr/bin/env python3
"""Randomized routes with exact global expert balance.

The construction is a randomized regular bipartite graph between global tokens
and experts. Every token has ``topk`` distinct experts and every expert has the
same global degree. Unlike the historical affine route, token/expert and
source/destination ordering is randomized.
"""

from __future__ import annotations

import torch


def make_random_exact_balanced_topk(
    rank: int,
    world_size: int,
    device: torch.device,
    *,
    num_local_tokens: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> torch.Tensor:
    """Return this rank's slice of a seeded random regular route matrix."""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, {world_size})")
    if not 0 < topk <= num_experts:
        raise ValueError(
            f"topk={topk} must be positive and no larger than experts={num_experts}"
        )

    num_global_tokens = world_size * num_local_tokens
    total_routes = num_global_tokens * topk
    routes_per_expert, route_remainder = divmod(total_routes, num_experts)
    if route_remainder:
        raise ValueError(
            f"total routes={total_routes} must be divisible by "
            f"experts={num_experts}"
        )

    generator = torch.Generator(device=device).manual_seed(seed)

    token_blocks, token_remainder = divmod(num_global_tokens, num_experts)
    if token_remainder == 0:
        # This branch preserves the established construction and fingerprints.
        expert_permutation = torch.rand(
            (token_blocks, num_experts), generator=generator, device=device
        ).argsort(dim=1)
        shifts = torch.rand(
            (token_blocks, num_experts), generator=generator, device=device
        ).argsort(dim=1)[:, :topk]
        position = torch.arange(num_experts, device=device).view(1, num_experts, 1)
        gather_index = (position + shifts[:, None, :]) % num_experts
        block_routes = torch.gather(
            expert_permutation[:, :, None].expand(-1, -1, topk),
            1,
            gather_index,
        ).reshape(num_global_tokens, topk)
    elif num_experts % topk == 0:
        # DSV4 has total_routes divisible by experts although global_tokens is
        # not. Each flattened expert-permutation block contains whole tokens,
        # so every token still selects distinct experts.
        block_routes = torch.rand(
            (routes_per_expert, num_experts),
            generator=generator,
            device=device,
        ).argsort(dim=1).reshape(num_global_tokens, topk)
    else:
        raise ValueError(
            "random exact-balanced construction requires either global tokens "
            "or experts to be divisible by topk-compatible blocks; got "
            f"global_tokens={num_global_tokens}, experts={num_experts}, topk={topk}"
        )

    # Randomize which global/source token receives each regular-graph row. This
    # avoids forcing every source rank to contribute an identical deterministic
    # sequence to every destination rank.
    token_permutation = torch.randperm(
        num_global_tokens, generator=generator, device=device
    )
    global_topk = torch.empty_like(block_routes)
    global_topk[token_permutation] = block_routes

    begin = rank * num_local_tokens
    return global_topk[begin : begin + num_local_tokens].contiguous()
