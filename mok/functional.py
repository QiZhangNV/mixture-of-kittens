import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .ops import (
    all_gather_top_experts,
    barrier_all,
    bwd_epilogue,
    dispatch_mlp_swiglu_combine_bwd_mxfp8,
    dispatch_mlp_swiglu_combine_bwd_bf16,
    dispatch_mlp_swiglu_combine_bwd_mxfp8_accum,
    dispatch_mlp_swiglu_combine_bwd_bf16_accum,
    recompute_forward_context_mxfp8,
    recompute_forward_context_bf16,
    dispatch_mlp_swiglu_combine_fwd_mxfp8,
    dispatch_mlp_swiglu_combine_fwd_bf16,
    fwd_epilogue,
    schedule,
)


@dataclass(frozen=True, slots=True)
class MoKConfig:
    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 0.5
    all_gather_top_experts_chunk_bytes: int = 2048


@dataclass(frozen=True, slots=True)
class MoKSchedule:
    peer_rank: torch.Tensor          # (schedule_capacity,) int32
    peer_token_idx: torch.Tensor     # (schedule_capacity,) int32
    num_tokens: torch.Tensor         # (1,) int32
    tokens_per_expert: torch.Tensor  # (num_local_experts,) int32
    top_experts: torch.Tensor  # (num_local_tokens, topk) int32; -1 marks no route


@dataclass(frozen=True, slots=True)
class MoKForwardContext:
    x_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    up_shared: torch.Tensor
    up_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class SplitRoutedWeight:
    """One logical routed matrix stored in independent per-expert allocations.

    ``data`` is a representative expert tensor used for dtype/shape metadata;
    ``storage_table`` contains one TK/TMA layout per local expert and owns no
    weight payload. MXFP8 callers additionally provide one tcgen05-layout
    scale tensor per expert plus a descriptor table, and the native or legacy
    columnwise payload used by dgrad. ``scale_tensors`` fields retain ownership
    of generated scale layouts; descriptor tables do not own their payloads.

    Gate and up may intentionally share the same full ``[2I, H]`` data,
    descriptor table, and scale tensor. MoK then applies its existing logical
    gate/up row offsets without repacking that FC1 parameter.
    """

    data: torch.Tensor
    storage_table: torch.Tensor
    scale: torch.Tensor | None = None
    scale_storage_table: torch.Tensor | None = None
    scale_tensors: tuple[torch.Tensor, ...] | None = None
    transposed_data: torch.Tensor | None = None
    transposed_scale: torch.Tensor | None = None
    transposed_storage_table: torch.Tensor | None = None
    transposed_scale_storage_table: torch.Tensor | None = None
    transposed_scale_tensors: tuple[torch.Tensor, ...] | None = None
    native_columnwise: bool = False


@dataclass(slots=True)
class MoKWorkspace:
    group_name: str                                   # () str
    ep_rank: int                                      # () int
    ep_size: int                                      # () int
    device: torch.device                              # () torch.device
    num_local_tokens: int                             # () int
    hidden_size: int                                  # () int
    topk: int                                         # () int
    schedule_capacity: int                            # () int
    x_buffer: torch.Tensor                            # (num_local_tokens, hidden_size) bfloat16
    x_buffer_handle: Any                              # () SymmetricMemory handle
    x_buffer_ptrs: list[int]                          # (ep_size,) uintptr64
    combine_buffer: torch.Tensor                      # (num_local_tokens * topk, hidden_size) bfloat16
    combine_buffer_handle: Any                        # () SymmetricMemory handle
    combine_buffer_ptrs: list[int]                    # (ep_size,) uintptr64
    d_y_buffer: torch.Tensor                          # (num_local_tokens, hidden_size) bfloat16
    d_y_buffer_handle: Any                            # () SymmetricMemory handle
    d_y_buffer_ptrs: list[int]                        # (ep_size,) uintptr64
    d_x_routed_buffer: torch.Tensor                   # (num_local_tokens * topk, hidden_size) bfloat16
    d_x_routed_buffer_handle: Any                     # () SymmetricMemory handle
    d_x_routed_buffer_ptrs: list[int]                 # (ep_size,) uintptr64
    router_weight_buffer: torch.Tensor                # (num_local_tokens, topk) float32
    router_weight_buffer_handle: Any                  # () SymmetricMemory handle
    router_weight_buffer_ptrs: list[int]              # (ep_size,) uintptr64
    d_router_weight_buffer: torch.Tensor              # (num_local_tokens, topk) float32
    d_router_weight_buffer_handle: Any                # () SymmetricMemory handle
    d_router_weight_buffer_ptrs: list[int]            # (ep_size,) uintptr64
    all_gather_top_experts_buffer: torch.Tensor       # (ep_size, num_local_tokens, topk) int32
    all_gather_top_experts_buffer_handle: Any         # () SymmetricMemory handle
    all_gather_top_experts_buffer_multicast_ptr: int  # () uintptr64
    barrier_buffer: torch.Tensor                      # (1,) int32
    barrier_buffer_handle: Any                        # () SymmetricMemory handle
    barrier_buffer_ptrs: list[int]                    # (ep_size,) uintptr64
    barrier_buffer_multicast_ptr: int                 # () uintptr64
    barrier_target: torch.Tensor                      # (1,) int32


_WORKSPACE_CACHE: dict[tuple[str, int, int, int, int, int], MoKWorkspace] = {}


def validate_workspace_args(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> None:
    """Validates the arguments used to create or retrieve a workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        None
    """
    if (
        type(config.schedule_capacity_multiplier) not in (int, float)
        or not math.isfinite(config.schedule_capacity_multiplier)
        or config.schedule_capacity_multiplier <= 0
    ):
        raise ValueError("schedule_capacity_multiplier must be a positive finite number")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError("MoK workspace device must be the current CUDA device")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("MoK currently requires an SM100 or SM103 GPU")
    device_properties = torch.cuda.get_device_properties(device)
    if type(num_local_tokens) is not int or num_local_tokens < 512:
        raise ValueError("num_local_tokens must be an integer at least 512")
    if num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be divisible by 256")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if hidden_size % 256 != 0:
        raise ValueError("hidden_size must be divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    fwd_epilogue_smem_bytes = 2 * ((topk + 1) * 2048 + 2 * topk * 4) + 1024
    if fwd_epilogue_smem_bytes > device_properties.shared_memory_per_block_optin:
        raise ValueError("topk requires more dynamic shared memory than the device supports")

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("MoK EP size must be one of 1, 4, 8, 16, 32, 64")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError("current process is not a member of the EP process group")


def create_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Creates a new caller-owned workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    local_shape = torch.tensor([num_local_tokens, hidden_size, topk], dtype=torch.int64, device=device)
    gathered_shapes = torch.empty(ep_size * local_shape.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(gathered_shapes == local_shape).item():  # .item() here is fine since this is one-time setup
        raise ValueError("MoK requires identical token, hidden, and top-k shapes on every EP rank")
    if ep_size > 1:
        symm_mem.enable_symm_mem_for_group(group_name)

    schedule_capacity = num_local_tokens * topk * schedule_capacity_factor

    def allocate_buffer(*shape: int, dtype: torch.dtype, zero: bool = False) -> tuple[torch.Tensor, Any, list[int]]:
        if ep_size == 1:
            if zero:
                buffer = torch.zeros(*shape, dtype=dtype, device=device)
            else:
                buffer = torch.empty(*shape, dtype=dtype, device=device)
            handle = None
            ptrs = [int(buffer.data_ptr())]
        else:
            buffer = symm_mem.empty(*shape, dtype=dtype, device=device)
            if zero:
                buffer.zero_()
            handle = symm_mem.rendezvous(buffer, group_name)
            ptrs = [int(handle.buffer_ptrs[peer_rank]) for peer_rank in range(ep_size)]
        return buffer, handle, ptrs

    x_buffer, x_buffer_handle, x_buffer_ptrs = allocate_buffer(num_local_tokens, hidden_size, dtype=torch.bfloat16)
    combine_buffer, combine_buffer_handle, combine_buffer_ptrs = allocate_buffer(num_local_tokens * topk, hidden_size, dtype=torch.bfloat16)
    d_y_buffer, d_y_buffer_handle, d_y_buffer_ptrs = allocate_buffer(num_local_tokens, hidden_size, dtype=torch.bfloat16)
    d_x_routed_buffer, d_x_routed_buffer_handle, d_x_routed_buffer_ptrs = allocate_buffer(num_local_tokens * topk, hidden_size, dtype=torch.bfloat16)
    router_weight_buffer, router_weight_buffer_handle, router_weight_buffer_ptrs = allocate_buffer(num_local_tokens, topk, dtype=torch.float32)
    d_router_weight_buffer, d_router_weight_buffer_handle, d_router_weight_buffer_ptrs = allocate_buffer(num_local_tokens, topk, dtype=torch.float32)

    all_gather_top_experts_buffer, all_gather_top_experts_buffer_handle, _ = allocate_buffer(ep_size, num_local_tokens, topk, dtype=torch.int32)
    all_gather_top_experts_buffer_multicast_ptr = int(
        all_gather_top_experts_buffer.data_ptr()
        if all_gather_top_experts_buffer_handle is None
        else all_gather_top_experts_buffer_handle.multicast_ptr
    )

    barrier_buffer, barrier_buffer_handle, barrier_buffer_ptrs = allocate_buffer(1, dtype=torch.int32, zero=True)
    barrier_buffer_multicast_ptr = int(
        barrier_buffer.data_ptr()
        if barrier_buffer_handle is None
        else barrier_buffer_handle.multicast_ptr
    )
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)

    dist.barrier(group=group, async_op=True, device_ids=[device_index]).block_current_stream()

    workspace = MoKWorkspace(
        group_name=group_name, ep_rank=ep_rank, ep_size=ep_size, device=device,
        num_local_tokens=num_local_tokens, hidden_size=hidden_size, topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=x_buffer, x_buffer_handle=x_buffer_handle, x_buffer_ptrs=x_buffer_ptrs,
        combine_buffer=combine_buffer, combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        d_y_buffer=d_y_buffer, d_y_buffer_handle=d_y_buffer_handle,
        d_y_buffer_ptrs=d_y_buffer_ptrs,
        d_x_routed_buffer=d_x_routed_buffer, d_x_routed_buffer_handle=d_x_routed_buffer_handle,
        d_x_routed_buffer_ptrs=d_x_routed_buffer_ptrs,
        router_weight_buffer=router_weight_buffer,
        router_weight_buffer_handle=router_weight_buffer_handle,
        router_weight_buffer_ptrs=router_weight_buffer_ptrs,
        d_router_weight_buffer=d_router_weight_buffer,
        d_router_weight_buffer_handle=d_router_weight_buffer_handle,
        d_router_weight_buffer_ptrs=d_router_weight_buffer_ptrs,
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=all_gather_top_experts_buffer_handle,
        all_gather_top_experts_buffer_multicast_ptr=all_gather_top_experts_buffer_multicast_ptr,
        barrier_buffer=barrier_buffer, barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
    )
    return workspace


def get_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Returns a cached workspace, creating and caching one if absent.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(2, math.ceil(ep_size * config.schedule_capacity_multiplier))

    cache_key = (group_name, device_index, num_local_tokens, hidden_size, topk, schedule_capacity_factor)
    cached_workspace = _WORKSPACE_CACHE.get(cache_key)
    if cached_workspace is not None:
        return cached_workspace

    workspace = create_workspace(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    _WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def clear_workspace_cache() -> None:
    """Clears cached workspaces after all participating ranks synchronize.

    Inputs:
        None

    Outputs:
        None
    """
    for workspace in _WORKSPACE_CACHE.values():
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                    workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        torch.cuda.synchronize(workspace.device)
    _WORKSPACE_CACHE.clear()


def build_schedule(
    workspace: MoKWorkspace,
    config: MoKConfig,
    top_experts: torch.Tensor,
    *,
    num_local_experts: int,
) -> MoKSchedule:
    """All-gathers routing choices and builds this rank's padded expert schedule.

    Inputs:
        workspace:         MoKWorkspace
        config:            MoKConfig
        top_experts:       int32 or int64 [num_local_tokens, topk]
        num_local_experts: int

    Outputs:
        schedule: MoKSchedule
    """
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    device_properties = torch.cuda.get_device_properties(workspace.device)
    if type(config.fwd_num_comm_sms) is not int or config.fwd_num_comm_sms <= 0:
        raise ValueError("fwd_num_comm_sms must be a positive integer")
    if config.fwd_num_comm_sms % 2 != 0:
        raise ValueError("fwd_num_comm_sms must be even")
    if type(config.bwd_num_comm_sms) is not int or config.bwd_num_comm_sms <= 0:
        raise ValueError("bwd_num_comm_sms must be a positive integer")
    if config.bwd_num_comm_sms % 2 != 0:
        raise ValueError("bwd_num_comm_sms must be even")
    if (config.fwd_num_comm_sms >= device_properties.multi_processor_count
            or config.bwd_num_comm_sms >= device_properties.multi_processor_count):
        raise ValueError("communication-SM counts must leave at least one compute SM")
    if (type(config.minibatch_size) is not int or config.minibatch_size <= 0
            or config.minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(config.macrobatch_size) is not int or config.macrobatch_size <= 0
            or config.macrobatch_size % config.minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if (type(config.all_gather_top_experts_chunk_bytes) is not int
            or config.all_gather_top_experts_chunk_bytes <= 0
            or config.all_gather_top_experts_chunk_bytes % 16 != 0):
        raise ValueError("all_gather_top_experts_chunk_bytes must be a positive multiple of 16")
    if (config.all_gather_top_experts_chunk_bytes + 1024
            > device_properties.shared_memory_per_block_optin):
        raise ValueError("all_gather_top_experts_chunk_bytes exceeds device dynamic shared-memory capacity")
    route_buffer_bytes = workspace.num_local_tokens * workspace.topk * 4
    if route_buffer_bytes % config.all_gather_top_experts_chunk_bytes != 0:
        raise ValueError("all_gather_top_experts_chunk_bytes must divide one rank's route-buffer bytes")
    if not top_experts.is_cuda or top_experts.device != workspace.device:
        raise ValueError("top_experts must be on the workspace CUDA device")
    if top_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("top_experts must have dtype torch.int32 or torch.int64")
    if not top_experts.is_contiguous():
        raise ValueError("top_experts must be contiguous")
    if tuple(top_experts.shape) != (workspace.num_local_tokens, workspace.topk):
        raise ValueError("top_experts must have shape (num_local_tokens, topk)")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")

    top_experts_int32 = (
        top_experts if top_experts.dtype == torch.int32 else top_experts.to(torch.int32)
    )
    all_gather_top_experts(
        top_experts_int32, workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr, workspace.ep_rank,
        config.all_gather_top_experts_chunk_bytes,
    )
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    (schedule_peer_rank, schedule_peer_token_idx,
     num_tokens, tokens_per_expert) = schedule(
        workspace.all_gather_top_experts_buffer, num_local_experts,
        workspace.schedule_capacity, workspace.ep_rank)
    return MoKSchedule(
        peer_rank=schedule_peer_rank,
        peer_token_idx=schedule_peer_token_idx,
        num_tokens=num_tokens,
        tokens_per_expert=tokens_per_expert,
        top_experts=top_experts_int32,
    )


def validate_inputs(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor | None = None,
    grad_output: torch.Tensor | None = None,
) -> None:
    """Validates runtime inputs against the workspace and schedule.

    Inputs:
        config:         MoKConfig
        workspace:      MoKWorkspace
        schedule:       MoKSchedule
        x:              bfloat16 [num_local_tokens, hidden_size]
        router_weights: float32 [num_local_tokens, topk] | None
        grad_output:    bfloat16 [num_local_tokens, hidden_size] | None

    Outputs:
        None
    """
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    expected_activation_shape = (workspace.num_local_tokens, workspace.hidden_size)
    tensors = [("x", x, torch.bfloat16, expected_activation_shape)]
    if grad_output is not None:
        tensors.append(("grad_output", grad_output, torch.bfloat16, expected_activation_shape))
    if router_weights is not None:
        tensors.append(
            (
                "router_weights",
                router_weights,
                torch.float32,
                (workspace.num_local_tokens, workspace.topk),
            )
        )
    tensors.append(
        (
            "schedule.top_experts",
            schedule.top_experts,
            torch.int32,
            (workspace.num_local_tokens, workspace.topk),
        )
    )
    for tensor_name, tensor, expected_dtype, expected_shape in tensors:
        if not tensor.is_cuda or tensor.device != workspace.device or tensor.dtype != expected_dtype or not tensor.is_contiguous():
            raise ValueError(f"{tensor_name} must be contiguous {expected_dtype} on the workspace CUDA device")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} shape does not match the workspace")
    if schedule.peer_rank.numel() != workspace.schedule_capacity:
        raise ValueError("schedule capacity does not match the workspace")


def forward(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | SplitRoutedWeight,
    routed_up_weights: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | SplitRoutedWeight,
    routed_down_weights: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | SplitRoutedWeight,
    swiglu_limit: float | None = None,
) -> tuple[
    torch.Tensor,
    MoKForwardContext,
]:
    """Runs the MoE forward pass.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 representation
        swiglu_limit:        float | None

    Outputs:
        output:          bfloat16 [num_local_tokens, hidden_size]
        forward_context: MoKForwardContext
    """
    validate_inputs(config, workspace, schedule, x, router_weights)

    shared_fc2_weights = shared_down_weights
    routed_fc2_weights = routed_down_weights

    workspace.x_buffer.copy_(x)  # TODO: we can remove this
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

    split_weights = isinstance(routed_gate_weights, SplitRoutedWeight)
    if split_weights != isinstance(
        routed_up_weights, SplitRoutedWeight
    ) or split_weights != isinstance(routed_fc2_weights, SplitRoutedWeight):
        raise TypeError(
            "routed gate/up/down weights must use the same storage representation"
        )
    routed_precision_is_mxfp8 = (
        routed_gate_weights.scale is not None
        if split_weights
        else isinstance(routed_gate_weights, tuple)
    )
    if split_weights:
        split_triplet = (routed_gate_weights, routed_up_weights, routed_fc2_weights)
        if any(
            (weight.scale is not None) != routed_precision_is_mxfp8
            for weight in split_triplet
        ):
            raise ValueError("split routed gate/up/down weights must use one precision")

    if routed_precision_is_mxfp8:
        if split_weights:
            if any(weight.scale_storage_table is None for weight in split_triplet):
                raise ValueError(
                    "all split MXFP8 routed weights must provide scale descriptor tables"
                )
            routed_gate_weights_fp8 = routed_gate_weights.data
            routed_gate_weights_sc = routed_gate_weights.scale
            routed_up_weights_fp8 = routed_up_weights.data
            routed_up_weights_sc = routed_up_weights.scale
            routed_fc2_weights_fp8 = routed_fc2_weights.data
            routed_fc2_weights_sc = routed_fc2_weights.scale
            routed_storage_tables = (
                routed_gate_weights.storage_table,
                routed_up_weights.storage_table,
                routed_fc2_weights.storage_table,
            )
            routed_scale_storage_tables = (
                routed_gate_weights.scale_storage_table,
                routed_up_weights.scale_storage_table,
                routed_fc2_weights.scale_storage_table,
            )
        else:
            routed_gate_weights_fp8, routed_gate_weights_sc = routed_gate_weights
            routed_up_weights_fp8, routed_up_weights_sc = routed_up_weights
            routed_fc2_weights_fp8, routed_fc2_weights_sc = routed_fc2_weights
            routed_storage_tables = (None, None, None)
            routed_scale_storage_tables = (None, None, None)
        (
            x_fp8_t_routed,
            x_sc_t_routed,
            gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            up_shared,
            up_fp8_routed,
            up_sc_routed,
            hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            y_shared,
            y_routed,
        ) = dispatch_mlp_swiglu_combine_fwd_mxfp8(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            shared_gate_weights,
            routed_gate_weights_fp8,
            routed_gate_weights_sc,
            shared_up_weights,
            routed_up_weights_fp8,
            routed_up_weights_sc,
            shared_fc2_weights,
            routed_fc2_weights_fp8,
            routed_fc2_weights_sc,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            swiglu_limit,
            config.fwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            *routed_storage_tables,
            *routed_scale_storage_tables,
        )
        forward_context = MoKForwardContext(
            x_routed=(x_fp8_t_routed, x_sc_t_routed),
            gate_shared=gate_shared,
            gate_routed=(gate_fp8_routed, gate_sc_routed),
            up_shared=up_shared,
            up_routed=(up_fp8_routed, up_sc_routed),
            hidden_shared=hidden_shared,
            hidden_routed=(hidden_fp8_t_routed, hidden_sc_t_routed),
        )
    else:
        if split_weights:
            routed_gate_data = routed_gate_weights.data
            routed_up_data = routed_up_weights.data
            routed_fc2_data = routed_fc2_weights.data
            routed_storage_tables = (
                routed_gate_weights.storage_table,
                routed_up_weights.storage_table,
                routed_fc2_weights.storage_table,
            )
        else:
            routed_gate_data = routed_gate_weights
            routed_up_data = routed_up_weights
            routed_fc2_data = routed_fc2_weights
            routed_storage_tables = (None, None, None)
        (
            x_routed,
            gate_shared,
            gate_routed,
            up_shared,
            up_routed,
            hidden_shared,
            hidden_routed,
            y_shared,
            y_routed,
        ) = dispatch_mlp_swiglu_combine_fwd_bf16(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            shared_gate_weights,
            routed_gate_data,
            shared_up_weights,
            routed_up_data,
            shared_fc2_weights,
            routed_fc2_data,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            swiglu_limit,
            config.fwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            *routed_storage_tables,
        )
        forward_context = MoKForwardContext(
            x_routed=x_routed,
            gate_shared=gate_shared,
            gate_routed=gate_routed,
            up_shared=up_shared,
            up_routed=up_routed,
            hidden_shared=hidden_shared,
            hidden_routed=hidden_routed,
        )

    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    output = fwd_epilogue(
        y_shared,
        workspace.combine_buffer,
        workspace.router_weight_buffer,
        schedule.top_experts,
    )
    return output, forward_context


def recompute_forward_context(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
) -> MoKForwardContext:
    """Recomputes the intermediates needed by the MoK backward pass.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        x:                   bfloat16 [num_local_tokens, hidden_size]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 representation
        swiglu_limit:        float | None

    Outputs:
        forward_context: MoKForwardContext
    """
    validate_inputs(config, workspace, schedule, x)
    if isinstance(routed_gate_weights, tuple) != isinstance(routed_up_weights, tuple):
        raise TypeError("routed gate and up weights must use the same precision representation")
    if isinstance(routed_gate_weights, tuple) and (len(routed_gate_weights) != 2 or len(routed_up_weights) != 2):
        raise ValueError("MXFP8 routed gate and up weights must be data/scale pairs")

    workspace.x_buffer.copy_(x)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

    if isinstance(routed_gate_weights, tuple):
        routed_gate_weights_fp8, routed_gate_weights_sc = routed_gate_weights
        routed_up_weights_fp8, routed_up_weights_sc = routed_up_weights
        (x_fp8_t_routed, x_sc_t_routed,
         gate_shared, gate_fp8_routed, gate_sc_routed,
         up_shared, up_fp8_routed, up_sc_routed,
         hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed) = recompute_forward_context_mxfp8(
            workspace.x_buffer, workspace.x_buffer_ptrs,
            shared_gate_weights, routed_gate_weights_fp8, routed_gate_weights_sc,
            shared_up_weights, routed_up_weights_fp8, routed_up_weights_sc,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.fwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )
        x_routed = (x_fp8_t_routed, x_sc_t_routed)
        gate_routed = (gate_fp8_routed, gate_sc_routed)
        up_routed = (up_fp8_routed, up_sc_routed)
        hidden_routed = (hidden_fp8_t_routed, hidden_sc_t_routed)
    else:
        (x_routed, gate_shared, gate_routed, up_shared, up_routed,
         hidden_shared, hidden_routed) = recompute_forward_context_bf16(
            workspace.x_buffer, workspace.x_buffer_ptrs,
            shared_gate_weights, routed_gate_weights,
            shared_up_weights, routed_up_weights,
            schedule.peer_rank, schedule.peer_token_idx,
            schedule.num_tokens, schedule.tokens_per_expert,
            workspace.topk, swiglu_limit, config.fwd_num_comm_sms,
            config.macrobatch_size, config.minibatch_size,
        )

    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    return MoKForwardContext(
        x_routed=x_routed,
        gate_shared=gate_shared,
        gate_routed=gate_routed,
        up_shared=up_shared,
        up_routed=up_routed,
        hidden_shared=hidden_shared,
        hidden_routed=hidden_routed,
    )


def backward(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    forward_context: MoKForwardContext,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor,
    shared_up_weights: torch.Tensor,
    shared_down_weights: torch.Tensor,
    routed_gate_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    routed_up_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    routed_down_weights: torch.Tensor | tuple[torch.Tensor, ...] | SplitRoutedWeight,
    swiglu_limit: float | None = None,
    main_grads: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    main_grad_storage_tables: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the MoE backward pass.

    Inputs:
        config:              MoKConfig
        workspace:           MoKWorkspace
        schedule:            MoKSchedule
        forward_context:     MoKForwardContext
        grad_output:         bfloat16 [num_local_tokens, hidden_size]
        x:                   bfloat16 [num_local_tokens, hidden_size]
        router_weights:      float32 [num_local_tokens, topk]
        shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        shared_down_weights: bfloat16 [hidden_size, intermediate_size]
        routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 tensor tuple
        routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size] or MXFP8 tensor tuple
        routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size] or MXFP8 tensor tuple
        swiglu_limit:        float | None
        main_grads:          optional FP32 or BF16 buffers ordered as shared gate,
                             routed gate, shared up, routed up, shared down,
                             routed down; wgrad is accumulated in-place

    Outputs:
        d_x:                   bfloat16 [num_local_tokens, hidden_size]
        d_router_weights:      float32 [num_local_tokens, topk]
        d_routed_gate_weights: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_up_weights:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_routed_down_weights: bfloat16 [num_local_experts, hidden_size, intermediate_size]
        d_shared_gate_weights: bfloat16 [intermediate_size, hidden_size]
        d_shared_up_weights:   bfloat16 [intermediate_size, hidden_size]
        d_shared_down_weights: bfloat16 [hidden_size, intermediate_size]
    """
    validate_inputs(config, workspace, schedule, x, router_weights, grad_output)
    shared_fc2_weights = shared_down_weights
    routed_fc2_weights = routed_down_weights
    if not isinstance(forward_context, MoKForwardContext):
        raise TypeError("forward_context must be a MoKForwardContext")

    workspace.d_y_buffer.copy_(grad_output)  # TODO: we can remove this
    workspace.x_buffer.copy_(x)  # TODO: we can remove this
    workspace.router_weight_buffer.copy_(router_weights)  # TODO: we can remove this
    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    split_weights = isinstance(routed_gate_weights, SplitRoutedWeight)
    if split_weights != isinstance(
        routed_up_weights, SplitRoutedWeight
    ) or split_weights != isinstance(routed_fc2_weights, SplitRoutedWeight):
        raise TypeError(
            "routed gate/up/down weights must use the same storage representation"
        )
    if split_weights and main_grads is None:
        raise ValueError("split routed weights require fused main-grad accumulation")
    if (main_grad_storage_tables is not None) != split_weights:
        raise ValueError(
            "split routed weights require three split main-grad storage tables"
        )
    routed_precision_is_mxfp8 = (
        routed_gate_weights.scale is not None
        if split_weights
        else isinstance(routed_gate_weights, tuple)
    )
    if routed_precision_is_mxfp8:
        if split_weights:
            split_triplet = (
                routed_gate_weights,
                routed_up_weights,
                routed_fc2_weights,
            )
            if any(weight.scale is None for weight in split_triplet):
                raise ValueError("all split MXFP8 routed weights must provide scales")
            if any(
                weight.transposed_data is None
                or weight.transposed_scale is None
                or weight.transposed_storage_table is None
                or weight.scale_storage_table is None
                or weight.transposed_scale_storage_table is None
                for weight in split_triplet
            ):
                raise ValueError(
                    "split MXFP8 routed weights require rowwise/columnwise data, "
                    "scales, and descriptors"
                )
            native_columnwise_values = {
                weight.native_columnwise for weight in split_triplet
            }
            if len(native_columnwise_values) != 1:
                raise ValueError(
                    "split MXFP8 routed weights must agree on columnwise layout"
                )
            routed_weights_are_native_columnwise = routed_gate_weights.native_columnwise
            routed_gate_weights_fp8 = routed_gate_weights.data
            routed_gate_weights_sc = routed_gate_weights.scale
            routed_gate_weights_t_fp8 = routed_gate_weights.transposed_data
            routed_gate_weights_t_sc = routed_gate_weights.transposed_scale
            routed_up_weights_fp8 = routed_up_weights.data
            routed_up_weights_sc = routed_up_weights.scale
            routed_up_weights_t_fp8 = routed_up_weights.transposed_data
            routed_up_weights_t_sc = routed_up_weights.transposed_scale
            routed_fc2_weights_t_fp8 = routed_fc2_weights.transposed_data
            routed_fc2_weights_t_sc = routed_fc2_weights.transposed_scale
            weight_storage_tables = (
                routed_gate_weights.storage_table,
                routed_up_weights.storage_table,
                routed_gate_weights.transposed_storage_table,
                routed_up_weights.transposed_storage_table,
                routed_fc2_weights.transposed_storage_table,
            )
            scale_storage_tables = (
                routed_gate_weights.scale_storage_table,
                routed_up_weights.scale_storage_table,
                routed_gate_weights.transposed_scale_storage_table,
                routed_up_weights.transposed_scale_storage_table,
                routed_fc2_weights.transposed_scale_storage_table,
            )
        elif len(routed_gate_weights) == 5:
            (
                routed_gate_weights_fp8,
                routed_gate_weights_sc,
                routed_gate_weights_t_fp8,
                routed_gate_weights_t_sc,
                routed_weights_are_native_columnwise,
            ) = routed_gate_weights
            if len(routed_up_weights) != 5 or len(routed_fc2_weights) != 3:
                raise ValueError(
                    "native-columnwise MXFP8 gate/up/down tuples must have lengths 5/5/3"
                )
            (
                routed_up_weights_fp8,
                routed_up_weights_sc,
                routed_up_weights_t_fp8,
                routed_up_weights_t_sc,
                routed_up_is_native_columnwise,
            ) = routed_up_weights
            (
                routed_fc2_weights_t_fp8,
                routed_fc2_weights_t_sc,
                routed_fc2_is_native_columnwise,
            ) = routed_fc2_weights
            if (
                routed_weights_are_native_columnwise is not True
                or routed_up_is_native_columnwise is not True
                or routed_fc2_is_native_columnwise is not True
            ):
                raise ValueError("all native-columnwise MXFP8 flags must be True")
            weight_storage_tables = None
            scale_storage_tables = None
        elif len(routed_gate_weights) == 4:
            (
                routed_gate_weights_fp8,
                routed_gate_weights_sc,
                routed_gate_weights_t_fp8,
                routed_gate_weights_t_sc,
            ) = routed_gate_weights
            (
                routed_up_weights_fp8,
                routed_up_weights_sc,
                routed_up_weights_t_fp8,
                routed_up_weights_t_sc,
            ) = routed_up_weights
            routed_fc2_weights_t_fp8, routed_fc2_weights_t_sc = routed_fc2_weights
            routed_weights_are_native_columnwise = False
            weight_storage_tables = None
            scale_storage_tables = None
        else:
            raise ValueError("MXFP8 gate weight tuple must contain four or five values")
        x_fp8_t_routed, x_sc_t_routed = forward_context.x_routed
        gate_fp8_routed, gate_sc_routed = forward_context.gate_routed
        up_fp8_routed, up_sc_routed = forward_context.up_routed
        hidden_fp8_t_routed, hidden_sc_t_routed = forward_context.hidden_routed
        mxfp8_bwd_args = (
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights,
            routed_gate_weights_t_fp8,
            routed_gate_weights_t_sc,
            shared_up_weights,
            routed_up_weights_t_fp8,
            routed_up_weights_t_sc,
            shared_fc2_weights,
            routed_fc2_weights_t_fp8,
            routed_fc2_weights_t_sc,
            x_fp8_t_routed,
            x_sc_t_routed,
            forward_context.gate_shared,
            gate_fp8_routed,
            gate_sc_routed,
            forward_context.up_shared,
            up_fp8_routed,
            up_sc_routed,
            forward_context.hidden_shared,
            hidden_fp8_t_routed,
            hidden_sc_t_routed,
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            routed_gate_weights_fp8,
            routed_gate_weights_sc,
            routed_up_weights_fp8,
            routed_up_weights_sc,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            routed_weights_are_native_columnwise,
        )
        if main_grads is None:
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_fp8_routed,
                d_gate_sc_routed,
                d_up_shared,
                d_up_fp8_routed,
                d_up_sc_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_fp8_routed,
                d_y_sc_routed,
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_fc2,
                d_w_routed_fc2,
            ) = dispatch_mlp_swiglu_combine_bwd_mxfp8(*mxfp8_bwd_args)
        else:
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_fp8_routed,
                d_gate_sc_routed,
                d_up_shared,
                d_up_fp8_routed,
                d_up_sc_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_fp8_routed,
                d_y_sc_routed,
            ) = dispatch_mlp_swiglu_combine_bwd_mxfp8_accum(
                *mxfp8_bwd_args,
                main_grads=main_grads,
                weight_storage_tables=weight_storage_tables,
                scale_storage_tables=scale_storage_tables,
                main_grad_storage_tables=main_grad_storage_tables,
            )
            (
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_fc2,
                d_w_routed_fc2,
            ) = main_grads
    else:
        if split_weights:
            if any(
                weight.scale is not None
                for weight in (
                    routed_gate_weights,
                    routed_up_weights,
                    routed_fc2_weights,
                )
            ):
                raise ValueError("split BF16 routed weights must not provide scales")
            routed_gate_data = routed_gate_weights.data
            routed_up_data = routed_up_weights.data
            routed_fc2_data = routed_fc2_weights.data
            weight_storage_tables = (
                routed_gate_weights.storage_table,
                routed_up_weights.storage_table,
                routed_fc2_weights.storage_table,
            )
        else:
            routed_gate_data = routed_gate_weights
            routed_up_data = routed_up_weights
            routed_fc2_data = routed_fc2_weights
            weight_storage_tables = None
        x_routed = forward_context.x_routed
        gate_routed = forward_context.gate_routed
        up_routed = forward_context.up_routed
        hidden_routed = forward_context.hidden_routed
        bwd_args = (
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights,
            routed_gate_data,
            shared_up_weights,
            routed_up_data,
            shared_fc2_weights,
            routed_fc2_data,
            x_routed,
            forward_context.gate_shared,
            gate_routed,
            forward_context.up_shared,
            up_routed,
            forward_context.hidden_shared,
            hidden_routed,
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        if main_grads is None:
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_routed,
                d_up_shared,
                d_up_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_routed,
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_fc2,
                d_w_routed_fc2,
            ) = dispatch_mlp_swiglu_combine_bwd_bf16(*bwd_args)
        else:
            (
                d_x_shared,
                d_x_routed,
                d_gate_shared,
                d_gate_routed,
                d_up_shared,
                d_up_routed,
                d_hidden_shared,
                d_hidden_routed,
                d_y_routed,
            ) = dispatch_mlp_swiglu_combine_bwd_bf16_accum(
                *bwd_args,
                main_grads=main_grads,
                weight_storage_tables=weight_storage_tables,
                main_grad_storage_tables=main_grad_storage_tables,
            )
            (
                d_w_shared_gate,
                d_w_routed_gate,
                d_w_shared_up,
                d_w_routed_up,
                d_w_shared_fc2,
                d_w_routed_fc2,
            ) = main_grads

    barrier_all(
        workspace.barrier_buffer,
        workspace.barrier_buffer_ptrs,
        workspace.barrier_buffer_multicast_ptr,
        workspace.barrier_target,
    )
    d_x = bwd_epilogue(
        d_x_shared,
        workspace.d_x_routed_buffer,
        schedule.top_experts,
    )
    d_router_weights = (
        workspace.d_router_weight_buffer.clone()
    )  # TODO: we can remove this
    d_router_weights.masked_fill_(schedule.top_experts < 0, 0.0)
    return (
        d_x,
        d_router_weights,
        d_w_routed_gate,
        d_w_routed_up,
        d_w_routed_fc2,
        d_w_shared_gate,
        d_w_shared_up,
        d_w_shared_fc2,
    )
