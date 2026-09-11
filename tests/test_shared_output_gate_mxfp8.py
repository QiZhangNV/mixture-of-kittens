"""MXFP8 routed experts with the unchanged BF16 shared-output-gate contract.

Default cases are small EP4 CUDA tests. MOK_TEST_QWEN_MXFP8_GATE_SHAPE=1 adds
T4096/H4096/I1024/E32/K10, EP4/local8, split/native-columnwise accumulation.

Two intentionally separate oracles are used:
* Complete MoE: independent BF16 shared S/reference; routed outputs/gradients
  use existing MXFP8 tolerance, shared S/gradients and gate dW use BF16 tolerance.
* Actual-S local gate check: actual saved S is an input fixture, but dZ is
  independently derived by a local autograd graph. Its FP64 dW oracle checks
  the gate consumer/FP32 producer, NOT independent accuracy of S itself.

No golden is formed from MOK's dZ. Exact rounding/non-BF16-output witnesses
prevent the broad routed tolerance from masking broken gate arithmetic.
"""

import hashlib
import json
import os

import pytest
import torch
import torch.distributed as dist

from mok import functional, ops

from .test_shared_output_gate_reference import (GATED_RESULT_NAMES,
                                                _bf16_product_rounding_witness,
                                                _check_saved_gate,
                                                _require_mok_gate_api)
from .test_split_routed_weights import _quantize_experts
from .utils import (BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness,
                    generate_inputs, run_reference_bf16,
                    run_shared_output_gate_reference, run_swiglu_reference)


def _inputs(context, qwen=False):
    rank, world, device = context
    assert world == 4, "initial MXFP8 gate tests intentionally require four EP ranks"
    experts, topk, tokens, hidden, intermediate = (32, 10, 4096, 4096, 1024) if qwen else (8, 2, 512, 256, 256)
    inputs = generate_inputs(rank, device, experts, experts // world, topk, tokens, hidden, intermediate)
    generator = torch.Generator(device=device).manual_seed(5919 + rank)
    gate = torch.randn(1, hidden, generator=generator, device=device, dtype=torch.bfloat16) * hidden ** -0.5
    return inputs, gate


def _schedule(context, inputs, macro=4096, qwen=False):
    x, ids, probs = inputs[:3]
    config = functional.MoKConfig(
        fwd_num_comm_sms=48 if qwen else 2, bwd_num_comm_sms=56 if qwen else 2,
        minibatch_size=4096 if qwen else 256, macrobatch_size=macro,
        schedule_capacity_multiplier=0.5 if qwen else 1.5,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=context[2], num_local_tokens=x.shape[0],
        hidden_size=x.shape[1], topk=probs.shape[1],
    )
    schedule = functional.build_schedule(workspace, config, ids, num_local_experts=inputs[6].shape[0])
    padded_rows = schedule.num_tokens.item()
    if macro == 512:
        assert padded_rows > macro, "the replay case must actually exceed the routed ring"
    else:
        assert padded_rows <= macro, "the no-replay control must fit the routed ring"
    return config, workspace, schedule


def _split_weight(experts):
    rows, scales, columns, column_scales = _quantize_experts(experts)
    view = functional.SplitRoutedWeight(
        data=rows[0], storage_table=ops.make_routed_weight_storage_table_mxfp8(rows),
        scale=scales[0], scale_storage_table=ops.make_routed_scale_storage_table(scales),
        scale_tensors=tuple(scales), transposed_data=columns[0], transposed_scale=column_scales[0],
        transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(columns),
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(column_scales),
        transposed_scale_tensors=tuple(column_scales), native_columnwise=True,
    )
    # Descriptor tables do not own payload allocations. Keep EVERY expert alive.
    return view, (rows, scales, columns, column_scales)


def _routed(inputs, layout):
    gate, up, down = inputs[6:9]
    if layout == "legacy":
        q_gate, q_up, q_down = [ops.mxfp8_quantize(weight, True, True) for weight in (gate, up, down)]
        return {"layout": layout, "forward": (q_gate[:2], q_up[:2], q_down[:2]),
                "backward": (q_gate, q_up, q_down[2:])}
    fc1 = torch.cat((gate, up), dim=1)
    if layout == "native":
        q_fc1 = ops.mxfp8_quantize(fc1, True, True)
        q_down = ops.mxfp8_quantize(down, True, True)
        fwd_fc1 = q_fc1[:2]
        bwd_fc1 = (q_fc1[0], q_fc1[1], q_fc1[2].transpose(-2, -1).contiguous(), q_fc1[3], True)
        bwd_down = (q_down[2].transpose(-2, -1).contiguous(), q_down[3], True)
        return {"layout": layout, "forward": (fwd_fc1, fwd_fc1, q_down[:2]),
                "backward": (bwd_fc1, bwd_fc1, bwd_down)}
    assert layout == "split_native"
    fc1_view, fc1_owners = _split_weight([weight.clone() for weight in fc1])
    down_view, down_owners = _split_weight([weight.clone() for weight in down])
    views = (fc1_view, fc1_view, down_view)
    return {"layout": layout, "forward": views, "backward": views,
            "owners": (fc1_owners, down_owners)}


def _main_grads(inputs, gate, layout):
    seed = 0.25
    shared = [torch.full_like(weight, seed, dtype=torch.float32) for weight in inputs[3:6]]
    gate_grad = torch.full_like(gate, seed, dtype=torch.float32)
    tables = None
    if layout == "legacy":
        routed = [torch.full_like(weight, seed, dtype=torch.float32) for weight in inputs[6:9]]
        main = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
        return {"main": main, "tables": tables, "gate": gate_grad, "routed": routed,
                "layout": layout, "seed": seed}
    fc1 = torch.cat((inputs[6], inputs[7]), dim=1)
    if layout == "native":
        routed_fc1 = torch.full_like(fc1, seed, dtype=torch.float32)
        routed_down = torch.full_like(inputs[8], seed, dtype=torch.float32)
        main_fc1, main_down = routed_fc1, routed_down
    else:
        routed_fc1 = [torch.full_like(weight, seed, dtype=torch.float32) for weight in fc1]
        routed_down = [torch.full_like(weight, seed, dtype=torch.float32) for weight in inputs[8]]
        fc1_table = ops.make_routed_d_weight_storage_table(routed_fc1)
        down_table = ops.make_routed_d_weight_storage_table(routed_down)
        tables = (fc1_table, fc1_table, down_table)
        main_fc1, main_down = routed_fc1[0], routed_down[0]
    return {"main": (shared[0], main_fc1, shared[1], main_fc1, shared[2], main_down),
            "tables": tables, "gate": gate_grad, "fc1": routed_fc1, "down": routed_down,
            "layout": layout, "seed": seed}


def _forward(inputs, gate, routed, state):
    return functional.forward(*state, inputs[0], inputs[2], *inputs[3:6], *routed["forward"],
                              shared_output_gate_weight=gate)


def _backward(inputs, gate, routed, state, saved, main=None):
    return functional.backward(
        *state, saved, inputs[-1], inputs[0], inputs[2], *inputs[3:6], *routed["backward"],
        main_grads=main["main"] if main else None,
        main_grad_storage_tables=main["tables"] if main else None,
        shared_output_gate_weight=gate, shared_output_gate_main_grad=main["gate"] if main else None,
    )


def _logical_results(output, gradients, main, intermediate, calls):
    assert len(gradients) == 9
    if main is None:
        assert all(value.dtype == torch.bfloat16 for value in gradients[2:8])
        return (output, *gradients)
    # Returned split routed entries are representatives, so inspect ALL buffers.
    expected_aliases = (main["main"][1], main["main"][3], main["main"][5],
                        main["main"][0], main["main"][2], main["main"][4], main["gate"])
    assert all(actual is expected for actual, expected in zip(gradients[2:], expected_aliases, strict=True))
    if main["layout"] == "legacy":
        gate, up, down = main["routed"]
    else:
        fc1 = torch.stack(main["fc1"]) if isinstance(main["fc1"], list) else main["fc1"]
        down = torch.stack(main["down"]) if isinstance(main["down"], list) else main["down"]
        gate, up = fc1.split(intermediate, dim=1)
    weights = (gate, up, down, main["main"][0], main["main"][2], main["main"][4], main["gate"])
    # Compare per-call contributions, not a large seed that could hide errors.
    normalized = tuple((value - main["seed"]) / calls for value in weights)
    return (output, gradients[0], gradients[1], *normalized)


def _independent_shared(inputs):
    x = inputs[0]
    gate, up, down = inputs[3:6]
    return run_swiglu_reference(x @ gate.T, x @ up.T) @ down.T


def _check_saved_and_independent_s(context, inputs, weight, saved):
    _check_saved_gate(saved, inputs[0], weight)
    for value in (saved.gate_shared, saved.up_shared, saved.hidden_shared):
        assert value.dtype == torch.bfloat16
    check_correctness("MXFP8-gated/independent-S", _independent_shared(inputs), saved.shared_output,
                      BF16_TOLERANCE, print_stats=context[0] == 0)


def _local_gate_fp64(inputs, weight, saved):
    # Actual S is ONLY the input fixture for this local consumer check.
    # dZ_ref comes from an independent local autograd graph, never from MOK.
    _, _, _, dz_ref, _, dw_fp64 = run_shared_output_gate_reference(
        inputs[0], saved.shared_output, weight, inputs[-1])
    u = torch.finfo(torch.float32).eps / 2
    length = inputs[0].shape[0]
    gamma_k = length * u / (1 - length * u)
    bound = gamma_k * (dz_ref.double().abs().T @ inputs[0].double().abs())
    return dw_fp64, bound


def _check_local_gate(actual, before, golden):
    expected, bound = golden
    if before is not None:
        expected = before.double() + expected
        u = torch.finfo(torch.float32).eps / 2
        bound = bound + u * (before.double().abs() + golden[0].abs() + bound)
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    assert torch.isfinite(expected).all() and torch.isfinite(bound).all()
    assert torch.all((actual.double() - expected).abs() <= bound), "actual-S local gate FP64 producer check failed"


def _compare_independent(context, actual, reference, label):
    assert len(actual) == len(reference) == 10
    for index, (name, value, golden) in enumerate(zip(GATED_RESULT_NAMES, actual, reference, strict=True)):
        # Routed dX/dP/W have FP8 error; the shared branch and gate do not.
        tolerance = MXFP8_TOLERANCE if index < 6 else BF16_TOLERANCE
        check_correctness(f"{label}/independent-S/{name}", golden, value, tolerance, print_stats=context[0] == 0)


def _run_case(context, layout, accumulate, macro, qwen=False):
    _require_mok_gate_api()
    inputs, gate = _inputs(context, qwen)
    if not accumulate:
        inputs[0].requires_grad_(True)
        gate.requires_grad_(True)
    routed = _routed(inputs, layout)
    state = _schedule(context, inputs, macro, qwen)
    main = _main_grads(inputs, gate, layout) if accumulate else None
    reference = run_reference_bf16(*inputs, shared_output_gate_weight=gate)
    for iteration in range(2 if accumulate else 1):
        output, saved = _forward(inputs, gate, routed, state)
        _check_saved_and_independent_s(context, inputs, gate, saved)
        golden = _local_gate_fp64(inputs, gate, saved)
        before = main["gate"].clone() if main else None
        gradients = _backward(inputs, gate, routed, state, saved, main)
        _check_local_gate(gradients[-1], before, golden)
        actual = _logical_results(output, gradients, main, inputs[3].shape[0], iteration + 1)
        _compare_independent(context, actual, reference, f"{layout}/accum={accumulate}/macro={macro}")


@pytest.mark.parametrize("layout,accumulate,macro", [
    ("legacy", False, 4096), ("legacy", False, 512),
    ("legacy", True, 4096), ("legacy", True, 512),
    ("native", False, 4096), ("native", False, 512),
    ("native", True, 4096), ("native", True, 512),
    ("split_native", True, 4096), ("split_native", True, 512),
])
def test_mxfp8_gated_independent_reference_and_fp32_gate_consumer(context, layout, accumulate, macro):
    _run_case(context, layout, accumulate, macro)


@pytest.mark.parametrize("accumulate", [False, True])
def test_mxfp8_gated_exact_bf16_product_rounding_witness(context, accumulate):
    inputs, _ = _inputs(context)
    x, shared, gate, dy = _bf16_product_rounding_witness(context[2], 512, 256)
    weights = inputs[3:9]
    for weight in weights:
        weight.zero_()
    weights[0][0, 0] = 16
    weights[1][0, 0] = 1
    weights[2][0, 0] = (1 + 1 / 128) / 16
    weights[2][1, 0] = 1 / 16
    inputs = (x, inputs[1], inputs[2], *weights, dy)
    routed, state = _routed(inputs, "legacy"), _schedule(context, inputs)
    main = _main_grads(inputs, gate, "legacy") if accumulate else None
    old_dg = (dy.float() * shared.float()).sum(-1, keepdim=True).to(torch.bfloat16)
    torch.testing.assert_close(old_dg, torch.full_like(old_dg, 1 / 16384), rtol=0, atol=0)
    _, _, dg_ref, _, _, dw64 = run_shared_output_gate_reference(x, shared, gate, dy)
    torch.testing.assert_close(dg_ref, torch.zeros_like(dg_ref), rtol=0, atol=0)
    torch.testing.assert_close(dw64, torch.zeros_like(dw64), rtol=0, atol=0)
    for _ in range(2 if accumulate else 1):
        output, saved = _forward(inputs, gate, routed, state)
        torch.testing.assert_close(saved.shared_output, shared, rtol=0, atol=0)
        torch.testing.assert_close(output, shared * 0.5, rtol=0, atol=0)
        gradients = _backward(inputs, gate, routed, state, saved, main)
        assert len(gradients) == 9 and gradients[-1].dtype == torch.float32
        expected = torch.full_like(gate, main["seed"] if main else 0, dtype=torch.float32)
        torch.testing.assert_close(gradients[-1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("qwen", [
    pytest.param(False, id="small"),
    pytest.param(True, id="qwen-target", marks=pytest.mark.skipif(
        os.environ.get("MOK_TEST_QWEN_MXFP8_GATE_SHAPE") != "1",
        reason="set MOK_TEST_QWEN_MXFP8_GATE_SHAPE=1 for the full Qwen shape",
    )),
])
def test_mxfp8_gated_exact_fp32_wgrad_without_bf16_intermediate(context, accumulate, qwen):
    inputs, gate = _inputs(context, qwen)
    x, dy = torch.zeros_like(inputs[0]), torch.zeros_like(inputs[-1])
    x[:, :2] = 1
    x[-1, 1] = 1 / 64
    dy[:, 0] = 4
    gate.zero_()  # G=0.5, not an ungated/None branch.
    weights = inputs[3:9]
    for weight in weights:
        weight.zero_()
    weights[0][0, 0], weights[1][0, 0], weights[2][0, 0] = 16, 1, 1 / 16
    shared = torch.zeros_like(x)
    shared[:, 0] = 1
    inputs = (x, inputs[1], inputs[2], *weights, dy)
    _, _, _, dz_ref, _, dw64 = run_shared_output_gate_reference(x, shared, gate, dy)
    torch.testing.assert_close(dz_ref, torch.ones_like(dz_ref), rtol=0, atol=0)
    assert dw64[0, 1].item() == x.shape[0] - 1 + 1 / 64
    assert not torch.equal(dw64.float(), dw64.to(torch.bfloat16).float())
    routed = _routed(inputs, "legacy")
    state = _schedule(context, inputs, macro=65536 if qwen else 4096, qwen=qwen)
    main = _main_grads(inputs, gate, "legacy") if accumulate else None
    expected = torch.full_like(gate, main["seed"] if main else 0, dtype=torch.float32)
    for _ in range(2 if accumulate else 1):
        output, saved = _forward(inputs, gate, routed, state)
        torch.testing.assert_close(saved.shared_output, shared, rtol=0, atol=0)
        torch.testing.assert_close(output, shared * 0.5, rtol=0, atol=0)
        gradients = _backward(inputs, gate, routed, state, saved, main)
        expected.add_(dw64.float())
        assert len(gradients) == 9 and gradients[-1].dtype == torch.float32
        torch.testing.assert_close(gradients[-1], expected, rtol=0, atol=0)


@pytest.mark.parametrize("macro", [4096, 512])
def test_mxfp8_gated_multiple_live_contexts(context, macro):
    inputs, gate = _inputs(context)
    second = list(inputs)
    second[0], second[-1] = inputs[0] * -0.5, inputs[-1] * 2
    routed = _routed(inputs, "legacy")
    pending = []
    for batch in (inputs, tuple(second)):
        state = _schedule(context, batch, macro)
        output, saved = _forward(batch, gate, routed, state)
        _check_saved_and_independent_s(context, batch, gate, saved)
        pending.append((batch, state, output, saved, saved.shared_output.clone(), saved.shared_output_gate.clone()))
    assert pending[0][3].shared_output.data_ptr() != pending[1][3].shared_output.data_ptr()
    assert pending[0][3].shared_output_gate.data_ptr() != pending[1][3].shared_output_gate.data_ptr()
    for batch, state, output, saved, shared_snapshot, gate_snapshot in reversed(pending):
        torch.testing.assert_close(saved.shared_output, shared_snapshot, rtol=0, atol=0)
        torch.testing.assert_close(saved.shared_output_gate, gate_snapshot, rtol=0, atol=0)
        golden = _local_gate_fp64(batch, gate, saved)
        gradients = _backward(batch, gate, routed, state, saved)
        _check_local_gate(gradients[-1], None, golden)
        reference = run_reference_bf16(*batch, shared_output_gate_weight=gate)
        _compare_independent(context, (output, *gradients), reference, f"live-context/macro={macro}")


def test_mxfp8_ungated_reference_and_fingerprints(context):
    """Use -k ungated on 04d84cf/new checkouts; no gated execution.

    Both revisions have the nine-item/optional-gate API. This is not an adapter
    for older upstream eight-item APIs. Cross-checkout hashes are evidence,
    while the same-build repeat reports bitwise agreement without assuming it.
    """
    inputs, _ = _inputs(context)
    routed, state = _routed(inputs, "legacy"), _schedule(context, inputs)
    reference = run_reference_bf16(*inputs)
    output, saved = functional.forward(*state, inputs[0], inputs[2], *inputs[3:6], *routed["forward"])
    gradients = functional.backward(*state, saved, inputs[-1], inputs[0], inputs[2],
                                    *inputs[3:6], *routed["backward"])
    assert len(gradients) == 9 and gradients[-1] is None
    assert saved.shared_output is None and saved.shared_output_gate is None
    first = (output, *gradients[:8])
    fingerprint = {}
    for name, actual, golden in zip(GATED_RESULT_NAMES[:-1], first, reference, strict=True):
        check_correctness(f"ungated/{name}", golden, actual, MXFP8_TOLERANCE, print_stats=context[0] == 0)
        fingerprint[name] = hashlib.sha256(actual.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    explicit_output, explicit_saved = _forward(inputs, None, routed, state)
    explicit = _backward(inputs, None, routed, state, explicit_saved)
    assert len(explicit) == 9 and explicit[-1] is None
    assert explicit_saved.shared_output is None and explicit_saved.shared_output_gate is None
    for name, first_value, second_value in zip(
        GATED_RESULT_NAMES[:-1], first, (explicit_output, *explicit[:8]), strict=True
    ):
        check_correctness(f"ungated/explicit-None/{name}", first_value, second_value,
                          MXFP8_TOLERANCE, print_stats=context[0] == 0)
    print(json.dumps({"event": "mxfp8_ungated_fingerprints", "rank": context[0],
                      "functional_source": functional.__file__, "sha256": fingerprint,
                      "implicit_vs_explicit_none_bitwise": [
                          torch.equal(a, b) for a, b in zip(first, (explicit_output, *explicit[:8]), strict=True)
                      ]}), flush=True)


@pytest.mark.skipif(os.getenv("MOK_TEST_QWEN_MXFP8_GATE_SHAPE") != "1",
                    reason="opt in to Qwen E32/EP4/local8 MXFP8 gate shape")
def test_qwen_mxfp8_gated_split_native_fp32_accumulation(context):
    _run_case(context, "split_native", True, 65536, qwen=True)
