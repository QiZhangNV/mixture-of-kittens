import pytest
import torch

from mok import functional


def _tensor() -> torch.Tensor:
    return torch.empty(0)


def test_normalize_contiguous_backward_weights() -> None:
    gate, up, down = _tensor(), _tensor(), _tensor()
    normalized = functional._normalize_backward_weights(gate, up, down, None, None)
    assert isinstance(normalized, functional._BF16BackwardWeights)
    assert normalized.gate_data is gate
    assert normalized.up_data is up
    assert normalized.down_data is down
    assert normalized.storage_tables is None

    gate_row, gate_row_scale, gate_column, gate_column_scale = (
        _tensor(),
        _tensor(),
        _tensor(),
        _tensor(),
    )
    up_row, up_row_scale, up_column, up_column_scale = (
        _tensor(),
        _tensor(),
        _tensor(),
        _tensor(),
    )
    down_column, down_column_scale = _tensor(), _tensor()
    legacy = functional._normalize_backward_weights(
        (gate_row, gate_row_scale, gate_column, gate_column_scale),
        (up_row, up_row_scale, up_column, up_column_scale),
        (down_column, down_column_scale),
        None,
        None,
    )
    assert isinstance(legacy, functional._MXFP8BackwardWeights)
    assert legacy.gate_row_data is gate_row
    assert legacy.gate_column_data is gate_column
    assert legacy.up_row_data is up_row
    assert legacy.up_column_data is up_column
    assert legacy.down_column_data is down_column
    assert legacy.native_columnwise is False
    assert legacy.storage_tables is None
    assert legacy.scale_storage_tables is None

    native = functional._normalize_backward_weights(
        (gate_row, gate_row_scale, gate_column, gate_column_scale, True),
        (up_row, up_row_scale, up_column, up_column_scale, True),
        (down_column, down_column_scale, True),
        None,
        None,
    )
    assert isinstance(native, functional._MXFP8BackwardWeights)
    assert native.gate_column_data is gate_column
    assert native.up_column_data is up_column
    assert native.down_column_data is down_column
    assert native.native_columnwise is True


def test_normalize_split_backward_weights() -> None:
    main_grad = _tensor()
    main_grad_tables = (_tensor(), _tensor(), _tensor())

    bf16_data = (_tensor(), _tensor(), _tensor())
    bf16_tables = (_tensor(), _tensor(), _tensor())
    bf16_weights = tuple(
        functional.SplitRoutedWeight(data, table)
        for data, table in zip(bf16_data, bf16_tables, strict=True)
    )
    bf16 = functional._normalize_backward_weights(
        *bf16_weights,
        (main_grad,),
        main_grad_tables,
    )
    assert isinstance(bf16, functional._BF16BackwardWeights)
    assert bf16.gate_data is bf16_data[0]
    assert bf16.up_data is bf16_data[1]
    assert bf16.down_data is bf16_data[2]
    assert bf16.storage_tables is not None
    assert all(
        actual is expected
        for actual, expected in zip(bf16.storage_tables, bf16_tables, strict=True)
    )

    split_weights = []
    for _ in range(3):
        split_weights.append(
            functional.SplitRoutedWeight(
                data=_tensor(),
                storage_table=_tensor(),
                scale=_tensor(),
                scale_storage_table=_tensor(),
                transposed_data=_tensor(),
                transposed_scale=_tensor(),
                transposed_storage_table=_tensor(),
                transposed_scale_storage_table=_tensor(),
                native_columnwise=True,
            )
        )
    gate, up, down = split_weights
    mxfp8 = functional._normalize_backward_weights(
        gate,
        up,
        down,
        (main_grad,),
        main_grad_tables,
    )
    assert isinstance(mxfp8, functional._MXFP8BackwardWeights)
    assert mxfp8.gate_row_data is gate.data
    assert mxfp8.gate_column_data is gate.transposed_data
    assert mxfp8.up_row_data is up.data
    assert mxfp8.up_column_data is up.transposed_data
    assert mxfp8.down_column_data is down.transposed_data
    assert mxfp8.native_columnwise is True
    expected_storage_tables = (
        gate.storage_table,
        up.storage_table,
        gate.transposed_storage_table,
        up.transposed_storage_table,
        down.transposed_storage_table,
    )
    assert mxfp8.storage_tables is not None
    assert all(
        actual is expected
        for actual, expected in zip(
            mxfp8.storage_tables, expected_storage_tables, strict=True
        )
    )
    expected_scale_storage_tables = (
        gate.scale_storage_table,
        up.scale_storage_table,
        gate.transposed_scale_storage_table,
        up.transposed_scale_storage_table,
        down.transposed_scale_storage_table,
    )
    assert mxfp8.scale_storage_tables is not None
    assert all(
        actual is expected
        for actual, expected in zip(
            mxfp8.scale_storage_tables, expected_scale_storage_tables, strict=True
        )
    )


def test_normalize_backward_weights_rejects_mixed_encodings() -> None:
    with pytest.raises(TypeError, match="same storage representation"):
        functional._normalize_backward_weights(
            _tensor(),
            (_tensor(), _tensor(), _tensor(), _tensor()),
            _tensor(),
            None,
            None,
        )
