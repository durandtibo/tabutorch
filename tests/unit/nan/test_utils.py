from __future__ import annotations

import pytest
import torch

from tabutorch.nan import check_all_nan

###################################
#     Tests for check_all_nan     #
###################################

ALL_NAN_TENSORS = [
    torch.tensor(float("nan")),
    torch.tensor([float("nan")]),
    torch.tensor([float("nan"), float("nan"), float("nan")]),
]


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([1.0, 2.0, 3.0, float("nan")]),
        torch.tensor([1.0, 2.0, float("nan"), float("nan")]),
        torch.tensor([1.0, float("nan"), float("nan"), float("nan")]),
    ],
)
def test_check_all_nan_no_nan(x: torch.Tensor) -> None:
    check_all_nan(x)


@pytest.mark.parametrize("x", ALL_NAN_TENSORS)
def test_check_all_nan_all_nan(x: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="tensor contains only NaN values"):
        check_all_nan(x)


@pytest.mark.parametrize("x", ALL_NAN_TENSORS)
def test_check_all_nan_all_nan_warn(x: torch.Tensor) -> None:
    with pytest.raises(RuntimeWarning, match="tensor contains only NaN values"):
        check_all_nan(x, warn=True)
