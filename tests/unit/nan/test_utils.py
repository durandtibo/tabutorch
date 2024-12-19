from __future__ import annotations

import pytest
import torch

from tabutorch.nan import check_all_nan, check_any_nan

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
        torch.tensor(1.0),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.ones(2, 3),
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


###################################
#     Tests for check_any_nan     #
###################################

ALL_WITH_TENSORS = [
    torch.tensor([1.0, 2.0, 3.0, float("nan")]),
    torch.tensor([1.0, 2.0, float("nan"), float("nan")]),
    torch.tensor([1.0, float("nan"), float("nan"), float("nan")]),
    torch.tensor(float("nan")),
    torch.tensor([float("nan")]),
    torch.tensor([float("nan"), float("nan"), float("nan")]),
]


@pytest.mark.parametrize("x", [torch.tensor(1.0), torch.tensor([1.0, 2.0, 3.0]), torch.ones(2, 3)])
def test_check_any_nan_no_nan(x: torch.Tensor) -> None:
    check_any_nan(x)


@pytest.mark.parametrize("x", ALL_WITH_TENSORS)
def test_check_any_nan_with_nan(x: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="tensor contains at least one NaN value"):
        check_any_nan(x)


@pytest.mark.parametrize("x", ALL_WITH_TENSORS)
def test_check_any_nan_with_nan_warn(x: torch.Tensor) -> None:
    with pytest.raises(RuntimeWarning, match="tensor contains at least one NaN value"):
        check_any_nan(x, warn=True)
