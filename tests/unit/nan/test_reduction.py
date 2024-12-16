from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from tabutorch.nan import mean

##########################
#     Tests for mean     #
##########################


def test_mean_no_nan() -> None:
    assert objects_are_equal(mean(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), torch.tensor(3.0))


def test_mean_nan_omit() -> None:
    assert objects_are_equal(
        mean(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]), nan_policy="omit"), torch.tensor(3.0)
    )


def test_mean_nan_omit_args() -> None:
    assert objects_are_equal(
        mean(
            torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]),
            dim=1,
            keepdim=True,
            nan_policy="omit",
        ),
        torch.tensor([[1.5], [5.0]]),
    )


def test_mean_nan_propagate() -> None:
    assert objects_are_equal(
        mean(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0])),
        torch.tensor(float("nan")),
        equal_nan=True,
    )


def test_mean_nan_propagate_args() -> None:
    assert objects_are_equal(
        mean(torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]), dim=1, keepdim=True),
        torch.tensor([[float("nan")], [5.0]]),
        equal_nan=True,
    )


def test_mean_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        mean(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]), nan_policy="incorrect")
