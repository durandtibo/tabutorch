from __future__ import annotations

import pytest
import torch
from coola import objects_are_equal

from tabutorch.utils.nan import check_nan_policy, contains_nan, mean

NAN_POLICIES = ["omit", "propagate", "raise"]

######################################
#     Tests for check_nan_policy     #
######################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_check_nan_policy_valid(nan_policy: str) -> None:
    check_nan_policy(nan_policy)


def test_check_nan_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        check_nan_policy("incorrect")


##################################
#     Tests for contains_nan     #
##################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_contains_nan_no_nan(nan_policy: str) -> None:
    assert not contains_nan(torch.tensor([1, 2, 3, 4, 5]), nan_policy=nan_policy)


def test_contains_nan_omit() -> None:
    assert contains_nan(torch.tensor([1, 2, 3, 4, float("nan")]), nan_policy="omit")


def test_contains_nan_propagate() -> None:
    assert contains_nan(torch.tensor([1, 2, 3, 4, float("nan")]), nan_policy="propagate")


def test_contains_nan_raise() -> None:
    with pytest.raises(ValueError, match="input tensor contains at least one NaN value"):
        contains_nan(torch.tensor([1, 2, 3, 4, float("nan")]), nan_policy="raise")


def test_contains_nan_raise_name() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        contains_nan(torch.tensor([1, 2, 3, 4, float("nan")]), nan_policy="raise", name="'x'")


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
