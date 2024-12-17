from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from tabutorch.nan import mean, nanvar

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


############################
#     Tests for nanvar     #
############################


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, float("nan"), 9.0]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]),
        torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]]),
    ],
)
def test_nanvar_correction_0(x: torch.Tensor) -> None:
    assert objects_are_allclose(nanvar(x, correction=0), torch.tensor(7.654321193695068))


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, float("nan"), 9.0]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]),
        torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]]),
    ],
)
def test_nanvar_correction_1(x: torch.Tensor) -> None:
    assert objects_are_allclose(nanvar(x), torch.tensor(8.61111068725586))


def test_nanvar_correction_0_dim_0() -> None:
    assert objects_are_allclose(
        nanvar(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            correction=0,
            dim=0,
        ),
        torch.tensor([4.0, 10.666666984558105, 10.666666984558105, 10.666666984558105]),
    )


def test_nanvar_correction_0_dim_1() -> None:
    assert objects_are_allclose(
        nanvar(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            correction=0,
            dim=1,
        ),
        torch.tensor([1.25, 1.25, 0.6666666865348816]),
    )


def test_nanvar_correction_1_dim_0() -> None:
    assert objects_are_allclose(
        nanvar(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            dim=0,
        ),
        torch.tensor([8.0, 16.0, 16.0, 16.0]),
    )


def test_nanvar_correction_1_dim_1() -> None:
    assert objects_are_allclose(
        nanvar(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            dim=1,
        ),
        torch.tensor([1.6666666269302368, 1.6666666269302368, 1.0]),
    )


def test_nanvar_correction_keepdim_2d() -> None:
    assert objects_are_allclose(
        nanvar(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            keepdim=True,
            dim=0,
        ),
        torch.tensor([[8.0, 16.0, 16.0, 16.0]]),
    )


def test_nanvar_correction_keepdim_3d() -> None:
    assert objects_are_allclose(
        nanvar(
            torch.tensor(
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [float("nan"), 9.0, 10.0, 11.0],
                    ],
                    [
                        [10.0, float("nan"), 12.0, 13.0],
                        [14.0, 15.0, 16.0, 17.0],
                        [18.0, 19.0, 20.0, 21.0],
                    ],
                ]
            ),
            keepdim=True,
            dim=1,
        ),
        torch.tensor([[[8.0, 16.0, 16.0, 16.0]], [[16.0, 8.0, 16.0, 16.0]]]),
    )


def test_nanvar_constant() -> None:
    assert objects_are_allclose(nanvar(torch.ones(2, 3, 4)), torch.tensor(0.0))


def test_nanvar_constant_nan() -> None:
    assert objects_are_allclose(
        nanvar(torch.full((2, 3), float("nan"))), torch.tensor(float("nan")), equal_nan=True
    )
