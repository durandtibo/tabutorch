from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from tabutorch.nan import mean, nanstd, nanvar, std

##########################
#     Tests for mean     #
##########################


def test_mean_no_nan() -> None:
    assert objects_are_equal(mean(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), torch.tensor(3.0))


def test_mean_nan_omit() -> None:
    assert objects_are_equal(
        mean(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]), nan_policy="omit"), torch.tensor(3.0)
    )


def test_mean_nan_omit_dim() -> None:
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


def test_mean_nan_propagate_dim() -> None:
    assert objects_are_equal(
        mean(torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]), dim=1, keepdim=True),
        torch.tensor([[float("nan")], [5.0]]),
        equal_nan=True,
    )


def test_mean_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        mean(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]), nan_policy="incorrect")


#########################
#     Tests for std     #
#########################


def test_std_no_nan() -> None:
    assert objects_are_allclose(
        std(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), torch.tensor(1.5811388492584229)
    )


def test_std_nan_omit() -> None:
    assert objects_are_allclose(
        std(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]), nan_policy="omit"),
        torch.tensor(1.8257418870925903),
    )


def test_std_nan_omit_dim() -> None:
    assert objects_are_allclose(
        std(
            torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]),
            dim=1,
            keepdim=True,
            nan_policy="omit",
        ),
        torch.tensor([[0.7071067690849304], [1.0]]),
    )


def test_std_nan_omit_correction_0() -> None:
    assert objects_are_allclose(
        std(
            torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]),
            dim=1,
            correction=0,
            nan_policy="omit",
        ),
        torch.tensor([0.5, 0.8164966106414795]),
    )


def test_std_nan_propagate() -> None:
    assert objects_are_allclose(
        std(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0])),
        torch.tensor(float("nan")),
        equal_nan=True,
    )


def test_std_nan_propagate_dim() -> None:
    assert objects_are_allclose(
        std(torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]), dim=1, keepdim=True),
        torch.tensor([[float("nan")], [1.0]]),
        equal_nan=True,
    )


def test_std_nan_propagate_correction_0() -> None:
    assert objects_are_allclose(
        std(torch.tensor([[1.0, 2.0, float("nan")], [4.0, 5.0, 6.0]]), dim=1, correction=0),
        torch.tensor([float("nan"), 0.8164966106414795]),
        equal_nan=True,
    )


def test_std_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        std(torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]), nan_policy="incorrect")


############################
#     Tests for nanstd     #
############################


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, float("nan"), 9.0]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]),
        torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]]),
    ],
)
def test_nanstd_correction_0(x: torch.Tensor) -> None:
    assert objects_are_allclose(nanstd(x, correction=0), torch.tensor(2.766644392345187))


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, float("nan"), 9.0]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]),
        torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, float("nan"), 9.0]]]),
    ],
)
def test_nanstd_correction_1(x: torch.Tensor) -> None:
    assert objects_are_allclose(nanstd(x), torch.tensor(2.9344694047230853))


def test_nanstd_correction_0_dim_0() -> None:
    assert objects_are_allclose(
        nanstd(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            correction=0,
            dim=0,
        ),
        torch.tensor([2.0, 3.2659863723778924, 3.2659863723778924, 3.2659863723778924]),
    )


def test_nanstd_correction_0_dim_1() -> None:
    assert objects_are_allclose(
        nanstd(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            correction=0,
            dim=1,
        ),
        torch.tensor([1.118033988749895, 1.118033988749895, 0.8164965930944731]),
    )


def test_nanstd_correction_1_dim_0() -> None:
    assert objects_are_allclose(
        nanstd(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            dim=0,
        ),
        torch.tensor([2.8284271247461903, 4.0, 4.0, 4.0]),
    )


def test_nanstd_correction_1_dim_1() -> None:
    assert objects_are_allclose(
        nanstd(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            dim=1,
        ),
        torch.tensor([1.2909944333459524, 1.2909944333459524, 1.0]),
    )


def test_nanstd_correction_keepdim_2d() -> None:
    assert objects_are_allclose(
        nanstd(
            torch.tensor(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [float("nan"), 9.0, 10.0, 11.0]]
            ),
            keepdim=True,
            dim=0,
        ),
        torch.tensor([[2.8284271247461903, 4.0, 4.0, 4.0]]),
    )


def test_nanstd_correction_keepdim_3d() -> None:
    assert objects_are_allclose(
        nanstd(
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
        torch.tensor(
            [[[2.8284271247461903, 4.0, 4.0, 4.0]], [[4.0, 2.8284271247461903, 4.0, 4.0]]]
        ),
    )


def test_nanstd_constant() -> None:
    assert objects_are_allclose(nanstd(torch.ones(2, 3, 4)), torch.tensor(0.0))


def test_nanstd_constant_nan() -> None:
    assert objects_are_allclose(
        nanstd(torch.full((2, 3), float("nan"))), torch.tensor(float("nan")), equal_nan=True
    )


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
