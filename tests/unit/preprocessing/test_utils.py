from __future__ import annotations

import torch
from coola import objects_are_equal

from tabutorch.preprocessing.utils import handle_zeros_in_scale, is_constant_feature

###########################################
#     Tests for handle_zeros_in_scale     #
###########################################


def test_handle_zeros_in_scale_1d() -> None:
    assert objects_are_equal(
        handle_zeros_in_scale(scale=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])),
        torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0]),
    )


def test_handle_zeros_in_scale_2d() -> None:
    assert objects_are_equal(
        handle_zeros_in_scale(scale=torch.tensor([[4.0, 3.0, 2.0], [1.0, 0.0, 1e-9]])),
        torch.tensor([[4.0, 3.0, 2.0], [1.0, 1.0, 1.0]]),
    )


def test_handle_zeros_in_scale_copy_true() -> None:
    scale = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    out = handle_zeros_in_scale(scale)
    assert scale is not out
    assert objects_are_equal(scale, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]))
    assert objects_are_equal(out, torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0]))


def test_handle_zeros_in_scale_copy_false() -> None:
    scale = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    out = handle_zeros_in_scale(scale, copy=False)
    assert scale is out
    assert objects_are_equal(scale, torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0]))
    assert objects_are_equal(out, torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0]))


def test_handle_zeros_in_scale_constant_mask() -> None:
    assert objects_are_equal(
        handle_zeros_in_scale(
            scale=torch.tensor([4.0, 1e-10, 2.0, 1.0, 0.0]),
            constant_mask=torch.tensor([False, True, False, False, True]),
        ),
        torch.tensor([4.0, 1.0, 2.0, 1.0, 1.0]),
    )


#########################################
#     Tests for is_constant_feature     #
#########################################


def test_is_constant_feature_1d() -> None:
    assert objects_are_equal(
        is_constant_feature(
            mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 2.0]),
            var=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
            n_samples=torch.tensor([10, 11, 12, 13, 10]),
        ),
        torch.tensor([False, False, False, False, True]),
    )


def test_is_constant_feature_2d() -> None:
    assert objects_are_equal(
        is_constant_feature(
            mean=torch.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 5.0]]),
            var=torch.tensor([[4.0, 3.0, 2.0], [1.0, 0.0, 1e-9]]),
            n_samples=torch.tensor([[10, 11, 12], [13, 10, 100]]),
        ),
        torch.tensor([[False, False, False], [False, True, True]]),
    )


def test_is_constant_feature_n_samples_scalar() -> None:
    assert objects_are_equal(
        is_constant_feature(
            mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 2.0]),
            var=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
            n_samples=10,
        ),
        torch.tensor([False, False, False, False, True]),
    )
