from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from tabutorch.nan import nanmax, nanstd, nanvar

if is_numpy_available():
    import numpy as np


def nan_values(tensor: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    mask = torch.rand_like(tensor) < ratio
    tensor[mask] = float("nan")
    return tensor


############################
#     Tests for nanmax     #
############################


@numpy_available
def test_nanmax() -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(nanmax(x), torch.tensor(np.nanmax(x)))


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_nanmax_dim(dim: int, keepdim: bool) -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(
        nanmax(x, dim=dim, keepdim=keepdim)[0].numpy(),
        np.nanmax(x.numpy(), axis=dim, keepdims=keepdim),
    )


############################
#     Tests for nanstd     #
############################


@numpy_available
@pytest.mark.parametrize("correction", [0, 1])
def test_nanstd(correction: int) -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(
        nanstd(x, correction=correction),
        torch.tensor(np.nanstd(x, ddof=correction)),
    )


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_nanstd_dim(dim: int, correction: int, keepdim: bool) -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(
        nanstd(x, correction=correction, dim=dim, keepdim=keepdim).numpy(),
        np.nanstd(x.numpy(), ddof=correction, axis=dim, keepdims=keepdim),
    )


############################
#     Tests for nanvar     #
############################


@numpy_available
@pytest.mark.parametrize("correction", [0, 1])
def test_nanvar(correction: int) -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(
        nanvar(x, correction=correction),
        torch.tensor(np.nanvar(x, ddof=correction)),
    )


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_nanvar_dim(dim: int, correction: int, keepdim: bool) -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(
        nanvar(x, correction=correction, dim=dim, keepdim=keepdim).numpy(),
        np.nanvar(x.numpy(), ddof=correction, axis=dim, keepdims=keepdim),
    )
