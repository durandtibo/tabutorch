from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from tabutorch.nan import nanmax, nanmin, nanstd, nanvar

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


@numpy_available
@pytest.mark.filterwarnings("ignore:All-NaN axis encountered:RuntimeWarning")
def test_nanmax_full_nan() -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(nanmax(x), torch.tensor(np.nanmax(x)), equal_nan=True)


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_nanmax_full_nan_dim(dim: int, keepdim: bool) -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(
        nanmax(x, dim=dim, keepdim=keepdim)[0].numpy(),
        np.nanmax(x.numpy(), axis=dim, keepdims=keepdim),
        equal_nan=True,
    )


############################
#     Tests for nanmin     #
############################


@numpy_available
def test_nanmin() -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(nanmin(x), torch.tensor(np.nanmin(x)))


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_nanmin_dim(dim: int, keepdim: bool) -> None:
    x = nan_values(torch.randn(100, 10), ratio=0.2)
    assert objects_are_allclose(
        nanmin(x, dim=dim, keepdim=keepdim)[0].numpy(),
        np.nanmin(x.numpy(), axis=dim, keepdims=keepdim),
    )


@numpy_available
@pytest.mark.filterwarnings("ignore:All-NaN axis encountered:RuntimeWarning")
def test_nanmin_full_nan() -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(nanmin(x), torch.tensor(np.nanmin(x)), equal_nan=True)


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_nanmin_full_nan_dim(dim: int, keepdim: bool) -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(
        nanmin(x, dim=dim, keepdim=keepdim)[0].numpy(),
        np.nanmin(x.numpy(), axis=dim, keepdims=keepdim),
        equal_nan=True,
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
        equal_nan=True,
    )


@numpy_available
@pytest.mark.parametrize("correction", [0, 1])
def test_nanstd_full_nan(correction: int) -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(
        nanstd(x, correction=correction),
        torch.tensor(np.nanstd(x, ddof=correction)),
        equal_nan=True,
    )


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_nanstd_full_nan_dim(dim: int, correction: int, keepdim: bool) -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(
        nanstd(x, correction=correction, dim=dim, keepdim=keepdim).numpy(),
        np.nanstd(x.numpy(), ddof=correction, axis=dim, keepdims=keepdim),
        equal_nan=True,
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


@numpy_available
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
def test_nanvar_full_nan(correction: int) -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(
        nanvar(x, correction=correction),
        torch.tensor(np.nanvar(x, ddof=correction)),
        equal_nan=True,
    )


@numpy_available
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice:RuntimeWarning")
def test_nanvar_full_nan_dim(dim: int, correction: int, keepdim: bool) -> None:
    x = torch.full((10, 5), float("nan"))
    assert objects_are_allclose(
        nanvar(x, correction=correction, dim=dim, keepdim=keepdim).numpy(),
        np.nanvar(x.numpy(), ddof=correction, axis=dim, keepdims=keepdim),
        equal_nan=True,
    )
