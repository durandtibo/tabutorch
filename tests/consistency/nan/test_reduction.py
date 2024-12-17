from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from tabutorch.nan import nanvar

if is_numpy_available():
    import numpy as np


def nan_values(tensor: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    mask = torch.rand_like(tensor) < ratio
    tensor[mask] = float("nan")
    return tensor


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
        show_difference=True,
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
        show_difference=True,
    )
