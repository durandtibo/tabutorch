r"""Contain functions to manage tensor with NaN values."""

from __future__ import annotations

__all__ = [
    "check_all_nan",
    "check_nan_policy",
    "contains_nan",
    "mean",
    "nanmax",
    "nanmin",
    "nanstd",
    "nanvar",
    "nmax",
    "nmin",
    "std",
    "var",
]

from tabutorch.nan.policy import check_nan_policy, contains_nan
from tabutorch.nan.reduction import (
    mean,
    nanmax,
    nanmin,
    nanstd,
    nanvar,
    nmax,
    nmin,
    std,
    var,
)
from tabutorch.nan.utils import check_all_nan
