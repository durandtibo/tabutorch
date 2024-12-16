r"""Contain utility functions to manage tensor with NaN values."""

from __future__ import annotations

__all__ = ["check_nan_policy", "contains_nan", "mean", "std"]

from tabutorch.nan.policy import check_nan_policy, contains_nan
from tabutorch.nan.reduction import mean, std