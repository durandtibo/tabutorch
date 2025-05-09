r"""Contain utility functions to manage tensors with NaN values."""

from __future__ import annotations

__all__ = ["check_all_nan", "check_any_nan"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def check_all_nan(x: torch.Tensor, warn: bool = False) -> None:
    r"""Check if the input tensor contains only NaN values.

    Args:
        x: The tensor to check.
        warn: If ``True``, a warning message is raised,
            otherwise an error message is raised.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import check_all_nan
    >>> check_all_nan(torch.tensor([1.0, 2.0, 3.0]))

    ```
    """
    if not x.isnan().all():
        return
    msg = "tensor contains only NaN values"
    if warn:
        raise RuntimeWarning(msg)
    raise RuntimeError(msg)


def check_any_nan(x: torch.Tensor, warn: bool = False) -> None:
    r"""Check if the input tensor contains at least one NaN values.

    Args:
        x: The tensor to check.
        warn: If ``True``, a warning message is raised,
            otherwise an error message is raised.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import check_any_nan
    >>> check_any_nan(torch.tensor([1.0, 2.0, 3.0]))

    ```
    """
    if not x.isnan().any():
        return
    msg = "tensor contains at least one NaN value"
    if warn:
        raise RuntimeWarning(msg)
    raise RuntimeError(msg)
