r"""Contain reduction functions to manage tensors with NaN values."""

from __future__ import annotations

__all__ = ["mean"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def mean(x: torch.Tensor, *args: Any, nan_policy: str = "propagate", **kwargs: Any) -> torch.Tensor:
    r"""Return the mean values.

    Args:
        x: The input tensor.
        *args: Positional arguments of ``torch.mean`` or
            ``torch.nanmean``.
        nan_policy: The policy on how to handle NaN values in the input
            tensor when estimating the mean. The following options are
            available: ``'omit'`` and ``'propagate'``.
        **kwargs: Keyword arguments of ``torch.mean`` or
            ``torch.nanmean``.

    Returns:
        Returns the mean values.

    Raises:
        ValueError: if the ``nan_policy`` value is incorrect.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import mean
    >>> mean(torch.tensor([1.0, 2.0, 3.0]))
    tensor(2.)
    >>> mean(torch.tensor([1.0, 2.0, float("nan")]))
    tensor(nan)
    >>> mean(torch.tensor([1.0, 2.0, float("nan")]), nan_policy="omit")
    tensor(1.5000)

    ```
    """
    if nan_policy == "propagate":
        return x.mean(*args, **kwargs)
    if nan_policy == "omit":
        return x.nanmean(*args, **kwargs)
    msg = f"Incorrect 'nan_policy': {nan_policy}. The valid values are: 'omit' and 'propagate'"
    raise ValueError(msg)
