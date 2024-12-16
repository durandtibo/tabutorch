r"""Contain utility functions to manage tensor with NaN values."""

from __future__ import annotations

__all__ = ["check_nan_policy", "contains_nan", "mean"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def check_nan_policy(nan_policy: str) -> None:
    r"""Check the NaN policy.

    Args:
        nan_policy: The NaN policy.

    Raises:
        ValueError: if ``nan_policy`` is not ``'omit'``,
            ``'propagate'``, or ``'raise'``.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.nan import check_nan_policy
    >>> check_nan_policy(nan_policy="omit")

    ```
    """
    if nan_policy not in {"omit", "propagate", "raise"}:
        msg = (
            f"Incorrect 'nan_policy': {nan_policy}. The valid values are: "
            f"'omit', 'propagate', 'raise'"
        )
        raise ValueError(msg)


def contains_nan(
    tensor: torch.Tensor, nan_policy: str = "propagate", name: str = "input tensor"
) -> bool:
    r"""Indicate if the given tensor contains at least one NaN value.

    Args:
        tensor: The tensor to check.
        nan_policy: The NaN policy. The valid values are ``'omit'``,
            ``'propagate'``, or ``'raise'``.
        name: An optional name to be more precise about the tensor when
            the exception is raised.

    Returns:
        ``True`` if the tensor contains at least one NaN value.

    Raises:
        ValueError: if the tensor contains at least one NaN value and
            ``nan_policy`` is ``'raise'``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.utils.nan import contains_nan
    >>> contains_nan(torch.tensor([1.0, 2.0, 3.0]))
    False
    >>> contains_nan(torch.tensor([1.0, 2.0, float("nan")]))
    True

    ```
    """
    check_nan_policy(nan_policy)
    isnan = tensor.isnan().any().item()
    if isnan and nan_policy == "raise":
        msg = f"{name} contains at least one NaN value"
        raise ValueError(msg)
    return isnan


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
    >>> from tabutorch.utils.nan import mean
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
