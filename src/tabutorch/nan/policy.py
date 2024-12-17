r"""Contain utility functions to manage tensor with NaN values."""

from __future__ import annotations

__all__ = ["check_nan_policy", "contains_nan"]

from typing import TYPE_CHECKING

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

    >>> from tabutorch.nan import check_nan_policy
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
    >>> from tabutorch.nan import contains_nan
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
