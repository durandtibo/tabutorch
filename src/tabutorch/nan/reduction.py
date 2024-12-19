r"""Contain reduction functions to manage tensors with NaN values."""

from __future__ import annotations

__all__ = [
    "mean",
    "nanmax",
    "nanmin",
    "nanstd",
    "nanvar",
    "nmax",
    "std",
    "var",
]

import torch
from typing_extensions import overload


@overload
def nmax(
    x: torch.Tensor, dim: None = None, nan_policy: str = "propagate"
) -> torch.Tensor: ...  # pragma: no cover


@overload
def nmax(
    x: torch.Tensor,
    dim: int | tuple[int, ...],
    *,
    keepdim: bool = False,
    nan_policy: str = "propagate",
) -> torch.return_types.max: ...  # pragma: no cover


def nmax(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
    nan_policy: str = "propagate",
) -> torch.Tensor | torch.return_types.max:
    r"""Return the maximum values.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        correction: The difference between the sample size and sample
            degrees of freedom.
        keepdim: Whether the output tensor has dim retained or not.
        nan_policy: The policy on how to handle NaN values in the input
            tensor when estimating the max. The following options are
            available: ``'omit'`` and ``'propagate'``.

    Returns:
        Returns the maximum values.

    Raises:
        ValueError: if the ``nan_policy`` value is incorrect.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import nmax
    >>> nmax(torch.tensor([1.0, 2.0, 3.0]))
    tensor(3.)
    >>> torch.max(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> nmax(torch.tensor([1.0, 2.0, 3.0, float("nan")]), nan_policy="omit")
    tensor(3.)

    ```
    """
    if nan_policy == "propagate":
        return x.max() if dim is None else x.max(dim=dim, keepdim=keepdim)
    if nan_policy == "omit":
        return nanmax(x, dim=dim, keepdim=keepdim)
    msg = f"Incorrect 'nan_policy': {nan_policy}. The valid values are: 'omit' and 'propagate'"
    raise ValueError(msg)


def mean(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
    nan_policy: str = "propagate",
) -> torch.Tensor:
    r"""Return the mean values.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        keepdim: Whether the output tensor has dim retained or not.
        nan_policy: The policy on how to handle NaN values in the input
            tensor when estimating the mean. The following options are
            available: ``'omit'`` and ``'propagate'``.

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
        return x.mean(dim=dim, keepdim=keepdim)
    if nan_policy == "omit":
        return x.nanmean(dim=dim, keepdim=keepdim)
    msg = f"Incorrect 'nan_policy': {nan_policy}. The valid values are: 'omit' and 'propagate'"
    raise ValueError(msg)


def std(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
    nan_policy: str = "propagate",
) -> torch.Tensor:
    r"""Return the standard deviation values.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        correction: The difference between the sample size and sample
            degrees of freedom.
        keepdim: Whether the output tensor has dim retained or not.
        nan_policy: The policy on how to handle NaN values in the input
            tensor when estimating the standard deviation.
            The following options are available: ``'omit'`` and
            ``'propagate'``.

    Returns:
        Returns the standard deviation values.

    Raises:
        ValueError: if the ``nan_policy`` value is incorrect.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import std
    >>> std(torch.tensor([1.0, 2.0, 3.0]))
    tensor(1.)
    >>> std(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> std(torch.tensor([1.0, 2.0, 3.0, float("nan")]), nan_policy="omit")
    tensor(1.)

    ```
    """
    if nan_policy == "propagate":
        return x.std(dim=dim, correction=correction, keepdim=keepdim)
    if nan_policy == "omit":
        return nanstd(x, dim=dim, correction=correction, keepdim=keepdim)
    msg = f"Incorrect 'nan_policy': {nan_policy}. The valid values are: 'omit' and 'propagate'"
    raise ValueError(msg)


def var(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
    nan_policy: str = "propagate",
) -> torch.Tensor:
    r"""Return the variance values.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        correction: The difference between the sample size and sample
            degrees of freedom.
        keepdim: Whether the output tensor has dim retained or not.
        nan_policy: The policy on how to handle NaN values in the input
            tensor when estimating the variance.
            The following options are available: ``'omit'`` and
            ``'propagate'``.

    Returns:
        Returns the variance values.

    Raises:
        ValueError: if the ``nan_policy`` value is incorrect.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import var
    >>> var(torch.tensor([1.0, 2.0, 3.0]))
    tensor(1.)
    >>> var(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> var(torch.tensor([1.0, 2.0, 3.0, float("nan")]), nan_policy="omit")
    tensor(1.)

    ```
    """
    if nan_policy == "propagate":
        return x.var(dim=dim, correction=correction, keepdim=keepdim)
    if nan_policy == "omit":
        return nanvar(x, dim=dim, correction=correction, keepdim=keepdim)
    msg = f"Incorrect 'nan_policy': {nan_policy}. The valid values are: 'omit' and 'propagate'"
    raise ValueError(msg)


@overload
def nanmax(x: torch.Tensor, dim: None = None) -> torch.Tensor: ...  # pragma: no cover


@overload
def nanmax(
    x: torch.Tensor, dim: int | tuple[int, ...], *, keepdim: bool = False
) -> torch.return_types.max: ...  # pragma: no cover


def nanmax(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
) -> torch.Tensor | torch.return_types.max:
    r"""Compute the maximum, while ignoring NaNs.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The maximum, while ignoring NaNs.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import nanmax
    >>> nanmax(torch.tensor([1.0, 2.0, 3.0]))
    tensor(3.)
    >>> torch.max(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> nanmax(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(3.)

    ```
    """
    if dim is None:
        return _nanmax_without_dim(x)
    return _nanmax_with_dim(x, dim=dim, keepdim=keepdim)


def _nanmax_without_dim(x: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(x.dtype).min
    mask = x.isnan()
    if mask.all():
        return torch.tensor(float("nan"))
    return x.nan_to_num(min_value).max()


def _nanmax_with_dim(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
) -> torch.return_types.max:
    min_value = torch.finfo(x.dtype).min
    res = x.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    mask = x.isnan().all(dim=dim, keepdim=keepdim)
    if mask.any():
        res[0][mask] = float("nan")
    return res


@overload
def nanmin(x: torch.Tensor, dim: None = None) -> torch.Tensor: ...  # pragma: no cover


@overload
def nanmin(
    x: torch.Tensor, dim: int | tuple[int, ...], *, keepdim: bool = False
) -> torch.return_types.min: ...  # pragma: no cover


def nanmin(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
) -> torch.Tensor | torch.return_types.min:
    r"""Compute the minimum, while ignoring NaNs.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The minimum, while ignoring NaNs.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import nanmin
    >>> nanmin(torch.tensor([1.0, 2.0, 3.0]))
    tensor(1.)
    >>> torch.min(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> nanmin(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(1.)

    ```
    """
    if dim is None:
        return _nanmin_without_dim(x)
    return _nanmin_with_dim(x, dim=dim, keepdim=keepdim)


def _nanmin_without_dim(x: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(x.dtype).max
    mask = x.isnan()
    if mask.all():
        return torch.tensor(float("nan"))
    return x.nan_to_num(min_value).min()


def _nanmin_with_dim(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
) -> torch.return_types.min:
    min_value = torch.finfo(x.dtype).max
    res = x.nan_to_num(min_value).min(dim=dim, keepdim=keepdim)
    mask = x.isnan().all(dim=dim, keepdim=keepdim)
    if mask.any():
        res[0][mask] = float("nan")
    return res


def nanstd(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the standard deviation, while ignoring NaNs.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        correction: The difference between the sample size and sample
            degrees of freedom.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The standard deviation, while ignoring NaNs.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import nanstd
    >>> nanstd(torch.tensor([1.0, 2.0, 3.0]))
    tensor(1.)
    >>> torch.var(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> nanstd(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(1.)

    ```
    """
    return nanvar(x=x, dim=dim, correction=correction, keepdim=keepdim).sqrt()


def nanvar(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    *,
    correction: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the variance, while ignoring NaNs.

    Args:
        x: The input tensor.
        dim: The dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.
        correction: The difference between the sample size and sample
            degrees of freedom.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        The variance, while ignoring NaNs.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.nan import nanvar
    >>> nanvar(torch.tensor([1.0, 2.0, 3.0]))
    tensor(1.)
    >>> torch.var(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(nan)
    >>> nanvar(torch.tensor([1.0, 2.0, 3.0, float("nan")]))
    tensor(1.)

    ```
    """
    mean = x.nanmean(dim=dim, keepdim=True)
    var = (x - mean).square().nansum(dim=dim, keepdim=keepdim)
    count = x.isnan().logical_not().sum(dim=dim, keepdim=keepdim)
    return var.div((count - correction).clamp_min(0))
