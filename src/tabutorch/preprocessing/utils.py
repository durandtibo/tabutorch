r"""Contain utility functions to preprocess data."""

from __future__ import annotations

__all__ = ["handle_zeros_in_scale", "is_constant_feature"]

import torch


def handle_zeros_in_scale(
    scale: torch.Tensor, copy: bool = True, constant_mask: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.

    Args:
        scale: The tensor with the scale values.
        copy: If ``True``, it creates a copy of ``scale`` before to
            update the values.
        constant_mask: An optional tensor that indicates the features
            that are constant (``True``) or not (``False``). The tensor
            must have the same shape as the ``scale`` input.

    Returns:
        The tensor with the new scale values.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.preprocessing.utils import handle_zeros_in_scale
    >>> scale = handle_zeros_in_scale(
    ...     scale=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
    ... )
    >>> scale
    tensor([4., 3., 2., 1., 1.])

    ```
    """
    if constant_mask is None:
        # Detect near constant values to avoid dividing by a very small
        # value that could lead to surprising results and numerical
        # stability issues.
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale


def is_constant_feature(
    mean: torch.Tensor, var: torch.Tensor, n_samples: torch.Tensor | int
) -> torch.Tensor:
    """Detect if a feature is indistinguishable from a constant feature.

    The detection is based on its computed variance and on the
    theoretical error bounds of the '2 pass algorithm' for variance
    computation.

    See "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Args:
        mean: The tensor with the mean values.
        var: The tensor with the variance values. This tensor must
            have the same shape as the ``mean`` input.
        n_samples: The number of samples used to compute the mean and
            variance. This tensor must have the same shape as the
            ``mean`` and ``var`` inputs.

    Returns:
        A boolean tensor that indicates the features that are constant
            (``True``) or not (``False``). The tensor must have the
            same shape as the ``mean`` and ``var`` inputs.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.preprocessing.utils import is_constant_feature
    >>> mask = is_constant_feature(
    ...     mean=torch.tensor([1.0, 2.0, 3.0, 4.0, 2.0]),
    ...     var=torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]),
    ...     n_samples=torch.tensor([10, 11, 12, 13, 10]),
    ... )
    >>> mask
    tensor([False, False, False, False,  True])

    ```
    """
    # Implementation based on ``_is_constant_feature`` in sklearn.
    eps = torch.finfo(mean.dtype).eps
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound
