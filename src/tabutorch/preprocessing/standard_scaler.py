r"""Contain the implementation of a standard scaler."""

from __future__ import annotations

__all__ = ["StandardScaler"]

import torch
from coola.utils.format import repr_mapping_line
from torch import Tensor

from tabutorch.preprocessing import BaseTransformer
from tabutorch.preprocessing.utils import handle_zeros_in_scale, is_constant_feature
from tabutorch.utils.nan import check_nan_policy, contains_nan


class StandardScaler(BaseTransformer[Tensor]):
    r"""Standardize features by removing the mean and scaling to unit
    variance.

    The standard score of a sample ``x`` is calculated as:
    ``z = (x - m) / s`` where ``m`` is the mean of the training
    samples, and ``s`` is the standard deviation of the training
    samples. The mean and standard deviation are computed when
    ``fit`` or ``fit_transform`` are called. Calling ``forward``
    does not update the mean and standard deviation.

    Args:
        num_features: The number of features or channels of the input.
        std_correction: The difference between the sample size and
            sample degrees of freedom when estimating the standard
            deviation.
        nan_policy: The policy on how to handle NaN values in the input
            tensor when estimating the mean and standard deviation.
            The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.preprocessing import StandardScaler
    >>> x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    >>> module = StandardScaler(num_features=3)
    >>> module
    StandardScaler(num_features=3, std_correction=1, nan_policy='propagate')
    >>> module.fit(x)
    >>> module.transform(x)
    tensor([[-1.1619, -1.1619, -1.1619],
            [-0.3873, -0.3873, -0.3873],
            [ 0.3873,  0.3873,  0.3873],
            [ 1.1619,  1.1619,  1.1619]])

    ```

    Note:
        ``std_correction`` needs to be set to ``0`` to be equivalent
            to ``sklearn.preprocessing.StandardScaler``.
    """

    def __init__(
        self,
        num_features: int,
        std_correction: int = 1,
        nan_policy: str = "propagate",
    ) -> None:
        super().__init__()
        self.register_buffer(name="mean", tensor=torch.zeros(num_features))
        self.register_buffer(name="scale", tensor=torch.ones(num_features))

        self._std_correction = std_correction

        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy

    def extra_repr(self) -> str:
        return repr_mapping_line(
            {
                "num_features": self.mean.numel(),
                "std_correction": self._std_correction,
                "nan_policy": self._nan_policy,
            }
        )

    def fit(self, x: Tensor) -> None:
        contains_nan(x, nan_policy=self._nan_policy, name="'x'")
        dim = self.mean.shape[0]
        x = x.view(-1, dim)
        self.mean = x.mean(dim=0)
        std = x.std(dim=0, correction=self._std_correction)
        constant_mask = is_constant_feature(mean=self.mean, var=std.pow(2), n_samples=x.shape[0])
        self.scale = handle_zeros_in_scale(std, copy=False, constant_mask=constant_mask)

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        return x.sub(self.mean).div(self.scale)
