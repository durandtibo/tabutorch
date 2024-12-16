r"""Contain the implementation of a standard scaler."""

from __future__ import annotations

__all__ = ["StandardScaler"]

import torch
from torch import Tensor

from tabutorch.preprocessing import BaseTransformer


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

    Example usage:

    ```pycon

    >>> import torch
    >>> from tabutorch.preprocessing import StandardScaler
    >>> x = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    >>> module = StandardScaler(num_features=3)
    >>> module
    StandardScaler(num_features=3, std_correction=1)
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

    def __init__(self, num_features: int, std_correction: int = 1) -> None:
        super().__init__()
        self.register_buffer(name="mean", tensor=torch.zeros(num_features))
        self.register_buffer(name="std", tensor=torch.ones(num_features))

        self._std_correction = std_correction

    def extra_repr(self) -> str:
        return f"num_features={self.mean.numel()}, std_correction={self._std_correction}"

    def fit(self, x: Tensor) -> None:
        dim = self.mean.shape[0]
        x = x.view(-1, dim)
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0, correction=self._std_correction)

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        return x.sub(self.mean).div(self.std)
