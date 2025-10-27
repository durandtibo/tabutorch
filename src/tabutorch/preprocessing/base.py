r"""Contain the base class to implement a data transformer."""

from __future__ import annotations

__all__ = ["BaseTransformer"]

from abc import abstractmethod
from typing import Generic, TypeVar

from torch.nn import Module

T = TypeVar("T")


class BaseTransformer(Module, Generic[T]):
    r"""Define the base class to implement a data transformer."""

    def forward(self, x: T) -> T:
        return self.transform(x)

    @abstractmethod
    def fit(self, x: T) -> None:
        pass

    @abstractmethod
    def fit_transform(self, x: T) -> T:
        pass

    @abstractmethod
    def transform(self, x: T) -> T:
        pass
