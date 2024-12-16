r"""Contain the base class to implement a data transformer."""

from __future__ import annotations

__all__ = ["BaseTransformer"]

from abc import abstractmethod
from typing import Generic, TypeVar

from torch.nn import Module

T = TypeVar("T")


class BaseTransformer(Generic[T], Module):
    r"""Define the base class to implement a data transformer."""

    @abstractmethod
    def fit(self, data: T) -> None:
        pass

    @abstractmethod
    def fit_transform(self, data: T) -> T:
        pass

    @abstractmethod
    def transform(self, data: T) -> T:
        pass
