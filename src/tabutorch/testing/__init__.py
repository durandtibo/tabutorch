r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = [
    "cuda_available",
    "distributed_available",
    "gloo_available",
    "nccl_available",
    "objectory_available",
    "sklearn_available",
    "two_gpus_available",
]

from tabutorch.testing.fixtures import (
    cuda_available,
    distributed_available,
    gloo_available,
    nccl_available,
    objectory_available,
    sklearn_available,
    two_gpus_available,
)
