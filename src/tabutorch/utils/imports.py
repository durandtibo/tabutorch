r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_objectory",
    "check_sklearn",
    "is_objectory_available",
    "is_sklearn_available",
    "objectory_available",
    "sklearn_available",
]

from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available, package_available

if TYPE_CHECKING:
    from collections.abc import Callable


#####################
#     objectory     #
#####################


def is_objectory_available() -> bool:
    r"""Indicate if the ``objectory`` package is installed or not.

    Returns:
        ``True`` if ``objectory`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.imports import is_objectory_available
    >>> is_objectory_available()

    ```
    """
    return package_available("objectory")


def check_objectory() -> None:
    r"""Check if the ``objectory`` package is installed.

    Raises:
        RuntimeError: if the ``objectory`` package is not installed.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.imports import check_objectory
    >>> check_objectory()

    ```
    """
    if not is_objectory_available():
        msg = (
            "'objectory' package is required but not installed. "
            "You can install 'objectory' package with the command:\n\n"
            "pip install objectory\n"
        )
        raise RuntimeError(msg)


def objectory_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``objectory``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``objectory`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.imports import objectory_available
    >>> @objectory_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_objectory_available)


###################
#     sklearn     #
###################


def is_sklearn_available() -> bool:
    r"""Indicate if the ``sklearn`` package is installed or not.

    Returns:
        ``True`` if ``sklearn`` is available otherwise ``False``.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.imports import is_sklearn_available
    >>> is_sklearn_available()

    ```
    """
    return package_available("sklearn")


def check_sklearn() -> None:
    r"""Check if the ``sklearn`` package is installed.

    Raises:
        RuntimeError: if the ``sklearn`` package is not installed.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.imports import check_sklearn
    >>> check_sklearn()

    ```
    """
    if not is_sklearn_available():
        msg = (
            "'sklearn' package is required but not installed. "
            "You can install 'sklearn' package with the command:\n\n"
            "pip install scikit-learn\n"
        )
        raise RuntimeError(msg)


def sklearn_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``sklearn``
    package is installed.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``sklearn`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon

    >>> from tabutorch.utils.imports import sklearn_available
    >>> @sklearn_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_sklearn_available)
