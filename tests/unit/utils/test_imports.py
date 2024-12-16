from __future__ import annotations

from unittest.mock import patch

import pytest

from tabutorch.utils.imports import (
    check_objectory,
    check_sklearn,
    is_objectory_available,
    is_sklearn_available,
    objectory_available,
    sklearn_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


#####################
#     objectory     #
#####################


def test_check_objectory_with_package() -> None:
    with patch("tabutorch.utils.imports.is_objectory_available", lambda: True):
        check_objectory()


def test_check_objectory_without_package() -> None:
    with (
        patch("tabutorch.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="'objectory' package is required but not installed."),
    ):
        check_objectory()


def test_is_objectory_available() -> None:
    assert isinstance(is_objectory_available(), bool)


def test_objectory_available_with_package() -> None:
    with patch("tabutorch.utils.imports.is_objectory_available", lambda: True):
        fn = objectory_available(my_function)
        assert fn(2) == 44


def test_objectory_available_without_package() -> None:
    with patch("tabutorch.utils.imports.is_objectory_available", lambda: False):
        fn = objectory_available(my_function)
        assert fn(2) is None


def test_objectory_available_decorator_with_package() -> None:
    with patch("tabutorch.utils.imports.is_objectory_available", lambda: True):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_objectory_available_decorator_without_package() -> None:
    with patch("tabutorch.utils.imports.is_objectory_available", lambda: False):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


###################
#     sklearn     #
###################


def test_check_sklearn_with_package() -> None:
    with patch("tabutorch.utils.imports.is_sklearn_available", lambda: True):
        check_sklearn()


def test_check_sklearn_without_package() -> None:
    with (
        patch("tabutorch.utils.imports.is_sklearn_available", lambda: False),
        pytest.raises(RuntimeError, match="'sklearn' package is required but not installed."),
    ):
        check_sklearn()


def test_is_sklearn_available() -> None:
    assert isinstance(is_sklearn_available(), bool)


def test_sklearn_available_with_package() -> None:
    with patch("tabutorch.utils.imports.is_sklearn_available", lambda: True):
        fn = sklearn_available(my_function)
        assert fn(2) == 44


def test_sklearn_available_without_package() -> None:
    with patch("tabutorch.utils.imports.is_sklearn_available", lambda: False):
        fn = sklearn_available(my_function)
        assert fn(2) is None


def test_sklearn_available_decorator_with_package() -> None:
    with patch("tabutorch.utils.imports.is_sklearn_available", lambda: True):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_sklearn_available_decorator_without_package() -> None:
    with patch("tabutorch.utils.imports.is_sklearn_available", lambda: False):

        @sklearn_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
