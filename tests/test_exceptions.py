"""Tests for core.exceptions -- custom exception hierarchy.

Trivial tests: exceptions can be instantiated, raised, caught,
and follow the inheritance chain.
"""

from __future__ import annotations

import pytest

from core.exceptions import CompassError, MemoryReadError, ProcessNotFoundError


class TestExceptionHierarchy:
    def test_compass_error_is_exception(self) -> None:
        assert issubclass(CompassError, Exception)

    def test_memory_read_error_is_compass_error(self) -> None:
        assert issubclass(MemoryReadError, CompassError)

    def test_process_not_found_error_is_compass_error(self) -> None:
        assert issubclass(ProcessNotFoundError, CompassError)


class TestExceptionInstantiation:
    def test_compass_error_message(self) -> None:
        err = CompassError("something broke")
        assert str(err) == "something broke"

    def test_memory_read_error_raises(self) -> None:
        with pytest.raises(MemoryReadError):
            raise MemoryReadError("read failed at 0xDEADBEEF")

    def test_process_not_found_error_raises(self) -> None:
        with pytest.raises(ProcessNotFoundError):
            raise ProcessNotFoundError("eqgame.exe not found")

    def test_catch_via_base_class(self) -> None:
        with pytest.raises(CompassError):
            raise MemoryReadError("caught as base")
