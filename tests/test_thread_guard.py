"""Tests for util.thread_guard -- thread ownership assertions.

Warning-only system: violations are logged, not raised. Tests verify
that set_brain_thread, assert_brain_thread, and violation_count work
correctly, including cross-thread detection and cooldown behavior.
"""

from __future__ import annotations

import threading

import util.thread_guard as tg


def _reset_module() -> None:
    """Reset module-level state for test isolation."""
    tg._brain_thread_id = 0
    tg._violation_count = 0
    tg._last_violation_time = 0.0


class TestSetBrainThread:
    def test_sets_current_thread(self) -> None:
        _reset_module()
        tg.set_brain_thread()
        assert tg._brain_thread_id == threading.current_thread().ident

    def test_violation_count_zero_after_set(self) -> None:
        _reset_module()
        tg.set_brain_thread()
        assert tg.violation_count() == 0


class TestAssertBrainThread:
    def test_no_violation_on_correct_thread(self) -> None:
        _reset_module()
        tg.set_brain_thread()  # register current thread as brain thread
        tg.assert_brain_thread("test_field")
        assert tg.violation_count() == 0

    def test_skipped_before_init(self) -> None:
        """Before set_brain_thread(), assert_brain_thread is a no-op."""
        _reset_module()
        tg.assert_brain_thread("test_field")
        assert tg.violation_count() == 0

    def test_violation_from_different_thread(self) -> None:
        """Calling from a non-brain thread increments violation count."""
        _reset_module()
        tg.set_brain_thread()  # main thread is "brain"

        violation_detected = threading.Event()

        def worker() -> None:
            tg.assert_brain_thread("my_field")
            violation_detected.set()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5.0)

        assert violation_detected.is_set()
        assert tg.violation_count() >= 1

    def test_multiple_violations_counted(self) -> None:
        _reset_module()
        tg.set_brain_thread()

        results: list[int] = []

        def worker() -> None:
            tg.assert_brain_thread("field_a")
            tg.assert_brain_thread("field_b")
            results.append(tg.violation_count())

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5.0)

        assert tg.violation_count() >= 2


class TestViolationCount:
    def test_returns_module_level_count(self) -> None:
        _reset_module()
        assert tg.violation_count() == 0

    def test_increments_on_violation(self) -> None:
        _reset_module()
        tg.set_brain_thread()

        done = threading.Event()

        def worker() -> None:
            tg.assert_brain_thread("x")
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5.0)

        assert done.is_set()
        assert tg.violation_count() == 1
