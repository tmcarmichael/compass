"""Timing variation utilities for non-blocking agent control.

Uses log-normal distribution: most actions complete near the base time,
occasionally taking 1.5-2x longer.  Provides interruptible sleep for
FLEE-aware delays (polls a predicate so the brain can re-evaluate
under threat).
"""

import random
import time
from collections.abc import Callable

# When True, all sleeps in this module become no-ops.  Used by the
# headless simulator to run routines at full speed without patching
# time.sleep globally.
_suppress_sleep: bool = False


def varying_sleep(base: float, sigma: float = 0.3) -> None:
    """Sleep with log-normal timing variation.

    Most sleeps land near *base*; occasionally 1.5-2x longer.
    Never shorter than 60% of base.
    """
    if _suppress_sleep:
        return
    multiplier = max(0.6, random.lognormvariate(0, sigma))
    time.sleep(base * multiplier)


def interruptible_sleep(
    base: float,
    interrupt_fn: Callable[[], bool] | None = None,
    poll_interval: float = 0.1,
    sigma: float = 0.3,
) -> bool:
    """Sleep with log-normal variation and periodic interrupt check.

    Like varying_sleep but polls *interrupt_fn* every *poll_interval* seconds.
    Returns True if interrupted early, False if slept the full duration.

    Used to make blocking delays FLEE-aware: if the interrupt predicate
    fires (e.g., low HP, adds detected), the sleep breaks immediately
    so the brain can re-evaluate and activate emergency rules.
    """
    if _suppress_sleep:
        return False
    multiplier = max(0.6, random.lognormvariate(0, sigma))
    deadline = time.time() + base * multiplier
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        if interrupt_fn is not None:
            try:
                if interrupt_fn():
                    return True
            except Exception:
                pass
        time.sleep(min(poll_interval, max(0.001, remaining)))


def jittered_value(base: float, sigma: float) -> float:
    """Return a value jittered by gaussian noise, for thresholds etc."""
    return base + random.gauss(0, sigma)
