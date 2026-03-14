"""Response curves for consideration-based utility scoring.

Pure functions that map a continuous input value to a 0.0-1.0 signal.
Used by score functions to convert game state (HP%, mana%, distance) into
utility scores for both rule selection and target scoring.

All functions return values clamped to [0.0, 1.0].
"""

import math


def linear(value: float, lo: float, hi: float) -> float:
    """Linear ramp from 0.0 at lo to 1.0 at hi, clamped [0, 1]."""
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    t = (value - lo) / (hi - lo)
    return max(0.0, min(1.0, t))


def inverse_linear(value: float, lo: float, hi: float) -> float:
    """Linear ramp from 1.0 at lo to 0.0 at hi, clamped [0, 1]."""
    return 1.0 - linear(value, lo, hi)


def logistic(value: float, midpoint: float, k: float = 10.0) -> float:
    """S-curve from 0 to 1 centered at midpoint. Steepness controlled by k.

    k > 0: rising (value > midpoint -> high score)
    Higher k = steeper transition.
    """
    x = k * (value - midpoint)
    # Clamp to prevent overflow
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def inverse_logistic(value: float, midpoint: float, k: float = 10.0) -> float:
    """S-curve from 1 to 0 centered at midpoint. High urgency when value is LOW."""
    return 1.0 - logistic(value, midpoint, k)


def polynomial(value: float, lo: float, hi: float, exp: float = 2.0) -> float:
    """Power curve from 0.0 at lo to 1.0 at hi, clamped [0, 1].

    exp > 1: slow start, fast finish (quadratic, cubic)
    exp < 1: fast start, slow finish (diminishing returns, sqrt-like)
    exp = 1: identical to linear()
    """
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    t = (value - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return float(t**exp)


def bell(value: float, center: float, width: float) -> float:
    """Bell curve centered at center, clamped [0, 1].

    Returns 1.0 at center, drops to ~0.0 at center +/- width.
    Models 'sweet spot' preferences (ideal distance, ideal difficulty).
    Uses Gaussian shape: exp(-((x - center) / sigma)^2)
    where sigma = width / 2.5 so that width gives ~95% dropoff.
    """
    if width <= 0:
        return 1.0 if value == center else 0.0
    sigma = width / 2.5
    return math.exp(-(((value - center) / sigma) ** 2))
