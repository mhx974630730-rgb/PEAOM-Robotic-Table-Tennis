#!/usr/bin/env python3
"""ROS-independent timing conversion shared by simulation and real control.

The paper model gives an arrival duration

    f(k) = (1 - exp(-k * (x_hit - x0))) / (k * vx0).

The physical delay relative to the predicted hit is therefore

    t_delay = f(k) + m,
    m = state_t0 - predicted_hit_time.

Because f(k) depends on the current ball state, one fixed delay interval
cannot correspond to K_RANGE for every ball.  In t_delay mode we consequently
derive the physical delay endpoints from K_RANGE for each inference.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple


def arrival_time_from_k(x0: float, vx0: float, x_hit: float,
                        k: float) -> Optional[float]:
    """Return f(k), using a numerically stable k -> 0 limit."""
    dx = float(x_hit - x0)
    vx0 = float(vx0)
    k = float(k)
    if not all(math.isfinite(value) for value in (dx, vx0, k)):
        return None
    if abs(vx0) < 1e-9:
        return None
    if abs(k) < 1e-9:
        result = dx / vx0
    else:
        result = -math.expm1(-k * dx) / (k * vx0)
    return float(result) if math.isfinite(result) and result > 0.0 else None


def k_from_arrival_time(x0: float, vx0: float, x_hit: float,
                        arrival_time: float) -> Optional[float]:
    """Invert f(k) by monotonic bisection and return the real solution."""
    dx = float(x_hit - x0)
    vx0 = float(vx0)
    arrival_time = float(arrival_time)
    if not all(math.isfinite(value) for value in (dx, vx0, arrival_time)):
        return None
    if abs(dx) < 1e-12 or abs(vx0) < 1e-12 or arrival_time <= 0.0:
        return None

    constant_velocity_time = dx / vx0
    if constant_velocity_time <= 0.0:
        return None
    ratio = arrival_time / constant_velocity_time
    if not math.isfinite(ratio) or ratio <= 0.0:
        return None
    if abs(ratio - 1.0) < 1e-12:
        return 0.0

    def residual(y: float) -> float:
        if abs(y) < 1e-10:
            value = 1.0 - 0.5 * y + (y * y) / 6.0
        else:
            try:
                value = -math.expm1(-y) / y
            except OverflowError:
                return math.inf
        return float(value - ratio)

    if ratio < 1.0:
        lo, hi = 0.0, 1.0
        while residual(hi) > 0.0 and hi < 512.0:
            hi *= 2.0
    else:
        lo, hi = -1.0, 0.0
        while residual(lo) < 0.0 and lo > -512.0:
            lo *= 2.0

    f_lo, f_hi = residual(lo), residual(hi)
    if not math.isfinite(f_lo) or not math.isfinite(f_hi) or f_lo * f_hi > 0.0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < 1e-12:
            return float(mid / dx)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(0.5 * (lo + hi) / dx)


def _checked_k_range(k_range: Sequence[float]) -> Tuple[float, float]:
    if len(k_range) != 2:
        raise ValueError("k_range must contain exactly two values")
    k_min, k_max = (float(k_range[0]), float(k_range[1]))
    if not math.isfinite(k_min) or not math.isfinite(k_max) or k_min >= k_max:
        raise ValueError("k_range must be finite and strictly increasing")
    return k_min, k_max


def dynamic_t_delay_bounds(x0: float, vx0: float, x_hit: float,
                           state_t0: float, predicted_hit_time: float,
                           k_range: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Return the per-ball physical delay interval corresponding to K_RANGE.

    The return order is ``(earliest_delay, latest_delay)``.  For the normal
    forward ball motion, earliest_delay corresponds to k_max and latest_delay
    corresponds to k_min.
    """
    k_min, k_max = _checked_k_range(k_range)
    arrival_at_k_min = arrival_time_from_k(x0, vx0, x_hit, k_min)
    arrival_at_k_max = arrival_time_from_k(x0, vx0, x_hit, k_max)
    if arrival_at_k_min is None or arrival_at_k_max is None:
        return None
    offset_m = float(state_t0) - float(predicted_hit_time)
    delay_at_k_min = arrival_at_k_min + offset_m
    delay_at_k_max = arrival_at_k_max + offset_m
    if not math.isfinite(delay_at_k_min) or not math.isfinite(delay_at_k_max):
        return None
    return (min(delay_at_k_min, delay_at_k_max),
            max(delay_at_k_min, delay_at_k_max))


def resolve_policy_timing(*, mode: str, timing_value: float,
                          x0: float, vx0: float, x_hit: float,
                          state_t0: float, predicted_hit_time: float,
                          k_range: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Resolve one policy timing output.

    Returns ``(equivalent_k, physical_t_delay, scheduled_hit_rel)``.

    * ``mode='k'``: timing_value is the mapped k.
    * ``mode='t_delay'``: timing_value is a normalized selector in [-1, 1].
      -1 selects the earliest physical delay (k_max), +1 selects the latest
      delay (k_min), and intermediate values interpolate in arrival time.
    """
    k_min, k_max = _checked_k_range(k_range)
    predicted_hit_time = float(predicted_hit_time)
    state_t0 = float(state_t0)
    if not math.isfinite(predicted_hit_time) or not math.isfinite(state_t0):
        return None, None, None

    if mode == "k":
        k = float(timing_value)
        if not math.isfinite(k):
            return None, None, None
        tolerance = 1e-12 * max(1.0, abs(k_min), abs(k_max))
        if k < k_min - tolerance or k > k_max + tolerance:
            return None, None, None
        k = min(max(k, k_min), k_max)
        arrival_time = arrival_time_from_k(x0, vx0, x_hit, k)
        if arrival_time is None:
            return None, None, None
    elif mode == "t_delay":
        selector = float(timing_value)
        if not math.isfinite(selector):
            return None, None, None
        selector = min(max(selector, -1.0), 1.0)
        arrival_earliest = arrival_time_from_k(x0, vx0, x_hit, k_max)
        arrival_latest = arrival_time_from_k(x0, vx0, x_hit, k_min)
        if arrival_earliest is None or arrival_latest is None:
            return None, None, None
        alpha = 0.5 * (selector + 1.0)
        arrival_time = arrival_earliest + alpha * (arrival_latest - arrival_earliest)
        k = k_from_arrival_time(x0, vx0, x_hit, arrival_time)
        if k is None:
            return None, None, None
        tolerance = 1e-8 * max(1.0, abs(k_min), abs(k_max))
        if k < k_min - tolerance or k > k_max + tolerance:
            return None, None, None
        k = min(max(float(k), k_min), k_max)
    else:
        raise ValueError("mode must be 'k' or 't_delay'")

    scheduled_hit_rel = state_t0 + arrival_time
    t_delay = scheduled_hit_rel - predicted_hit_time
    if not all(math.isfinite(value) for value in (k, t_delay, scheduled_hit_rel)):
        return None, None, None
    return float(k), float(t_delay), float(scheduled_hit_rel)
