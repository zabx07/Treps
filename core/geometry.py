from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterable, Optional, Sequence


def calculate_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle


def calculate_vertical_lean(top: Sequence[float], bottom: Sequence[float]) -> float:
    tx, ty = top
    bx, by = bottom
    dx = tx - bx
    dy = ty - by
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-6:
        return default
    return numerator / denominator


def update_rolling_average(history: Deque[float], value: float, window: int) -> float:
    if history.maxlen != window:
        history = deque(history, maxlen=window)
    history.append(value)
    return sum(history) / len(history)


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return sum(values) / len(values)


def as_pixel(point: Sequence[float], width: int, height: int) -> tuple[int, int]:
    return int(point[0] * width), int(point[1] * height)
