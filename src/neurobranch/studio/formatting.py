"""Consistent number formatting for the studio surface."""

from __future__ import annotations


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def fixed(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def interval(low: float, high: float, digits: int = 3) -> str:
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def seconds(value: float) -> str:
    return f"{value:.1f}s"


def integer(value: int) -> str:
    return f"{int(value):,}"


def bytes_human(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{value:,} B"
