"""Command line entry point."""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    """Import the Typer app lazily so ``python -m neurobranch.cli.main`` is clean."""
    if name == "app":
        from neurobranch.cli.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
