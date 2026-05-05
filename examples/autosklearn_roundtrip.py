"""Disabled autosklearn example.

auto-sklearn 0.15.0 requires scikit-learn<0.25, which is incompatible with this
project's modern Python/dependency baseline.
"""

from __future__ import annotations


def main() -> None:
    """Exit with a clear message explaining why this example is disabled."""
    raise SystemExit(
        "This example is intentionally disabled: auto-sklearn currently conflicts "
        "with the project's supported Python/dependency baseline."
    )


if __name__ == "__main__":
    main()
