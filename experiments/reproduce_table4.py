"""Print the B01-B45 registry."""

from __future__ import annotations

from experiments.registry import BASELINES


def main() -> None:
    for key, spec in BASELINES.items():
        print(key, spec)


if __name__ == "__main__":
    main()
