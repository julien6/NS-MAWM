#!/usr/bin/env python
from experiments.registry import BASELINES


def main() -> None:
    for key, spec in BASELINES.items():
        print(key, spec)


if __name__ == "__main__":
    main()
