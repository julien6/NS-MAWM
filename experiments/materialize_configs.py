"""Write smoke/full YAML configs for B01-B45."""

from __future__ import annotations

import argparse

from experiments.baseline_configs import materialize_baseline_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/configs/materialized")
    args = parser.parse_args()
    written = materialize_baseline_configs(args.output_dir)
    print(f"wrote {len(written)} configs under {args.output_dir}")


if __name__ == "__main__":
    main()
