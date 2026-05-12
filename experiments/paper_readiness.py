"""Assess whether run logs are complete enough for reported-result analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.matrix_status import build_status_table
from experiments.reproduction_check import ReproductionCheckResult, validate_reproduction_logs
from experiments.registry import BASELINES


ARTICLE_STAGES = ("train", "rollout", "planning", "rule_dropout", "noisy_rules")


@dataclass(frozen=True)
class ReadinessReport:
    runs: str
    mode: str
    baselines: tuple[str, ...]
    required_seeds: int
    required_horizons: tuple[int, ...]
    required_stages: tuple[str, ...]
    complete_baselines: tuple[str, ...]
    incomplete_baselines: tuple[str, ...]
    validation: ReproductionCheckResult

    @property
    def ok(self) -> bool:
        return not self.incomplete_baselines and self.validation.ok

    def to_json_dict(self) -> dict[str, object]:
        validation = asdict(self.validation)
        return {
            "runs": self.runs,
            "mode": self.mode,
            "baselines": list(self.baselines),
            "required_seeds": self.required_seeds,
            "required_horizons": list(self.required_horizons),
            "required_stages": list(self.required_stages),
            "complete_baselines": list(self.complete_baselines),
            "incomplete_baselines": list(self.incomplete_baselines),
            "validation": validation,
            "ok": self.ok,
        }


def build_readiness_report(
    runs: str | Path,
    *,
    mode: str = "full",
    baselines: tuple[str, ...] | None = None,
    required_seeds: int | None = None,
    required_horizons: tuple[int, ...] | None = None,
    required_stages: tuple[str, ...] = ARTICLE_STAGES,
) -> tuple[ReadinessReport, object]:
    selected = baselines or tuple(BASELINES)
    seed_count = required_seeds if required_seeds is not None else (1 if mode == "smoke" else 5)
    horizons = required_horizons or ((3, 5) if mode == "smoke" else (10, 25, 50))
    status = build_status_table(
        runs,
        mode=mode,
        stages=list(required_stages),
        baselines=selected,
        required_seeds=seed_count,
        required_horizons=horizons,
    )
    complete = tuple(str(row.baseline_id) for row in status.itertuples() if bool(row.complete))
    incomplete = tuple(str(row.baseline_id) for row in status.itertuples() if not bool(row.complete))
    validation = validate_reproduction_logs(
        runs,
        baselines=selected,
        required_seeds=seed_count,
        required_horizons=horizons,
    )
    report = ReadinessReport(
        runs=str(runs),
        mode=mode,
        baselines=selected,
        required_seeds=seed_count,
        required_horizons=horizons,
        required_stages=required_stages,
        complete_baselines=complete,
        incomplete_baselines=incomplete,
        validation=validation,
    )
    return report, status


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--baselines", default="")
    parser.add_argument("--required-seeds", type=int, default=None)
    parser.add_argument("--required-horizons", default="")
    parser.add_argument("--required-stages", default=",".join(ARTICLE_STAGES))
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report, status = build_readiness_report(
        args.runs,
        mode=args.mode,
        baselines=_parse_csv(args.baselines) or None,
        required_seeds=args.required_seeds,
        required_horizons=_parse_int_csv(args.required_horizons) or None,
        required_stages=_parse_csv(args.required_stages),
    )
    print(status.to_string(index=False))
    print(
        f"\nreadiness: {'ok' if report.ok else 'incomplete'}; "
        f"complete={len(report.complete_baselines)}/{len(report.baselines)}; "
        f"missing_metrics={len(report.validation.missing_metrics)}; "
        f"missing_horizons={len(report.validation.missing_horizons)}; "
        f"insufficient_seeds={len(report.validation.insufficient_seeds)}"
    )
    if args.write_artifacts:
        out = Path(args.runs)
        out.mkdir(parents=True, exist_ok=True)
        status.to_csv(out / "paper_readiness_status.csv", index=False)
        (out / "paper_readiness_report.json").write_text(
            json.dumps(report.to_json_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"wrote {out / 'paper_readiness_status.csv'}")
        print(f"wrote {out / 'paper_readiness_report.json'}")
    if args.strict and not report.ok:
        raise SystemExit("paper readiness incomplete")


if __name__ == "__main__":
    main()
