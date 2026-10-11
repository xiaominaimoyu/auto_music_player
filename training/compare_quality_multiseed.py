"""Compare matched quality-mask pilots across seeds with paired work bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap(values: list[float], *, seed: int, rounds: int) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "n_works": 0,
            "mean": None,
            "lower_95": None,
            "upper_95": None,
        }
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(rounds, len(array)))].mean(axis=1)
    return {
        "n_works": int(len(array)),
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _load_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    report_path = Path(path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"audit report is not complete: {report_path}")
    if report.get("work_unit") != "performance_dataset::piece_group":
        raise ValueError(f"audit report has an unexpected work unit: {report_path}")
    if not isinstance(report.get("work_values"), dict):
        raise ValueError(f"audit report has no selected work values: {report_path}")
    return report_path, report


def _pair_summary(
    *,
    baseline: dict[str, Any],
    soft_mask: dict[str, Any],
    seed: str,
    rounds: int,
) -> dict[str, Any]:
    baseline_values = baseline["work_values"]
    soft_values = soft_mask["work_values"]
    target_names = sorted(
        set(baseline_values).intersection(soft_values)
    )
    result: dict[str, Any] = {
        "seed": seed,
        "baseline": baseline.get("checkpoint"),
        "soft_mask": soft_mask.get("checkpoint"),
        "sample_id_count_baseline": baseline.get("sample_id_count"),
        "sample_id_count_soft_mask": soft_mask.get("sample_id_count"),
        "targets": {},
    }
    for target in target_names:
        base_by_work = {
            str(key): float(value) for key, value in baseline_values[target].items()
        }
        soft_by_work = {
            str(key): float(value) for key, value in soft_values[target].items()
        }
        common_works = sorted(set(base_by_work).intersection(soft_by_work))
        deltas = [soft_by_work[key] - base_by_work[key] for key in common_works]
        result["targets"][target] = {
            "n_works": len(common_works),
            "baseline_mean": float(np.mean([base_by_work[key] for key in common_works]))
            if common_works
            else None,
            "soft_mask_mean": float(np.mean([soft_by_work[key] for key in common_works]))
            if common_works
            else None,
            "delta_soft_minus_baseline": float(np.mean(deltas)) if deltas else None,
            "paired_bootstrap": _bootstrap(
                deltas,
                seed=20261003 + int(seed) % 10000 + len(result["targets"]),
                rounds=rounds,
            ),
        }
    return result


def compare(
    *,
    pairs: list[tuple[str, str | Path, str | Path]],
    validation_sample_ids: str | Path,
    output: str | Path,
    rounds: int = 2000,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one baseline/soft-mask pair is required")
    if rounds <= 0:
        raise ValueError("bootstrap rounds must be positive")
    validation_path = Path(validation_sample_ids).resolve()
    validation_sha256 = _sha256(validation_path)
    reports = []
    for seed, baseline_path, soft_path in pairs:
        base_path, base = _load_report(baseline_path)
        soft_path_resolved, soft = _load_report(soft_path)
        if base.get("split") != "validation" or soft.get("split") != "validation":
            raise ValueError(f"pair {seed} is not a validation audit")
        if base.get("sample_id_file_sha256") != soft.get("sample_id_file_sha256"):
            raise ValueError(f"pair {seed} uses different validation sample IDs")
        if (
            base.get("sample_id_file_sha256") != validation_sha256
            or soft.get("sample_id_file_sha256") != validation_sha256
        ):
            raise ValueError(f"pair {seed} does not use the supplied validation ID file")
        reports.append(
            {
                "seed": str(seed),
                "baseline_path": str(base_path),
                "baseline_sha256": _sha256(base_path),
                "soft_mask_path": str(soft_path_resolved),
                "soft_mask_sha256": _sha256(soft_path_resolved),
                "summary": _pair_summary(
                    baseline=base,
                    soft_mask=soft,
                    seed=str(seed),
                    rounds=rounds,
                ),
            }
        )

    target_sets = [set(item["summary"]["targets"]) for item in reports]
    target_names = sorted(set.intersection(*target_sets))
    aggregate: dict[str, Any] = {}
    for target in target_names:
        per_seed_targets = [item["summary"]["targets"][target] for item in reports]
        # Keep only works present in every seed so the aggregate remains paired.
        work_maps: list[tuple[dict[str, float], dict[str, float]]] = []
        for item in reports:
            base_report = json.loads(
                Path(item["baseline_path"]).read_text(encoding="utf-8")
            )
            soft_report = json.loads(
                Path(item["soft_mask_path"]).read_text(encoding="utf-8")
            )
            work_maps.append(
                (
                    {
                        str(key): float(value)
                        for key, value in base_report["work_values"][target].items()
                    },
                    {
                        str(key): float(value)
                        for key, value in soft_report["work_values"][target].items()
                    },
                )
            )
        common_works = set(work_maps[0][0]).intersection(work_maps[0][1])
        for base_by_work, soft_by_work in work_maps[1:]:
            common_works.intersection_update(base_by_work, soft_by_work)
        ordered_works = sorted(common_works)
        baseline_by_work = {
            work: float(np.mean([base_by_work[work] for base_by_work, _ in work_maps]))
            for work in ordered_works
        }
        soft_by_work = {
            work: float(np.mean([soft_by_work[work] for _, soft_by_work in work_maps]))
            for work in ordered_works
        }
        deltas = [soft_by_work[work] - baseline_by_work[work] for work in ordered_works]
        aggregate[target] = {
            "n_seeds": len(reports),
            "n_works": len(ordered_works),
            "baseline_mean": float(np.mean(list(baseline_by_work.values())))
            if ordered_works
            else None,
            "soft_mask_mean": float(np.mean(list(soft_by_work.values())))
            if ordered_works
            else None,
            "delta_soft_minus_baseline": float(np.mean(deltas)) if deltas else None,
            "paired_work_bootstrap": _bootstrap(
                deltas,
                seed=20262003 + len(aggregate),
                rounds=rounds,
            ),
            "per_seed_delta": [
                {
                    "seed": reports[index]["seed"],
                    "n_works": per_seed_targets[index]["n_works"],
                    "delta_soft_minus_baseline": per_seed_targets[index][
                        "delta_soft_minus_baseline"
                    ],
                }
                for index in range(len(reports))
            ],
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "comparison": "matched validation sample IDs, three seeds, piece_group paired bootstrap",
        "validation_sample_ids": str(validation_path),
        "validation_sample_ids_sha256": validation_sha256,
        "pair_count": len(reports),
        "bootstrap_rounds": rounds,
        "pairs": reports,
        "aggregate": aggregate,
        "interpretation": "diagnostic only; matched pilot works are not a generalization estimate and no test split was used",
    }
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["output"] = str(output_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path.with_name(output_path.name + ".sha256")).write_text(
        _sha256(output_path) + "\n", encoding="ascii"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("SEED", "BASELINE_AUDIT", "SOFT_MASK_AUDIT"),
        required=True,
    )
    parser.add_argument("--validation-sample-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=2000)
    args = parser.parse_args(argv)
    pairs = [(str(seed), baseline, soft_mask) for seed, baseline, soft_mask in args.pair]
    result = compare(
        pairs=pairs,
        validation_sample_ids=args.validation_sample_ids,
        output=args.output,
        rounds=args.rounds,
    )
    print(json.dumps({"status": result["status"], "output": result["output"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compare", "main"]
