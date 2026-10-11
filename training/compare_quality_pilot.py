"""Summarize matched no-mask versus quality-mask diagnostic pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_summary(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not metrics:
        raise ValueError(f"run has no metrics: {root}")
    best_epoch = int(config["best_epoch"])
    record = next(item for item in metrics if int(item["epoch"]) == best_epoch)
    validation = record["validation"]
    train = record["train"]
    return {
        "path": str(root),
        "best_epoch": best_epoch,
        "best_validation_loss": float(config["best_validation_loss"]),
        "validation": {
            key: validation[key]
            for key in (
                "loss",
                "note_log_key_duration_ratio_mae",
                "note_micro_offset_ms_mae",
                "note_velocity_0_1_mae",
                "chord_log_ioi_ratio_mae",
            )
            if key in validation
        },
        "train_quality_weight_sum": train.get("quality_weight_sum"),
        "train_quality_affected_target_count": train.get(
            "quality_affected_target_count"
        ),
        "quality_mask_path": config.get("quality_mask_path"),
        "quality_mask_sha256": config.get("quality_mask_sha256"),
        "train_sample_id_count": config.get("train_sample_id_count"),
        "validation_sample_id_count": config.get("validation_sample_id_count"),
        "artifacts": {
            name: _sha256(root / name)
            for name in ("run_config.json", "metrics.jsonl", "checkpoint_best.pt")
        },
        "complete": (root / "_SUCCESS").is_file() and not (root / "_INCOMPLETE").exists(),
    }


def compare(
    *,
    baseline: str | Path,
    soft_mask: str | Path,
    pilot_selection: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    baseline_summary = _run_summary(baseline)
    soft_summary = _run_summary(soft_mask)
    if not baseline_summary["complete"] or not soft_summary["complete"]:
        raise ValueError("pilot runs must both have _SUCCESS and no _INCOMPLETE")
    baseline_loss = float(baseline_summary["best_validation_loss"])
    soft_loss = float(soft_summary["best_validation_loss"])
    delta = soft_loss - baseline_loss
    result = {
        "schema_version": 1,
        "status": "complete",
        "comparison": "same seed, architecture, optimizer, sample IDs and validation IDs",
        "baseline": baseline_summary,
        "soft_mask": soft_summary,
        "validation_loss_delta_soft_minus_baseline": delta,
        "validation_loss_relative_delta": delta / baseline_loss if baseline_loss else None,
        "pilot_selection": str(Path(pilot_selection).resolve()),
        "pilot_selection_sha256": _sha256(Path(pilot_selection).resolve()),
        "interpretation": "diagnostic only; the 18-work train set and 4-work validation set are not a generalization estimate",
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
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--soft-mask", required=True)
    parser.add_argument("--pilot-selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare(
        baseline=args.baseline,
        soft_mask=args.soft_mask,
        pilot_selection=args.pilot_selection,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "output": result["output"] if "output" in result else args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compare", "main"]
