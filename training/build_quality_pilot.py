"""Build deterministic train/validation diagnostic sets for quality-mask pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from training.quality_mask import load_quality_mask


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    flagged_ids: set[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    by_id = {str(row["id"]): row for row in rows}
    reviewed = sorted(
        sample_id
        for sample_id in flagged_ids
        if by_id.get(sample_id, {}).get("split") == split
    )
    controls: list[str] = []
    selections: list[dict[str, Any]] = []
    used_controls: set[str] = set()
    for sample_id in reviewed:
        target = by_id[sample_id]
        source = target.get("performance_dataset")
        target_piece = target.get("piece_group")
        candidates = [
            row
            for row in rows
            if row.get("split") == split
            and row.get("performance_dataset") == source
            and str(row.get("id")) not in flagged_ids
            and str(row.get("id")) not in used_controls
            and row.get("piece_group") != target_piece
            and not row.get("warning_flags")
            and float(row.get("refined_recall") or 0.0) >= 0.95
            and float(row.get("interpolation_rate") or 1.0) <= 0.05
        ]
        if not candidates:
            raise ValueError(f"no clean control candidate for {sample_id}")
        target_notes = max(1, int(target.get("note_count") or 1))
        candidates.sort(
            key=lambda row: (
                abs(math.log(max(1, int(row.get("note_count") or 1)) / target_notes)),
                str(row.get("id")),
            )
        )
        control = candidates[0]
        control_id = str(control["id"])
        used_controls.add(control_id)
        controls.append(control_id)
        selections.append(
            {
                "reviewed_sample_id": sample_id,
                "control_sample_id": control_id,
                "source": source,
                "reviewed_piece_group": target_piece,
                "control_piece_group": control.get("piece_group"),
                "reviewed_note_count": target.get("note_count"),
                "control_note_count": control.get("note_count"),
            }
        )
    return reviewed, controls, selections


def build(
    *, dataset: str | Path, quality_mask: str | Path, output: str | Path
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    mask_data = load_quality_mask(quality_mask)
    manifest_path = dataset_root / "manifest.jsonl"
    rows = _read_manifest(manifest_path)
    flagged_ids = set(mask_data.decisions)
    reviewed_train, train_controls, train_selections = _select_split(
        rows, split="train", flagged_ids=flagged_ids
    )
    if not reviewed_train:
        raise ValueError("quality mask has no reviewed train samples")
    reviewed_validation, validation_controls, validation_selections = _select_split(
        rows, split="validation", flagged_ids=flagged_ids
    )
    selected_ids = reviewed_train + train_controls
    validation_ids = reviewed_validation + validation_controls
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    validation_output_path = output_path.with_name("validation_sample_ids.txt")
    validation_output_path.write_text(
        "\n".join(validation_ids) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": str(dataset_root),
        "manifest_sha256": _sha256(manifest_path),
        "quality_mask": mask_data.path,
        "quality_mask_sha256": mask_data.sha256,
        "quality_mask_policy_version": mask_data.policy_version,
        "reviewed_train_count": len(reviewed_train),
        "control_count": len(train_controls),
        "selected_count": len(selected_ids),
        "reviewed_train_ids": reviewed_train,
        "control_ids": train_controls,
        "selections": train_selections,
        "reviewed_validation_count": len(reviewed_validation),
        "validation_control_count": len(validation_controls),
        "validation_selected_count": len(validation_ids),
        "reviewed_validation_ids": reviewed_validation,
        "validation_control_ids": validation_controls,
        "validation_selections": validation_selections,
        "output": str(output_path),
        "validation_output": str(validation_output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path.with_name(output_path.name + ".sha256").write_text(
        _sha256(output_path) + "\n", encoding="ascii"
    )
    validation_output_path.with_name(validation_output_path.name + ".sha256").write_text(
        _sha256(validation_output_path) + "\n", encoding="ascii"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quality-mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(
        dataset=args.dataset,
        quality_mask=args.quality_mask,
        output=args.output,
    )
    print(json.dumps({"status": result["status"], "selected_count": result["selected_count"], "output": result["output"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build", "main"]
