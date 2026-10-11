"""Audit true-bar geometry and normalized labels without training or inputs.

Only locked train/validation timing references are accepted via their existing
integrity loader. New artifacts are evaluation-only, not masks or model inputs.
Historical fixed-score adoption reports and checkpoints are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.bar_supervision import (
    DEFAULT_BAR_HORIZONS, NORMALIZED_TARGET_DOMAIN, SCORE_GEOMETRY_FIELDS,
    BarSupervisionError, build_bar_spans, decompose_bar_targets,
    evaluate_normalized_timing, span_coverage, supervised_span_eligibility,
    validate_horizons,
)
from training.pianocore_dataset import write_deterministic_npz
from training.tempo_reference import _canonical, _safe_path, _sha256, load_timing_reference


def _aggregate(rows: Sequence[dict[str, Any]], horizons: Sequence[int]) -> dict[str, Any]:
    result = {"sample_count": len(rows), "horizons": {}}
    count_fields = (
        "exact_geometry_window_count", "missing_end_boundary_count",
        "partial_bar_excluded_window_count", "untrusted_start_count",
        "invalid_ioi_after_trusted_start_count", "eligible_window_count",
        "max_chords_excluded_window_count", "capacity_retained_window_count",
        "same_core_window_count", "cross_core_window_count",
    )
    for horizon in horizons:
        key = str(horizon)
        records = [row["coverage"]["horizons"][key] for row in rows]
        counts = {field: sum(record[field] for record in records) for field in count_fields}
        for field in ("eligible_window_count", "capacity_retained_window_count", "same_core_window_count"):
            counts[field.removesuffix("_window_count") + "_covered_sample_count"] = sum(record[field] > 0 for record in records)
        result["horizons"][key] = counts
    result["complete_one_bar_decomposition_count"] = sum(row["decomposition"]["bar_count"] for row in rows)
    result["decomposition_covered_interval_count"] = sum(row["decomposition"]["covered_interval_count"] for row in rows)
    for field in ("normalization_max_relative_error", "reconstruction_max_error_seconds"):
        values = [row["decomposition"][field] for row in rows if row["decomposition"][field] is not None]
        result[field] = max(values) if values else None
    roundtrip = [row["native_target_normalized_roundtrip"]["native_target_roundtrip_max_error_seconds"]
                 for row in rows if row["native_target_normalized_roundtrip"]["native_target_roundtrip_max_error_seconds"] is not None]
    result["normalized_native_human_roundtrip_max_error_seconds"] = max(roundtrip) if roundtrip else None
    return result


def audit_bar_supervision(*, timing_reference: str | Path, output_dir: str | Path,
                          horizons: Sequence[int] = DEFAULT_BAR_HORIZONS,
                          core_size: int = 128, max_chords: int | None = None) -> dict[str, Any]:
    """Write a new sealed diagnostic; never edit references/datasets/checkpoints."""
    selected = validate_horizons(horizons)
    if 1 not in selected:
        raise BarSupervisionError("audit requires one-bar geometry for decomposition")
    root, output = Path(timing_reference).resolve(), Path(output_dir).resolve()
    if root.is_file():
        root = root.parent
    if output == root or output.is_relative_to(root):
        raise BarSupervisionError("output must not be inside the timing reference")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise BarSupervisionError("output directory must be new or empty")
    implementation_paths = [Path(__file__).resolve().parent / name for name in (
        "audit_bar_supervision.py", "bar_supervision.py", "tempo_reference.py",
        "midi_features.py", "pianocore_dataset.py",
    )]
    implementations = {path.name: _sha256(path) for path in implementation_paths}
    summary_path = root / "summary.json"
    source_summary_hash = _sha256(summary_path)
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    references = load_timing_reference(root)
    if _sha256(summary_path) != source_summary_hash:
        raise BarSupervisionError("source timing summary changed while loading")
    if output.is_relative_to(Path(source["dataset_root"]).resolve()):
        raise BarSupervisionError("output must not be inside the source dataset")
    metadata = {str(row["id"]): row for row in source["samples"]}
    if set(metadata) != set(references):
        raise BarSupervisionError("source metadata and verified references differ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "_INCOMPLETE").write_text("bar geometry audit in progress\n", encoding="ascii")
    rows = []
    for position, (sample_id, reference) in enumerate(sorted(references.items()), start=1):
        spans = build_bar_spans(reference, selected)
        native = reference["chord_native_masks"][:, 1]
        trusted = reference["chord_trusted"]
        target = reference["chord_log_ioi_ratio"]
        coverage = span_coverage(spans, native, trusted, core_size=core_size, max_chords=max_chords)
        eligibility = supervised_span_eligibility(spans, native, trusted)
        decomposition = decompose_bar_targets(spans, native, trusted, target)
        human_ioi = np.diff(reference["chord_performance_seconds"], prepend=reference["chord_performance_seconds"][0])
        diagnostics = dict(human_ioi_seconds=human_ioi, global_tempo_scale=float(reference["global_tempo_scale"]))
        identity = evaluate_normalized_timing(spans, np.zeros(spans.chord_count), target, native, trusted, **diagnostics)
        roundtrip = evaluate_normalized_timing(spans, target, target, native, trusted, **diagnostics)
        relative = f"samples/{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()}.npz"
        arrays = {
            "span_starts": spans.starts, "span_ends": spans.ends, "span_horizons": spans.horizons,
            "span_eligible": eligibility["eligible"], "span_untrusted_start": eligibility["untrusted_start"],
            "span_invalid_ioi_after_trusted_start": eligibility["invalid_ioi_after_trusted_start"],
            "span_same_core": spans.starts // coverage["core_size"] == spans.ends // coverage["core_size"],
            "span_capacity_retained": np.ones(len(spans.starts), dtype=bool) if max_chords is None
                else spans.ends - spans.starts + 1 <= max_chords,
            "score_ioi_seconds": spans.score_ioi_seconds,
            **{name: decomposition[name] for name in (
                "bar_starts", "bar_ends", "bar_tempo_log_ratio", "local_log_ioi_ratio", "local_available")},
        }
        digest = write_deterministic_npz(output / relative, arrays)
        info = metadata[sample_id]
        rows.append({"id": sample_id, "split": info["split"],
            "performance_dataset": info["performance_dataset"], "piece_group": info["piece_group"],
            "path": relative, "sha256": digest, "source_reference_path": info["path"],
            "source_reference_sha256": info["sha256"], "coverage": coverage,
            "decomposition": {name: value for name, value in decomposition.items() if not isinstance(value, np.ndarray)},
            "identity_normalized_reference": identity, "native_target_normalized_roundtrip": roundtrip})
        if position % 25 == 0 or position == len(references):
            print(f"bar_geometry_audit={position}/{len(references)}", flush=True)
    if any(_sha256(path) != implementations[path.name] for path in implementation_paths):
        raise BarSupervisionError("implementation changed during audit; incomplete output must not be used")
    if _sha256(summary_path) != source_summary_hash:
        raise BarSupervisionError("source timing summary changed during audit")
    if _sha256(Path(source["dataset_root"]) / "manifest.jsonl") != source["dataset_manifest_sha256"]:
        raise BarSupervisionError("source dataset manifest changed during audit")
    for info in metadata.values():
        if _sha256(_safe_path(root, info["path"])) != info["sha256"]:
            raise BarSupervisionError("source timing sample changed during audit")
    report = {"schema_version": 1, "kind": "bar_supervision_diagnostic", "status": "complete",
        "timing_reference_root": str(root), "timing_reference_summary_sha256": source_summary_hash,
        "dataset_manifest_sha256": source["dataset_manifest_sha256"],
        "sample_count": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)),
        "implementation_hashes": implementations, "score_geometry_allowlist": list(SCORE_GEOMETRY_FIELDS),
        "horizons": list(selected), "core_size": core_size, "max_chords": max_chords,
        "target_domain": NORMALIZED_TARGET_DOMAIN, "domain_frozen_for_future_diagnostics_only": True,
        "pickup_policy": "unknown; MIDI tick zero assumed meter-grid bar start; no printed measure numbering claim",
        "boundary_policy": "exact recorded chord endpoints only; missing endpoints never imputed; partial bars excluded",
        "capacity_policy": "max_chords counts both endpoints; capacity exclusions never alter native masks",
        "decomposition_policy": "complete native-valid one-bar targets only; bar tempo plus normalized local residual; no slow curve fitted",
        "reference_policy": "human/global_scale diagnostic labels; predictions never receive performer scale",
        "historical_evaluation_policy": "old selection is historical coverage/label audit, not a new adoption or blind-test domain",
        "training_changed": False, "model_input": False, "adoption_gate_changed": False,
        "deployment_evidence": False, "samples": rows,
        "by_split": {split: _aggregate([row for row in rows if row["split"] == split], selected)
                     for split in sorted({row["split"] for row in rows})},
        "by_source": {name: _aggregate([row for row in rows if row["performance_dataset"] == name], selected)
                      for name in sorted({row["performance_dataset"] for row in rows})},
    }
    (output / "summary.json").write_bytes(_canonical(report) + b"\n")
    (output / "_SUCCESS").write_text(_sha256(output / "summary.json") + "\n", encoding="ascii")
    (output / "_INCOMPLETE").unlink()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_BAR_HORIZONS))
    parser.add_argument("--core-size", type=int, default=128)
    parser.add_argument("--max-chords", type=int)
    args = parser.parse_args(argv)
    report = audit_bar_supervision(timing_reference=args.timing_reference, output_dir=args.output_dir,
                                  horizons=args.horizons, core_size=args.core_size, max_chords=args.max_chords)
    print(json.dumps({name: report[name] for name in ("status", "sample_count", "split_counts", "by_split")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
