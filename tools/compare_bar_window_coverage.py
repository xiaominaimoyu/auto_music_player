"""Read-only comparison of fixed-128 versus bar-aligned cores on the balanced dev set.

Writes a diagnostic artifact under E: and never touches datasets, checkpoints, the
player, or Android.  Coverage comes from the same two implementations the trainer
and the auditor use, so the numbers cannot drift from the shipped code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.bar_supervision import build_bar_spans, span_coverage, supervised_span_eligibility, validate_horizons
from training.bar_windows import DEFAULT_MAX_WINDOW_CHORDS, bar_boundary_indices, bar_window_coverage, plan_bar_windows
from training.tempo_reference import load_timing_reference
from training.train_expression import _sha256

REFERENCE_ENV = "AMP_TIMING_REFERENCE"
OUTPUT_ENV = "AMP_BAR_COVERAGE_OUTPUT"
HORIZONS = (1, 4, 16)
CONTEXT_SIZE = 16
FIXED_CORE_SIZE = 128


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-reference", type=Path,
                        default=Path(os.environ[REFERENCE_ENV]) if os.environ.get(REFERENCE_ENV) else None,
                        required=not bool(os.environ.get(REFERENCE_ENV)),
                        help="sealed tempo_reference directory, e.g. the balanced 250/50 development reference")
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ[OUTPUT_ENV]) if os.environ.get(OUTPUT_ENV) else None,
                        required=not bool(os.environ.get(OUTPUT_ENV)),
                        help="new directory for the diagnostic artifact; never overwrites")
    arguments = parser.parse_args(argv)
    reference_root, output = arguments.timing_reference, arguments.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    references = load_timing_reference(reference_root)
    summary = json.loads((reference_root / "summary.json").read_text(encoding="utf-8"))
    split_of = {str(row["id"]): str(row["split"]) for row in summary["samples"]}
    source_hashes = {str(row["id"]): str(row["sha256"]) for row in summary["samples"]}
    rows = []
    for sample_id, reference in sorted(references.items()):
        native = reference["chord_native_masks"][:, 1]
        trusted = reference["chord_trusted"]
        spans = build_bar_spans(reference, HORIZONS)
        boundaries = bar_boundary_indices(reference)
        windows = plan_bar_windows(spans.chord_count, boundaries, spans, CONTEXT_SIZE, DEFAULT_MAX_WINDOW_CHORDS)
        eligible = supervised_span_eligibility(spans, native, trusted)["eligible"]
        contained = np.searchsorted([w.core_start for w in windows], spans.starts, side="right") - 1
        contained = contained == np.searchsorted([w.core_start for w in windows], spans.ends, side="right") - 1
        fixed = span_coverage(spans, native, trusted, core_size=FIXED_CORE_SIZE)["horizons"]
        coverage = bar_window_coverage(windows, spans, boundaries)["horizons"]
        per_horizon = {}
        for horizon in validate_horizons(HORIZONS):
            key = str(horizon)
            selected = spans.horizons == horizon
            per_horizon[key] = {
                "geometry_spans": int(selected.sum()),
                "eligible_spans": int((selected & eligible).sum()),
                "fixed_core_eligible_same_core": int(fixed[key]["same_core_window_count"]),
                "bar_core_retained": int(coverage[key]["retained_window_count"]),
                "bar_core_retained_eligible": int((selected & eligible & contained).sum()),
                "bar_crosses_bar_aligned_cut": int(coverage[key]["crosses_bar_aligned_cut_window_count"]),
                "bar_capacity_forced_split": int(coverage[key]["capacity_forced_split_window_count"]),
            }
        rows.append({"id": sample_id, "split": split_of[sample_id], "chord_count": spans.chord_count,
                     "core_count": len(windows), "reference_sha256": source_hashes.get(sample_id),
                     "horizons": per_horizon})
    aggregate = {}
    for split in sorted({row["split"] for row in rows}):
        selected_rows = [row for row in rows if row["split"] == split]
        aggregate[split] = {
            "sample_count": len(selected_rows), "chord_count": sum(row["chord_count"] for row in selected_rows),
            "core_count": sum(row["core_count"] for row in selected_rows),
            "horizons": {key: {field: sum(row["horizons"][key][field] for row in selected_rows)
                               for field in selected_rows[0]["horizons"][key]}
                         for key in (str(horizon) for horizon in HORIZONS)},
        }
    report = {"kind": "bar_window_coverage_comparison", "status": "complete", "schema_version": 1,
              "timing_reference_root": str(reference_root),
              "timing_reference_summary_sha256": _sha256(reference_root / "summary.json"),
              "context_size": CONTEXT_SIZE, "max_window_chords": DEFAULT_MAX_WINDOW_CHORDS,
              "fixed_core_size_baseline": FIXED_CORE_SIZE, "horizons": list(HORIZONS),
              "definition": {"eligible_spans": "complete exact-boundary spans with every interval native-valid",
                             "fixed_core_eligible_same_core": "eligible spans whose endpoints share one 128-chord core",
                             "bar_core_retained_eligible": "eligible spans whose endpoints share one bar-aligned core"},
              "training_started": False, "model_input": False, "deployment_evidence": False,
              "adoption_gate_changed": False, "aggregate": aggregate, "samples": rows,
              "implementation_sha256": {name: _sha256(PROJECT_ROOT / "training" / name)
                                        for name in ("bar_windows.py", "bar_objective.py", "bar_supervision.py",
                                                     "sequence_expression.py", "train_sequence_expression.py")}}
    output.mkdir(parents=True)
    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    (output / "summary.json").write_bytes(payload + b"\n")
    digest = hashlib.sha256(payload).hexdigest()
    (output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "summary_sha256": digest,
                      "validation_h16": aggregate["validation"]["horizons"]["16"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
