"""Is the per-performance label offset a property of the score, or of the performer?

`tools/audit_target_channels.py` showed that removing each performance's own median takes
19.2% off the `log_key_duration_ratio` error, so a large piece of that channel is a constant
shift per recording.  Centering the training target only pays if that shift is a property of
the *score* -- that is, if two different people playing the same piece produce the same
offset.  This probe measures exactly that, per channel, with a one-way intraclass
correlation ICC(1) over one offset per (piece, performer):

    ICC = (MSB - MSW) / (MSB + (k_mean - 1) * MSW)

ICC near 1 means the offset is a stable score property a score-only model can learn; ICC
near 0 means it is performer idiosyncrasy, and centering the target would only move the
error around rather than remove it.  It reads labels only, trains nothing, and never touches
the locked test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from tools.audit_cross_performance_reproducibility import PairBuffer, score_groups, select_performances
from tools.audit_target_channels import CHANNELS
from training.sequence_expression import load_selected_samples
from training.train_expression import TrainingRunError, _sha256, _write_json

OFFSET_CHANNELS = tuple((name, kind, column) for kind, name, column in CHANNELS
                        if name in ("onset_delta_ms", "log_ioi_ratio", "log_key_duration_ratio"))


def performance_offsets(sample) -> dict[str, tuple[float, int] | None]:
    """Median supervised label value per channel, or None when nothing is supervised."""
    offsets: dict[str, tuple[float, int] | None] = {}
    for name, kind, column in OFFSET_CHANNELS:
        targets = sample.chord_targets if kind == "chord" else sample.note_targets
        masks = sample.chord_masks if kind == "chord" else sample.note_masks
        values = np.asarray(targets, dtype=np.float64)[:, column][np.asarray(masks, dtype=bool)[:, column]]
        finite = values[np.isfinite(values)]
        offsets[name] = (float(np.median(finite)), int(finite.size)) if finite.size else None
    return offsets


def icc1(groups: list[np.ndarray]) -> dict[str, object]:
    """One-way intraclass correlation for unbalanced groups of offsets.

    Ragged on purpose: pieces hold different numbers of usable performances, so nothing is
    stacked into a rectangular matrix -- the sums are accumulated group by group.
    """
    kept = [np.asarray(group, dtype=np.float64) for group in groups if len(group) >= 2]
    if len(kept) < 2:
        return {"note": "fewer than two pieces with two or more offsets"}
    sizes = np.asarray([len(group) for group in kept], dtype=np.float64)
    means = np.asarray([group.mean() for group in kept], dtype=np.float64)
    count = float(sizes.sum())
    grand_mean = float((sizes * means).sum()) / count
    between_sum = float((sizes * (means - grand_mean) ** 2).sum())
    within_sum = float(sum(((group - mean) ** 2).sum() for group, mean in zip(kept, means)))
    pieces = len(kept)
    msb = between_sum / (pieces - 1)
    msw = within_sum / (count - pieces)
    mean_size = count / pieces
    result = {"pieces": pieces, "offsets": int(count), "mean_offsets_per_piece": mean_size,
              "between_piece_variance": msb, "within_piece_variance": msw,
              # total sum of squares splits exactly into the between and within parts
              "offset_sd_overall": float(np.sqrt((between_sum + within_sum) / (count - 1.0)))}
    denominator = msb + (mean_size - 1.0) * msw
    result["icc1"] = None if denominator <= 0 else float(np.clip((msb - msw) / denominator, -1.0, 1.0))
    return result


def summarize_offset_pairs(buffer: PairBuffer) -> dict[str, object]:
    """Within-piece agreement between two different performers' offsets."""
    pair = buffer.concatenated()
    if pair is None or buffer.size < 2:
        return {"offset_pairs": buffer.size}
    first = np.asarray(pair[0], dtype=np.float64)
    second = np.asarray(pair[1], dtype=np.float64)
    difference = np.abs(first - second)
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = float(np.corrcoef(first, second)[0, 1])
    return {"offset_pairs": int(buffer.size),
            "median_abs_offset_pair_difference": float(np.median(difference)),
            "p90_abs_offset_pair_difference": float(np.percentile(difference, 90)),
            "offset_pearson_across_performers": pearson if np.isfinite(pearson) else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-groups", type=int, default=0, help="0 uses every eligible group")
    parser.add_argument("--max-performances-per-group", type=int, default=8)
    parser.add_argument("--groups-per-load", type=int, default=16,
                        help="performances from this many groups are loaded per shard pass")
    parser.add_argument("--minimum-offset-rows", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.groups_per_load < 1 or arguments.max_performances_per_group < 2:
        raise SystemExit("groups per load must be >= 1 and performances per group must be >= 2")
    dataset_root = arguments.dataset.resolve()
    groups = score_groups(dataset_root / "manifest.jsonl", arguments.split)
    if not groups:
        raise TrainingRunError(f"no {arguments.split} piece groups with two or more equal-length performances")
    candidates = []
    single_performer = 0
    for group, rows in groups:
        ids = select_performances(rows, arguments.max_performances_per_group)
        if len(ids) < 2:
            single_performer += 1
            continue
        candidates.append((group, ids))
    if arguments.max_groups > 0:
        candidates = candidates[:arguments.max_groups]

    offsets: dict[str, dict[str, list[float]]] = {name: {} for name, _kind, _column in OFFSET_CHANNELS}
    loaded_groups = 0
    for start in range(0, len(candidates), arguments.groups_per_load):
        batch = candidates[start:start + arguments.groups_per_load]
        wanted = {sample_id: group for group, ids in batch for sample_id in ids}
        samples = load_selected_samples(dataset_root, arguments.split, frozenset(wanted))
        for sample in samples:
            for name, value in performance_offsets(sample).items():
                if value is None or value[1] < arguments.minimum_offset_rows:
                    continue
                offsets[name].setdefault(wanted[sample.sample_id], []).append(value[0])
        loaded_groups += len(batch)
        del samples
    if not any(offsets[name] for name, _kind, _column in OFFSET_CHANNELS):
        raise TrainingRunError("no performance offsets were measured in this selection")

    channels = {}
    for name, _kind, _column in OFFSET_CHANNELS:
        per_piece = [np.asarray(values, dtype=np.float64) for _group, values in sorted(offsets[name].items())]
        report = icc1(per_piece)
        pairs = PairBuffer()
        for values in per_piece:
            for first_index in range(len(values)):
                for second_index in range(first_index + 1, len(values)):
                    pairs.add(np.asarray([values[first_index]]), np.asarray([values[second_index]]))
        report.update(summarize_offset_pairs(pairs))
        report["pieces_measured"] = len(per_piece)
        channels[name] = report

    summary = {
        "kind": "piece_level_label_offset_reproducibility", "status": "complete",
        "dataset": str(dataset_root), "split": arguments.split,
        "eligible_groups": len(groups), "candidate_groups": len(candidates), "groups_loaded": loaded_groups,
        "groups_skipped_for_one_distinct_performer": single_performer,
        "offset_definition": "median of the supervised target rows of one channel inside one performance",
        "selection_rule": ("at most max_performances_per_group performances per piece, at most one per distinct "
                           "performer, so every within-piece comparison is between two different people"),
        "minimum_offset_rows": arguments.minimum_offset_rows,
        "reading": {"icc1": "near 1 means the offset is a stable score property a score-only model can predict; "
                            "near 0 means it is performer idiosyncrasy, so centering the target cannot remove it",
                    "model_quality_claim": False, "adoption_evidence": False},
        "channels": channels, "trained": False, "model_inputs_modified": False, "adoption_evidence": False,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", summary)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "channels": {name: {key: channels[name].get(key) for key in
                                         ("pieces_measured", "icc1", "offset_pairs",
                                          "median_abs_offset_pair_difference",
                                          "offset_pearson_across_performers",
                                          "between_piece_variance", "within_piece_variance")}
                                   for name in channels}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
