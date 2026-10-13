"""Read-only ceiling probe: how much of each label repeats across performances.

PianoCoRe holds many performances of the same score -- a single ``piece_group`` can carry
over 1,700 of them -- so for one score position we can compare what two different
performances say the label is.  A component that does not repeat across performances is
performer idiosyncrasy or alignment noise: no score-only model can predict it, and more
data or bigger weights will not change that.

Two switches decide what is being measured, and they answer different questions:

``--granularity chord|note``  which rows are paired.  Chord rows carry the onset and IOI
    targets; note rows carry duration, micro offset and velocity.  The model is supervised
    per note for duration, so only ``note`` gives a floor comparable with its error -- the
    chord mean averages independent per-note noise away and so understates it.  All 932
    eligible train groups have identical note counts, so note rows align by index.
``--centre-per-performance``  subtract each performance's own median before pairing, which
    removes the recording's level and leaves its profile.  Read it together with
    ``tools/audit_piece_offset_reproducibility.py``, which measures the level that was
    subtracted.

Memory is the reason this file is written the way it is.  Comparing every performance of
every group is O(n^2) per group and, multiplied by the positions of each pair, is what
exhausted the commit limit on 2026-10-11.  Here each group contributes at most
``--max-performances-per-group`` performances, at most one per distinct ``performer`` value,
and matched positions accumulate in float32 numpy blocks rather than Python tuples.  The
implied pair volume is printed before anything is loaded and refused above
``--max-pair-positions``.  Nothing is retrained, rewritten, or read out of the locked test
split.
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

from training.piece_groups import score_groups, select_performances
from training.sequence_expression import load_selected_samples
from training.train_expression import TrainingRunError, _sha256, _write_json

SCORE_ONSET_COLUMN = 0
NOTE_ONSET_BEAT_COLUMN = 3
ONSET_CHANNEL = 0
IOI_CHANNEL = 1
DURATION_NOTE_COLUMN = 1
BEAT_TOLERANCE = 1e-6
# (channel, row source, column); "chord-mean" aggregates note rows onto their chord
CHORD_CHANNELS = (("onset_delta_ms", "chord", 0), ("log_ioi_ratio", "chord", 1),
                  ("log_key_duration_ratio", "chord-mean", DURATION_NOTE_COLUMN))
NOTE_CHANNELS = (("log_key_duration_ratio", "note", 1), ("micro_offset_ms", "note", 0),
                 ("velocity_robust_z", "note", 3))
CHANNELS = tuple(name for name, _source, _column in CHORD_CHANNELS)


# ``score_groups`` and ``select_performances`` moved to ``training.piece_groups`` when the
# shared-profile training target started consuming the same eligibility rules: one implementation,
# so the probe and the training target cannot drift apart while claiming to measure the same thing.


def chord_labels(sample, notes_to_chord: np.ndarray) -> dict[str, np.ndarray]:
    """Per-chord label values, NaN where the dataset does not supervise that channel."""
    valid_chord = np.asarray(sample.chord_masks[:, ONSET_CHANNEL], dtype=bool)
    valid_ioi = np.asarray(sample.chord_masks[:, IOI_CHANNEL], dtype=bool)
    duration_sum = np.zeros(len(sample.chord_features), dtype=np.float64)
    duration_count = np.zeros(len(sample.chord_features), dtype=np.float64)
    note_valid = np.asarray(sample.note_masks[:, DURATION_NOTE_COLUMN], dtype=bool)
    mapped = np.asarray(notes_to_chord, dtype=np.int64)[note_valid]
    values = np.asarray(sample.note_targets[note_valid, DURATION_NOTE_COLUMN], dtype=np.float64)
    np.add.at(duration_sum, mapped, values)
    np.add.at(duration_count, mapped, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        duration_mean = np.where(duration_count > 0, duration_sum / np.maximum(duration_count, 1.0), np.nan)
    return {"onset_delta_ms": np.where(valid_chord,
                                       np.asarray(sample.chord_targets[:, ONSET_CHANNEL], dtype=np.float64), np.nan),
            "log_ioi_ratio": np.where(valid_ioi,
                                      np.asarray(sample.chord_targets[:, IOI_CHANNEL], dtype=np.float64), np.nan),
            "log_key_duration_ratio": duration_mean,
            "score_onset_beats": np.asarray(sample.chord_features[:, SCORE_ONSET_COLUMN], dtype=np.float64)}


def note_labels(sample) -> dict[str, np.ndarray]:
    """Per-note label values, NaN where the dataset does not supervise that note."""
    labels: dict[str, np.ndarray] = {}
    targets = np.asarray(sample.note_targets, dtype=np.float64)
    masks = np.asarray(sample.note_masks, dtype=bool)
    for name, _source, column in NOTE_CHANNELS:
        labels[name] = np.where(masks[:, column], targets[:, column], np.nan)
    labels["score_onset_beats"] = np.asarray(sample.note_features[:, NOTE_ONSET_BEAT_COLUMN], dtype=np.float64)
    return labels


def centre_labels(labels: dict[str, np.ndarray], channels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Remove this recording's own level from each channel, leaving only its profile.

    The median comes from that one performance's supervised rows, so NaN positions stay NaN
    and no cross-recording information is used.  Centred pairing answers a different
    question than raw pairing: how much of the *within-recording shape* repeats across
    performers, which is the part a model can still be blamed for missing.
    """
    centred = dict(labels)
    for channel in channels:
        values = np.asarray(labels[channel], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            centred[channel] = values - float(np.median(finite))
    return centred


class PairBuffer:
    """Float32 blocks of matched (first, second) label values, one pair per position."""

    def __init__(self) -> None:
        self.first: list[np.ndarray] = []
        self.second: list[np.ndarray] = []
        self.size = 0

    def add(self, left: np.ndarray, right: np.ndarray) -> None:
        if left.size == 0:
            return
        self.first.append(left.astype(np.float32))
        self.second.append(right.astype(np.float32))
        self.size += int(left.size)

    def concatenated(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.size == 0:
            return None
        return (np.concatenate(self.first), np.concatenate(self.second))


def summarize_channel(buffer: PairBuffer, minimum_positions: int) -> dict[str, object]:
    pair = buffer.concatenated()
    if pair is None or buffer.size < minimum_positions:
        return {"pair_positions": buffer.size, "note": f"fewer than {minimum_positions} matched positions"}
    first = np.asarray(pair[0], dtype=np.float64)
    second = np.asarray(pair[1], dtype=np.float64)
    difference = np.abs(first - second)
    rms = float(np.sqrt((difference ** 2).mean()))
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = float(np.corrcoef(first, second)[0, 1])
        rank_pearson = float(np.corrcoef(np.argsort(np.argsort(first)), np.argsort(np.argsort(second)))[0, 1])
    reliability = 2.0 * pearson / (1.0 + pearson) if np.isfinite(pearson) and pearson > 0 else None
    return {
        "pair_positions": int(buffer.size),
        "median_abs_cross_performance_difference": float(np.median(difference)),
        "p90_abs_cross_performance_difference": float(np.percentile(difference, 90)),
        "cross_performance_rms": rms,
        "pearson_correlation": pearson if np.isfinite(pearson) else None,
        "pearson_on_ranks": rank_pearson if np.isfinite(rank_pearson) else None,
        # Two independent measurements of one underlying value differ by sqrt(2) times the
        # per-measurement noise, so the scatter a score-only model cannot reach is ~rms/sqrt(2).
        "implied_per_performance_rms": rms / float(np.sqrt(2.0)),
        "split_half_reliability_estimate": float(reliability) if reliability is not None else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train", help="which split's performances to compare")
    parser.add_argument("--granularity", choices=("chord", "note"), default="chord",
                        help="chord pairs onset/IOI and the chord-mean duration; note pairs per-note duration")
    parser.add_argument("--max-groups", type=int, default=0, help="0 uses every eligible group")
    parser.add_argument("--max-performances-per-group", type=int, default=4)
    parser.add_argument("--max-pair-positions", type=int, default=40_000_000,
                        help="abort before loading if the estimated matched-position count exceeds this")
    parser.add_argument("--minimum-pair-positions", type=int, default=1000)
    parser.add_argument("--centre-per-performance", action="store_true",
                        help="compare profiles, not levels: subtract each performance's own median first")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.max_groups < 0 or arguments.max_performances_per_group < 2:
        raise SystemExit("max groups must be >= 0 and performances per group must be >= 2")
    spec = NOTE_CHANNELS if arguments.granularity == "note" else CHORD_CHANNELS
    names = tuple(name for name, _source, _column in spec)
    dataset_root = arguments.dataset.resolve()
    groups = score_groups(dataset_root / "manifest.jsonl", arguments.split)
    if not groups:
        raise TrainingRunError(f"no {arguments.split} piece groups with two or more equal-length performances")
    selected_groups = []
    single_performer_groups = 0
    for group, rows in groups:
        ids = select_performances(rows, arguments.max_performances_per_group)
        if len(ids) < 2:
            single_performer_groups += 1
            continue
        selected_groups.append((group, ids, int(rows[0]["chord_count"])))
    if arguments.max_groups > 0:
        selected_groups = selected_groups[:arguments.max_groups]
    estimated_positions = sum((len(ids) * (len(ids) - 1) // 2) * chords for _group, ids, chords in selected_groups)
    if estimated_positions > arguments.max_pair_positions:
        raise SystemExit(f"estimated {estimated_positions:,} matched positions exceeds "
                         f"--max-pair-positions={arguments.max_pair_positions:,}; lower --max-groups "
                         "or --max-performances-per-group")
    print(json.dumps({"eligible_groups": len(groups), "groups_to_compare": len(selected_groups),
                      "granularity": arguments.granularity,
                      "estimated_pair_positions": estimated_positions}, indent=2), flush=True)

    buffers = {name: PairBuffer() for name in names}
    used_groups = 0
    compared_pairs = 0
    skipped_length_mismatch = 0
    for group, ids, _chords in selected_groups:
        samples = {sample.sample_id: sample for sample in
                   load_selected_samples(dataset_root, arguments.split, frozenset(ids))}
        if len(samples) < 2:
            continue
        if arguments.granularity == "note":
            labels = {sample_id: note_labels(sample) for sample_id, sample in samples.items()}
        else:
            labels = {sample_id: chord_labels(sample, np.asarray(sample.note_to_chord, dtype=np.int64))
                      for sample_id, sample in samples.items()}
        if arguments.centre_per_performance:
            labels = {sample_id: centre_labels(label, names) for sample_id, label in labels.items()}
        if len({int(label["score_onset_beats"].size) for label in labels.values()}) != 1:
            skipped_length_mismatch += 1
            continue
        ordered = sorted(labels)
        for first_index in range(len(ordered)):
            for second_id in ordered[first_index + 1:]:
                first, second = labels[ordered[first_index]], labels[second_id]
                aligned = (np.abs(first["score_onset_beats"] - second["score_onset_beats"]) <= BEAT_TOLERANCE)
                for name in names:
                    left, right = first[name], second[name]
                    usable = aligned & np.isfinite(left) & np.isfinite(right)
                    buffers[name].add(left[usable], right[usable])
                compared_pairs += 1
        used_groups += 1
        samples.clear()
        labels.clear()

    report = {
        "kind": "cross_performance_label_reproducibility", "status": "complete",
        "dataset": str(dataset_root), "split": arguments.split,
        "granularity": arguments.granularity, "channels_compared": list(names),
        "eligible_groups": len(groups), "candidate_groups": len(selected_groups),
        "groups_used": used_groups, "pairs_compared": compared_pairs,
        "groups_skipped_for_length_mismatch": skipped_length_mismatch,
        "groups_skipped_for_one_distinct_performer": single_performer_groups,
        "caps": {"max_performances_per_group": arguments.max_performances_per_group,
                 "max_groups": arguments.max_groups, "max_pair_positions": arguments.max_pair_positions,
                 "estimated_pair_positions": estimated_positions},
        "label_transform": ("each performance's channel median removed before pairing, so every number below "
                            "describes the within-performance profile, not its level"
                            if arguments.centre_per_performance else "raw labels, level included"),
        "matching_rule": ("equal chord counts plus position-wise equal score onset beats at the chosen "
                          "granularity; unmatched or unmasked positions are dropped, never interpolated"),
        "pairing_rule": "at most one performance per distinct performer value and at most "
                        "max_performances_per_group performances per score, so every compared pair is two "
                        "different performers; scores whose recordings all come from one performer value "
                        "cannot say anything about reproducibility and are dropped",
        "channels": {name: summarize_channel(buffers[name], arguments.minimum_pair_positions) for name in names},
        "reading": {
            "ceiling": "a score-only model cannot predict the part of a label that does not repeat across "
                       "performances of the same score; compare implied_per_performance_rms with the model's "
                       "validation error for the same channel at the same granularity",
            "assumption": "treats the two performances as the same underlying expression plus independent "
                          "noise; real performers share habits, so this is an optimistic view of how much of "
                          "the residual is truly unpredictable",
        },
        "trained": False, "model_inputs_modified": False, "adoption_evidence": False,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "groups_used": used_groups, "pairs_compared": compared_pairs,
                      "channels": report["channels"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
