"""Which score quantities actually explain the duration shape a piece shares across performers?

The previous rounds established, in this order: the duration channel is the only one with
headroom above its noise floor; removing the per-recording level from the target does not
recover it; and 14x more capacity does not either.  The remaining question is *what variable
describes the shared shape* -- and the cheap, honest way to answer it is to fit the
piece-mean shape profile directly instead of training a GRU and hoping.

Method
------
For every eligible piece (>= 2 accepted performances, identical note counts, one recording per
distinct performer) each recording's duration label is centred on its own median, then
averaged across recordings per note position.  That average is ``S``: the part of the
duration profile that different people agree on, i.e. what a score-only model can chase.
The explanatory variables are the shipped shape block itself (``training/shape_features.py``,
computed from pitches, onsets, score durations and chord membership -- never from a label),
so this tool measures exactly what a run would train on rather than a wider candidate list
whose losers would have to be re-justified later.  Two quantities were screened out before
that block was fixed and are documented in the library instead: ``rest_after`` (structurally
zero, because this corpus writes every note at least up to the next onset) and the gap to the
next note onset (|r| ~ 0.005 with ``S``).

Two reports per variable set, both with grouped cross-validation (whole pieces held out,
which is the deployment situation: the game's scores were never in training):

* ``pooled``            -- explains ``S`` including differences between pieces;
* ``within_piece``      -- ``S`` and the variables demeaned per piece, i.e. the note-level
                        shape inside a score, which is the part rules cannot fake.

A variable set earns a retrain only if it raises the within-piece out-of-fold R-squared.
Samples are loaded with ``relative_position=False`` on purpose: deriving geometry while the
positions have been rescaled to 0-1 and the durations are still in beats mixes units and
silently changes every overlap threshold in the block.  Nothing trains, rewrites, or reads
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

from tools.audit_cross_performance_reproducibility import score_groups, select_performances
from training.midi_features import NOTE_FEATURE_NAMES
from training.sequence_expression import load_selected_samples
from training.shape_features import SHAPE_FEATURE_NAMES, derive_shape_features
from training.train_expression import TrainingRunError, _sha256, _write_json

DURATION_COLUMN = 1
EXISTING_BASELINE = "existing_features"
# the shipped block itself: what this tool measures is exactly what a run would train on
CANDIDATE_NAMES = SHAPE_FEATURE_NAMES


def grouped_folds(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out whole groups (pieces), never individual rows."""
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f"need at least {folds} groups to build {folds} folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    blocks = np.array_split(unique, folds)
    splits = []
    for block in blocks:
        test = np.isin(groups, block)
        splits.append((~test, test))
    return splits


def ridge_out_of_fold_r2(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray, *,
                         alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                         folds: int = 5, seed: int = 20261003) -> dict[str, object]:
    """Ridge regression out-of-fold R^2, standardised on the training part of every split.

    Each split fits the penalty on its own training rows and predicts the held-out rows, so
    the reported R^2 is a generalisation estimate for whole unseen pieces, not a fit quality.
    """
    if len(target) != matrix.shape[0] or len(groups) != len(target):
        raise ValueError("matrix, target and groups must share their leading dimension")
    splits = grouped_folds(groups, folds, seed)
    best: dict[str, object] | None = None
    for alpha in alphas:
        predictions = np.zeros_like(target)
        for train_mask, test_mask in splits:
            centre = matrix[train_mask].mean(axis=0)
            scale = matrix[train_mask].std(axis=0)
            scale[scale < 1e-9] = 1.0
            trained = (matrix[train_mask] - centre) / scale
            held = (matrix[test_mask] - centre) / scale
            target_centre = float(target[train_mask].mean())
            design = np.column_stack((trained, np.ones(len(trained))))
            penalty = np.eye(design.shape[1]) * alpha
            penalty[-1, -1] = 0.0
            weights = np.linalg.solve(design.T @ design + penalty, design.T @ (target[train_mask] - target_centre))
            predictions[test_mask] = np.column_stack((held, np.ones(len(held)))) @ weights + target_centre
        variance = float(((target - target.mean()) ** 2).sum())
        r2 = 1.0 - float(((target - predictions) ** 2).sum()) / variance if variance > 0 else None
        residual_mae = float(np.abs(target - predictions).mean())
        if best is None or (r2 is not None and (best["r2"] is None or r2 > best["r2"])):
            best = {"alpha": alpha, "r2": None if r2 is None else float(r2), "residual_mae": residual_mae}
    return {"folds": folds, "seed": seed, "rows": int(len(target)), **best}


def _shape_block(sample) -> np.ndarray:
    """The shipped shape block for one recording, via the same code training uses.

    Loaded with ``relative_position=False`` so positions and score durations stay in beats:
    deriving candidates from rescaled positions while durations stay in beats would mix units
    and quietly change every threshold in the geometry (it did exactly that in an earlier
    revision of this file, which is why its numbers were recomputed).
    """
    return derive_shape_features(np.asarray(sample.note_features, dtype=np.float64),
                                 np.asarray(sample.chord_features[:, 0], dtype=np.float64),
                                 np.asarray(sample.note_to_chord, dtype=np.int64), CANDIDATE_NAMES)


def collect_profiles(dataset_root: Path, split: str, candidates: list[tuple[str, list[str]]]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build per-position arrays: shared shape profile S, existing features, candidates, group ids."""
    existing_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    profile_rows: list[np.ndarray] = []
    used = 0
    skipped = 0
    for group, ids in candidates:
        samples = {sample.sample_id: sample for sample in
                   load_selected_samples(dataset_root, split, frozenset(ids), relative_position=False)}
        if len(samples) < 2:
            skipped += 1
            continue
        centred: list[np.ndarray] = []
        base: np.ndarray | None = None
        extra: np.ndarray | None = None
        lengths: list[int] = []
        for sample_id in sorted(samples):
            sample = samples[sample_id]
            targets = np.asarray(sample.note_targets, dtype=np.float64)
            masks = np.asarray(sample.note_masks, dtype=bool)
            values = np.where(masks[:, DURATION_COLUMN], targets[:, DURATION_COLUMN], np.nan)
            finite = values[np.isfinite(values)]
            if not finite.size:
                continue
            centred.append(values - float(np.median(finite)))
            lengths.append(len(values))
            if base is None:
                base = np.asarray(sample.note_features, dtype=np.float64)
                extra = _shape_block(sample)
            else:
                # candidates are taken from one recording per piece, which is only legitimate if
                # the score geometry is shared; compare it rather than assuming it
                other = np.asarray(sample.note_features, dtype=np.float64)
                if other.shape != base.shape or not np.array_equal(other[:, [0, 3, 4]], base[:, [0, 3, 4]]):
                    continue
                if not np.allclose(_shape_block(sample), extra, atol=1e-4):
                    continue
        if base is None or extra is None or len(centred) < 2 or len(set(lengths)) != 1:
            skipped += 1
            continue
        stack = np.vstack([np.nan_to_num(block, nan=0.0) for block in centred])
        supervised = np.vstack([np.isfinite(block) for block in centred])
        profile = np.where(supervised.any(axis=0),
                           np.where(supervised, stack, 0.0).sum(axis=0) / np.maximum(supervised.sum(axis=0), 1),
                           np.nan)
        keep = np.isfinite(profile)
        existing_rows.append(base[keep])
        candidate_rows.append(extra[keep])
        profile_rows.append(profile[keep])
        used += 1
        del samples, centred
    if not used:
        raise TrainingRunError("no piece profiles could be built")
    return (np.concatenate(profile_rows), np.vstack(existing_rows), np.vstack(candidate_rows),
            np.concatenate([np.full(len(row), index) for index, row in enumerate(profile_rows)]), used)


def demean_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    centred = values.astype(np.float64, copy=True)
    for group in np.unique(groups):
        rows = groups == group
        centred[rows] -= centred[rows].mean(axis=0)
    return centred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-groups", type=int, default=250)
    parser.add_argument("--max-performances-per-group", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.max_groups < arguments.folds:
        raise SystemExit("max groups must be at least the fold count")
    dataset_root = arguments.dataset.resolve()
    groups = score_groups(dataset_root / "manifest.jsonl", arguments.split)
    candidates = []
    for group, rows in groups:
        ids = select_performances(rows, arguments.max_performances_per_group)
        if len(ids) >= 2:
            candidates.append((group, ids))
    candidates = candidates[:arguments.max_groups]
    profile, existing, extra, group_ids, used = collect_profiles(dataset_root, arguments.split, candidates)

    sets = {EXISTING_BASELINE: existing, "candidates_only": extra, "existing_plus_candidates":
            np.column_stack((existing, extra))}
    reports: dict[str, object] = {}
    for name, matrix in sets.items():
        pooled = ridge_out_of_fold_r2(matrix, profile, group_ids, folds=arguments.folds)
        within = ridge_out_of_fold_r2(demean_within_groups(matrix, group_ids),
                                      demean_within_groups(profile.reshape(-1, 1), group_ids).ravel(),
                                      group_ids, folds=arguments.folds)
        reports[name] = {"pooled": pooled, "within_piece": within}
    correlations = {}
    within_profile = demean_within_groups(profile.reshape(-1, 1), group_ids).ravel()
    for index, name in enumerate(CANDIDATE_NAMES):
        column = extra[:, index]
        within_column = demean_within_groups(column.reshape(-1, 1), group_ids).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled_r = float(np.corrcoef(column, profile)[0, 1])
            within_r = float(np.corrcoef(within_column, within_profile)[0, 1])
        # a constant column has no correlation to report; say so instead of leaving a bare null,
        # because "this score style has no inter-note rests at all" is itself a finding
        degenerate = float(column.std()) <= 1e-12
        correlations[name] = {"pooled_pearson": None if not np.isfinite(pooled_r) else pooled_r,
                              "within_piece_pearson": None if not np.isfinite(within_r) else within_r,
                              "column_sd": float(column.std()),
                              "degenerate_no_variance": degenerate}

    summary = {
        "kind": "duration_shape_explanatory_power", "status": "complete", "trained": False,
        "dataset": str(dataset_root), "split": arguments.split,
        "groups_requested": len(candidates), "groups_used": used, "positions": int(len(profile)),
        "target": "per-note-position mean of each recording's median-centred log_key_duration_ratio, "
                  "averaged across performers of the same score -- the shared shape a score-only "
                  "model could chase",
        "shipped_block_variables": list(CANDIDATE_NAMES),
        "candidate_derivation": "pitches, onsets and score durations recovered from note_features columns "
                                "0/3/4 plus note_to_chord; no label column is read, so nothing here can "
                                "leak the target",
        "variables_per_set": {EXISTING_BASELINE: list(NOTE_FEATURE_NAMES),
                              "candidates_only": list(CANDIDATE_NAMES),
                              "existing_plus_candidates": list(NOTE_FEATURE_NAMES) + list(CANDIDATE_NAMES)},
        "ridge_grouped_cv": reports,
        "candidate_correlations": correlations,
        "reading": {"decision_rule": "candidates earn a retrain only if existing_plus_candidates raises the "
                                     "within_piece r2 over existing_features by more than a rounding margin; "
                                     "pooled r2 alone can be bought with piece identity and does not help the "
                                     "player",
                    "model_quality_claim": False, "adoption_evidence": False},
        "model_inputs_modified": False, "adoption_evidence": False,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", summary)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "groups_used": used, "positions": int(len(profile)),
                      "ridge_grouped_cv": reports,
                      "top_within_piece_candidates": sorted(
                          ((name, value["within_piece_pearson"]) for name, value in correlations.items()),
                          key=lambda item: -abs(item[1] if item[1] is not None else 0.0))[:6]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
