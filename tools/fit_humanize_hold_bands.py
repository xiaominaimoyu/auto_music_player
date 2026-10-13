"""Re-estimate the humanizer's hold bands from the corpus, then measure that rule as a candidate.

Why this is the cheapest adoption experiment on the table.  The piece-level bootstrap of
2026-10-11 found that the shipping humanizer is *worse than playing the score as written* on
piano material: its hold bands sit at 0.70-0.85 of the element span while the real/written
median of this corpus is 0.922.  That says the incumbent rule is badly calibrated, not that
expression enhancement is impossible -- and re-calibrating three constants needs no training,
no runtime, no packaging change, and no new input columns.

Method
------
The rule applies ``played = hold x element span`` while the label says
``played = exp(label) x written``, so the hold a recording asks for is exactly
``exp(label) x written / span``.  For every supervised note in the **fit split** the tool
computes that quantity, groups it by the band the humanizer would actually pick
(``training/rules_baseline.tier_of``), and takes the median per band.  Medians, not means: the
labels carry alignment noise and a handful of bad fits must not drag a constant that ships to
every player.  Each band is then widened symmetrically around its median, kept inside the
module's own validator (``0 < lo <= hi <= 1``), and fed back through the *same* projection and
the *same* piece-level paired bootstrap that scored the model -- so the candidate is measured
rather than described, and "model" in the output table means the candidate rule.

Discipline: the fit split is the training sample list, the evaluation split is the development
validation list, and the locked test split is refused.  Nothing here edits
``playback_adoption.POLICY``: proposing that the gate be pinned to ``identity`` instead of to
whichever of the three required baselines happens to be weakest is the project owner's call,
and this tool's output is evidence for that decision, not the decision.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.humanize import HumanizeParams
from tools.bootstrap_sequence_vs_baselines import (CHANNELS, piece_identities, rule_projection_errors,
                                                   summarise_channel)
from training.midi_features import NOTE_TARGET_NAMES
from training.playback_adoption import POLICY
from training.rules_baseline import element_gaps, tier_of
from training.sequence_expression import load_selected_samples
from training.tempo_reference import load_timing_reference
from training.train_expression import TrainingRunError, _load_sample_id_file, _sha256, _write_json

BAND_NAMES = ("short_hold", "mid_hold", "long_hold")
DURATION_CHANNEL = "log_key_duration_ratio"
DURATION_COLUMN = NOTE_TARGET_NAMES.index("log_key_duration_ratio")
CANDIDATE_NAME = "candidate"


def hold_demand(reference_row: dict, labels: np.ndarray,
                supervised: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The hold ratio each supervised note is asking for, and the band it belongs to.

    ``exp(label) x written / span``: seconds on both sides, so the reference tempo cancels and
    nothing here depends on how fast the recording actually was.  ``labels`` is the whole note
    column and ``supervised`` the matching mask -- the shard carries notes with no trusted
    label, and those are dropped here rather than by pre-slicing, so every note array can still
    be checked against the others before anything is thrown away.
    """
    mapping = np.asarray(reference_row["note_to_chord"], dtype=np.int64)
    written = np.asarray(reference_row["note_score_duration_seconds"], dtype=np.float64)
    gaps = element_gaps(reference_row["chord_score_beats"], reference_row["note_score_duration_beats"], mapping)
    tempo = float(np.asarray(reference_row["score_seconds_per_beat_reference"]).item())
    if tempo <= 0 or not np.isfinite(tempo):
        raise TrainingRunError("reference tempo must be finite and positive")
    span = gaps[mapping] * tempo
    labels = np.asarray(labels, dtype=np.float64)
    if span.shape != written.shape or labels.shape != written.shape:
        raise TrainingRunError("note arrays must all share one length")
    keep = np.ones(len(written), dtype=bool) if supervised is None else np.asarray(supervised, dtype=bool)
    if keep.shape != written.shape:
        raise TrainingRunError("the supervised mask must match the note arrays")
    if np.any(keep & ((span <= 0) | (written <= 0))):
        raise TrainingRunError("supervised notes must have positive written duration and span")
    bands = np.asarray([tier_of(value) for value in gaps], dtype=object)[mapping]
    with np.errstate(divide="ignore", invalid="ignore"):
        demand = np.exp(labels) * written / span
    usable = keep & np.isfinite(demand) & (demand > 0)
    return demand[usable], bands[usable]


def fit_hold_bands(demands: list[tuple[np.ndarray, np.ndarray]], *, band_width: float = 0.06,
                   shipping: HumanizeParams | None = None) -> dict[str, dict[str, object]]:
    """Median demand per band, clipped into the range the humanizer allows.

    A band with no observations keeps the shipping value and says so instead of inventing a
    number, and clipping at 1.0 is deliberate: the player cannot make a note longer than the
    span it occupies without colliding with the next element.
    """
    if not 0 < band_width < 0.5:
        raise ValueError("band_width must sit inside (0, 0.5)")
    active = shipping or HumanizeParams()
    fitted: dict[str, dict[str, object]] = {}
    for name in BAND_NAMES:
        parts = [demand[bands == name] for demand, bands in demands if len(demand)]
        values = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        if not values.size:
            fitted[name] = {"observations": 0, "median_hold": None, "clipped_hold": None,
                            "kept_shipping_value": True, "shipping_band": list(getattr(active, name))}
            continue
        median = float(np.median(values))
        fitted[name] = {"observations": int(values.size), "median_hold": median,
                        "clipped_hold": float(min(max(median, 1e-3), 1.0)),
                        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
                        "kept_shipping_value": False}
    return fitted


def candidate_params(fitted: dict[str, dict[str, object]], *, base: HumanizeParams | None = None,
                     band_width: float = 0.06) -> HumanizeParams:
    """A parameter set whose bands are centred on the fitted medians.

    Bands keep a spread instead of collapsing to a constant, because the note-to-note variation
    is part of what makes the result read as human; the width is symmetric so the band mean
    equals the fitted median wherever no clipping was needed.
    """
    active = base or HumanizeParams()
    fields = {field.name: getattr(active, field.name) for field in dataclasses.fields(HumanizeParams)}
    for name in BAND_NAMES:
        entry = fitted.get(name) or {}
        centre = entry.get("clipped_hold")
        if centre is None:
            continue
        low, high = max(float(centre) - band_width, 1e-3), min(float(centre) + band_width, 1.0)
        fields[name] = (low, high) if low <= high else (float(centre), float(centre))
    return HumanizeParams(**fields)


def _duration_column(sample) -> tuple[np.ndarray, np.ndarray]:
    """The full-length duration label column and its supervised mask."""
    mask = np.asarray(sample.note_masks[:, DURATION_COLUMN], dtype=bool)
    label = np.asarray(sample.note_targets[:, DURATION_COLUMN], dtype=np.float64)
    if mask.shape != label.shape:
        raise TrainingRunError(f"{sample.sample_id}: duration mask and label disagree")
    return label, mask


def _supervised_by_channel(sample) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for channel, kind, column in CHANNELS:
        mask = np.asarray(getattr(sample, f"{kind}_masks")[:, column], dtype=bool)
        label = np.asarray(getattr(sample, f"{kind}_targets")[:, column], dtype=np.float64)[mask]
        if label.size:
            result[channel] = label
    return result


def _params_snapshot(params: HumanizeParams) -> dict[str, object]:
    return {name: getattr(params, name) for name in (*BAND_NAMES, "jitter_ms", "breath_ms", "jitter_correlation")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--fit-id-file", type=Path, required=True,
                        help="train split: the only labels allowed to shape the constants")
    parser.add_argument("--sample-id-file", type=Path, required=True,
                        help="development split the candidate is measured on")
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--evaluate-split", default="validation")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--band-width", type=float, default=0.06)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if "test" in (arguments.fit_split, arguments.evaluate_split):
        raise SystemExit("the locked test split may not be used to fit or to evaluate")
    if arguments.sessions < 1 or arguments.draws < 100:
        raise SystemExit("sessions must be at least 1 and draws at least 100")
    dataset_root = arguments.dataset.resolve()
    reference = load_timing_reference(arguments.timing_reference)
    shipping = HumanizeParams()

    fit_ids = frozenset(str(value) for value in _load_sample_id_file(arguments.fit_id_file))
    demands: list[tuple[np.ndarray, np.ndarray]] = []
    fit_rows = 0
    for sample in load_selected_samples(dataset_root, arguments.fit_split, fit_ids):
        row = reference[sample.sample_id]
        label, mask = _duration_column(sample)
        demands.append(hold_demand(row, label, mask))
        fit_rows += int(np.count_nonzero(mask))
    fitted = fit_hold_bands(demands, band_width=arguments.band_width, shipping=shipping)
    candidate = candidate_params(fitted, base=shipping, band_width=arguments.band_width)

    evaluate_ids = frozenset(str(value) for value in _load_sample_id_file(arguments.sample_id_file))
    identities = piece_identities(dataset_root, evaluate_ids)
    supervised: dict[str, dict[str, np.ndarray]] = {channel: {} for channel, _kind, _column in CHANNELS}
    errors: dict[tuple[str, str, str], float] = {}
    for sample in load_selected_samples(dataset_root, arguments.evaluate_split, evaluate_ids):
        labels = _supervised_by_channel(sample)
        for channel, label in labels.items():
            supervised[channel][sample.sample_id] = label
        row = reference[sample.sample_id]
        incumbent = rule_projection_errors(sample, row, labels, sessions=arguments.sessions, seed=arguments.seed,
                                           duration_centring=False, params=shipping)
        proposal = rule_projection_errors(sample, row, labels, sessions=arguments.sessions, seed=arguments.seed,
                                          duration_centring=False, params=candidate)
        for channel in labels:
            errors[(channel, "rules_session", sample.sample_id)] = incumbent[channel]["rules_session"]
            errors[(channel, "rules_centred", sample.sample_id)] = incumbent[channel]["rules_centred"]
            # the candidate is scored on the same sessions and the same rows as the incumbent,
            # so the paired difference contains the recalibration and nothing else
            errors[(channel, CANDIDATE_NAME, sample.sample_id)] = proposal[channel]["rules_session"]
    baselines = ("predict_zero", "predict_recording_median", "rules_session", "rules_centred")
    channels = {channel: summarise_channel(channel, supervised, errors, [CANDIDATE_NAME], identities,
                                           draws=arguments.draws, bootstrap_seed=arguments.seed,
                                           baselines=baselines)
                for channel, _kind, _column in CHANNELS}

    verdict = {}
    for channel, summary in channels.items():
        probe = summary.get("policy_gate_probe")
        if not probe:
            continue
        versus_identity = probe["measured_improvement_over_identity"]
        verdict[channel] = {"improvement_over_identity": versus_identity,
                            "interval_supports_improvement_over_identity":
                                probe["interval_supports_improvement_over_identity"],
                            "reaches_policy_fraction": bool(versus_identity is not None
                                                            and versus_identity >= POLICY["primary_improvement_fraction"]),
                            "improvement_over_shipped_rules": probe["measured_improvement_over_rules"]}
    report = {"kind": "humanize_hold_band_refit", "status": "complete", "trained": False,
              "dataset": str(dataset_root), "fit_split": arguments.fit_split,
              "evaluate_split": arguments.evaluate_split,
              "fit_sample_id_file": str(arguments.fit_id_file.resolve()),
              "fit_sample_id_file_sha256": _sha256(arguments.fit_id_file),
              "fit_supervised_notes": fit_rows,
              "evaluate_sample_id_file": str(arguments.sample_id_file.resolve()),
              "evaluate_sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "timing_reference_sha256": _sha256(arguments.timing_reference / "summary.json"),
              "shipping_params": _params_snapshot(shipping),
              "fitted_bands": fitted, "candidate_params": _params_snapshot(candidate),
              "model_column_means": "in `channels`, `model` is the candidate rule, not a trained model",
              "protocol": {"unit": "performance_dataset::piece_group", "piece_weighting": "equal",
                           "draws": arguments.draws, "bootstrap_seed": arguments.seed,
                           "rule_sessions_per_recording": arguments.sessions,
                           "sign": "a negative paired difference means the candidate errs less"},
              "channels": channels, "verdict": verdict,
              "reading": {"adoption_evidence": False,
                          "note": "a candidate rule that beats identity by >=10% with interval support is still not "
                                  "shippable: it needs the compiler/game expressibility check, listening evidence, "
                                  "and the rule path kept reversible. Nothing here changes POLICY's text."},
              "model_inputs_modified": False, "adoption_evidence": False}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "fitted_median_hold": {name: fitted[name]["median_hold"] for name in BAND_NAMES},
                      "candidate_params": report["candidate_params"], "verdict": verdict}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
