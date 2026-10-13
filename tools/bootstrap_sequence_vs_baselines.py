"""Piece-level paired bootstrap of the sequence model against its simple baselines.

The registered numeric gate for this line is not "validation loss went down": it is
``timing``/``duration`` improving by at least 5% over the strongest simple baseline, with a
piece-level 95% bootstrap interval that supports the improvement and no source regressing by
more than 2%.  Piece-level matters because one recording can contribute thousands of chords
while another contributes a few hundred -- row weighting would let a handful of long
recordings decide the verdict, and it hides the fact that recordings of the same piece share
everything a score-only model is able to predict.

Protocol (fixed here, matching the older flat-row pipeline's convention):
* the resampling unit is ``performance_dataset::piece_group``;
* inside a piece, recordings are averaged, then seeds are averaged, so seed noise cannot
  inflate the sample size;
* pieces carry equal weight inside a draw;
* 2,000 draws with a registered seed, percentile interval at 2.5/97.5;
* fewer than five pieces and no interval is reported at all.

Baselines.  ``predict_zero`` (play the score as written, which is also POLICY's
``identity`` and ``zero_model_adapter`` in label units) and
``predict_recording_median`` (one constant per recording, the most a per-recording offset
could explain) come from the labels themselves.  ``rules_session`` and ``rules_centred``
are the humanizer that already ships, projected into these units by
``training/rules_baseline.py``; POLICY's ``required_baselines`` names all three of
identity, rules and zero_model_adapter, so a report without ``rules`` cannot say anything
about adoption.  ``rules_session`` averages the error over independent playbacks -- one
realized session is what a listener hears -- while ``rules_centred`` averages the
predictions first, which cancels most of the jitter and is therefore the stronger of the
two: beating only the first is not a claim.  The strongest baseline present is what the 5%
requirement is measured against.  Nothing trains, and the locked test split is refused.
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
import torch

from core.humanize import HumanizeParams
from training.audit_sequence_expression import load_model
from training.playback_adoption import POLICY
from training.rules_baseline import (duration_log_ratio_predictions, element_gaps, ioi_log_ratio_predictions,
                                     leap_rule_coverage, onset_predictions, rule_timings)
from training.sequence_expression import checkpoint_preparation_flags, load_selected_samples, predict_sample
from training.train_expression import TrainingRunError, _load_sample_id_file, _sha256, _write_json
from training.tempo_reference import load_timing_reference

# (channel, row kind, column) -- the same map tools/audit_target_channels.py uses
CHANNELS = (("onset_delta_ms", "chord", 0), ("log_ioi_ratio", "chord", 1),
            ("log_key_duration_ratio", "note", 1))
# the two that follow from the labels alone
LABEL_BASELINES = ("predict_zero", "predict_recording_median")
# the shipped rule humanizer, projected by training/rules_baseline.py
RULE_BASELINES = ("rules_session", "rules_centred")
BASELINES = LABEL_BASELINES + RULE_BASELINES
MINIMUM_PIECES = 5
# a projection that cannot emit more than this share of a channel's supervised rows is not a
# baseline of that channel, it is a different measurement, so the run refuses instead
MAXIMUM_RULE_MISSING_FRACTION = 0.01


def bootstrap_difference(pairs: list[tuple[float, float]], draws: int, seed: int) -> dict[str, object]:
    """Percentile interval for the paired difference (model minus baseline) across pieces."""
    if len(pairs) < MINIMUM_PIECES:
        return {"pieces": len(pairs), "note": f"fewer than {MINIMUM_PIECES} pieces, no interval reported"}
    differences = np.asarray([(model - baseline) for model, baseline in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[indices].mean(axis=1)
    point = float(differences.mean())
    low, high = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"pieces": int(differences.size), "draws": draws, "seed": seed,
            "mean_paired_difference": point, "interval_low": low, "interval_high": high,
            "probability_model_worse": float((means > 0).mean()),
            "supports_improvement": bool(point < 0.0 and high < 0.0)}


def _highest_pitch_per_chord(sample) -> list[float]:
    """One pitch per element: the voice the element cannot release before.

    Used only for the leap-coverage disclosure, never for a prediction -- the shipping leap
    rule reads the game's degree names and has nothing to read here.
    """
    pitches = np.rint(60.0 + 24.0 * np.asarray(sample.note_features[:, 0], dtype=np.float64))
    mapping = np.asarray(sample.note_to_chord, dtype=np.int64)
    return [float(pitches[mapping == chord].max()) for chord in np.unique(mapping)]


def rule_projection_errors(sample, reference_row: dict, labels: dict[str, np.ndarray], *,
                           sessions: int, seed: int, duration_centring: bool,
                           params: HumanizeParams | None = None) -> dict[str, dict[str, float]]:
    """Errors of the shipping rule humanizer on one recording, in the label's own units.

    ``labels`` maps channel to that channel's supervised label vector for this recording --
    the same vectors the model is scored against, so every predictor sees one row set.
    Everything the projection needs is score-side and comes from the locked timing
    reference, so the baseline sees no more information than a score-only model is allowed
    to see.  When the arm being evaluated was trained on median-centred duration labels the
    prediction is centred the same way, because comparing a centred label against an
    uncentred prediction would measure the level the arm was built to forget.
    """
    chord_count = len(reference_row["chord_score_seconds"])
    if chord_count != len(sample.chord_features):
        raise TrainingRunError(f"{sample.sample_id}: reference chord count differs from the sample")
    if len(reference_row["note_to_chord"]) != len(sample.note_features):
        raise TrainingRunError(f"{sample.sample_id}: reference note count differs from the sample")
    tempo = float(np.asarray(reference_row["score_seconds_per_beat_reference"]).item())
    gaps = element_gaps(reference_row["chord_score_beats"], reference_row["note_score_duration_beats"],
                        reference_row["note_to_chord"])
    timings = rule_timings(reference_row["chord_score_seconds"], gaps, seconds_per_beat=tempo,
                           sessions=sessions, seed=seed, params=params)
    predictions = {"onset_delta_ms": onset_predictions(timings),
                   "log_ioi_ratio": ioi_log_ratio_predictions(timings, reference_row["chord_score_seconds"]),
                   "log_key_duration_ratio": duration_log_ratio_predictions(
                       timings, reference_row["note_to_chord"], gaps * tempo,
                       reference_row["note_score_duration_seconds"])}
    result: dict[str, dict[str, float]] = {}
    for channel, _kind, column in CHANNELS:
        label = labels.get(channel)
        if label is None:
            continue
        rows = getattr(sample, f"{_kind}_masks")[:, column]
        # predictions are (sessions, rows) while the mask is one boolean per row: indexing the
        # first axis would select sessions instead of supervised notes, which numpy only
        # complains about when the two counts happen to differ
        drawn = np.asarray(predictions[channel], dtype=np.float64)[:, rows]
        # a row counts only if *every* session can emit it, so that all four predictors are
        # scored on one shared row set instead of each quietly keeping its own easier rows
        emittable = np.isfinite(drawn).all(axis=0)
        missing = int(np.count_nonzero(~emittable))
        if missing and missing / max(len(label), 1) > MAXIMUM_RULE_MISSING_FRACTION:
            raise TrainingRunError(f"{sample.sample_id}: the rule projection cannot emit "
                                   f"{missing}/{len(label)} supervised rows of {channel}")
        if not np.any(emittable):
            raise TrainingRunError(f"{sample.sample_id}: no emittable rule prediction for {channel}")
        kept = drawn[:, emittable]
        target = label[emittable]
        if duration_centring and channel == "log_key_duration_ratio":
            # preparation already took the recording median out of the label; the baseline has to
            # lose its own level too, otherwise the comparison silently measures the level this arm
            # was trained to forget.  Both sides are re-centred here so the subtraction is exact
            # even when the two medians were taken over slightly different row sets.
            kept = kept - np.median(kept, axis=1, keepdims=True)
            target = target - float(np.median(target))
        result[channel] = {
            "rules_session": float(np.abs(kept - target[None, :]).mean()),
            "rules_centred": float(np.abs(kept.mean(axis=0) - target).mean()),
            "rules_rows": float(len(target)), "rules_missing_rows": float(missing),
        }
    return result


def collapse_to_pieces(channel: str, supervised: dict[str, np.ndarray],
                       errors: dict[tuple[str, str, str], float], run_names: list[str],
                       identities: dict[str, tuple[str, str]], *,
                       baselines: tuple[str, ...] = BASELINES) -> dict[str, dict[str, float]]:
    """One row per ``source::piece_group``: recordings averaged, then seeds averaged.

    Averaging happens twice before anything is compared, so neither a long recording nor a
    lucky seed can carry more weight than a piece of music deserves.  ``baselines`` names the
    predictors that must be present: the two label-derived ones are computed here, and every
    other name (the shipped rules, or a candidate rule) has to arrive in ``errors`` for every
    scored recording, because a paired test with a missing arm is not paired.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for sample_id, label in supervised.items():
        source, group = identities[sample_id]
        model_values = [errors[(channel, run, sample_id)] for run in run_names
                        if (channel, run, sample_id) in errors]
        if not model_values:
            continue
        bucket = buckets.setdefault(f"{source}::{group}", {"model": [], "recordings": [], "rows": []})
        bucket["model"].append(float(np.mean(model_values)))
        for baseline in baselines:
            if baseline in LABEL_BASELINES:
                zero, median = float(np.abs(label).mean()), float(np.abs(label - np.median(label)).mean())
                bucket.setdefault(baseline, []).append(zero if baseline == "predict_zero" else median)
            else:
                key = (channel, baseline, sample_id)
                if key not in errors:
                    raise TrainingRunError(f"missing {baseline} error for {sample_id} on {channel}")
                bucket.setdefault(baseline, []).append(errors[key])
        bucket["recordings"].append(1.0)
        bucket["rows"].append(float(label.size))
    pieces: dict[str, dict[str, float]] = {}
    for piece, data in sorted(buckets.items()):
        merged = {key: float(np.mean(data[key])) for key in ("model", *baselines) if key in data}
        # counts add up, errors average: mixing the two is how a long recording would otherwise
        # quietly buy itself extra weight inside its own piece
        merged["recordings"] = int(len(data["model"]))
        merged["rows"] = int(np.sum(data["rows"]))
        pieces[piece] = merged
    return pieces


def piece_identities(dataset_root: Path, sample_ids: frozenset[str]) -> dict[str, tuple[str, str]]:
    """``sample id -> (source, piece group)`` straight from the dataset manifest."""
    identities: dict[str, tuple[str, str]] = {}
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in sample_ids:
                identities[sample_id] = (str(row.get("performance_dataset") or "unknown"),
                                         str(row.get("piece_group") or sample_id))
    if set(identities) != set(sample_ids):
        raise TrainingRunError("dataset manifest does not cover every selected sample")
    return identities


def summarise_channel(channel: str, supervised: dict[str, dict[str, np.ndarray]],
                      errors: dict[tuple[str, str, str], float], run_names: list[str],
                      identities: dict[str, tuple[str, str]], *, draws: int, bootstrap_seed: int,
                      baselines: tuple[str, ...] = BASELINES) -> dict[str, object]:
    """The piece-level table and gates for one channel, shared by the bootstrap and rule tools.

    ``identity`` gets its own explicit column because POLICY's three required baselines are
    not equally strong on piano material: the shipping humanizer is *worse* than playing the
    score as written on the duration channel, so an improvement reported only over ``rules``
    can look like a pass while nothing musical has been gained.
    """
    pieces = collapse_to_pieces(channel, supervised[channel], errors, run_names, identities,
                                baselines=baselines)
    summary: dict[str, object] = {"pieces": len(pieces),
                                  "recordings": int(sum(value["recordings"] for value in pieces.values())),
                                  "rows": int(sum(value["rows"] for value in pieces.values()))}
    if not pieces:
        summary["note"] = "no supervised rows for this channel"
        return summary
    model_mae = float(np.mean([value["model"] for value in pieces.values()]))
    baseline_mae = {baseline: float(np.mean([value[baseline] for value in pieces.values()]))
                    for baseline in baselines}
    strongest = min(baseline_mae, key=lambda baseline: baseline_mae[baseline])
    comparisons = {baseline: bootstrap_difference([(value["model"], value[baseline])
                                                   for value in pieces.values()], draws, bootstrap_seed)
                   for baseline in baselines}
    improvement = 1.0 - model_mae / baseline_mae[strongest] if baseline_mae[strongest] > 0 else None
    versus_rules = _ratio_improvement(model_mae, baseline_mae.get("rules_session"))
    versus_identity = _ratio_improvement(model_mae, baseline_mae.get("predict_zero"))
    summary.update({
        "piece_equal_weight_mae": {"model": model_mae, **baseline_mae},
        "strongest_baseline": strongest,
        # a baseline of exactly zero means every recording was one row long: "predict its own
        # median" then reproduces it trivially, and no relative improvement can be defined
        "baseline_is_degenerate": baseline_mae[strongest] <= 0.0,
        "improvement_over_strongest_baseline": improvement,
        # POLICY's binding comparison: the humanizer that already ships, at 10%
        "improvement_over_shipped_rules": versus_rules,
        "improvement_over_identity": versus_identity,
        "paired_bootstrap": comparisons,
        "gate": {"required_relative_improvement": 0.05,
                 "relative_improvement_reached": improvement is not None and improvement >= 0.05,
                 "interval_supports_improvement": bool(comparisons[strongest].get("supports_improvement", False)),
                 "passed_numeric_gate": bool(improvement is not None and improvement >= 0.05
                                             and comparisons[strongest].get("supports_improvement", False))},
        "policy_gate_probe": {
            "required_baselines": list(POLICY["required_baselines"]),
            "required_relative_improvement": POLICY["primary_improvement_fraction"],
            "measured_improvement_over_rules": versus_rules,
            "measured_improvement_over_identity": versus_identity,
            "shortfall_against_policy": None if versus_rules is None
                                        else float(POLICY["primary_improvement_fraction"] - versus_rules),
            "interval_supports_improvement_over_rules": bool(
                comparisons.get("rules_session", {}).get("supports_improvement", False)),
            "interval_supports_improvement_over_identity": bool(
                comparisons.get("predict_zero", {}).get("supports_improvement", False)),
            "note": "identity and zero_model_adapter are the same all-zero prediction in label "
                    "units, which is predict_zero; only the metric scale differs from POLICY's "
                    "projected milliseconds",
        },
    })
    return summary


def _ratio_improvement(model_mae: float, baseline_mae: float | None) -> float | None:
    if baseline_mae is None or baseline_mae <= 0:
        return None
    return float(1.0 - model_mae / baseline_mae)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="one directory per seed")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rule-sessions", type=int, default=8,
                        help="playbacks of the shipping rule humanizer averaged per recording")
    parser.add_argument("--rule-seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.split == "test":
        raise SystemExit("the locked test split may only be read once the model is frozen")
    if arguments.draws < 100:
        raise SystemExit("draws must be at least 100")
    if arguments.rule_sessions < 1:
        raise SystemExit("rule sessions must be at least 1")
    dataset_root = arguments.dataset.resolve()
    sample_ids = frozenset(str(value) for value in _load_sample_id_file(arguments.sample_id_file))
    identities = piece_identities(dataset_root, sample_ids)
    reference = load_timing_reference(arguments.timing_reference)
    device = torch.device(arguments.device if torch.cuda.is_available() or arguments.device == "cpu" else "cpu")

    def _collect_labels(rows) -> dict[str, dict[str, np.ndarray]]:
        """Supervised label column per (channel, recording)."""
        collected: dict[str, dict[str, np.ndarray]] = {name: {} for name, _kind, _column in CHANNELS}
        for sample in rows:
            for name, kind, column in CHANNELS:
                mask = np.asarray(getattr(sample, f"{kind}_masks")[:, column], dtype=bool)
                label = np.asarray(getattr(sample, f"{kind}_targets")[:, column], dtype=np.float64)[mask]
                if label.size:
                    collected[name][sample.sample_id] = label
        return collected

    # one preparation per distinct flag combination: labels and inputs must be in the units
    # the checkpoint was trained in, or the arm comparison silently mixes unit systems
    prepared: dict[tuple[bool, bool, bool], list] = {}
    labels_by_flags: dict[tuple[bool, bool, bool], dict[str, dict[str, np.ndarray]]] = {}
    # (channel, baseline or run name, sample id) -> mean absolute error of that predictor
    errors: dict[tuple[str, str, str], float] = {}
    rule_coverage: dict[str, dict[str, float]] = {"missing_rows": {}, "rows": {}}
    top_voices: list[list[float]] = []
    sample_count = 0
    for run_directory in arguments.runs:
        checkpoint = (run_directory / arguments.checkpoint).resolve()
        model, config = load_model(checkpoint, device)
        flags = checkpoint_preparation_flags(config)
        if flags not in prepared:
            prepared[flags] = load_selected_samples(dataset_root, arguments.split, sample_ids,
                                                    timing_reference=reference, relative_position=flags[2],
                                                    duration_centering=flags[0], shape_feature_names=flags[1])
            if not prepared[flags]:
                raise TrainingRunError("no samples selected for the bootstrap")
            labels_by_flags[flags] = _collect_labels(prepared[flags])
            sample_count = len(prepared[flags])
            for sample in prepared[flags]:
                labels = {channel: by_sample[sample.sample_id]
                          for channel, by_sample in labels_by_flags[flags].items()
                          if sample.sample_id in by_sample}
                observed = rule_projection_errors(sample, reference[sample.sample_id], labels,
                                                  sessions=arguments.rule_sessions, seed=arguments.rule_seed,
                                                  duration_centring=flags[0])
                for channel, values in observed.items():
                    for baseline in RULE_BASELINES:
                        errors[(channel, baseline, sample.sample_id)] = values[baseline]
                    rule_coverage["rows"][f"{channel}::{sample.sample_id}"] = values["rules_rows"]
                    rule_coverage["missing_rows"][f"{channel}::{sample.sample_id}"] = values["rules_missing_rows"]
                top_voices.append(_highest_pitch_per_chord(sample))
        if len({group[0] for group in prepared}) > 1:
            raise TrainingRunError("the baselines are computed in one label unit; report centred and "
                                   "uncentred arms separately")
        samples = prepared[flags]
        supervised = labels_by_flags[flags]
        for sample in samples:
            note_output, chord_output = predict_sample(model, sample, config=config, device=device)
            arrays = {"chord": chord_output, "note": note_output}
            for name, kind, column in CHANNELS:
                label = supervised[name].get(sample.sample_id)
                if label is None:
                    continue
                prediction = np.asarray(arrays[kind][:, column], dtype=np.float64)[
                    np.asarray(getattr(sample, f"{kind}_masks")[:, column], dtype=bool)]
                errors[(name, run_directory.name, sample.sample_id)] = float(np.abs(prediction - label).mean())
        del model

    run_names = [run.name for run in arguments.runs]
    channels: dict[str, object] = {}
    for name, _kind, _column in CHANNELS:
        channels[name] = summarise_channel(name, supervised, errors, run_names, identities,
                                           draws=arguments.draws, bootstrap_seed=arguments.bootstrap_seed)

    rules_report = {
        "projector": "training/rules_baseline.py: core.humanize.build_plan on the shipping "
                     "parameters, plus the chained anti-overlap bound of core/compiler.py, all "
                     "on the locked score timeline (fixed_score), never the recording's tempo",
        "sessions_per_recording": arguments.rule_sessions, "session_seed_start": arguments.rule_seed,
        "params": {field.name: getattr(HumanizeParams(), field.name)
                   for field in dataclasses.fields(HumanizeParams())},
        "supervised_rows": {channel: int(np.sum([value for key, value in rule_coverage["rows"].items()
                                                 if key.startswith(f"{channel}::")]))
                            for channel, _kind, _column in CHANNELS},
        "rows_the_projection_could_not_emit": {channel: int(np.sum([value for key, value
                                                                    in rule_coverage["missing_rows"].items()
                                                                    if key.startswith(f"{channel}::")]))
                                               for channel, _kind, _column in CHANNELS},
        "leap_rule": {**leap_rule_coverage(top_voices),
                      "fires_on_this_corpus": False,
                      "reason": "core.humanize._pitch_value parses only the 21-key degree names "
                                "(low/mid/high + 1..7); a MIDI pitch is unparseable, and "
                                "unparseable means no extra delay. coverage is the semitone "
                                "surrogate statistic, disclosure only"},
        "not_reproduced": ["gap_ms, settle_ms and release_settle_ms (profile latency, not shaping)",
                           "max_hold_ms (profile cap)", "rest elements (PianoCoRe writes every note "
                           "at least to the next onset, so the breath-after-rest branch never runs)",
                           "element duration is the gap to the next chord, the game's own meaning; "
                           "using written lengths instead makes the anti-overlap chain snowball and "
                           "put the rules' onset error at ~22 s on this corpus"],
    }
    report = {"kind": "piece_level_paired_bootstrap_vs_baselines", "status": "complete", "trained": False,
              "dataset": str(dataset_root), "split": arguments.split, "device": str(device),
              "runs": run_names, "sample_count": sample_count,
              "baselines": list(BASELINES),
              "rules_projection": rules_report,
              "sample_id_file": str(arguments.sample_id_file.resolve()),
              "sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "timing_reference_sha256": _sha256(arguments.timing_reference / "summary.json"),
              "protocol": {"unit": "performance_dataset::piece_group", "recordings_and_seeds_averaged_per_piece": True,
                           "piece_weighting": "equal", "draws": arguments.draws,
                           "bootstrap_seed": arguments.bootstrap_seed, "interval": [2.5, 97.5],
                           "sign": "a negative paired difference means the model errs less than the baseline"},
              "channels": channels,
              "reading": {"adoption_evidence": False,
                          "note": "passing this numeric gate is necessary but not sufficient: player adoption also "
                                  "requires human A/B preferences and the compiler/game expressibility check, with "
                                  "the rule-based path kept as the fallback. POLICY's own bar is 10% over these "
                                  "baselines on projected milliseconds, and `rules` is the behaviour the app already "
                                  "ships -- losing to it means replacing working expression with worse expression"},
              "model_inputs_modified": False, "adoption_evidence": False}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "channels": {name: {key: value[key] for key in
                                         ("pieces", "piece_equal_weight_mae", "strongest_baseline",
                                          "improvement_over_strongest_baseline", "improvement_over_shipped_rules",
                                          "policy_gate_probe", "gate") if key in value}
                                   for name, value in channels.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
