"""Evaluate trusted raw predictions through safety and the real offline compiler."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from core.profile import load_profiles, resolve_profile
from training.audit_duration_reference import evaluate_duration_prediction, load_duration_reference
from training.midi_features import parse_midi_bytes
from training.pianocore_dataset import RefinedArchive
from training.playback_adoption import POLICY, adoption_decision, compare_work_metrics, verify_inventory
from training.shadow_expression import shadow_projection
from training.tempo_reference import _safe_path, evaluate_timing_prediction, load_timing_reference
from training.train_expression import _sha256, _write_json
from training.train_sequence_expression import write_integrity


SHADOW_CONFIG = {"strength": 0.5, "cumulative_onset_limit_s": 0.15,
                 "profile_id": "delta_force_harmonica", "scenario_id": "free_play",
                 "bpm_units_only": 120, "settle_ms": 30.0, "release_settle_ms": 20.0,
                 "hold_ratio": 1.0, "max_hold_ms": 3000.0, "gap_ms": 0.0}


def assert_reference_binding(timing_root: Path, duration_root: Path) -> None:
    summary = json.loads((duration_root / "summary.json").read_text(encoding="utf-8"))
    if summary["timing_reference_summary_sha256"] != _sha256(timing_root / "summary.json"):
        raise ValueError("duration and timing references have different source bindings")


def verify_parent_child(root: Path, kind: str, label: str, parent: dict) -> str:
    digest = verify_inventory(root / kind / label)
    if parent.get("child_integrity_sha256", {}).get(f"{kind}/{label}") != digest:
        raise ValueError("child inventory differs from sealed parent binding")
    return digest


def evaluate_onsets(reference: dict, onsets: np.ndarray, *, calibration: str) -> dict:
    """Unrepresented/nonpositive intervals are missing coverage, not zero error."""
    onsets = np.asarray(onsets, dtype=float)
    score_ioi = np.diff(reference["chord_score_seconds"], prepend=reference["chord_score_seconds"][0])
    if onsets.shape != score_ioi.shape:
        raise ValueError("shadow onset shape mismatch")
    predicted = np.diff(onsets, prepend=onsets[0])
    valid = np.isfinite(predicted) & (predicted > 0)
    valid[0] = False
    ref = dict(reference)
    ref["chord_native_masks"] = reference["chord_native_masks"].copy()
    ref["chord_native_masks"][:, 1] &= valid
    ref["chord_trusted"] = reference["chord_trusted"] & np.isfinite(onsets)
    # Filler is never measured: native metrics above explicitly exclude it.
    ratio = np.zeros(len(score_ioi))
    ratio[valid] = np.log(predicted[valid] / score_ioi[valid])
    result = evaluate_timing_prediction(ref, ratio, calibration=calibration, fixed_scale=1.0)
    result["missing_native_ioi_count"] = int(np.sum(reference["chord_native_masks"][:, 1] & ~valid))
    return result


def metrics_for_plan(plan: dict, timing: dict, duration: dict) -> dict:
    metrics = {}
    selected = plan["selected_note_indices"]
    for stage, prefix in (("raw", "raw"), ("safe", "desired"), ("projected", "actual")):
        onset, hold = plan[f"{prefix}_onsets_s"], plan[f"{prefix}_holds_s"]
        note_hold = np.full(len(duration["note_score_duration_seconds"]), np.nan)
        note_valid = np.zeros(len(note_hold), dtype=bool)
        valid_chord = np.isfinite(onset) & np.isfinite(hold) & (hold >= 0)
        note_hold[selected[valid_chord]] = hold[valid_chord]
        note_valid[selected[valid_chord]] = True
        for mode in ("fixed_score", "oracle_global"):
            scale = float(duration["global_tempo_scale"]) if mode == "oracle_global" else 1.0
            t = evaluate_onsets(timing, onset, calibration=mode)
            d = evaluate_duration_prediction(duration, predicted_duration_seconds=note_hold * scale,
                                             prediction_valid=note_valid, calibration=mode, fixed_scale=1.0)
            metrics[f"{stage}:{mode}:ioi_mae_ms"] = {"value": t["ioi_mae_ms"], "count": t["valid_ioi_count"]}
            metrics[f"{stage}:{mode}:key_duration_mae_ms"] = {"value": d["key_duration"]["mae_ms"],
                                                            "count": d["key_duration"]["count"]}
            for h, values in t["bar_drift"].items():
                metrics[f"{stage}:{mode}:bar{h}_mae_ms"] = {"value": values["mae_ms"], "count": values["window_count"]}
    return metrics


def aggregate_samples(sample_metrics: dict, identities: dict) -> dict:
    parts = defaultdict(list)
    metric_names = {name for values in sample_metrics.values() for name in values}
    for sample_id, metrics in sample_metrics.items():
        source, work = identities[sample_id]
        for name, metric in metrics.items():
            if metric["value"] is not None and metric["count"]:
                parts[(name, work)].append((float(metric["value"]), int(metric["count"])))
    values = {name: {} for name in sorted(metric_names)}
    for (name, work), observed in parts.items():
        values[name][work] = sum(value * count for value, count in observed) / sum(count for _, count in observed)
    return {"work_values": values, "source_by_work": {work: source for source, work in identities.values()},
            "metric_summary": {name: {"work_equal_mean": float(np.mean(list(by_work.values()))) if by_work else None,
                                     "n_works": len(by_work),
                                     "sample_count": sum(sample[name]["value"] is not None for sample in sample_metrics.values()),
                                     "event_or_window_count": sum(sample[name]["count"] for sample in sample_metrics.values())}
                               for name, by_work in values.items()}}


def _trace(path: Path, plan: dict) -> None:
    arrays = {key: plan[key] for key in ("raw_onsets_s", "raw_holds_s", "desired_onsets_s", "desired_holds_s",
                                       "actual_onsets_s", "actual_holds_s", "selected_note_indices", "coverage_mask")}
    arrays.update(event_time_ms=np.array([event.t_ms for event in plan["events"]]),
                  event_device=np.array([event.device for event in plan["events"]]),
                  event_key=np.array([event.key for event in plan["events"]]),
                  event_action=np.array([event.action for event in plan["events"]]),
                  event_source=np.array([event.source_index if event.source_index is not None else -1 for event in plan["events"]]))
    np.savez_compressed(path, **arrays)


def _contract_check(root: Path, registration: dict, jobs: list[dict], timing_root: Path) -> dict:
    """Supplementary post-run binding; never rewrite the original registration."""
    reference = json.loads((timing_root / "summary.json").read_text(encoding="utf-8"))
    hashes = {name: _sha256(Path(__file__).parent / name) for name in
              ("sequence_expression.py", "train_sequence_expression.py", "rhythm_objective.py")}
    for job in jobs:
        config = json.loads((root / "runs" / job["label"] / "run_config.json").read_text(encoding="utf-8"))
        for key in ("architecture", "seed", "cumulative_weight"):
            if config[key] != job[key]:
                raise ValueError(f"run changed registered job {key}")
        for key in ("epochs", "micro_weight", "onset_weight", "duration_weight", "ioi_weight", "velocity_weight",
                    "pedal_weight", "batch_size", "core_size", "context_size", "hidden_dim", "dropout",
                    "learning_rate", "weight_decay"):
            if config[key] != registration[key]:
                raise ValueError(f"run changed registered hyperparameter {key}")
        # Compare the names this check actually computes, not the whole record: the trainer has
        # been recording more implementation files than the three bound here, and a dictionary
        # equality test made every current checkpoint fail its own contract check.
        recorded = config["implementation_sha256"]
        if not all(recorded.get(name) == digest for name, digest in hashes.items()):
            raise ValueError("sequence implementation changed across registered jobs")
        if config["quality_mask"] is not None or config["target_shuffle"]:
            raise ValueError("unregistered mask or shuffle")
        for kind in ("train", "validation"):
            if config[f"{kind}_sample_id_file_sha256"] != registration["selection"][f"{kind}_sha256"]:
                raise ValueError("registered selection changed")
        from training.train_sequence_expression import reference_hash
        if config["timing_reference_sha256"] != reference_hash(timing_root):
            raise ValueError("reference changed across jobs")
        if config["dataset_success_sha256"] != _sha256(Path(reference["dataset_root"]) / "_SUCCESS"):
            raise ValueError("dataset changed across jobs")
        cache = Path(config["dataset_validation_cache"])
        if config["dataset_validation_cache_sha256"] != _sha256(cache):
            raise ValueError("validation cache changed across jobs")
        if config["cumulative_spans"] != [1, 8, 32] or config["cumulative_log_ratio_clip"] != 4.0:
            raise ValueError("cumulative objective defaults changed across jobs")
    return {"checked": True, "implementation_sha256": hashes,
            "binding_stage": "post-run supplementary verification, not retroactive preregistration",
            "limitation": "v1 preregistration did not pre-bind all implementations/cache/reference; all children checked before shadow evaluation"}


def evaluate(*, pilot: Path, previous_pilot: Path, timing_reference: Path,
             duration_reference: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("shadow output must be a new directory")
    verify_inventory(pilot)
    assert_reference_binding(timing_reference, duration_reference)
    timing = load_timing_reference(timing_reference)
    duration = load_duration_reference(duration_reference)
    summary = json.loads((pilot / "pilot_summary.json").read_text(encoding="utf-8"))
    registration = json.loads((pilot / "preregistration.json").read_text(encoding="utf-8"))
    if summary["preregistration_sha256"] != _sha256(pilot / "preregistration.json"):
        raise ValueError("pilot registration changed")
    bindings = _contract_check(pilot, registration, summary["runs"], timing_reference)
    previous_summary_path = previous_pilot / "pilot_summary.json"
    if (previous_pilot / "_SUCCESS").read_text(encoding="ascii").strip() != _sha256(previous_summary_path):
        raise ValueError("historical pilot parent fingerprint mismatch")
    old = json.loads(previous_summary_path.read_text(encoding="utf-8"))
    jobs = [(pilot, row) for row in summary["runs"]]
    jobs += [(previous_pilot, row) for row in old["runs"]
             if row["label"].startswith(("mlp_tempo_seed", "gru_tempo_seed"))]
    all_audits = []
    selection, selected_ids = None, None
    for root, job in jobs:
        audit_root = root / "audits" / job["label"]
        parent = summary if root == pilot else old
        verify_parent_child(root, "runs", job["label"], parent)
        digest = verify_parent_child(root, "audits", job["label"], parent)
        report = json.loads((audit_root / "report.json").read_text(encoding="utf-8"))
        if report["seed"] != job["seed"]:
            raise ValueError("audit seed differs from actual run")
        config = json.loads((root / "runs" / job["label"] / "run_config.json").read_text(encoding="utf-8"))
        expected_checkpoint = (root / "runs" / job["label"] / "checkpoint_best.pt").resolve()
        if (config["seed"] != job["seed"] or Path(report["checkpoint"]).resolve() != expected_checkpoint
                or report["checkpoint_sha256"] != _sha256(expected_checkpoint)):
            raise ValueError("model/run pairing differs")
        current_ids = set(report["prediction_files"])
        if selection is None:
            selection, selected_ids = report["sample_id_file_sha256"], current_ids
        if selection != report["sample_id_file_sha256"] or selected_ids != current_ids:
            raise ValueError("new/historical audit selection mismatch")
        all_audits.append((audit_root, job, report, digest))
    reference_summary = json.loads((timing_reference / "summary.json").read_text(encoding="utf-8"))
    source_rows = {}
    with (Path(reference_summary["dataset_root"]) / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in selected_ids:
                if row["split"] != "validation" or row.get("official_split") == "test":
                    raise ValueError("shadow selection must be validation, never test")
                source_rows[row["id"]] = row
    if set(source_rows) != selected_ids:
        raise ValueError("source selection incomplete")
    identities = {id_: (r["performance_dataset"], f"{r['performance_dataset']}::{r['piece_group']}") for id_, r in source_rows.items()}
    profile = resolve_profile(load_profiles(str(Path(__file__).parents[1] / "profiles")), SHADOW_CONFIG["profile_id"])
    if profile.id != SHADOW_CONFIG["profile_id"] or profile.legacy_keymap is not None:
        raise ValueError("requested compiler profile unavailable; no fallback to legacy")
    params = profile.build_compile_params(bpm=120, settle_ms=30, release_settle_ms=20,
                                          hold_ratio=1, max_hold_ms=3000, gap_ms=0)
    output.mkdir(parents=True)
    (output / "_INCOMPLETE").touch()
    prereg = {"policy": POLICY, "shadow_config": SHADOW_CONFIG, "selection_sha256": selection,
              "timing_summary_sha256": _sha256(timing_reference / "summary.json"),
              "duration_summary_sha256": _sha256(duration_reference / "summary.json"),
              "audit_integrity_sha256": {job["label"]: digest for _, job, _, digest in all_audits},
              "implementation_sha256": {name: _sha256(Path(__file__).parent / name) for name in
                                        ("shadow_expression.py", "evaluate_playback_adoption.py", "playback_adoption.py", "audit_duration_reference.py")},
              "profile_sha256": _sha256(Path(__file__).parents[1] / "profiles" / "delta_force_harmonica.yaml"),
              "core_implementation_sha256": {name: _sha256(Path(__file__).parents[1] / "core" / name)
                                             for name in ("compiler.py", "humanize.py", "scenario.py", "profile.py", "ir.py")},
              "candidate_families": [f"{a}_hold_ioi_cum{c:g}" for a in ("mlp", "gru") for c in (0.0, 0.5)],
              "baseline_families": POLICY["required_baselines"], "training_supplementary_binding": bindings}
    _write_json(output / "shadow_preregistration.json", prereg)
    scores = {}
    with RefinedArchive(reference_summary["archive_path"]) as archive:
        for sample_id, row in source_rows.items():
            data = archive.read(row["score_midi_path"])
            if hashlib.sha256(data).hexdigest() != row["score_midi_sha256"]:
                raise ValueError("score source member modified")
            scores[sample_id] = parse_midi_bytes(data)
    baselines, reports = {}, {}
    seeds = sorted({row["seed"] for _, row, _, _ in all_audits})
    for seed in seeds:
        for strategy in POLICY["required_baselines"]:
            sample_metrics, baseline_projection = {}, {}
            for sample_id in sorted(selected_ids):
                score, ref = scores[sample_id], timing[sample_id]
                kw = {"strategy": strategy, "seed": seed, "params": params}
                if strategy == "zero_model_adapter":
                    kw = {**kw, "strategy": "model", "note_prediction": np.zeros((len(score.notes), 6)),
                          "chord_prediction": np.zeros((len(ref["chord_score_seconds"]), 2))}
                plan = shadow_projection(score, ref["note_to_chord"], **kw)
                sample_metrics[sample_id] = metrics_for_plan(plan, ref, duration[sample_id])
                baseline_projection[sample_id] = {key: plan[key] for key in
                    ("actual_inter_event_gap_policy", "minimum_actual_preparation_gap_ms",
                     "actual_full_release_gap_violation_count", "required_model_full_release_gap_ms")}
            baselines[(strategy, seed)] = {"seed": seed, "selection_sha256": selection,
                                            **aggregate_samples(sample_metrics, identities), "projection": baseline_projection}
            _write_json(output / f"baseline_{strategy}_seed{seed}.json", baselines[(strategy, seed)])
    for audit_root, job, report, _ in all_audits:
        label = job["label"]
        print(f"SHADOW_START {label}", flush=True)
        trace_root = output / "traces" / label
        trace_root.mkdir(parents=True)
        sample_metrics, details = {}, {}
        for sample_id in sorted(selected_ids):
            with np.load(_safe_path(audit_root, report["prediction_files"][sample_id]), allow_pickle=False) as stored:
                ny, cy, mapping = stored["note_prediction"], stored["chord_prediction"], stored["note_to_chord"]
                if not np.array_equal(mapping, timing[sample_id]["note_to_chord"]) or not np.array_equal(stored["note_masks"], duration[sample_id]["note_native_masks"]):
                    raise ValueError("prediction/reference original ordering or masks differ")
            plan = shadow_projection(scores[sample_id], mapping, ny, cy, strategy="model", seed=job["seed"], params=params)
            sample_metrics[sample_id] = metrics_for_plan(plan, timing[sample_id], duration[sample_id])
            beyond = np.abs(plan["actual_onsets_s"] - plan["score_onsets_s"]) > SHADOW_CONFIG["cumulative_onset_limit_s"] + 1e-9
            details[sample_id] = {key: plan[key] for key in ("clipping", "covered_chord_count", "source_chord_count", "source_note_count",
                                                            "unrepresented_source_voices", "compiler_delayed_chords", "max_compiler_onset_change_ms",
                                                            "all_events_balanced", "rest_element_count", "long_note_truncations",
                                                            "actual_inter_event_gap_policy", "minimum_actual_preparation_gap_ms",
                                                            "actual_full_release_gap_violation_count", "required_model_full_release_gap_ms")}
            details[sample_id]["folded_note_count"] = len(plan["folded_note_indices"])
            details[sample_id]["projected_cumulative_budget_exceeded_count"] = int(beyond.sum())
            _trace(trace_root / f"{sample_id}.npz", plan)
        reports[label] = {"seed": job["seed"], "selection_sha256": selection,
                          **aggregate_samples(sample_metrics, identities), "samples": sample_metrics, "projection": details}
        _write_json(output / f"{label}.json", reports[label])
        print(f"SHADOW_END {label}", flush=True)
    families = {}
    for family in prereg["candidate_families"] + ["mlp_tempo", "gru_tempo"]:
        candidates = [reports[f"{family}_seed{s}"] for s in seeds]
        # Raw rule jitter can invalidate different intervals/windows per seed.
        # Do not silently intersect that coverage or use it for promotion;
        # adoption concerns the fully compiled common-coverage output only.
        def gate_fields(report):
            return {**report, "work_values": {name: report["work_values"][name]
                                              for name in POLICY["primary_metrics"] + POLICY["noninferiority_metrics"]}}
        comparisons = {strategy: compare_work_metrics([gate_fields(baselines[(strategy, s)]) for s in seeds],
                                                    [gate_fields(r) for r in candidates])
                       for strategy in POLICY["required_baselines"]}
        safety = all(d["all_events_balanced"] and d["covered_chord_count"] == d["source_chord_count"]
                     and d["projected_cumulative_budget_exceeded_count"] == 0
                     and d["actual_full_release_gap_violation_count"] == 0
                     for r in candidates for d in r["projection"].values())
        families[family] = {"contrasts": comparisons, "decision": adoption_decision(comparisons,
                             safety_passed=safety, expected_sources=sorted({source for source, _ in identities.values()})),
                            "summary": {name: float(np.mean([r["metric_summary"][name]["work_equal_mean"] for r in candidates]))
                                        for name in candidates[0]["metric_summary"]},
                            "safe_scope": "compiler-only, selected highest voice, pitch degradation disclosed; no actual input"}
    result = {"status": "complete", "production_approved": False, "actual_game_input_sent": False,
              "test_read": False, "selection_sha256": selection, "sample_count": len(selected_ids),
              "work_count": len(set(work for _, work in identities.values())), "families": families,
              "baseline_summary": {strategy: {name: float(np.mean([baselines[(strategy, s)]["metric_summary"][name]["work_equal_mean"] for s in seeds]))
                                               for name in next(iter(baselines.values()))["metric_summary"]}
                                   for strategy in POLICY["required_baselines"]},
              "aggregation": "event/window weighted within work, equal works, seed mean; bootstrap conditional on fixed seeds",
              "training_supplementary_binding": bindings,
              "preregistration_sha256": _sha256(output / "shadow_preregistration.json")}
    _write_json(output / "results.json", result)
    write_integrity(output)
    (output / "_INCOMPLETE").unlink()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "previous-pilot", "timing-reference", "duration-reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = evaluate(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], "families": len(result["families"]), "production_approved": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
