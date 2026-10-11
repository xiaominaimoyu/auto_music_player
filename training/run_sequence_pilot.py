"""Run a locked, train/validation-only tempo and small-sequence experiment.

GPU jobs are sequential. Artifacts never belong in the source checkout. This
is a diagnostic pilot: no job is automatically promoted to desktop playback.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from training.audit_sequence_expression import audit
from training.compare_quality_multiseed import _bootstrap
from training.train_sequence_expression import train
from training.train_expression import _sha256


def paired_work_comparison(baselines: list[dict], candidates: list[dict], *, seed=20261003) -> dict:
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("need matched nonempty seed pairs")
    for base, candidate in zip(baselines, candidates):
        if base["sample_id_file_sha256"] != candidate["sample_id_file_sha256"]:
            raise ValueError("comparison selection mismatch")
    names = set(baselines[0]["work_values"]) & set(candidates[0]["work_values"])
    result = {}
    for name in sorted(n for n in names if n.startswith("model:")):
        keys = set(baselines[0]["work_values"][name])
        for report in baselines + candidates:
            if set(report["work_values"][name]) != keys:
                raise ValueError("comparison work sets mismatch")
        works = sorted(keys)
        if not works:
            continue
        base = np.array([[r["work_values"][name][w] for w in works] for r in baselines])
        candidate = np.array([[r["work_values"][name][w] for w in works] for r in candidates])
        delta = candidate - base
        result[name] = {"baseline_mean": float(base.mean()), "candidate_mean": float(candidate.mean()),
                        "delta_candidate_minus_baseline": float(delta.mean()),
                        "delta_by_seed": delta.mean(axis=1).tolist(),
                        "fixed_seed_work_bootstrap": _bootstrap(delta.mean(axis=0).tolist(), seed=seed, rounds=3000)}
    return {"seed_pairs": len(baselines), "negative_delta_is_better": True,
            "interval_scope": "work bootstrap conditional on these fixed seeds; not total seed uncertainty",
            "targets": result}


def timing_summary(report: dict) -> dict:
    """Macro average samples; window coverage remains explicit and separate."""
    result = {}
    samples = report.get("timing_metrics", {})
    for mode in ("oracle_global", "fixed_score"):
        for model in ("model", "identity"):
            values = [s[model][mode] for s in samples.values()]
            iois = [v["ioi_mae_ms"] for v in values if v["ioi_mae_ms"] is not None]
            bars = {}
            for horizon in ("1", "4", "16"):
                windows = [v["bar_drift"][horizon] for v in values]
                available = [v["mae_ms"] for v in windows if v["window_count"] > 0]
                bars[horizon] = {"sample_macro_mae_ms": float(np.mean(available)) if available else None,
                                 "samples_with_windows": len(available),
                                 "window_count": sum(v["window_count"] for v in windows),
                                 "skipped_window_count": sum(v["skipped_window_count"] for v in windows)}
            result[f"{model}:{mode}"] = {"sample_macro_ioi_mae_ms": float(np.mean(iois)) if iois else None,
                                         "sample_count": len(iois), "bar_drift": bars}
    return result


def seal_pilot(output: Path, timing_reference: Path) -> dict:
    """Independently verify child artifacts and bind the completed parent review.

    The unconditioned model also gets physical-time diagnostics, post-hoc only:
    the reference is never passed to that model or its training. Child run and
    audit files remain byte-for-byte unchanged.
    """
    from training.tempo_reference import load_timing_reference, evaluate_timing_prediction

    summary_path = output / "pilot_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    references = load_timing_reference(timing_reference)
    children = {}
    for run in summary["runs"]:
        label = run["label"]
        for kind in ("runs", "audits"):
            root = output / kind / label
            inventory_path = root / "integrity.json"
            if (root / "_INCOMPLETE").exists() or not (root / "_SUCCESS").is_file():
                raise ValueError(f"incomplete child: {root}")
            digest = _sha256(inventory_path)
            if (root / "_SUCCESS").read_text(encoding="ascii").strip() != digest:
                raise ValueError(f"child integrity marker mismatch: {root}")
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            for item in inventory["files"]:
                path = root / item["path"]
                if root.resolve() not in path.resolve().parents:
                    raise ValueError("child inventory path escape")
                if path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
                    raise ValueError(f"child artifact modified: {path}")
            children[f"{kind}/{label}"] = digest
        report = json.loads((output / "audits" / label / "report.json").read_text(encoding="utf-8"))
        if not report.get("timing_metrics"):
            metrics = {}
            for sample_id, relative in report["prediction_files"].items():
                with np.load(output / "audits" / label / relative, allow_pickle=False) as prediction:
                    cy = prediction["chord_prediction"][:, 1]
                reference = references[sample_id]
                metrics[sample_id] = {
                    name: {mode: evaluate_timing_prediction(reference, values, calibration=mode, fixed_scale=1.0)
                           for mode in ("oracle_global", "fixed_score")}
                    for name, values in (("model", cy), ("identity", np.zeros(len(cy))))}
            run["timing"] = timing_summary({"timing_metrics": metrics})
            run["timing_reference_used_posthoc_only"] = True
    summary["child_integrity_sha256"] = children
    summary["posthoc_timing_reference_summary_sha256"] = _sha256(timing_reference / "summary.json")
    summary["parent_integrity_verified"] = True
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "_SUCCESS").write_text(_sha256(summary_path) + "\n", encoding="ascii")
    return summary


def run_pilot(*, dataset: Path, output: Path, timing_reference: Path,
              train_sample_id_file: Path, validation_sample_id_file: Path,
              validation_cache: Path, epochs: int = 5, device: str = "cuda",
              seeds: tuple[int, ...] = (20261003, 20261004, 20261005)) -> dict:
    if output.exists():
        raise ValueError("pilot output must be a new directory")
    output.mkdir(parents=True)
    (output / "_INCOMPLETE").touch()
    jobs = [("mlp_no_tempo", "mlp", seeds[0], None, False, 64)]
    jobs += [(f"{architecture}_tempo", architecture, seed, timing_reference, False, 64)
             for seed in seeds for architecture in ("mlp", "gru")]
    jobs += [("gru_tempo_shuffled", "gru", seeds[0], timing_reference, True, 64),
             ("mlp_tempo_capacity128", "mlp", seeds[0], timing_reference, False, 128)]
    reports, runs = {}, []
    for name, architecture, seed, reference, shuffled, hidden in jobs:
        label = f"{name}_seed{seed}"
        started = time.monotonic()
        print(f"JOB_START {label}", flush=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = train(dataset=dataset, output=output / "runs" / label, architecture=architecture,
                       train_sample_id_file=train_sample_id_file, validation_sample_id_file=validation_sample_id_file,
                       timing_reference=reference, validation_cache=validation_cache, device=device,
                       epochs=epochs, batch_size=32, core_size=128, context_size=16,
                       hidden_dim=hidden, seed=seed, patience=epochs, target_shuffle=shuffled, cpu_threads=4)
        report = audit(dataset=dataset, checkpoint=Path(result["output"]) / "checkpoint_best.pt",
                       output=output / "audits" / label, sample_id_file=validation_sample_id_file,
                       validation_cache=validation_cache, timing_reference=reference, device=device, seed=seed)
        reports[label] = report
        runs.append({**result, "label": label, "seed": seed,
                     "elapsed_seconds": time.monotonic() - started,
                     "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
                     "timing": timing_summary(report)})
        print(f"JOB_END {label} {runs[-1]['elapsed_seconds']:.2f}s", flush=True)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    summary = {"status": "complete", "production_approved": False, "test_read": False,
               "split": "selected train/validation", "runs": runs,
               "gru_minus_mlp": paired_work_comparison(
                   [reports[f"mlp_tempo_seed{s}"] for s in seeds],
                   [reports[f"gru_tempo_seed{s}"] for s in seeds]),
               "tempo_minus_no_tempo_mlp_single_seed": paired_work_comparison(
                   [reports[f"mlp_no_tempo_seed{seeds[0]}"]], [reports[f"mlp_tempo_seed{seeds[0]}"]]),
               "gru_minus_shuffled_single_seed": paired_work_comparison(
                   [reports[f"gru_tempo_shuffled_seed{seeds[0]}"]], [reports[f"gru_tempo_seed{seeds[0]}"]]),
               "gru_minus_capacity_mlp_single_seed": paired_work_comparison(
                   [reports[f"mlp_tempo_capacity128_seed{seeds[0]}"]], [reports[f"gru_tempo_seed{seeds[0]}"]]),
               "capacity_note": "same inputs/windows/epoch budget, not equal parameters; additional MLP128 control is one seed only"}
    (output / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "_INCOMPLETE").unlink()
    (output / "_SUCCESS").touch()
    return seal_pilot(output, timing_reference)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "output", "timing-reference", "train-sample-id-file", "validation-sample-id-file", "validation-cache"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result = run_pilot(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
