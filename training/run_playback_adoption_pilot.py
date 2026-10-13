"""Pre-registered timing/hold pilots; offline only, no deployment or test split."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from training.audit_sequence_expression import audit
from training.playback_adoption import POLICY, verify_inventory
from training.run_sequence_pilot import paired_work_comparison, timing_summary
from training.train_expression import _sha256, _write_json
from training.train_sequence_expression import reference_hash, train, write_integrity


SOURCE_NAMES = (
    "run_playback_adoption_pilot.py", "playback_adoption.py", "sequence_expression.py",
    "train_sequence_expression.py", "rhythm_objective.py", "audit_sequence_expression.py",
    "run_sequence_pilot.py", "train_expression.py", "tempo_reference.py", "midi_features.py",
    "audit_expression.py", "compare_quality_multiseed.py", "validate_pianocore_dataset.py",
    "pianocore_dataset.py",
)
TRAIN_IMPLEMENTATION_NAMES = ("sequence_expression.py", "train_sequence_expression.py", "rhythm_objective.py")
TRAIN_OPTION_NAMES = (
    "epochs", "batch_size", "core_size", "context_size", "hidden_dim", "dropout",
    "learning_rate", "weight_decay", "patience", "cpu_threads", "target_shuffle",
    "max_steps_per_epoch", "micro_weight", "onset_weight", "duration_weight",
    "ioi_weight", "velocity_weight", "pedal_weight", "cumulative_spans", "cumulative_log_ratio_clip",
)


def _validate_jobs(jobs: list[dict]) -> None:
    if not jobs:
        raise ValueError("registration must contain jobs")
    labels, factors = set(), set()
    for job in jobs:
        label = job["label"]
        factor = (job["architecture"], job["seed"], job["cumulative_weight"])
        if (label in labels or factor in factors or not isinstance(label, str)
                or not label or Path(label).name != label or "/" in label or "\\" in label
                or label in {".", ".."}):
            raise ValueError("duplicate or unsafe registered job")
        labels.add(label)
        factors.add(factor)


def build_registration(*, dataset: Path, timing_reference: Path, train_sample_id_file: Path,
                       validation_sample_id_file: Path, validation_cache: Path,
                       epochs: int, device: str, source_root: Path | None = None) -> dict:
    """Freeze every input fingerprint and training option before starting a job.

    Source snapshots and bindings are reproducibility evidence, not signatures
    and not a new authorization to deploy a model or read the test split.
    """
    if epochs <= 0 or device not in {"cpu", "cuda"}:
        raise ValueError("positive epochs and explicit cpu/cuda device required")
    source_root = (source_root or Path(__file__).parent).resolve()
    dataset, timing_reference, validation_cache = dataset.resolve(), timing_reference.resolve(), validation_cache.resolve()
    train_sample_id_file, validation_sample_id_file = train_sample_id_file.resolve(), validation_sample_id_file.resolve()
    seeds = (20261003, 20261004, 20261005)
    jobs = [{"label": f"{architecture}_hold_ioi_cum{cumulative:g}_seed{seed}",
             "architecture": architecture, "seed": seed, "cumulative_weight": cumulative}
            for seed in seeds for architecture in ("mlp", "gru") for cumulative in (0.0, 0.5)]
    _validate_jobs(jobs)
    files = {
        "dataset_success": dataset / "_SUCCESS", "dataset_manifest": dataset / "manifest.jsonl",
        "dataset_normalizer": dataset / "normalizer.json",
        "timing_summary": timing_reference / "summary.json", "timing_success": timing_reference / "_SUCCESS",
        "validation_cache": validation_cache, "train_selection": train_sample_id_file,
        "validation_selection": validation_sample_id_file,
        **{f"source:{name}": source_root / name for name in SOURCE_NAMES},
    }
    bindings = {name: {"path": str(path), "sha256": _sha256(path)} for name, path in files.items()}
    return {
        "schema_version": 2, "kind": "timing_hold_adoption_pilot",
        "policy": json.loads(json.dumps(POLICY)), "jobs": jobs,
        "epochs": epochs, "device": device, "quality_mask": None, "test_read": False,
        "micro_weight": 0.0, "onset_weight": 0.0, "duration_weight": 1.0,
        "ioi_weight": 1.0, "velocity_weight": 0.0, "pedal_weight": 0.0,
        "batch_size": 32, "core_size": 128, "context_size": 16, "hidden_dim": 64,
        "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001,
        "patience": epochs, "cpu_threads": 4, "target_shuffle": False,
        "max_steps_per_epoch": None, "relative_position": True,
        "cumulative_spans": [1, 8, 32], "cumulative_log_ratio_clip": 4.0,
        "dataset_root": str(dataset), "timing_reference": str(timing_reference),
        "timing_reference_sha256": reference_hash(timing_reference),
        "validation_cache": str(validation_cache),
        "selection": {"train_sha256": bindings["train_selection"]["sha256"],
                      "validation_sha256": bindings["validation_selection"]["sha256"]},
        "file_bindings": bindings,
        "implementation_sha256": {name: bindings[f"source:{name}"]["sha256"] for name in SOURCE_NAMES},
        "source_snapshots": {name: f"source_snapshot/training/{name}" for name in SOURCE_NAMES},
        "runtime": {"torch_version": str(torch.__version__)},
        "budget_note": "all 12 jobs have identical inputs/budget; objective and architecture factors are separate",
        "registration_note": "v2 pre-job binding; never rewrites or upgrades earlier v1 pilot evidence",
    }


def verify_registration_bindings(registration: dict, *, output: Path | None = None,
                                 registration_sha256: str | None = None) -> None:
    """Fail closed when any frozen input/source/snapshot changes mid-pilot."""
    if registration.get("schema_version") != 2:
        raise ValueError("expected v2 pilot registration")
    _validate_jobs(registration["jobs"])
    for name, binding in registration["file_bindings"].items():
        path = Path(binding["path"])
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise ValueError(f"registered binding changed: {name}")
    if output is not None:
        if registration_sha256 is None or _sha256(output / "preregistration.json") != registration_sha256:
            raise ValueError("frozen preregistration changed")
        if json.loads((output / "preregistration.json").read_text(encoding="utf-8")) != registration:
            raise ValueError("in-memory frozen preregistration changed")
        for name, relative in registration["source_snapshots"].items():
            path = output / relative
            if not path.resolve().is_relative_to(output.resolve()) or not path.is_file():
                raise ValueError("registered source snapshot missing or unsafe")
            if _sha256(path) != registration["implementation_sha256"][name]:
                raise ValueError(f"registered source snapshot changed: {name}")


def verify_registered_job(run_root: Path, job: dict, registration: dict,
                          *, report: dict | None = None) -> None:
    """Check actual child configuration/seed, never just a requested job label."""
    config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    for key in ("architecture", "seed", "cumulative_weight"):
        if config.get(key) != job[key]:
            raise ValueError(f"child changed registered job {key}")
    for key in TRAIN_OPTION_NAMES + ("device", "relative_position", "quality_mask"):
        if config.get(key) != registration[key]:
            raise ValueError(f"child changed registered option {key}")
    if str(config.get("torch_version")) != registration["runtime"]["torch_version"]:
        raise ValueError("child torch runtime changed")
    files = registration["file_bindings"]
    for kind in ("train", "validation"):
        if (config.get(f"{kind}_sample_id_file_sha256") != registration["selection"][f"{kind}_sha256"]
                or Path(config[f"{kind}_sample_id_file"]).resolve() != Path(files[f"{kind}_selection"]["path"])):
            raise ValueError("child changed registered selection")
    for config_key, registered_key in (("dataset_root", "dataset_root"),
                                       ("timing_reference", "timing_reference"),
                                       ("dataset_validation_cache", "validation_cache")):
        if Path(config[config_key]).resolve() != Path(registration[registered_key]):
            raise ValueError(f"child changed registered path {config_key}")
    for config_key, expected in (
        ("dataset_success_sha256", files["dataset_success"]["sha256"]),
        ("timing_reference_sha256", registration["timing_reference_sha256"]),
        ("dataset_validation_cache_sha256", files["validation_cache"]["sha256"]),
    ):
        if config.get(config_key) != expected:
            raise ValueError(f"child changed registered fingerprint {config_key}")
    expected_implementation = {name: registration["implementation_sha256"][name] for name in TRAIN_IMPLEMENTATION_NAMES}
    # the trainer records more implementation files than the pilot binds, so compare the bound
    # names rather than the whole dictionary -- full equality fails any checkpoint that records
    # extra files without any of the bound ones having changed
    recorded_implementation = config.get("implementation_sha256") or {}
    if not all(recorded_implementation.get(name) == digest for name, digest in expected_implementation.items()):
        raise ValueError("child changed registered training implementation")
    if report is not None:
        checkpoint = (run_root / "checkpoint_best.pt").resolve()
        if (report.get("status") != "complete" or report.get("split") != "validation"
                or report.get("seed") != job["seed"] or report.get("architecture") != job["architecture"]
                or report.get("target_shuffle_training") is not False
                or Path(report["checkpoint"]).resolve() != checkpoint
                or report.get("checkpoint_sha256") != _sha256(checkpoint)
                or report.get("sample_id_file_sha256") != registration["selection"]["validation_sha256"]
                or report.get("timing_reference_sha256") != registration["timing_reference_sha256"]
                or Path(report["dataset_root"]).resolve() != Path(registration["dataset_root"])):
            raise ValueError("child audit differs from registered run/selection")


def run_pilot(*, dataset: Path, output: Path, timing_reference: Path,
              train_sample_id_file: Path, validation_sample_id_file: Path,
              validation_cache: Path, epochs: int = 5, device: str = "cuda") -> dict:
    output = output.resolve()
    if output.exists():
        raise ValueError("pilot output must be a new directory")
    registration = build_registration(dataset=dataset, timing_reference=timing_reference,
                                      train_sample_id_file=train_sample_id_file,
                                      validation_sample_id_file=validation_sample_id_file,
                                      validation_cache=validation_cache, epochs=epochs, device=device)
    jobs = registration["jobs"]
    seeds = sorted({job["seed"] for job in jobs})
    verify_registration_bindings(registration)
    output.mkdir(parents=True)
    (output / "_INCOMPLETE").touch()
    _write_json(output / "preregistration.json", registration)
    registration_sha256 = _sha256(output / "preregistration.json")
    for name, relative in registration["source_snapshots"].items():
        snapshot = output / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(registration["file_bindings"][f"source:{name}"]["path"], snapshot)
    verify_registration_bindings(registration, output=output, registration_sha256=registration_sha256)
    (output / "logs").mkdir()
    runs, reports = [], {}
    for job in jobs:
        verify_registration_bindings(registration, output=output, registration_sha256=registration_sha256)
        label = job["label"]
        print(f"JOB_START {label}", flush=True)
        started = time.monotonic()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with (output / "logs" / f"{label}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            result = train(dataset=dataset, output=output / "runs" / label, architecture=job["architecture"],
                           train_sample_id_file=train_sample_id_file, validation_sample_id_file=validation_sample_id_file,
                           timing_reference=timing_reference, validation_cache=validation_cache, device=device,
                           seed=job["seed"], cumulative_weight=job["cumulative_weight"],
                           **{name: registration[name] for name in TRAIN_OPTION_NAMES})
            verify_registration_bindings(registration, output=output, registration_sha256=registration_sha256)
            verify_inventory(output / "runs" / label)
            verify_registered_job(output / "runs" / label, job, registration)
            report = audit(dataset=dataset, checkpoint=Path(result["output"]) / "checkpoint_best.pt",
                           output=output / "audits" / label, sample_id_file=validation_sample_id_file,
                           validation_cache=validation_cache, timing_reference=timing_reference,
                           device=device, seed=job["seed"])
        verify_registration_bindings(registration, output=output, registration_sha256=registration_sha256)
        verify_inventory(output / "audits" / label)
        verify_registered_job(output / "runs" / label, job, registration, report=report)
        reports[label] = report
        runs.append({**result, **job, "elapsed_seconds": time.monotonic() - started,
                     "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
                     "timing": timing_summary(report)})
        print(f"JOB_END {label} {runs[-1]['elapsed_seconds']:.2f}s", flush=True)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    contrasts = {}
    for cumulative in (0.0, 0.5):
        contrasts[f"gru_minus_mlp_cum{cumulative:g}"] = paired_work_comparison(
            [reports[f"mlp_hold_ioi_cum{cumulative:g}_seed{s}"] for s in seeds],
            [reports[f"gru_hold_ioi_cum{cumulative:g}_seed{s}"] for s in seeds])
    for architecture in ("mlp", "gru"):
        contrasts[f"cumulative_minus_local_{architecture}"] = paired_work_comparison(
            [reports[f"{architecture}_hold_ioi_cum0_seed{s}"] for s in seeds],
            [reports[f"{architecture}_hold_ioi_cum0.5_seed{s}"] for s in seeds])
    summary = {"status": "complete", "production_approved": False, "test_read": False,
               "registration_schema_version": 2, "registration_binding_checked": True,
               "runs": runs, "raw_target_contrasts": contrasts,
               "preregistration_sha256": registration_sha256,
               "child_integrity_sha256": {f"{kind}/{job['label']}": verify_inventory(output / kind / job["label"])
                                          for job in jobs for kind in ("runs", "audits")},
               "loss_comparison_note": "different objectives produce different losses; compare physical metrics, not loss magnitudes"}
    verify_registration_bindings(registration, output=output, registration_sha256=registration_sha256)
    for job in jobs:
        verify_registered_job(output / "runs" / job["label"], job, registration, report=reports[job["label"]])
    _write_json(output / "pilot_summary.json", summary)
    write_integrity(output)
    (output / "_INCOMPLETE").unlink()
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "timing-reference", "train-sample-id-file", "validation-sample-id-file", "validation-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result = run_pilot(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "production_approved": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
