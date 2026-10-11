"""Synthetic registration/inventory checks; never train or operate a GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from training import run_playback_adoption_pilot as pilot
from training.train_expression import _sha256


@pytest.fixture
def registered(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "_SUCCESS").write_text("synthetic dataset success\n")
    (dataset / "manifest.jsonl").write_text('{"id":"demo","split":"validation"}\n')
    (dataset / "normalizer.json").write_text('{"synthetic":true}\n')
    timing = tmp_path / "timing"
    timing.mkdir()
    (timing / "summary.json").write_text('{"synthetic":true}\n')
    (timing / "_SUCCESS").write_text(_sha256(timing / "summary.json") + "\n")
    cache = tmp_path / "validation-cache.json"
    cache.write_text('{"synthetic":true}\n')
    train_ids, val_ids = tmp_path / "train.txt", tmp_path / "val.txt"
    train_ids.write_text("train-work\n")
    val_ids.write_text("val-work\n")
    sources = tmp_path / "source-code"
    sources.mkdir()
    for name in pilot.SOURCE_NAMES:
        (sources / name).write_text(f"# synthetic source fixture: {name}\n")
    registration = pilot.build_registration(
        dataset=dataset, timing_reference=timing, train_sample_id_file=train_ids,
        validation_sample_id_file=val_ids, validation_cache=cache, epochs=1,
        device="cpu", source_root=sources,
    )
    return registration


def _config(registration, job):
    files = registration["file_bindings"]
    return {
        **{name: registration[name] for name in pilot.TRAIN_OPTION_NAMES},
        **{name: registration[name] for name in ("device", "relative_position", "quality_mask")},
        **{name: job[name] for name in ("architecture", "seed", "cumulative_weight")},
        "torch_version": registration["runtime"]["torch_version"],
        "train_sample_id_file": files["train_selection"]["path"],
        "validation_sample_id_file": files["validation_selection"]["path"],
        "train_sample_id_file_sha256": registration["selection"]["train_sha256"],
        "validation_sample_id_file_sha256": registration["selection"]["validation_sha256"],
        "dataset_root": registration["dataset_root"],
        "timing_reference": registration["timing_reference"],
        "dataset_validation_cache": registration["validation_cache"],
        "dataset_success_sha256": files["dataset_success"]["sha256"],
        "timing_reference_sha256": registration["timing_reference_sha256"],
        "dataset_validation_cache_sha256": files["validation_cache"]["sha256"],
        "implementation_sha256": {name: registration["implementation_sha256"][name]
                                  for name in pilot.TRAIN_IMPLEMENTATION_NAMES},
    }


def _write_run(root, registration, job):
    root.mkdir(parents=True)
    (root / "run_config.json").write_text(json.dumps(_config(registration, job)))
    (root / "checkpoint_best.pt").write_bytes(b"synthetic test checkpoint, not a model")
    pilot.write_integrity(root)


def _report(root, registration, job):
    return {
        "status": "complete", "split": "validation", "seed": job["seed"],
        "architecture": job["architecture"], "target_shuffle_training": False,
        "checkpoint": str((root / "checkpoint_best.pt").resolve()),
        "checkpoint_sha256": _sha256(root / "checkpoint_best.pt"),
        "sample_id_file_sha256": registration["selection"]["validation_sha256"],
        "timing_reference_sha256": registration["timing_reference_sha256"],
        "dataset_root": registration["dataset_root"],
        "work_values": {"model:log_ioi_ratio": {f"work-{index}": 0.1 for index in range(6)}},
        "timing_metrics": {},
    }


def test_v2_registration_binds_full_inputs_implementations_and_explicit_defaults(registered):
    r = registered
    assert r["schema_version"] == 2
    assert len(r["jobs"]) == 12
    assert r["cumulative_spans"] == [1, 8, 32]
    assert r["cumulative_log_ratio_clip"] == 4
    assert r["patience"] == r["epochs"] == 1
    assert r["target_shuffle"] is False
    assert r["quality_mask"] is None
    assert set(r["implementation_sha256"]) == set(pilot.SOURCE_NAMES)
    assert set(r["file_bindings"]) == {
        "dataset_success", "dataset_manifest", "dataset_normalizer", "timing_summary", "timing_success",
        "validation_cache", "train_selection", "validation_selection",
        *{f"source:{name}" for name in pilot.SOURCE_NAMES},
    }
    pilot.verify_registration_bindings(r)


@pytest.mark.parametrize("binding", ["validation_cache", "dataset_manifest", "dataset_normalizer", "timing_summary",
                                      "timing_success", "train_selection", "source:rhythm_objective.py"])
def test_changed_input_or_source_rejected(registered, binding):
    path = Path(registered["file_bindings"][binding]["path"])
    path.write_text("changed during pilot\n")
    with pytest.raises(ValueError, match="registered binding changed"):
        pilot.verify_registration_bindings(registered)


def test_duplicate_label_or_duplicate_factor_rejected(registered):
    for change in ("same-label", "same-factor"):
        copy_r = copy.deepcopy(registered)
        duplicate = copy.deepcopy(copy_r["jobs"][0])
        if change == "same-factor":
            duplicate["label"] += "-other-name"
        copy_r["jobs"].append(duplicate)
        with pytest.raises(ValueError, match="duplicate"):
            pilot.verify_registration_bindings(copy_r)


@pytest.mark.parametrize("field,value", [
    ("seed", 0), ("architecture", "gru"), ("cumulative_weight", 1),
    ("cumulative_spans", [1, 4, 16]), ("cumulative_log_ratio_clip", 2),
    ("dataset_success_sha256", "changed"), ("timing_reference_sha256", "changed"),
    ("dataset_validation_cache_sha256", "changed"),
    ("train_sample_id_file_sha256", "changed"), ("dropout", 0.8),
    ("implementation_sha256", {}), ("target_shuffle", True),
])
def test_child_actual_seed_option_or_binding_change_rejected(registered, tmp_path, field, value):
    root = tmp_path / "child"
    job = registered["jobs"][0]
    _write_run(root, registered, job)
    pilot.verify_registered_job(root, job, registered)
    config = json.loads((root / "run_config.json").read_text())
    config[field] = value
    (root / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="child changed"):
        pilot.verify_registered_job(root, job, registered)


def test_report_cannot_point_at_a_different_checkpoint_or_seed(registered, tmp_path):
    root = tmp_path / "child"
    job = registered["jobs"][0]
    _write_run(root, registered, job)
    report = _report(root, registered, job)
    pilot.verify_registered_job(root, job, registered, report=report)
    report["seed"] = 0
    with pytest.raises(ValueError, match="child audit"):
        pilot.verify_registered_job(root, job, registered, report=report)
    report = _report(root, registered, job)
    other = tmp_path / "different-checkpoint.pt"
    other.write_bytes((root / "checkpoint_best.pt").read_bytes())
    report["checkpoint"] = str(other)
    with pytest.raises(ValueError, match="child audit"):
        pilot.verify_registered_job(root, job, registered, report=report)


def _fake_jobs(monkeypatch, registration, *, mutate=None):
    calls = {"train": 0, "audit": 0}
    monkeypatch.setattr(pilot, "build_registration", lambda **kwargs: registration)

    def fake_train(**kwargs):
        calls["train"] += 1
        root = Path(kwargs["output"])
        job = next(row for row in registration["jobs"] if row["label"] == root.name)
        for field in pilot.TRAIN_OPTION_NAMES:
            assert kwargs[field] == registration[field]
        _write_run(root, registration, job)
        if mutate is not None:
            mutate(root)
        return {"status": "complete", "output": str(root), "best_epoch": 1,
                "best_validation_loss": 0.1, "parameter_count": 0}

    def fake_audit(**kwargs):
        calls["audit"] += 1
        root, output = Path(kwargs["checkpoint"]).parent, Path(kwargs["output"])
        job = next(row for row in registration["jobs"] if row["label"] == root.name)
        report = _report(root, registration, job)
        output.mkdir(parents=True)
        (output / "report.json").write_text(json.dumps(report))
        pilot.write_integrity(output)
        return report

    monkeypatch.setattr(pilot, "train", fake_train)
    monkeypatch.setattr(pilot, "audit", fake_audit)
    return calls


def _run(registration, output):
    files = registration["file_bindings"]
    return pilot.run_pilot(dataset=Path(registration["dataset_root"]), output=output,
                           timing_reference=Path(registration["timing_reference"]),
                           train_sample_id_file=Path(files["train_selection"]["path"]),
                           validation_sample_id_file=Path(files["validation_selection"]["path"]),
                           validation_cache=Path(registration["validation_cache"]), epochs=1, device="cpu")


def test_future_pilot_seals_bound_source_snapshots_without_real_training(registered, tmp_path, monkeypatch):
    calls = _fake_jobs(monkeypatch, registered)
    output = tmp_path / "new-pilot"
    result = _run(registered, output)
    assert calls == {"train": 12, "audit": 12}
    assert result["registration_schema_version"] == 2
    assert result["registration_binding_checked"]
    assert not result["production_approved"]
    assert not result["test_read"]
    assert not (output / "_INCOMPLETE").exists()
    pilot.verify_inventory(output)
    pilot.verify_registration_bindings(registered, output=output,
                                       registration_sha256=result["preregistration_sha256"])
    snapshot = output / registered["source_snapshots"]["rhythm_objective.py"]
    snapshot.write_text("altered snapshot\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        pilot.verify_registration_bindings(registered, output=output,
                                           registration_sha256=result["preregistration_sha256"])


def test_changed_cache_after_first_job_stops_before_audit_and_never_seals(registered, tmp_path, monkeypatch):
    cache = Path(registered["validation_cache"])
    calls = _fake_jobs(monkeypatch, registered, mutate=lambda root: cache.write_text("changed\n"))
    output = tmp_path / "failed-pilot"
    with pytest.raises(ValueError, match="binding changed: validation_cache"):
        _run(registered, output)
    assert calls == {"train": 1, "audit": 0}
    assert (output / "_INCOMPLETE").exists()
    assert not (output / "_SUCCESS").exists()


def test_preregistration_mutation_stops_future_job(registered, tmp_path, monkeypatch):
    def mutate(root):
        (root.parents[1] / "preregistration.json").write_text("{}")
    calls = _fake_jobs(monkeypatch, registered, mutate=mutate)
    output = tmp_path / "failed-pilot"
    with pytest.raises(ValueError, match="frozen preregistration changed"):
        _run(registered, output)
    assert calls == {"train": 1, "audit": 0}
    assert not (output / "_SUCCESS").exists()


def test_in_memory_registration_cannot_change_options_between_jobs(registered, tmp_path, monkeypatch):
    def mutate(root):
        registered["cumulative_spans"].append(64)
    calls = _fake_jobs(monkeypatch, registered, mutate=mutate)
    output = tmp_path / "failed-pilot"
    with pytest.raises(ValueError, match="in-memory frozen preregistration changed"):
        _run(registered, output)
    assert calls == {"train": 1, "audit": 0}
    assert not (output / "_SUCCESS").exists()
