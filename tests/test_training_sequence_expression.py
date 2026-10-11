from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from training.audit_sequence_expression import audit, load_model
from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.sequence_expression import (
    collate_windows, fit_input_normalizers, make_model, make_windows, predict_sample,
    prepare_sample, shuffle_training_targets,
)
from training.train_expression import TrainingRunError
from training.train_sequence_expression import _run_epoch, train


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _arrays(chords: int = 7) -> dict[str, np.ndarray]:
    mapping = np.repeat(np.arange(chords), 2)
    note = np.zeros((len(mapping), 15), dtype=np.float32)
    chord = np.zeros((chords, 11), dtype=np.float32)
    note[:, 0] = np.tile([-1.0, 1.0], chords)
    note[:, 3] = mapping / 2.0
    chord[:, 0] = np.arange(chords) / 2.0
    return {
        "note_features": note, "chord_features": chord, "note_to_chord": mapping,
        "note_targets": np.arange(len(mapping) * 6, dtype=np.float32).reshape(-1, 6) / 100.0,
        "chord_targets": np.arange(chords * 2, dtype=np.float32).reshape(-1, 2) / 100.0,
        "note_masks": np.ones((len(mapping), 6), dtype=bool),
        "chord_masks": np.ones((chords, 2), dtype=bool),
    }


def _config(sample, architecture="gru"):
    return {
        "architecture": architecture, "hidden_dim": 8, "dropout": 0.0,
        "core_size": 3, "context_size": 1, "batch_size": 2,
        "target_shuffle": False, "relative_position": True,
        "note_target_means": [0.0] * 6, "note_target_scales": [1.0] * 6,
        "chord_target_means": [0.0] * 2, "chord_target_scales": [1.0] * 2,
        "onset_weight": 0.1, "ioi_weight": 1.0, "duration_weight": 1.0,
        "velocity_weight": 1.0, "pedal_weight": 1.0,
        **fit_input_normalizers([sample]),
    }


def test_core_windows_cover_events_once_and_never_escape_sample():
    sample = prepare_sample("a", _arrays())
    windows = list(make_windows(sample, 3, 2))
    chord_counts = np.zeros(7)
    note_counts = np.zeros(14)
    for window in windows:
        assert 0 <= window.start <= window.core_start < window.core_end <= window.end <= 7
        chord_counts[window.core_start:window.core_end] += 1
        ni = (sample.note_to_chord >= window.core_start) & (sample.note_to_chord < window.core_end)
        note_counts[ni] += 1
    assert np.all(chord_counts == 1)
    assert np.all(note_counts == 1)


def test_padding_and_context_never_supervise_and_note_mapping_is_batch_local():
    a = prepare_sample("a", _arrays(7))
    b = prepare_sample("b", _arrays(1))
    windows = [list(make_windows(a, 3, 2))[1], list(make_windows(b, 3, 2))[0]]
    batch = collate_windows(windows, _config(a))
    assert batch["lengths"].tolist() == [6, 1]
    assert batch["chord_masks"][0, :2].sum() == 0
    assert batch["chord_masks"][0, 5:].sum() == 0
    assert batch["chord_masks"][1, 1:].sum() == 0
    assert batch["note_to_chord"].tolist() == [2, 2, 3, 3, 4, 4, 6, 6]


def test_short_and_single_chord_samples_predict_all_events_once():
    sample = prepare_sample("short", _arrays(1))
    config = _config(sample)
    model = make_model(config)
    note, chord = predict_sample(model, sample, config=config)
    assert note.shape == (2, 6)
    assert chord.shape == (1, 2)
    assert np.isfinite(note).all() and np.isfinite(chord).all()


@pytest.mark.parametrize("architecture", ["mlp", "gru"])
def test_padded_batch_cannot_change_real_prediction(architecture):
    short = prepare_sample("short", _arrays(1))
    long = prepare_sample("long", _arrays(7))
    config = _config(long, architecture)
    model = make_model(config).eval()
    short_window = list(make_windows(short))[0]
    long_window = list(make_windows(long))[0]
    with torch.inference_mode():
        a_note, a_chord = model(collate_windows([short_window], config))
        b_note, b_chord = model(collate_windows([short_window, long_window], config))
    assert torch.allclose(a_chord[0, 0], b_chord[0, 0], atol=1e-6)
    assert torch.allclose(a_note, b_note[:2], atol=1e-6)


def test_only_allowlisted_score_timing_can_change_features():
    arrays = _arrays()
    reference = {
        "score_seconds_per_beat_reference": np.asarray(0.5),
        "chord_tempo_log_ratio": np.zeros(7),
        "note_duration_tempo_log_ratio": np.zeros(14),
        "global_tempo_scale": np.asarray(99.0),
        "chord_performance_seconds": np.arange(7) * 1000.0,
        "note_to_chord": arrays["note_to_chord"],
    }
    a = prepare_sample("a", arrays, timing_reference=reference)
    changed = dict(reference, global_tempo_scale=np.asarray(0.01), chord_performance_seconds=np.arange(7) * -5.0)
    b = prepare_sample("a", arrays, timing_reference=changed)
    assert a.note_features.shape == (14, 17)
    assert a.chord_features.shape == (7, 13)
    assert np.array_equal(a.note_features, b.note_features)
    assert np.array_equal(a.chord_features, b.chord_features)


def test_native_mask_and_mapping_reference_mismatch_is_rejected():
    arrays = _arrays(1)
    reference = {"score_seconds_per_beat_reference": 0.5,
                 "chord_tempo_log_ratio": np.zeros(1), "note_duration_tempo_log_ratio": np.zeros(2),
                 "note_native_masks": np.zeros((2, 6), dtype=bool)}
    with pytest.raises(TrainingRunError, match="native masks mismatch"):
        prepare_sample("a", arrays, timing_reference=reference)
    arrays["note_to_chord"][0] = 1
    with pytest.raises(TrainingRunError, match="escapes sample"):
        prepare_sample("a", arrays)


def test_shuffle_keeps_target_mask_pairs_and_preserves_inputs():
    arrays = _arrays()
    arrays["note_masks"][::2, 0] = False
    arrays["chord_masks"][::2, 0] = False
    sample = prepare_sample("a", arrays)
    shuffled = shuffle_training_targets(sample, np.random.default_rng(123))
    for kind in ("note", "chord"):
        before = np.column_stack((getattr(sample, f"{kind}_targets"), getattr(sample, f"{kind}_masks")))
        after = np.column_stack((getattr(shuffled, f"{kind}_targets"), getattr(shuffled, f"{kind}_masks")))
        assert sorted(map(tuple, before)) == sorted(map(tuple, after))
    assert shuffled.note_features is sample.note_features
    assert shuffled.chord_features is sample.chord_features
    assert np.array_equal(shuffled.note_to_chord, sample.note_to_chord)
    assert not np.array_equal(shuffled.note_targets, sample.note_targets)


def test_validation_never_applies_training_negative_control():
    sample = prepare_sample("a", _arrays())
    config = _config(sample, "mlp")
    model = make_model(config)
    normal = _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=None, seed=3)
    config["target_shuffle"] = True
    shuffled_option = _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=None, seed=3)
    assert normal == shuffled_option
    assert not shuffled_option["target_shuffle_applied"]


def test_tiny_cpu_model_can_fit_supervised_data():
    torch.manual_seed(51)
    sample = prepare_sample("tiny", _arrays(5))
    config = _config(sample, "gru")
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    before = _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=None, seed=1)
    for epoch in range(25):
        _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=optimizer, seed=epoch)
    after = _run_epoch(model, [sample], config, device=torch.device("cpu"), optimizer=None, seed=1)
    assert after["loss"] < before["loss"] * 0.4
    assert after["target_counts"] == before["target_counts"]


def test_contexts_reset_at_sample_edges_and_keep_score_order():
    a = prepare_sample("a", _arrays(7))
    other_arrays = _arrays(2)
    other_arrays["chord_features"][:, 0] += 1000.0
    other_arrays["note_features"][:, 3] += 1000.0
    b = prepare_sample("b", other_arrays)
    config = _config(a)
    first = list(make_windows(a, 3, 16))[0]
    last = list(make_windows(b, 3, 16))[0]
    batch = collate_windows([first, last], config)
    cm = torch.tensor(config["chord_input_means"])
    cs = torch.tensor(config["chord_input_scales"])
    unnormalized = batch["chord_features"] * cs + cm
    assert torch.allclose(unnormalized[0, :7], torch.from_numpy(a.chord_features).to(unnormalized.dtype))
    assert torch.allclose(unnormalized[1, :2], torch.from_numpy(b.chord_features).to(unnormalized.dtype))
    assert a.chord_features[0, 0] == b.chord_features[0, 0] == 0.0
    assert batch["records"][0][0] == "a" and batch["records"][1][0] == "b"


def _write_dataset(root: Path):
    root.mkdir()
    (root / "_SUCCESS").write_text("synthetic test dataset\n", encoding="utf-8")
    manifest = []
    for split, sample_id in (("train", "train-a"), ("validation", "val-b")):
        arrays = _arrays(5)
        path = root / "shards" / f"split={split}" / "source=demo" / "part-00000.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, **arrays, sample_ids=np.asarray([sample_id]),
                 sample_note_offsets=np.asarray([0, 10], dtype=np.int64),
                 sample_chord_offsets=np.asarray([0, 5], dtype=np.int64))
        manifest.append({"id": sample_id, "split": split, "performance_dataset": "demo",
                         "piece_group": sample_id, "shard_path": path.relative_to(root).as_posix()})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in manifest), encoding="utf-8")
    normalizer = {"fit_split": "train",
                  "note_targets": {name: {"mean": 0.0, "std": 1.0} for name in NOTE_TARGET_NAMES},
                  "chord_targets": {name: {"mean": 0.0, "std": 1.0} for name in CHORD_TARGET_NAMES}}
    (root / "normalizer.json").write_text(json.dumps(normalizer), encoding="utf-8")
    cache = root.parent / "validation_cache.json"
    cache.write_text(json.dumps({"validation_report": {"status": "valid", "dataset_root": str(root.resolve())},
                                 "dataset_success_sha256": hashlib.sha256((root / "_SUCCESS").read_bytes()).hexdigest()}), encoding="utf-8")
    train_ids, val_ids = root.parent / "train_ids.txt", root.parent / "val_ids.txt"
    train_ids.write_text("train-a\n", encoding="utf-8")
    val_ids.write_text("val-b\n", encoding="utf-8")
    return cache, train_ids, val_ids


def test_cpu_train_checkpoint_and_validation_audit_artifact_contract(tmp_path):
    dataset = tmp_path / "dataset"
    cache, train_ids, val_ids = _write_dataset(dataset)
    output = tmp_path / "run"
    result = train(dataset=dataset, output=output, architecture="gru", train_sample_id_file=train_ids,
                   validation_sample_id_file=val_ids, validation_cache=cache, device="cpu",
                   epochs=1, batch_size=2, core_size=3, context_size=1, hidden_dim=8, dropout=0.0,
                   cpu_threads=2, target_shuffle=True)
    assert result["status"] == "complete"
    assert (output / "_SUCCESS").is_file()
    assert not (output / "_INCOMPLETE").exists()
    model, config = load_model(output / "checkpoint_best.pt")
    assert config["input_fit_split"] == "selected train only"
    assert config["timing_reference"] is None
    assert config["quality_mask"] is None
    audited = audit(dataset=dataset, checkpoint=output / "checkpoint_best.pt", output=tmp_path / "audit",
                    sample_id_file=val_ids, validation_cache=cache, device="cpu")
    assert audited["sample_id_count"] == 1
    assert "model:micro_offset_ms" in audited["work_values"]
    assert "identity:micro_offset_ms" in audited["work_values"]
    path = tmp_path / "audit" / audited["prediction_files"]["val-b"]
    with np.load(path, allow_pickle=False) as loaded:
        assert loaded["note_prediction"].shape == (10, 6)
        assert loaded["chord_prediction"].shape == (5, 2)
    with pytest.raises(TrainingRunError, match="must be empty"):
        train(dataset=dataset, output=output, architecture="mlp", train_sample_id_file=train_ids,
              validation_sample_id_file=val_ids, validation_cache=cache, device="cpu")
    checkpoint = output / "checkpoint_best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(TrainingRunError, match="differs from run integrity"):
        load_model(checkpoint)
