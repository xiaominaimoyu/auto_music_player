from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.shape_features import SHAPE_FEATURE_NAMES
from training.train_sequence_expression import SHAPE_FEATURE_NAMES as TRAINER_NAMES
import training.sequence_expression as se


class _Sentinel(Exception):
    pass


def _arrays(onsets: list[float], durations: list[float], pitches: list[float]) -> dict[str, np.ndarray]:
    count = len(onsets)
    note_features = np.zeros((count, 15), dtype=np.float32)
    for index, (pitch, onset, duration) in enumerate(zip(pitches, onsets, durations)):
        note_features[index, 0] = (pitch - 60.0) / 24.0
        note_features[index, 3] = onset
        note_features[index, 4] = math.log1p(duration)
    return {"note_features": note_features,
            "chord_features": np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
            "note_targets": np.zeros((count, len(NOTE_TARGET_NAMES)), dtype=np.float32),
            "chord_targets": np.zeros((2, len(CHORD_TARGET_NAMES)), dtype=np.float32),
            "note_masks": np.ones((count, len(NOTE_TARGET_NAMES)), dtype=bool),
            "chord_masks": np.ones((2, len(CHORD_TARGET_NAMES)), dtype=bool),
            "note_to_chord": np.array([0] * (count // 2 + count % 2) + [1] * (count // 2), dtype=np.int32)}


def test_the_trainer_and_the_module_agree_on_the_feature_names():
    assert TRAINER_NAMES == SHAPE_FEATURE_NAMES


def test_the_default_path_keeps_the_shipped_note_feature_width():
    arrays = _arrays([10.0, 11.0], [2.0, 1.0], [60.0, 62.0])
    legacy = se.prepare_sample("s", arrays)
    explicit = se.prepare_sample("s", arrays, shape_features=False)
    assert legacy.note_features.shape == explicit.note_features.shape == (2, 15)
    widened = se.prepare_sample("s", arrays, shape_features=True)
    assert widened.note_features.shape == (2, 15 + len(SHAPE_FEATURE_NAMES))
    # the shipped columns stay exactly where they were; only new trailing columns are added
    assert np.array_equal(widened.note_features[:, :15], legacy.note_features[:, :15])


def test_shape_columns_come_from_raw_beats_not_from_rescaled_positions():
    # chords span 10..20 while note durations are 1-2 beats: deriving after the
    # relative-position rescale would mix a 0-1 position with a raw beat duration and report
    # one voice still sounding where the score has none
    arrays = _arrays([10.0, 11.0], [2.0, 1.0], [60.0, 62.0])
    derived = se.prepare_sample("s", arrays, shape_features=True).note_features[:, 15:]
    assert derived[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(derived[:, :2], np.log1p(1.0), atol=1e-6)
    rescaled = se.prepare_sample("s", arrays, relative_position=False, shape_features=True).note_features[:, 15:]
    assert np.allclose(rescaled, derived, atol=1e-6)


def test_the_cli_flag_reaches_the_trainer(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import training.train_sequence_expression as trainer

    captured: dict[str, object] = {}
    monkeypatch.setattr(trainer, "train", lambda **kwargs: captured.update(kwargs) or {"ok": True})
    argv = ["--dataset", str(tmp_path), "--output", str(tmp_path / "out"),
            "--train-sample-id-file", str(tmp_path / "train.txt"),
            "--validation-sample-id-file", str(tmp_path / "validation.txt"),
            "--architecture", "gru", "--shape-features"]
    assert trainer.main(argv) == 0
    assert captured["shape_features"] is True
    assert trainer.main(argv[:-1]) == 0
    assert captured["shape_features"] is False


def test_inference_reads_both_target_transform_flags_back_out_of_the_checkpoint_config(
        monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}

    def _capture(sample_id, arrays, **kwargs):
        recorded.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(se, "prepare_sample", _capture)
    config = {"sample_id": "s", "relative_position": False, "note_shape_features": True,
              "duration_center_per_sample": True}
    with pytest.raises(_Sentinel):
        se.predict_sample(object(), {"note_features": np.zeros((1, 15))}, config=config)
    assert recorded["shape_features"] is True
    assert recorded["duration_centering"] is True
    assert recorded["relative_position"] is False


def test_a_checkpoint_config_without_the_keys_still_prepares_the_legacy_width(
        monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}

    def _capture(sample_id, arrays, **kwargs):
        recorded.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(se, "prepare_sample", _capture)
    with pytest.raises(_Sentinel):
        se.predict_sample(object(), {"note_features": np.zeros((1, 15))},
                          config={"sample_id": "s"})
    assert recorded == {"timing_reference": None, "relative_position": True,
                        "duration_centering": False, "shape_features": False}


def test_preparation_flags_come_from_the_checkpoint_and_default_to_legacy():
    assert se.checkpoint_preparation_flags({}) == (False, False, True)
    assert se.checkpoint_preparation_flags({"note_shape_features": True}) == (False, True, True)
    assert se.checkpoint_preparation_flags({"duration_center_per_sample": True, "relative_position": False,
                                            "note_shape_features": True}) == (True, True, False)
