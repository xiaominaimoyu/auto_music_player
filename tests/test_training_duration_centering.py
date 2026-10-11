from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.sequence_expression import (
    DURATION_TARGET_COLUMN,
    prepare_sample,
)


def _arrays(durations: list[float], duration_mask: list[bool]) -> dict[str, np.ndarray]:
    notes = len(durations)
    return {
        "note_features": np.zeros((notes, 4), dtype=np.float32),
        "chord_features": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "note_targets": np.column_stack([np.zeros(notes), durations, np.zeros((notes, len(NOTE_TARGET_NAMES) - 2))])
        .astype(np.float32),
        "chord_targets": np.zeros((2, len(CHORD_TARGET_NAMES)), dtype=np.float32),
        "note_masks": np.column_stack([np.ones(notes, dtype=bool), duration_mask,
                                       np.ones((notes, len(NOTE_TARGET_NAMES) - 2), dtype=bool)]),
        "chord_masks": np.ones((2, len(CHORD_TARGET_NAMES)), dtype=bool),
        "note_to_chord": np.array([0] * (notes // 2) + [1] * (notes - notes // 2), dtype=np.int64),
    }


def test_the_duration_column_is_resolved_from_the_shipped_names():
    assert NOTE_TARGET_NAMES[DURATION_TARGET_COLUMN] == "log_key_duration_ratio"


def test_default_path_is_untouched():
    arrays = _arrays([1.0, 2.0, 3.0, 4.0], [True, True, True, True])
    plain = prepare_sample("s", arrays)
    explicit = prepare_sample("s", arrays, duration_centering=False)
    assert np.array_equal(plain.note_targets, explicit.note_targets)
    assert np.array_equal(plain.note_targets, arrays["note_targets"])


def test_centering_removes_each_recording_s_own_level():
    durations = [1.0, 2.0, 3.0, 100.0]
    sample = prepare_sample("s", _arrays(durations, [True, True, True, True]), duration_centering=True)
    column = sample.note_targets[:, DURATION_TARGET_COLUMN]
    # median of 1/2/3/100 is 2.5, so the supervised level becomes zero-centred
    assert float(np.median(column)) == pytest.approx(0.0)
    assert column.tolist() == pytest.approx([-1.5, -0.5, 0.5, 97.5])


def test_unsupervised_rows_do_not_move_the_centre_but_are_still_shifted():
    # the 100 sits on an unsupervised row: with only 1/2/3 supervised the centre is 2.0
    sample = prepare_sample("s", _arrays([1.0, 2.0, 3.0, 100.0], [True, True, True, False]),
                            duration_centering=True)
    column = sample.note_targets[:, DURATION_TARGET_COLUMN]
    assert float(np.median(column[:3])) == pytest.approx(0.0)
    assert column[3] == pytest.approx(98.0)


def test_other_channels_are_not_relabelled():
    arrays = _arrays([1.0, 2.0, 3.0, 4.0], [True, True, True, True])
    arrays["note_targets"][:, 0] = [5.0, 6.0, 7.0, 8.0]
    sample = prepare_sample("s", arrays, duration_centering=True)
    assert sample.note_targets[:, 0].tolist() == pytest.approx([5.0, 6.0, 7.0, 8.0])
    assert np.array_equal(sample.chord_targets, arrays["chord_targets"])


def test_a_recording_without_supervised_duration_rows_is_left_alone():
    sample = prepare_sample("s", _arrays([1.0, 2.0], [False, False]), duration_centering=True)
    assert sample.note_targets[:, DURATION_TARGET_COLUMN].tolist() == pytest.approx([1.0, 2.0])


def test_the_cli_flag_reaches_the_trainer(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import training.train_sequence_expression as trainer

    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(trainer, "train", _fake_train)
    argv = ["--dataset", str(tmp_path), "--output", str(tmp_path / "out"),
            "--train-sample-id-file", str(tmp_path / "train.txt"),
            "--validation-sample-id-file", str(tmp_path / "validation.txt"),
            "--architecture", "gru", "--duration-centering"]
    assert trainer.main(argv) == 0
    assert captured["duration_centering"] is True
    assert trainer.main(argv[:-1]) == 0
    assert captured["duration_centering"] is False
