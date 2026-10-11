import numpy as np
import pytest

pytest.importorskip("torch")

from training.train_expression import (
    _prepare_features,
    _relative_position_features,
    _select_sample_arrays,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "note_features": np.asarray(
            [[0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 14.0], [0.0, 0.0, 0.0, 20.0]],
            dtype=np.float32,
        ),
        "sample_note_offsets": np.asarray([0, 2, 3], dtype=np.int64),
    }


def test_relative_position_is_reset_per_work() -> None:
    arrays = _arrays()
    transformed = _relative_position_features(
        arrays["note_features"], arrays, kind="note"
    )
    assert np.allclose(transformed[:, 3], [0.0, 1.0, 0.0])


def test_prepare_features_applies_train_statistics_after_position_transform() -> None:
    arrays = _arrays()
    transformed = _prepare_features(
        arrays,
        kind="note",
        relative_position=True,
        means=np.zeros(4, dtype=np.float32),
        scales=np.asarray([1.0, 1.0, 1.0, 0.5], dtype=np.float32),
    )
    assert np.allclose(transformed[:, 3], [0.0, 2.0, 0.0])


def test_select_sample_arrays_rebuilds_ragged_offsets() -> None:
    arrays = {
        "sample_ids": np.asarray(["a", "b"]),
        "sample_note_offsets": np.asarray([0, 2, 5], dtype=np.int64),
        "sample_chord_offsets": np.asarray([0, 1, 3], dtype=np.int64),
        "note_features": np.arange(10, dtype=np.float32).reshape(5, 2),
        "note_targets": np.arange(5, dtype=np.float32).reshape(5, 1),
        "note_masks": np.ones((5, 1), dtype=bool),
        "chord_features": np.arange(6, dtype=np.float32).reshape(3, 2),
        "chord_targets": np.arange(3, dtype=np.float32).reshape(3, 1),
        "chord_masks": np.ones((3, 1), dtype=bool),
        "note_to_chord": np.asarray([0, 0, 1, 2, 2], dtype=np.int64),
    }
    selected = _select_sample_arrays(arrays, frozenset({"b"}))
    assert selected is not None
    assert selected["sample_ids"].tolist() == ["b"]
    assert selected["sample_note_offsets"].tolist() == [0, 3]
    assert selected["sample_chord_offsets"].tolist() == [0, 2]
    assert selected["note_to_chord"].tolist() == [0, 1, 1]
