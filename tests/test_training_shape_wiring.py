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


def _arrays(count: int = 2) -> dict[str, np.ndarray]:
    return {"note_features": np.zeros((count, 15), dtype=np.float32),
            "chord_features": np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
            "note_targets": np.zeros((count, len(NOTE_TARGET_NAMES)), dtype=np.float32),
            "chord_targets": np.zeros((2, len(CHORD_TARGET_NAMES)), dtype=np.float32),
            "note_masks": np.ones((count, len(NOTE_TARGET_NAMES)), dtype=bool),
            "chord_masks": np.ones((2, len(CHORD_TARGET_NAMES)), dtype=bool),
            "note_to_chord": np.arange(count, dtype=np.int32)}


def test_the_trainer_and_the_module_agree_on_the_shipped_block():
    assert TRAINER_NAMES == SHAPE_FEATURE_NAMES


def test_the_default_path_keeps_the_shipped_note_feature_width():
    arrays = _arrays()
    legacy = se.prepare_sample("s", arrays)
    explicit = se.prepare_sample("s", arrays, shape_feature_names=())
    assert legacy.note_features.shape == explicit.note_features.shape == (2, 15)
    widened = se.prepare_sample("s", arrays, shape_feature_names=SHAPE_FEATURE_NAMES)
    assert widened.note_features.shape == (2, 15 + len(SHAPE_FEATURE_NAMES))
    # the shipped columns stay exactly where they were; only new trailing columns are added
    assert np.array_equal(widened.note_features[:, :15], legacy.note_features[:, :15])


def test_a_subset_of_names_produces_exactly_that_many_trailing_columns():
    arrays = _arrays()
    names = ("duration_versus_local_mean_log", "pitch_repeats_previous_chord")
    sample = se.prepare_sample("s", arrays, shape_feature_names=names)
    assert sample.note_features.shape == (2, 17)


def test_the_block_is_derived_before_positions_are_rescaled():
    # onset 0 and 100 with 2-beat lengths: in raw beats they are sequential, but if the block
    # were derived after rescaling positions to 0-1 while durations stayed in beats, the
    # first note would appear to overlap the second
    arrays = _arrays(2)
    arrays["note_features"][:, 3] = [0.0, 100.0]
    arrays["note_features"][:, 4] = [math.log1p(2.0)] * 2
    arrays["chord_features"][:, 0] = [0.0, 100.0]
    scaled = se.prepare_sample("s", arrays, relative_position=True, shape_feature_names=SHAPE_FEATURE_NAMES)
    raw = se.prepare_sample("s", arrays, relative_position=False, shape_feature_names=SHAPE_FEATURE_NAMES)
    index = 15 + SHAPE_FEATURE_NAMES.index("concurrent_notes_at_onset_log1p")
    assert np.array_equal(scaled.note_features[:, 15:], raw.note_features[:, 15:])
    assert scaled.note_features[:, index].tolist() == pytest.approx([0.0, 0.0])
    # the positions themselves really are rescaled, so equality above is not an accident
    assert scaled.note_features[:, 3].tolist() == pytest.approx([0.0, 1.0])


def test_the_recorded_input_names_follow_the_order_the_tensor_is_built_in():
    from training.midi_features import NOTE_FEATURE_NAMES
    from training.sequence_expression import TIMING_NOTE_FEATURE_NAMES, note_input_names

    assert note_input_names() == tuple(NOTE_FEATURE_NAMES) + TIMING_NOTE_FEATURE_NAMES
    without_reference = note_input_names(SHAPE_FEATURE_NAMES, has_timing_reference=False)
    assert without_reference == tuple(NOTE_FEATURE_NAMES) + SHAPE_FEATURE_NAMES
    with_reference = note_input_names(SHAPE_FEATURE_NAMES)
    # the block is inserted before the timing columns, so a name list that appends it last
    # would mislabel every column after position 15
    assert with_reference[15:15 + len(SHAPE_FEATURE_NAMES)] == SHAPE_FEATURE_NAMES
    assert with_reference[len(NOTE_FEATURE_NAMES) + len(SHAPE_FEATURE_NAMES):] == TIMING_NOTE_FEATURE_NAMES


def test_a_prepared_sample_is_exactly_as_wide_as_the_names_recorded_for_it():
    sample = se.prepare_sample("s", _arrays(), shape_feature_names=SHAPE_FEATURE_NAMES)
    assert sample.note_features.shape[1] == len(
        se.note_input_names(SHAPE_FEATURE_NAMES, has_timing_reference=False))
    plain = se.prepare_sample("s", _arrays())
    assert plain.note_features.shape[1] == len(se.note_input_names(has_timing_reference=False))


def test_other_channels_are_not_relabelled():
    arrays = _arrays()
    arrays["note_targets"][:, 0] = [5.0, 6.0]
    sample = se.prepare_sample("s", arrays, shape_feature_names=SHAPE_FEATURE_NAMES)
    assert sample.note_targets[:, 0].tolist() == pytest.approx([5.0, 6.0])
    assert np.array_equal(sample.chord_targets, arrays["chord_targets"])


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


def test_inference_reads_the_recorded_block_and_target_transform_from_the_checkpoint_config(
        monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}

    def _capture(sample_id, arrays, **kwargs):
        recorded.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(se, "prepare_sample", _capture)
    config = {"sample_id": "s", "relative_position": False, "duration_center_per_sample": True,
              "note_shape_feature_names": list(SHAPE_FEATURE_NAMES)}
    with pytest.raises(_Sentinel):
        se.predict_sample(object(), {"note_features": np.zeros((1, 15))}, config=config)
    assert recorded["shape_feature_names"] == SHAPE_FEATURE_NAMES
    assert recorded["duration_centering"] is True
    assert recorded["relative_position"] is False


def test_a_legacy_checkpoint_config_still_prepares_the_legacy_width(monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}

    def _capture(sample_id, arrays, **kwargs):
        recorded.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(se, "prepare_sample", _capture)
    with pytest.raises(_Sentinel):
        se.predict_sample(object(), {"note_features": np.zeros((1, 15))}, config={"sample_id": "s"})
    assert recorded == {"timing_reference": None, "relative_position": True,
                        "duration_centering": False, "shape_feature_names": ()}


def test_preparation_flags_return_names_and_refuse_an_unnamed_block():
    assert se.checkpoint_preparation_flags({}) == (False, (), True)
    assert se.checkpoint_preparation_flags({"note_shape_feature_names": ["a", "b"]}) == (False, ("a", "b"), True)
    with pytest.raises(se.TrainingRunError, match="without recording their names"):
        se.checkpoint_preparation_flags({"note_shape_features": True})
