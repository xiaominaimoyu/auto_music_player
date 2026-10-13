from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from training.piece_groups import group_membership, score_groups
from training.sequence_expression import SequenceSample
from training.shared_profile import (DURATION_COLUMN, SharedTarget, apply_shared_targets,
                                     profile_coverage, shared_duration_targets)
import training.train_sequence_expression as trainer


def _sample(sample_id: str, duration_labels: list[float], *, supervised: list[bool] | None = None,
            pitches: list[float] | None = None, onsets: list[float] | None = None,
            written: list[float] | None = None) -> SequenceSample:
    """One recording: rows are notes, the duration column carries its centred label."""
    count = len(duration_labels)
    features = np.zeros((count, 15), dtype=np.float32)
    features[:, 0] = pitches if pitches is not None else [0.0] * count
    features[:, 3] = onsets if onsets is not None else [float(index) for index in range(count)]
    features[:, 4] = written if written is not None else [math.log1p(1.0)] * count
    targets = np.zeros((count, 6), dtype=np.float32)
    targets[:, DURATION_COLUMN] = duration_labels
    masks = np.ones((count, 6), dtype=bool)
    if supervised is not None:
        masks[:, DURATION_COLUMN] = supervised
    return SequenceSample(sample_id, features, np.zeros((count, 2), dtype=np.float32), targets,
                          np.zeros((count, 2), dtype=np.float32), masks,
                          np.ones((count, 2), dtype=bool), np.arange(count, dtype=np.int32))


def test_the_shared_target_is_the_cross_performer_mean_at_each_note_position():
    first = _sample("a1", [0.2, -0.4, 0.1])
    second = _sample("a2", [0.4, -0.2, 0.5])
    targets = shared_duration_targets([first, second], {"a1": "pieceA", "a2": "pieceA"})
    assert targets["a1"].values.tolist() == pytest.approx([0.3, -0.3, 0.3])
    assert targets["a2"].values.tolist() == pytest.approx([0.3, -0.3, 0.3])
    assert targets["a1"].shared is True and targets["a1"].group_size == 2


def test_a_position_only_one_recording_supervises_is_still_the_shared_shape():
    # the unsupervised row must not be averaged as a zero: the other recording carries it alone
    first = _sample("a1", [0.2, 9.9], supervised=[True, False])
    second = _sample("a2", [0.6, 0.8])
    targets = shared_duration_targets([first, second], {"a1": "g", "a2": "g"})
    assert targets["a1"].values.tolist() == pytest.approx([0.4, 0.8])


def test_a_position_no_recording_supervises_falls_back_to_that_recording_s_own_value():
    first = _sample("a1", [0.2, -1.0], supervised=[True, False])
    second = _sample("a2", [0.6, -3.0], supervised=[True, False])
    targets = shared_duration_targets([first, second], {"a1": "g", "a2": "g"})
    assert targets["a1"].values.tolist() == pytest.approx([0.4, -1.0])
    assert targets["a2"].values.tolist() == pytest.approx([0.4, -3.0])


def test_a_recording_alone_in_its_split_keeps_its_own_labels_and_says_why():
    targets = shared_duration_targets([_sample("solo", [0.5, -0.5])], {"solo": "pieceZ"})
    entry = targets["solo"]
    assert entry.shared is False and entry.group_size == 1
    assert entry.values.tolist() == pytest.approx([0.5, -0.5])
    assert "alone" in entry.reason


def test_a_group_that_is_not_the_same_score_is_never_averaged():
    # same note count, different written durations: averaging positions here would mix two scores
    first = _sample("a1", [0.2, 0.4], written=[math.log1p(1.0), math.log1p(2.0)])
    second = _sample("a2", [0.6, 0.8], written=[math.log1p(3.0), math.log1p(1.0)])
    targets = shared_duration_targets([first, second], {"a1": "g", "a2": "g"})
    assert all(not entry.shared for entry in targets.values())
    assert targets["a1"].values.tolist() == pytest.approx([0.2, 0.4])
    assert "geometry" in targets["a1"].reason


def test_a_recording_without_a_known_group_is_reported_not_silently_shared():
    targets = shared_duration_targets([_sample("x", [0.1])], {})
    assert targets["x"].shared is False and "no piece group" in targets["x"].reason


def test_applying_the_target_touches_only_the_duration_column_and_keeps_the_masks():
    sample = _sample("a1", [0.2, -0.4])
    other = _sample("a2", [0.4, -0.2])
    targets = shared_duration_targets([sample, other], {"a1": "g", "a2": "g"})
    updated = apply_shared_targets([sample, other], targets)
    for index, result in enumerate(updated):
        assert np.array_equal(result.note_masks, [sample, other][index].note_masks)
        assert np.array_equal(result.note_features, [sample, other][index].note_features)
        assert result.note_targets[:, DURATION_COLUMN].tolist() == pytest.approx([0.3, -0.3])
        others = np.delete(result.note_targets, DURATION_COLUMN, axis=1)
        assert np.array_equal(others, np.delete([sample, other][index].note_targets, DURATION_COLUMN, axis=1))


def test_a_non_finite_shared_value_is_refused_instead_of_reaching_the_loss():
    sample = _sample("a1", [0.2, -0.4])
    broken = {"a1": SharedTarget(np.array([0.3, np.nan]), 2, True)}
    with pytest.raises(ValueError, match="non-finite"):
        apply_shared_targets([sample], broken)


def test_coverage_counts_the_shared_fraction_and_the_reasons():
    shared_a, shared_b = _sample("a1", [0.2]), _sample("a2", [0.4])
    lone = _sample("b1", [0.1])
    targets = shared_duration_targets([shared_a, shared_b, lone],
                                      {"a1": "g1", "a2": "g1", "b1": "g2"})
    coverage = profile_coverage(targets)
    assert coverage["recordings"] == 3 and coverage["shared_recordings"] == 2
    assert coverage["share_shared"] == pytest.approx(2 / 3)
    assert coverage["fallback_reasons"] == {"alone in this split": 1}


def test_the_trainer_refuses_a_shared_target_that_is_not_centred(tmp_path):
    with pytest.raises(trainer.TrainingRunError, match="median-centred"):
        trainer.train(dataset=tmp_path, output=tmp_path / "out", architecture="gru",
                      train_sample_id_file="a", validation_sample_id_file="b",
                      duration_target="shared_profile")


def test_the_cli_flag_reaches_the_trainer(tmp_path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(trainer, "train", lambda **kwargs: captured.update(kwargs) or {"ok": True})
    base = ["--dataset", str(tmp_path), "--output", str(tmp_path / "out"),
            "--train-sample-id-file", str(tmp_path / "train.txt"),
            "--validation-sample-id-file", str(tmp_path / "validation.txt"),
            "--architecture", "gru"]
    assert trainer.main(base + ["--duration-centering", "--duration-target", "shared_profile"]) == 0
    assert captured["duration_target"] == "shared_profile"
    assert trainer.main(base) == 0
    assert captured["duration_target"] == "label"


def test_group_membership_only_reports_ids_the_caller_is_allowed_to_see(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"id": "t1", "split": "train", "piece_group": "A", "chord_count": 2, "note_count": 3,
             "performer": "p1", "qc_status": "accepted"},
            {"id": "t2", "split": "train", "piece_group": "A", "chord_count": 2, "note_count": 3,
             "performer": "p2", "qc_status": "accepted"},
            {"id": "v1", "split": "validation", "piece_group": "A", "chord_count": 2, "note_count": 3,
             "performer": "p3", "qc_status": "accepted"}]
    manifest.write_text("\n".join(str(row) for row in rows).replace("'", '"'), encoding="utf-8")
    groups = score_groups(manifest, "train")
    assert [group for group, _ in groups] == ["A"]
    assert len(groups[0][1]) == 2, "the validation recording must not join the train profile"
    assert group_membership(groups, {"t1", "t2"}) == {"t1": "A", "t2": "A"}


def test_a_group_whose_note_counts_differ_is_not_eligible(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"a","split":"train","piece_group":"A","chord_count":2,"note_count":3,"performer":"p1",'
        '"qc_status":"accepted"}\n'
        '{"id":"b","split":"train","piece_group":"A","chord_count":2,"note_count":4,"performer":"p2",'
        '"qc_status":"accepted"}\n',
        encoding="utf-8")
    assert score_groups(manifest, "train") == []
