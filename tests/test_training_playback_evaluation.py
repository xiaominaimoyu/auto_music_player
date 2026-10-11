import io
import json
from pathlib import Path

import mido
import numpy as np
import pytest
pytest.importorskip("torch")

from training.audit_duration_reference import make_duration_reference
from training.evaluate_playback_adoption import assert_reference_binding, evaluate_onsets, metrics_for_plan, verify_parent_child
from training.tempo_reference import make_timing_reference
from training.train_expression import _sha256
from training.train_sequence_expression import write_integrity


def midi(tempo):
    song = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    song.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))
    for pitch in (60, 62, 64, 65):
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, time=480))
    buffer = io.BytesIO()
    song.save(file=buffer)
    return buffer.getvalue()


def references():
    score, performance = midi(500000), midi(1000000)
    alignment, interpolated = np.arange(4), np.zeros(4, bool)
    timing = make_timing_reference(score, performance, alignment, interpolated)
    duration = make_duration_reference(score, performance, alignment, interpolated, timing)
    return timing, duration


def test_oracle_is_diagnostic_not_fixed_score_gain():
    timing, duration = references()
    onsets = timing["chord_score_seconds"]
    plan = {"selected_note_indices": np.arange(4)}
    for prefix in ("raw", "desired", "actual"):
        plan[f"{prefix}_onsets_s"] = onsets
        plan[f"{prefix}_holds_s"] = np.full(4, .5)
    metrics = metrics_for_plan(plan, timing, duration)
    assert metrics["projected:oracle_global:ioi_mae_ms"]["value"] < 1e-6
    assert metrics["projected:fixed_score:ioi_mae_ms"]["value"] == pytest.approx(500)
    assert metrics["projected:oracle_global:key_duration_mae_ms"]["value"] < 1e-6
    assert metrics["projected:fixed_score:key_duration_mae_ms"]["value"] == pytest.approx(500)


def test_missing_projection_coverage_is_not_filled_as_identity():
    timing, _ = references()
    onsets = timing["chord_score_seconds"].copy()
    onsets[2] = np.nan
    result = evaluate_onsets(timing, onsets, calibration="fixed_score")
    assert result["missing_native_ioi_count"] == 2
    assert result["valid_ioi_count"] == 1


def test_duration_and_timing_binding_must_match(tmp_path: Path):
    timing, duration = tmp_path / "t", tmp_path / "d"
    timing.mkdir()
    duration.mkdir()
    (timing / "summary.json").write_text("{}")
    (duration / "summary.json").write_text(json.dumps({"timing_reference_summary_sha256": _sha256(timing / "summary.json")}))
    assert_reference_binding(timing, duration)
    (timing / "summary.json").write_text("different")
    with pytest.raises(ValueError, match="different source"):
        assert_reference_binding(timing, duration)


def test_child_seal_must_match_parent_seal(tmp_path: Path):
    child = tmp_path / "audits" / "child"
    child.mkdir(parents=True)
    (child / "report.json").write_text("{}")
    write_integrity(child)
    parent = {"child_integrity_sha256": {"audits/child": _sha256(child / "integrity.json")}}
    verify_parent_child(tmp_path, "audits", "child", parent)
    (child / "report.json").write_text("modified")
    write_integrity(child)
    with pytest.raises(ValueError, match="sealed parent"):
        verify_parent_child(tmp_path, "audits", "child", parent)
