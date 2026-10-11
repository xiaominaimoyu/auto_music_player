import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

import mido
import numpy as np
import pytest

from training.midi_features import (
    CHORD_TARGET_NAMES,
    NOTE_TARGET_NAMES,
    build_aligned_features,
    parse_midi_bytes,
)
from training.pianocore_dataset import (
    DatasetBuildError,
    PROCESSING_COLUMNS,
    TrainOnlyNormalizer,
    assign_split,
    build_dataset,
    connected_piece_groups,
    piece_key,
    score_semantic_fingerprint,
    select_rows,
    sha256_file,
    source_directory_name,
    validate_metadata_splits,
    write_deterministic_npz,
)
from training.validate_pianocore_dataset import validate_dataset


def _midi_bytes(
    notes,
    *,
    pedal=True,
    note_channel=0,
    pedal_channel=0,
    pedal_events=None,
    ticks_per_beat=480,
    tempo=500_000,
    time_signatures=None,
):
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(0, 0, mido.MetaMessage("set_tempo", tempo=tempo, time=0))]
    for order, (tick, numerator, denominator) in enumerate(time_signatures or ()):
        events.append(
            (
                tick,
                2 + order,
                mido.MetaMessage(
                    "time_signature",
                    numerator=numerator,
                    denominator=denominator,
                    time=0,
                ),
            )
        )
    if pedal:
        pedal_events = pedal_events or [(0, 127), (700, 0)]
        events.extend(
            (
                tick,
                1 + order,
                mido.Message(
                    "control_change",
                    control=64,
                    value=value,
                    channel=pedal_channel,
                    time=0,
                ),
            )
            for order, (tick, value) in enumerate(pedal_events)
        )
    for order, (start, end, pitch, velocity) in enumerate(notes):
        events.append(
            (
                start,
                10 + order,
                mido.Message(
                    "note_on",
                    note=pitch,
                    velocity=velocity,
                    channel=note_channel,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end,
                100 + order,
                mido.Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    channel=note_channel,
                    time=0,
                ),
            )
        )
    previous = 0
    for tick, _order, event in sorted(events, key=lambda item: (item[0], item[1])):
        event.time = tick - previous
        previous = tick
        track.append(event)
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


SCORE_NOTES = [
    (0, 480, 60, 33),
    (0, 480, 64, 33),
    (480, 960, 62, 33),
    (960, 1440, 67, 33),
]
PERFORMANCE_NOTES = [
    (0, 450, 60, 60),
    (10, 430, 64, 80),
    (520, 900, 62, 70),
    (1080, 1500, 67, 90),
]


def test_aligned_features_mask_interpolated_notes_and_keep_pedal_targets():
    score = parse_midi_bytes(_midi_bytes(SCORE_NOTES, pedal=False))
    performance = parse_midi_bytes(_midi_bytes(PERFORMANCE_NOTES, pedal=True))
    arrays = build_aligned_features(
        score,
        performance,
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([False, True, False, False], dtype=np.bool_),
    )

    assert arrays.note_features.shape[0] == 4
    assert arrays.note_targets.shape == (4, len(NOTE_TARGET_NAMES))
    assert arrays.chord_targets.shape == (3, len(CHORD_TARGET_NAMES))
    assert arrays.note_to_chord.tolist() == [0, 0, 1, 2]
    assert not arrays.note_masks[1].any()
    assert arrays.note_masks[[0, 2, 3], :4].all()
    assert arrays.note_masks[[0, 2, 3], 4:].all()
    assert arrays.diagnostics["interpolated_count"] == 1
    assert arrays.diagnostics["has_pedal"] is True


def test_alignment_dtypes_are_not_silently_coerced():
    score = parse_midi_bytes(_midi_bytes(SCORE_NOTES, pedal=False))
    performance = parse_midi_bytes(_midi_bytes(PERFORMANCE_NOTES, pedal=True))
    try:
        build_aligned_features(
            score,
            performance,
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.zeros(4, dtype=np.bool_),
        )
    except Exception as exc:
        assert "integer dtype" in str(exc)
    else:
        raise AssertionError("floating-point perf_idx was silently accepted")

    try:
        build_aligned_features(
            score,
            performance,
            np.arange(4, dtype=np.int32),
            np.zeros(4, dtype=np.uint8),
        )
    except Exception as exc:
        assert "boolean dtype" in str(exc)
    else:
        raise AssertionError("non-boolean interpolated flags were silently accepted")


def test_asynchronous_type_two_midi_is_rejected():
    midi = mido.MidiFile(type=2, ticks_per_beat=480)
    for pitch in (60, 64):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    output = io.BytesIO()
    midi.save(file=output)
    try:
        parse_midi_bytes(output.getvalue())
    except Exception as exc:
        assert "type" in str(exc)
    else:
        raise AssertionError("asynchronous Type 2 MIDI was silently merged")


def test_pedal_supervision_is_channel_specific_and_preserves_same_tick_order():
    score = parse_midi_bytes(_midi_bytes(SCORE_NOTES, pedal=False))
    other_channel = parse_midi_bytes(
        _midi_bytes(PERFORMANCE_NOTES, note_channel=1, pedal_channel=0)
    )
    arrays = build_aligned_features(
        score,
        other_channel,
        np.arange(4, dtype=np.int32),
        np.zeros(4, dtype=np.bool_),
    )
    assert not arrays.note_masks[:, 4:].any()

    same_tick = parse_midi_bytes(
        _midi_bytes(PERFORMANCE_NOTES, pedal_events=[(0, 127), (0, 0)])
    )
    arrays = build_aligned_features(
        score,
        same_tick,
        np.arange(4, dtype=np.int32),
        np.zeros(4, dtype=np.bool_),
    )
    assert arrays.note_masks[:, 4:].all()
    assert arrays.note_targets[0, NOTE_TARGET_NAMES.index("pedal_at_onset")] == 0.0


def test_interpolated_timing_cannot_change_supervised_targets():
    score_notes = [
        (0, 240, 60, 40),
        (480, 720, 62, 40),
        (960, 1200, 64, 40),
    ]
    base_performance = [
        (0, 240, 60, 70),
        (480, 720, 62, 70),
        (960, 1200, 64, 70),
    ]
    perturbed_performance = [
        (0, 240, 60, 70),
        (900, 1140, 62, 70),
        (960, 1200, 64, 70),
    ]
    score = parse_midi_bytes(_midi_bytes(score_notes, pedal=False))
    base = build_aligned_features(
        score,
        parse_midi_bytes(_midi_bytes(base_performance, pedal=False)),
        np.arange(3, dtype=np.int32),
        np.array([False, True, False], dtype=np.bool_),
    )
    perturbed = build_aligned_features(
        score,
        parse_midi_bytes(_midi_bytes(perturbed_performance, pedal=False)),
        np.arange(3, dtype=np.int32),
        np.array([False, True, False], dtype=np.bool_),
    )

    assert np.array_equal(base.note_masks, perturbed.note_masks)
    assert np.array_equal(base.chord_masks, perturbed.chord_masks)
    assert np.allclose(
        base.note_targets[base.note_masks],
        perturbed.note_targets[perturbed.note_masks],
    )
    assert np.allclose(
        base.chord_targets[base.chord_masks],
        perturbed.chord_targets[perturbed.chord_masks],
    )


def test_piece_split_is_stable_and_test_stays_locked():
    first = {
        "composer": "Composer",
        "composition": "Work",
        "movement": "I",
        "performance_id": "take-1",
        "split": "train",
    }
    second = {**first, "performance_id": "take-2"}
    assert piece_key(first) == piece_key(second)
    assert assign_split(first) == assign_split(second)
    assert assign_split({**first, "split": "test"}) == "test"


def test_official_piece_overlap_is_rejected():
    base = {
        "composer": "Composer",
        "composition": "Work",
        "movement": "I",
        "refined_score_midi_path": "work/score.mid",
    }
    try:
        validate_metadata_splits([{**base, "split": "train"}, {**base, "split": "test"}])
    except Exception as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("piece-level leakage was not rejected")


def test_connected_groups_cover_score_path_and_content_aliases():
    first = {
        "id": "first",
        "composer": "Composer A",
        "composition": "Work A",
        "movement": "I",
        "split": "train",
        "refined_score_midi_path": "scores/shared.mid",
    }
    renamed = {
        **first,
        "id": "renamed",
        "composer": "Different Metadata",
        "composition": "Different Work",
    }
    copied = {
        **first,
        "id": "copied",
        "refined_score_midi_path": "scores/copy.mid",
    }
    hashes = {
        "scores/shared.mid": "a" * 64,
        "scores/copy.mid": "a" * 64,
    }
    groups = connected_piece_groups([first, renamed, copied], hashes)
    assert len(set(groups.values())) == 1
    assert assign_split(first, group_key=groups["first"]) == assign_split(
        renamed, group_key=groups["renamed"]
    )


def test_official_semantic_score_alias_overlap_is_rejected():
    train = {
        "id": "train",
        "composer": "Composer A",
        "composition": "Work A",
        "movement": "I",
        "performer": "Performer A",
        "split": "train",
        "tier_a_star": "True",
        "quality_label": "high quality",
        "is_duplicate": "False",
        "is_refined": "True",
        "refined_score_midi_path": "scores/train.mid",
        "refined_performance_midi_path": "performances/train.mid",
        "refined_alignment_path": "alignments/train.npz",
    }
    test = {
        **train,
        "composer": "Renamed Composer",
        "composition": "Renamed Work",
        "id": "test",
        "performer": "Performer B",
        "split": "test",
        "refined_score_midi_path": "scores/test.mid",
        "refined_performance_midi_path": "performances/test.mid",
        "refined_alignment_path": "alignments/test.npz",
    }
    semantic_hashes = {
        "scores/train.mid": "a" * 64,
        "scores/test.mid": "a" * 64,
    }

    try:
        validate_metadata_splits([train, test], semantic_hashes)
    except DatasetBuildError as exc:
        assert "score_semantic_hashes=1" in str(exc)
    else:
        raise AssertionError("semantic score leakage was not rejected")


def test_source_directory_names_cannot_collide_after_sanitizing():
    assert source_directory_name("A/B") != source_directory_name("A_B")


def test_score_semantic_fingerprint_ignores_encoding_but_not_music():
    first = _midi_bytes(SCORE_NOTES, pedal=False)
    doubled_ppq = [
        (start * 2, end * 2, pitch, velocity + 7)
        for start, end, pitch, velocity in SCORE_NOTES
    ]
    equivalent = _midi_bytes(
        doubled_ppq,
        pedal=False,
        note_channel=3,
        ticks_per_beat=960,
        tempo=720_000,
        time_signatures=[(0, 2, 2), (3840, 4, 4)],
    )
    changed = _midi_bytes(
        [*SCORE_NOTES[:-1], (960, 1440, 68, 33)],
        pedal=False,
    )

    assert first != equivalent
    assert score_semantic_fingerprint(parse_midi_bytes(first)) == (
        score_semantic_fingerprint(parse_midi_bytes(equivalent))
    )
    assert score_semantic_fingerprint(parse_midi_bytes(first)) != (
        score_semantic_fingerprint(parse_midi_bytes(changed))
    )


def test_deterministic_npz_hash_is_repeatable():
    arrays = {
        "b": np.array([[1.0, 2.0]], dtype=np.float32),
        "a": np.array([3, 4], dtype=np.int32),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        first = Path(temp_dir) / "first.npz"
        second = Path(temp_dir) / "second.npz"
        first_hash = write_deterministic_npz(first, arrays)
        second_hash = write_deterministic_npz(second, arrays)
        assert first_hash == second_hash
        assert first.read_bytes() == second.read_bytes()


def test_normalizer_accepts_exact_histogram_upper_bound():
    score = parse_midi_bytes(_midi_bytes(SCORE_NOTES, pedal=False))
    performance = parse_midi_bytes(_midi_bytes(PERFORMANCE_NOTES, pedal=True))
    arrays = build_aligned_features(
        score,
        performance,
        np.arange(4, dtype=np.int32),
        np.zeros(4, dtype=np.bool_),
    )
    velocity_index = NOTE_TARGET_NAMES.index("velocity_0_1")
    arrays.note_targets[:, velocity_index] = 1.0
    normalizer = TrainOnlyNormalizer()
    normalizer.update(arrays, "train")
    assert normalizer.summary()["note_targets"]["velocity_0_1"]["count"] == 4


def _metadata_row(identifier, directory, split, performance_id):
    return {
        "id": identifier,
        "composer": f"Composer_{directory}",
        "composition": f"Work_{directory}",
        "movement": "I",
        "performance_id": performance_id,
        "split": split,
        "tier_a_star": "True",
        "performance_dataset": "ATEPP",
        "performer": "Test Performer",
        "quality_label": "high quality",
        "is_duplicate": "False",
        "is_refined": "True",
        "refined_score_midi_path": f"{directory}/score_refined.mid",
        "refined_score_note_count": "4",
        "refined_performance_midi_path": f"{directory}/{performance_id}_refined.mid",
        "refined_performance_note_count": "4",
        "refined_performance_interpolated_note_count": "1",
        "refined_alignment_path": f"{directory}/{performance_id}_refined_align.npz",
        "refined_recall": "0.95",
    }


def _write_tiny_dataset(root):
    rows = [
        _metadata_row("sample_train", "train_work", "train", "take_train"),
        _metadata_row("sample_test", "test_work", "test", "take_test"),
    ]
    metadata = root / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    alignment = io.BytesIO()
    np.savez(
        alignment,
        perf_idx=np.arange(4, dtype=np.int32),
        interpolated=np.array([False, True, False, False], dtype=np.bool_),
    )
    archive = root / "refined.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for row in rows:
            transpose = 1 if row["split"] == "test" else 0
            score_notes = [
                (start, end, pitch + transpose, velocity)
                for start, end, pitch, velocity in SCORE_NOTES
            ]
            performance_notes = [
                (start, end, pitch + transpose, velocity)
                for start, end, pitch, velocity in PERFORMANCE_NOTES
            ]
            prefix = "PianoCoRe/refined/"
            output.writestr(
                prefix + row["refined_score_midi_path"],
                _midi_bytes(score_notes, pedal=False),
            )
            output.writestr(
                prefix + row["refined_performance_midi_path"],
                _midi_bytes(performance_notes, pedal=True),
            )
            output.writestr(prefix + row["refined_alignment_path"], alignment.getvalue())
    return metadata, archive


def test_selected_rows_drop_unused_metadata_fields_before_processing():
    row = _metadata_row("sample_train", "train_work", "train", "take_train")
    row["unused_large_metadata"] = "x" * 10_000
    selected = select_rows(
        [row],
        mode="full",
        validation_fraction=1e-9,
        seed=20260928,
        smoke_per_group=1,
        max_records=None,
        score_semantic_hash_by_path={row["refined_score_midi_path"]: "a" * 64},
    )

    assert len(selected) == 1
    assert set(selected[0]) == {
        *PROCESSING_COLUMNS,
        "_piece_group",
        "_split",
        "_score_semantic_sha256",
    }
    assert "unused_large_metadata" not in selected[0]


def test_tiny_dataset_build_has_manifest_qc_and_repeatable_shards():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        metadata, archive = _write_tiny_dataset(root)
        first = root / "out_first"
        second = root / "out_second"
        parallel = root / "out_parallel"
        kwargs = dict(
            metadata_path=metadata,
            archive_path=archive,
            mode="full",
            validation_fraction=1e-9,
            strict=True,
            max_samples_per_shard=1,
            max_notes_per_shard=10,
            max_interpolation_rate=0.30,
        )
        summary_first = build_dataset(output_path=first, **kwargs)
        summary_second = build_dataset(output_path=second, **kwargs)
        summary_parallel = build_dataset(output_path=parallel, workers=2, **kwargs)

        assert summary_first == summary_second
        assert summary_first["accepted_count"] == 2
        assert summary_parallel["accepted_count"] == 2
        assert summary_first["piece_group_split_overlap"] == 0
        assert (first / "_SUCCESS").is_file()
        manifest = [json.loads(line) for line in (first / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(manifest) == 2
        for item in manifest:
            left = first / item["shard_path"]
            right = second / item["shard_path"]
            assert sha256_file(left) == sha256_file(right) == item["shard_sha256"]
        normalizer = json.loads((first / "normalizer.json").read_text(encoding="utf-8"))
        assert normalizer["fit_split"] == "train"
        for name in ("schema.json", "manifest.jsonl", "rejects.jsonl", "normalizer.json"):
            assert (first / name).read_bytes() == (parallel / name).read_bytes()
        for item in manifest:
            assert (first / item["shard_path"]).read_bytes() == (
                parallel / item["shard_path"]
            ).read_bytes()
        assert (first / "integrity.json").is_file()
        assert (first / "_SUCCESS").read_text(encoding="ascii").strip() == sha256_file(
            first / "integrity.json"
        )
        validation = validate_dataset(first)
        assert validation["status"] == "valid"
        assert validation["accepted_count"] == 2
        assert validation["note_count"] == summary_first["note_count"]


def test_all_rejected_build_stays_incomplete():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        metadata, archive = _write_tiny_dataset(root)
        output = root / "all-rejected"
        try:
            build_dataset(
                metadata_path=metadata,
                archive_path=archive,
                output_path=output,
                mode="full",
                validation_fraction=1e-9,
                strict=False,
                max_interpolation_rate=0.0,
            )
        except DatasetBuildError as exc:
            assert "all selected samples were rejected" in str(exc)
        else:
            raise AssertionError("all-rejected dataset was incorrectly marked successful")
        assert (output / "_INCOMPLETE").is_file()
        assert not (output / "_SUCCESS").exists()


def test_strict_rejection_closes_streams_and_stays_incomplete():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        metadata, archive = _write_tiny_dataset(root)
        output = root / "strict-rejected"
        with pytest.raises(DatasetBuildError, match="interpolation rate"):
            build_dataset(
                metadata_path=metadata,
                archive_path=archive,
                output_path=output,
                mode="full",
                validation_fraction=1e-9,
                strict=True,
                max_interpolation_rate=0.0,
            )
        assert (output / "_INCOMPLETE").is_file()
        assert not (output / "_SUCCESS").exists()
        assert (output / "manifest.jsonl").read_bytes() == b""
        rejects = (output / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rejects) == 1
        assert json.loads(rejects[0])["reason_type"] == "DatasetBuildError"
