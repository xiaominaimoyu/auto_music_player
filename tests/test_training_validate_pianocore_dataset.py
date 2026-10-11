import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training.midi_features import (
    CHORD_FEATURE_NAMES,
    CHORD_TARGET_NAMES,
    NOTE_FEATURE_NAMES,
    NOTE_TARGET_NAMES,
    PerformanceArrays,
)
from training.pianocore_dataset import (
    PIPELINE_NAME,
    PIPELINE_VERSION,
    SHARD_ARRAY_DTYPES,
    TrainOnlyNormalizer,
    source_directory_name,
)
from training.validate_pianocore_dataset import (
    DatasetValidationError,
    main,
    validate_dataset,
)


SOURCE = "Demo"
SOURCE_DIRECTORY = source_directory_name(SOURCE)
SHARD_RELATIVES = {
    split: f"shards/split={split}/source={SOURCE_DIRECTORY}/part-00000.npz"
    for split in ("train", "validation")
}


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arrays(split: str):
    if split == "train":
        note_count, chord_count = 2, 1
        note_targets = np.array(
            [
                [10.0, 0.1, 0.5, 0.25, 0.0, 0.75],
                [0.0, 0.0, 0.6, -0.25, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        note_masks = np.array(
            [
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
            ],
            dtype=np.bool_,
        )
        chord_targets = np.array([[25.0, 0.0]], dtype=np.float32)
        chord_masks = np.array([[True, False]], dtype=np.bool_)
        note_to_chord = np.array([0, 0], dtype=np.int32)
        sample_id = "sample-1"
    elif split == "validation":
        note_count, chord_count = 1, 1
        note_targets = np.array(
            [[-5.0, -0.1, 0.4, -0.5, 0.0, 0.0]], dtype=np.float32
        )
        note_masks = np.array(
            [[True, True, True, True, False, False]], dtype=np.bool_
        )
        chord_targets = np.array([[-10.0, 0.0]], dtype=np.float32)
        chord_masks = np.array([[True, False]], dtype=np.bool_)
        note_to_chord = np.array([0], dtype=np.int32)
        sample_id = "sample-2"
    else:  # pragma: no cover - fixture misuse guard
        raise ValueError(split)

    return {
        "note_features": (
            np.arange(note_count * len(NOTE_FEATURE_NAMES), dtype=np.float32).reshape(
                note_count, len(NOTE_FEATURE_NAMES)
            )
            / 10.0
        ),
        "note_targets": note_targets,
        "note_masks": note_masks,
        "chord_features": (
            np.arange(chord_count * len(CHORD_FEATURE_NAMES), dtype=np.float32).reshape(
                chord_count, len(CHORD_FEATURE_NAMES)
            )
            / 10.0
        ),
        "chord_targets": chord_targets,
        "chord_masks": chord_masks,
        "note_to_chord": note_to_chord,
        "sample_note_offsets": np.array([0, note_count], dtype=np.int64),
        "sample_chord_offsets": np.array([0, chord_count], dtype=np.int64),
        "sample_ids": np.array([sample_id]),
    }


def _write_shard(root: Path, split: str, arrays=None):
    shard = root / SHARD_RELATIVES[split]
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, **(arrays or _arrays(split)))
    return shard


def _manifest(split: str, shard: Path, arrays):
    is_train = split == "train"
    note_masks = arrays["note_masks"]
    chord_masks = arrays["chord_masks"]
    note_count = len(arrays["note_features"])
    chord_count = len(arrays["chord_features"])
    interpolated_count = int((~note_masks[:, 0]).sum())
    warning_flags = (
        ["interpolation_rate_over_10_percent"]
        if interpolated_count / note_count > 0.10
        else []
    )
    digest_chars = {
        "train": ("1", "a", "c", "e"),
        "validation": ("2", "b", "d", "f"),
    }[split]
    score_hash, semantic_hash, performance_hash, alignment_hash = (
        character * 64 for character in digest_chars
    )
    return {
        "id": "sample-1" if is_train else "sample-2",
        "source": "pianocore_a_star_refined",
        "performance_dataset": SOURCE,
        "performer": "Performer One" if is_train else "Performer Two",
        "performance_id": "performance-1" if is_train else "performance-2",
        "official_split": "train",
        "split": split,
        "piece_group": ("3" if is_train else "4") * 64,
        "composer": "Composer",
        "composition": "Composition One" if is_train else "Composition Two",
        "movement": "",
        "score_midi_path": f"scores/{split}.mid",
        "performance_midi_path": f"performances/{split}.mid",
        "alignment_path": f"alignments/{split}.npz",
        "score_midi_sha256": score_hash,
        "score_semantic_sha256": semantic_hash,
        "performance_midi_sha256": performance_hash,
        "alignment_sha256": alignment_hash,
        "score_midi_size_bytes": 10,
        "performance_midi_size_bytes": 20,
        "alignment_size_bytes": 30,
        "refined_recall": 0.99,
        "note_count": note_count,
        "chord_count": chord_count,
        "interpolated_count": interpolated_count,
        "interpolation_rate": interpolated_count / note_count,
        "supervised_note_count": int(note_masks[:, 0].sum()),
        "supervised_chord_count": int(chord_masks[:, 0].sum()),
        "onset_delta_aux_masked_count": 0,
        "global_tempo_scale": 1.0,
        "has_pedal": bool(note_masks[:, 4].any()),
        "warning_flags": warning_flags,
        "qc_status": "warning" if warning_flags else "accepted",
        "shard_path": SHARD_RELATIVES[split],
        "shard_sha256": _sha256(shard),
        "shard_size_bytes": shard.stat().st_size,
        "sample_index": 0,
        "note_offset": 0,
        "chord_offset": 0,
    }


def _normalizer(train_arrays):
    accumulator = TrainOnlyNormalizer()
    accumulator.update(
        PerformanceArrays(
            note_features=train_arrays["note_features"],
            note_targets=train_arrays["note_targets"],
            note_masks=train_arrays["note_masks"],
            chord_features=train_arrays["chord_features"],
            chord_targets=train_arrays["chord_targets"],
            chord_masks=train_arrays["chord_masks"],
            note_to_chord=train_arrays["note_to_chord"],
            diagnostics={},
        ),
        "train",
    )
    return accumulator.summary()


def _seal(root: Path):
    inventory_names = [
        "build_config.json",
        "schema.json",
        "manifest.jsonl",
        "rejects.jsonl",
        "normalizer.json",
        "qc_summary.json",
        *sorted(SHARD_RELATIVES.values()),
    ]
    _write_json(
        root / "integrity.json",
        {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": PIPELINE_VERSION,
            "files": [
                {
                    "path": name,
                    "size_bytes": (root / name).stat().st_size,
                    "sha256": _sha256(root / name),
                }
                for name in sorted(inventory_names)
            ],
        },
    )
    (root / "_SUCCESS").write_text(
        _sha256(root / "integrity.json") + "\n", encoding="ascii"
    )


def _rewrite_shard(root: Path, split: str, arrays):
    shard = _write_shard(root, split, arrays)
    manifests = _read_jsonl(root / "manifest.jsonl")
    for manifest in manifests:
        if manifest["shard_path"] == SHARD_RELATIVES[split]:
            manifest["shard_sha256"] = _sha256(shard)
            manifest["shard_size_bytes"] = shard.stat().st_size
    _write_jsonl(root / "manifest.jsonl", manifests)
    _seal(root)


def _dataset(root: Path) -> Path:
    arrays_by_split = {split: _arrays(split) for split in SHARD_RELATIVES}
    shards = {
        split: _write_shard(root, split, arrays)
        for split, arrays in arrays_by_split.items()
    }
    manifests = [
        _manifest(split, shards[split], arrays_by_split[split])
        for split in ("train", "validation")
    ]
    config = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "mode": "smoke",
        "validation_fraction": 0.1,
        "seed": 20260928,
        "smoke_per_group": 2,
        "max_records": None,
        "max_samples_per_shard": 256,
        "max_notes_per_shard": 250_000,
        "max_interpolation_rate": 0.5,
        "workers": 1,
        "max_score_cache_entries": 16,
        "strict": False,
        "metadata_sha256": "5" * 64,
        "metadata_size_bytes": 100,
        "archive_sha256": "6" * 64,
        "archive_size_bytes": 200,
        "score_semantic_file_count": 2,
        "score_semantic_inventory_sha256": "7" * 64,
        "selected_count": 3,
        "provenance": {
            "source_sha256": {
                "training/midi_features.py": "8" * 64,
                "training/pianocore_dataset.py": "9" * 64,
            },
            "runtime": {
                "python_implementation": "CPython",
                "python_version": "3.12.0",
                "numpy_version": "2.0.0",
                "mido_version": "1.3.3",
                "zlib_version": "1.3.1",
            },
        },
    }
    config["config_sha256"] = _canonical_hash(config)
    _write_json(root / "build_config.json", config)
    _write_json(
        root / "schema.json",
        {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": PIPELINE_VERSION,
            "shard_format": "deterministic_npz_ragged_v1",
            "note_features": list(NOTE_FEATURE_NAMES),
            "note_targets": list(NOTE_TARGET_NAMES),
            "chord_features": list(CHORD_FEATURE_NAMES),
            "chord_targets": list(CHORD_TARGET_NAMES),
            "array_dtypes": SHARD_ARRAY_DTYPES,
            "interpolated_policy": "context retained; all note targets masked; chord supervision requires >=50% real notes",
            "onset_delta_policy": "bounded auxiliary target; trusted chord and absolute residual <=2000ms; long-form rubato uses log_ioi_ratio",
            "playback_boundary": "timing and duration are executable candidates; velocity and pedal are retained research targets",
            "score_identity_policy": "normalized beat-relative notes plus exact per-note bar phase; ignores redundant time-signature events, PPQ, tempo, track, channel, velocity, and MIDI encoding",
        },
    )
    _write_jsonl(root / "manifest.jsonl", manifests)
    _write_jsonl(
        root / "rejects.jsonl",
        [
            {
                "id": "rejected",
                "performance_dataset": SOURCE,
                "split": "train",
                "reason_type": "DemoReject",
                "reason": "test",
            }
        ],
    )
    _write_json(root / "normalizer.json", _normalizer(arrays_by_split["train"]))
    _write_json(
        root / "qc_summary.json",
        {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": PIPELINE_VERSION,
            "config_sha256": config["config_sha256"],
            "selected_count": 3,
            "accepted_count": 2,
            "rejected_count": 1,
            "note_count": 3,
            "chord_count": 2,
            "split_counts": {"train": 1, "validation": 1},
            "source_counts": {SOURCE: 2},
            "reject_reason_counts": {"DemoReject": 1},
            "pass_rate": 2 / 3,
            "piece_group_split_overlap": 0,
            "score_path_split_overlap": 0,
            "score_semantic_split_overlap": 0,
            "performer_count": 2,
            "performer_split_overlap": 0,
            "split_audit": {
                "piece_group_count": 2,
                "score_path_count": 2,
                "official_piece_group_overlap": 0,
                "official_score_path_overlap": 0,
                "score_semantic_hash_count": 2,
                "official_score_semantic_overlap": 0,
                "performer_count": 2,
                "official_performer_overlap": 0,
            },
            "selected_score_path_count": 2,
            "score_cache_capacity_per_process": 16,
            "normalizer_fit_split": "train",
        },
    )
    _seal(root)
    return root


def test_validate_dataset_accepts_complete_consistent_output(tmp_path):
    report = validate_dataset(_dataset(tmp_path / "dataset"))
    assert report == {
        "status": "valid",
        "dataset_root": str((tmp_path / "dataset").resolve()),
        "accepted_count": 2,
        "rejected_count": 1,
        "shard_count": 2,
        "note_count": 3,
        "chord_count": 2,
        "split_counts": {"train": 1, "validation": 1},
        "source_counts": {SOURCE: 2},
    }


def test_cli_prints_report_and_returns_zero(tmp_path, capsys):
    root = _dataset(tmp_path / "dataset")
    assert main(["--dataset", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"


@pytest.mark.parametrize("marker", ["_SUCCESS", "integrity.json"])
def test_missing_completion_file_fails(tmp_path, marker):
    root = _dataset(tmp_path / "dataset")
    (root / marker).unlink()
    with pytest.raises(DatasetValidationError, match=marker):
        validate_dataset(root)


def test_incomplete_marker_always_fails(tmp_path):
    root = _dataset(tmp_path / "dataset")
    (root / "_INCOMPLETE").write_text("still building\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="_INCOMPLETE"):
        validate_dataset(root)


def test_tampering_is_caught_by_inventory_hash(tmp_path):
    root = _dataset(tmp_path / "dataset")
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(DatasetValidationError, match="size mismatch"):
        validate_dataset(root)


def test_resealed_but_self_inconsistent_build_config_fails(tmp_path):
    root = _dataset(tmp_path / "dataset")
    config = json.loads((root / "build_config.json").read_text(encoding="utf-8"))
    config["selected_count"] = 99
    _write_json(root / "build_config.json", config)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="config_sha256 mismatch"):
        validate_dataset(root)


def test_allow_pickle_false_and_bool_mask_are_enforced(tmp_path):
    root = _dataset(tmp_path / "dataset")
    arrays = _arrays("train")
    arrays["note_masks"] = arrays["note_masks"].astype(np.uint8)
    _rewrite_shard(root, "train", arrays)
    with pytest.raises(DatasetValidationError, match="note_masks dtype mismatch"):
        validate_dataset(root)


@pytest.mark.parametrize(
    ("array_name", "replacement", "message"),
    [
        (
            "sample_note_offsets",
            np.array([0, 3], dtype=np.int64),
            "terminal value",
        ),
        (
            "note_to_chord",
            np.array([0, 1], dtype=np.int32),
            "out-of-range",
        ),
    ],
)
def test_bad_offsets_and_note_mapping_fail(tmp_path, array_name, replacement, message):
    root = _dataset(tmp_path / "dataset")
    arrays = _arrays("train")
    arrays[array_name] = replacement
    _rewrite_shard(root, "train", arrays)
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(root)


def test_manifest_offset_is_reconciled_with_shard(tmp_path):
    root = _dataset(tmp_path / "dataset")
    manifests = _read_jsonl(root / "manifest.jsonl")
    manifests[0]["note_offset"] = 1
    _write_jsonl(root / "manifest.jsonl", manifests)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="note_offset mismatch"):
        validate_dataset(root)


def test_normalizer_is_fully_recomputed_from_train_masks(tmp_path):
    root = _dataset(tmp_path / "dataset")
    normalizer = json.loads((root / "normalizer.json").read_text(encoding="utf-8"))
    normalizer["note_targets"][NOTE_TARGET_NAMES[0]]["mean"] += 1.0
    _write_json(root / "normalizer.json", normalizer)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="normalizer.*mean"):
        validate_dataset(root)


def test_qc_totals_and_actual_split_overlap_are_recomputed(tmp_path):
    root = _dataset(tmp_path / "dataset")
    manifests = _read_jsonl(root / "manifest.jsonl")
    manifests[1]["piece_group"] = manifests[0]["piece_group"]
    _write_jsonl(root / "manifest.jsonl", manifests)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="piece_group_split_overlap"):
        validate_dataset(root)


def test_manifest_diagnostics_are_reconciled_with_masks(tmp_path):
    root = _dataset(tmp_path / "dataset")
    manifests = _read_jsonl(root / "manifest.jsonl")
    manifests[0]["supervised_note_count"] = 2
    _write_jsonl(root / "manifest.jsonl", manifests)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="supervised_note_count mismatch"):
        validate_dataset(root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda arrays: arrays["note_masks"].__setitem__((0, 1), False),
            "first four note target masks",
        ),
        (
            lambda arrays: arrays["note_masks"].__setitem__((0, 5), False),
            "pedal note target masks",
        ),
        (
            lambda arrays: arrays["chord_masks"].__setitem__((0, 1), True),
            "first chord cannot supervise",
        ),
    ],
)
def test_mask_semantics_are_enforced(tmp_path, mutate, message):
    root = _dataset(tmp_path / "dataset")
    arrays = _arrays("train")
    mutate(arrays)
    _rewrite_shard(root, "train", arrays)
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(root)


def test_bounded_onset_target_is_enforced(tmp_path):
    root = _dataset(tmp_path / "dataset")
    arrays = _arrays("train")
    arrays["chord_targets"][0, 0] = 2_001.0
    _rewrite_shard(root, "train", arrays)
    with pytest.raises(DatasetValidationError, match="outside the supported bound"):
        validate_dataset(root)


@pytest.mark.parametrize("target_name", ["velocity_0_1", "pedal_at_onset", "pedal_coverage"])
def test_bounded_note_targets_are_enforced(tmp_path, target_name):
    root = _dataset(tmp_path / "dataset")
    arrays = _arrays("train")
    arrays["note_targets"][0, NOTE_TARGET_NAMES.index(target_name)] = 1.01
    _rewrite_shard(root, "train", arrays)
    with pytest.raises(DatasetValidationError, match=r"within \[0, 1\]"):
        validate_dataset(root)


def test_build_config_v2_fields_are_required(tmp_path):
    root = _dataset(tmp_path / "dataset")
    config = json.loads((root / "build_config.json").read_text(encoding="utf-8"))
    del config["metadata_sha256"]
    config["config_sha256"] = _canonical_hash(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    _write_json(root / "build_config.json", config)
    summary = json.loads((root / "qc_summary.json").read_text(encoding="utf-8"))
    summary["config_sha256"] = config["config_sha256"]
    _write_json(root / "qc_summary.json", summary)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="metadata_sha256"):
        validate_dataset(root)


def test_strict_build_cannot_contain_rejects(tmp_path):
    root = _dataset(tmp_path / "dataset")
    config = json.loads((root / "build_config.json").read_text(encoding="utf-8"))
    config["strict"] = True
    config["config_sha256"] = _canonical_hash(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    _write_json(root / "build_config.json", config)
    summary = json.loads((root / "qc_summary.json").read_text(encoding="utf-8"))
    summary["config_sha256"] = config["config_sha256"]
    _write_json(root / "qc_summary.json", summary)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="strict=true"):
        validate_dataset(root)


def test_nonfinite_json_is_rejected_even_when_resealed(tmp_path):
    root = _dataset(tmp_path / "dataset")
    summary = json.loads((root / "qc_summary.json").read_text(encoding="utf-8"))
    summary["pass_rate"] = float("nan")
    _write_json(root / "qc_summary.json", summary)
    _seal(root)
    with pytest.raises(DatasetValidationError, match="NaN"):
        validate_dataset(root)


def test_unlisted_file_is_rejected(tmp_path):
    root = _dataset(tmp_path / "dataset")
    (root / "extra.bin").write_bytes(b"not inventoried")
    with pytest.raises(DatasetValidationError, match="absent from integrity inventory"):
        validate_dataset(root)
