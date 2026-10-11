"""Read-only integrity and structural validation for PianoCoRe shards.

This command never repairs or rewrites a dataset.  It is intended to be run
after :mod:`training.pianocore_dataset` finishes and before any training job is
allowed to consume the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

from training.midi_features import (
    CHORD_FEATURE_NAMES,
    CHORD_TARGET_NAMES,
    NOTE_FEATURE_NAMES,
    NOTE_TARGET_NAMES,
    PerformanceArrays,
)
from training.pianocore_dataset import TrainOnlyNormalizer, source_directory_name


REQUIRED_METADATA_FILES = {
    "build_config.json",
    "schema.json",
    "manifest.jsonl",
    "rejects.jsonl",
    "normalizer.json",
    "qc_summary.json",
}
SHARD_ARRAY_NAMES = {
    "chord_features",
    "chord_masks",
    "chord_targets",
    "note_features",
    "note_masks",
    "note_targets",
    "note_to_chord",
    "sample_chord_offsets",
    "sample_ids",
    "sample_note_offsets",
}
SHARD_ARRAY_DTYPES = {
    "chord_features": "float32",
    "chord_masks": "bool",
    "chord_targets": "float32",
    "note_features": "float32",
    "note_masks": "bool",
    "note_targets": "float32",
    "note_to_chord": "int32",
    "sample_chord_offsets": "int64",
    "sample_note_offsets": "int64",
    "sample_ids": "unicode",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PIPELINE_NAME = "auto_music_player.pianocore_expression"
SUPPORTED_PIPELINE_VERSION = 2


class DatasetValidationError(RuntimeError):
    """Raised when a built dataset fails a read-only acceptance check."""


def _fail(message: str) -> None:
    raise DatasetValidationError(message)


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            return json.load(handle, parse_constant=_reject_nonfinite_json_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        _fail(f"cannot read valid UTF-8 JSON from {path.name}: {exc}")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    _fail(f"{path.name}:{line_number}: blank JSONL records are not allowed")
                try:
                    value = json.loads(
                        line, parse_constant=_reject_nonfinite_json_constant
                    )
                except ValueError as exc:
                    _fail(f"{path.name}:{line_number}: invalid JSON: {exc}")
                if not isinstance(value, dict):
                    _fail(f"{path.name}:{line_number}: record must be a JSON object")
                yield value
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read UTF-8 JSONL from {path.name}: {exc}")


def _required_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be a JSON array")
    return value


def _required_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _required_positive_int(value: Any, label: str) -> int:
    result = _required_nonnegative_int(value, label)
    if result == 0:
        _fail(f"{label} must be positive")
    return result


def _required_sha256(value: Any, label: str) -> str:
    digest = _required_string(value, label)
    if SHA256_PATTERN.fullmatch(digest) is None:
        _fail(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _safe_relative_path(root: Path, value: Any, label: str) -> tuple[str, Path]:
    text = _required_string(value, label)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or "\\" in text
        or text != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{label} is not a canonical safe POSIX relative path: {text!r}")
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        _fail(f"{label} escapes the dataset root: {text!r}")
    return text, candidate


def _validate_count_map(value: Any, expected: Counter[str], label: str) -> None:
    mapping = _required_object(value, label)
    actual: dict[str, int] = {}
    for key, count in mapping.items():
        if not isinstance(key, str) or not key:
            _fail(f"{label} keys must be non-empty strings")
        actual[key] = _required_nonnegative_int(count, f"{label}[{key!r}]")
    if actual != dict(sorted(expected.items())):
        _fail(f"{label} mismatch: expected {dict(sorted(expected.items()))}, got {actual}")


def _validate_inventory(root: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if (root / "_INCOMPLETE").exists():
        _fail("dataset contains _INCOMPLETE and is not eligible for consumption")
    success_path = root / "_SUCCESS"
    integrity_path = root / "integrity.json"
    if not success_path.is_file():
        _fail("dataset is missing _SUCCESS")
    if not integrity_path.is_file():
        _fail("dataset is missing integrity.json")
    try:
        success_hash = success_path.read_text(encoding="ascii", errors="strict").strip()
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read _SUCCESS: {exc}")
    if SHA256_PATTERN.fullmatch(success_hash) is None:
        _fail("_SUCCESS must contain exactly one lowercase SHA-256 digest")
    integrity_hash = _sha256_file(integrity_path)
    if success_hash != integrity_hash:
        _fail(
            f"_SUCCESS digest does not match integrity.json: "
            f"expected {integrity_hash}, got {success_hash}"
        )

    integrity = _required_object(_load_json(integrity_path), "integrity.json")
    entries = _required_list(integrity.get("files"), "integrity.json.files")
    inventory: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _required_object(raw_entry, f"integrity.json.files[{index}]")
        relative, path = _safe_relative_path(
            root, entry.get("path"), f"integrity.json.files[{index}].path"
        )
        if relative in inventory:
            _fail(f"integrity.json contains duplicate path {relative!r}")
        expected_size = _required_nonnegative_int(
            entry.get("size_bytes"), f"integrity[{relative!r}].size_bytes"
        )
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or SHA256_PATTERN.fullmatch(expected_hash) is None:
            _fail(f"integrity[{relative!r}].sha256 is not a lowercase SHA-256 digest")
        if not path.is_file():
            _fail(f"inventory file is missing: {relative}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            _fail(
                f"inventory size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            _fail(
                f"inventory SHA-256 mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        inventory[relative] = {
            "path": path,
            "size_bytes": actual_size,
            "sha256": actual_hash,
        }

    present_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    allowed_outside_inventory = {"integrity.json", "_SUCCESS"}
    unlisted = present_files - set(inventory) - allowed_outside_inventory
    if unlisted:
        _fail(f"dataset contains files absent from integrity inventory: {sorted(unlisted)}")
    stale_entries = set(inventory) - present_files
    if stale_entries:
        _fail(f"integrity inventory references missing files: {sorted(stale_entries)}")

    missing_metadata = REQUIRED_METADATA_FILES - set(inventory)
    if missing_metadata:
        _fail(f"integrity inventory is missing required metadata: {sorted(missing_metadata)}")
    shard_paths = {path for path in inventory if path.endswith(".npz")}
    if not shard_paths:
        _fail("integrity inventory contains no NPZ shards")
    unexpected = set(inventory) - REQUIRED_METADATA_FILES - shard_paths
    if unexpected:
        _fail(f"integrity inventory contains unsupported files: {sorted(unexpected)}")
    return inventory, shard_paths


def _validate_schema(schema: Any) -> tuple[int, int, int, int]:
    value = _required_object(schema, "schema.json")
    if value.get("shard_format") != "deterministic_npz_ragged_v1":
        _fail("schema.json.shard_format must be deterministic_npz_ragged_v1")
    if value.get("array_dtypes") != SHARD_ARRAY_DTYPES:
        _fail("schema.json.array_dtypes does not match the supported shard contract")
    expected_policies = {
        "interpolated_policy": "context retained; all note targets masked; chord supervision requires >=50% real notes",
        "onset_delta_policy": "bounded auxiliary target; trusted chord and absolute residual <=2000ms; long-form rubato uses log_ioi_ratio",
        "playback_boundary": "timing and duration are executable candidates; velocity and pedal are retained research targets",
        "score_identity_policy": "normalized beat-relative notes plus exact per-note bar phase; ignores redundant time-signature events, PPQ, tempo, track, channel, velocity, and MIDI encoding",
    }
    for key, expected in expected_policies.items():
        if value.get(key) != expected:
            _fail(f"schema.json.{key} does not match the v2 contract")
    expected_names = {
        "note_features": list(NOTE_FEATURE_NAMES),
        "note_targets": list(NOTE_TARGET_NAMES),
        "chord_features": list(CHORD_FEATURE_NAMES),
        "chord_targets": list(CHORD_TARGET_NAMES),
    }
    dimensions: list[int] = []
    for key in ("note_features", "note_targets", "chord_features", "chord_targets"):
        names = _required_list(value.get(key), f"schema.json.{key}")
        if not names or any(not isinstance(name, str) or not name for name in names):
            _fail(f"schema.json.{key} must contain non-empty string names")
        if len(names) != len(set(names)):
            _fail(f"schema.json.{key} contains duplicate names")
        if names != expected_names[key]:
            _fail(f"schema.json.{key} does not match the v2 feature contract")
        dimensions.append(len(names))
    return tuple(dimensions)  # type: ignore[return-value]


def _validate_metadata_contract(
    *,
    build_config: Any,
    schema: Any,
    summary: Any,
    normalizer: Any,
    integrity: Any,
) -> None:
    config = dict(_required_object(build_config, "build_config.json"))
    schema_object = _required_object(schema, "schema.json")
    summary_object = _required_object(summary, "qc_summary.json")
    normalizer_object = _required_object(normalizer, "normalizer.json")
    integrity_object = _required_object(integrity, "integrity.json")

    for label, value in (
        ("build_config.json", config),
        ("schema.json", schema_object),
        ("qc_summary.json", summary_object),
        ("integrity.json", integrity_object),
    ):
        if value.get("pipeline") != PIPELINE_NAME:
            _fail(f"{label}.pipeline must be {PIPELINE_NAME!r}")
        version = _required_nonnegative_int(
            value.get("pipeline_version"), f"{label}.pipeline_version"
        )
        if version != SUPPORTED_PIPELINE_VERSION:
            _fail(
                f"{label}.pipeline_version {version} is unsupported; "
                f"expected {SUPPORTED_PIPELINE_VERSION}"
            )

    config_hash = config.pop("config_sha256", None)
    if not isinstance(config_hash, str) or SHA256_PATTERN.fullmatch(config_hash) is None:
        _fail("build_config.json.config_sha256 is not a lowercase SHA-256 digest")
    expected_config_hash = hashlib.sha256(_canonical_json(config)).hexdigest()
    if config_hash != expected_config_hash:
        _fail(
            "build_config.json.config_sha256 mismatch: "
            f"expected {expected_config_hash}, got {config_hash}"
        )

    if config.get("mode") not in {"smoke", "full"}:
        _fail("build_config.json.mode must be 'smoke' or 'full'")
    validation_fraction = _required_finite_number(
        config.get("validation_fraction"),
        "build_config.json.validation_fraction",
    )
    if not 0.0 < validation_fraction < 0.5:
        _fail("build_config.json.validation_fraction must be between zero and 0.5")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail("build_config.json.seed must be an integer")
    for key in (
        "smoke_per_group",
        "max_samples_per_shard",
        "max_notes_per_shard",
        "max_score_cache_entries",
        "metadata_size_bytes",
        "archive_size_bytes",
        "score_semantic_file_count",
    ):
        _required_positive_int(config.get(key), f"build_config.json.{key}")
    max_records = config.get("max_records")
    if max_records is not None:
        _required_positive_int(max_records, "build_config.json.max_records")
    max_interpolation_rate = _required_finite_number(
        config.get("max_interpolation_rate"),
        "build_config.json.max_interpolation_rate",
    )
    if not 0.0 <= max_interpolation_rate <= 1.0:
        _fail("build_config.json.max_interpolation_rate must be between zero and one")
    workers = _required_positive_int(config.get("workers"), "build_config.json.workers")
    if workers > 32:
        _fail("build_config.json.workers must not exceed 32")
    if not isinstance(config.get("strict"), bool):
        _fail("build_config.json.strict must be boolean")
    for key in (
        "metadata_sha256",
        "archive_sha256",
        "score_semantic_inventory_sha256",
    ):
        _required_sha256(config.get(key), f"build_config.json.{key}")
    provenance = _required_object(
        config.get("provenance"), "build_config.json.provenance"
    )
    source_hashes = _required_object(
        provenance.get("source_sha256"),
        "build_config.json.provenance.source_sha256",
    )
    expected_source_files = {
        "training/midi_features.py",
        "training/pianocore_dataset.py",
    }
    if set(source_hashes) != expected_source_files:
        _fail(
            "build_config.json.provenance.source_sha256 must identify the two "
            "v2 preprocessing source files"
        )
    for name, digest in source_hashes.items():
        _required_sha256(
            digest,
            f"build_config.json.provenance.source_sha256[{name!r}]",
        )
    runtime = _required_object(
        provenance.get("runtime"), "build_config.json.provenance.runtime"
    )
    expected_runtime_fields = {
        "python_implementation",
        "python_version",
        "numpy_version",
        "mido_version",
        "zlib_version",
    }
    if set(runtime) != expected_runtime_fields:
        _fail("build_config.json.provenance.runtime fields do not match v2")
    for key, value in runtime.items():
        _required_string(value, f"build_config.json.provenance.runtime.{key}")

    if summary_object.get("config_sha256") != config_hash:
        _fail("qc_summary.json.config_sha256 does not match build_config.json")
    config_selected = _required_nonnegative_int(
        config.get("selected_count"), "build_config.json.selected_count"
    )
    summary_selected = _required_nonnegative_int(
        summary_object.get("selected_count"), "qc_summary.json.selected_count"
    )
    if config_selected != summary_selected:
        _fail("selected_count differs between build_config.json and qc_summary.json")
    if config_selected == 0:
        _fail("build_config.json.selected_count must be positive")
    if normalizer_object.get("fit_split") != "train":
        _fail("normalizer.json.fit_split must be 'train'")

    split_audit = _required_object(
        summary_object.get("split_audit"), "qc_summary.json.split_audit"
    )
    expected_audit_fields = {
        "piece_group_count",
        "score_path_count",
        "official_piece_group_overlap",
        "official_score_path_overlap",
        "score_semantic_hash_count",
        "official_score_semantic_overlap",
        "performer_count",
        "official_performer_overlap",
    }
    if set(split_audit) != expected_audit_fields:
        _fail("qc_summary.json.split_audit fields do not match v2")
    for key, value in split_audit.items():
        count = _required_nonnegative_int(value, f"qc_summary.json.split_audit.{key}")
        if key in {
            "official_piece_group_overlap",
            "official_score_path_overlap",
            "official_score_semantic_overlap",
        } and count != 0:
            _fail(f"qc_summary.json.split_audit.{key} must be zero")
    _required_positive_int(
        summary_object.get("selected_score_path_count"),
        "qc_summary.json.selected_score_path_count",
    )
    if summary_object.get("score_cache_capacity_per_process") != config.get(
        "max_score_cache_entries"
    ):
        _fail(
            "qc_summary.json.score_cache_capacity_per_process does not match "
            "build_config.json.max_score_cache_entries"
        )
    if summary_object.get("normalizer_fit_split") != "train":
        _fail("qc_summary.json.normalizer_fit_split must be 'train'")


def _require_array_shape(
    array: np.ndarray,
    expected: tuple[int, ...],
    label: str,
) -> None:
    if array.shape != expected:
        _fail(f"{label} shape mismatch: expected {expected}, got {array.shape}")


def _require_finite(array: np.ndarray, label: str) -> None:
    if array.dtype.kind not in "fc":
        _fail(f"{label} must use a floating dtype, got {array.dtype}")
    if not bool(np.isfinite(array).all()):
        _fail(f"{label} contains NaN or infinity")


def _validate_offsets(array: np.ndarray, sample_count: int, total: int, label: str) -> None:
    _require_array_shape(array, (sample_count + 1,), label)
    if array.dtype.kind not in "iu":
        _fail(f"{label} must use an integer dtype, got {array.dtype}")
    if int(array[0]) != 0:
        _fail(f"{label} must start at zero")
    if bool(np.any(array[1:] < array[:-1])):
        _fail(f"{label} must be monotonically non-decreasing")
    if int(array[-1]) != total:
        _fail(f"{label} terminal value must equal {total}, got {int(array[-1])}")


def _required_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    return number


def _validate_manifest_record(
    root: Path,
    row: dict[str, Any],
    row_index: int,
    shard_paths: set[str],
) -> tuple[str, str, str, int, int]:
    prefix = f"manifest.jsonl row {row_index}"
    _required_string(row.get("id"), f"{prefix}.id")
    shard_path, _ = _safe_relative_path(
        root, row.get("shard_path"), f"{prefix}.shard_path"
    )
    if shard_path not in shard_paths:
        _fail(f"{prefix} references a shard absent from integrity.json: {shard_path}")
    _required_nonnegative_int(row.get("sample_index"), f"{prefix}.sample_index")
    _required_nonnegative_int(row.get("note_offset"), f"{prefix}.note_offset")
    _required_nonnegative_int(row.get("chord_offset"), f"{prefix}.chord_offset")
    note_count = _required_nonnegative_int(
        row.get("note_count"), f"{prefix}.note_count"
    )
    chord_count = _required_nonnegative_int(
        row.get("chord_count"), f"{prefix}.chord_count"
    )
    if note_count == 0 or chord_count == 0:
        _fail(f"{prefix} must contain at least one note and one chord")
    shard_size = _required_nonnegative_int(
        row.get("shard_size_bytes"), f"{prefix}.shard_size_bytes"
    )
    shard_hash = row.get("shard_sha256")
    if not isinstance(shard_hash, str) or SHA256_PATTERN.fullmatch(shard_hash) is None:
        _fail(f"{prefix}.shard_sha256 is not a lowercase SHA-256 digest")
    if shard_size == 0:
        _fail(f"{prefix}.shard_size_bytes must be positive")

    split = _required_string(row.get("split"), f"{prefix}.split")
    if split not in {"train", "validation", "test"}:
        _fail(f"{prefix}.split is unsupported: {split!r}")
    official_split = _required_string(
        row.get("official_split"), f"{prefix}.official_split"
    )
    if official_split == "test":
        if split != "test":
            _fail(f"{prefix} moves an official test sample into {split!r}")
    elif official_split == "train":
        if split not in {"train", "validation"}:
            _fail(f"{prefix} moves an official train sample into {split!r}")
    else:
        _fail(f"{prefix}.official_split is unsupported: {official_split!r}")

    source = _required_string(
        row.get("performance_dataset"), f"{prefix}.performance_dataset"
    )
    if row.get("source") != "pianocore_a_star_refined":
        _fail(f"{prefix}.source must be 'pianocore_a_star_refined'")
    expected_parent = PurePosixPath(
        "shards",
        f"split={split}",
        f"source={source_directory_name(source)}",
    )
    actual_path = PurePosixPath(shard_path)
    if actual_path.parent != expected_parent or re.fullmatch(
        r"part-[0-9]{5,}\.npz", actual_path.name
    ) is None:
        _fail(
            f"{prefix}.shard_path does not match its split/source contract: "
            f"{shard_path!r}"
        )

    required_strings = (
        "performance_id",
        "piece_group",
        "composer",
        "composition",
        "score_midi_path",
        "performance_midi_path",
        "alignment_path",
    )
    for required in required_strings:
        _required_string(row.get(required), f"{prefix}.{required}")
    if not isinstance(row.get("movement"), str) or not isinstance(
        row.get("performer"), str
    ):
        _fail(f"{prefix}.movement and performer must be strings")
    if row["performer"] != row["performer"].strip():
        _fail(f"{prefix}.performer must not contain surrounding whitespace")
    if SHA256_PATTERN.fullmatch(str(row["piece_group"])) is None:
        _fail(f"{prefix}.piece_group is not a lowercase SHA-256 digest")
    for digest_name in (
        "score_midi_sha256",
        "score_semantic_sha256",
        "performance_midi_sha256",
        "alignment_sha256",
    ):
        digest = _required_string(row.get(digest_name), f"{prefix}.{digest_name}")
        if SHA256_PATTERN.fullmatch(digest) is None:
            _fail(f"{prefix}.{digest_name} is not a lowercase SHA-256 digest")
    for size_name in (
        "score_midi_size_bytes",
        "performance_midi_size_bytes",
        "alignment_size_bytes",
    ):
        if _required_nonnegative_int(row.get(size_name), f"{prefix}.{size_name}") == 0:
            _fail(f"{prefix}.{size_name} must be positive")

    interpolated = _required_nonnegative_int(
        row.get("interpolated_count"), f"{prefix}.interpolated_count"
    )
    supervised_notes = _required_nonnegative_int(
        row.get("supervised_note_count"), f"{prefix}.supervised_note_count"
    )
    supervised_chords = _required_nonnegative_int(
        row.get("supervised_chord_count"), f"{prefix}.supervised_chord_count"
    )
    onset_masked = _required_nonnegative_int(
        row.get("onset_delta_aux_masked_count"),
        f"{prefix}.onset_delta_aux_masked_count",
    )
    if interpolated > note_count or supervised_notes > note_count:
        _fail(f"{prefix} note diagnostics exceed note_count")
    if supervised_chords > chord_count or onset_masked > chord_count:
        _fail(f"{prefix} chord diagnostics exceed chord_count")
    if onset_masked > chord_count - supervised_chords:
        _fail(
            f"{prefix}.onset_delta_aux_masked_count exceeds unsupervised chords"
        )
    interpolation_rate = _required_finite_number(
        row.get("interpolation_rate"), f"{prefix}.interpolation_rate"
    )
    if not math.isclose(
        interpolation_rate,
        interpolated / note_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _fail(f"{prefix}.interpolation_rate does not match its counts")
    recall = _required_finite_number(row.get("refined_recall"), f"{prefix}.refined_recall")
    if not 0.0 <= recall <= 1.0:
        _fail(f"{prefix}.refined_recall must be between zero and one")
    if _required_finite_number(
        row.get("global_tempo_scale"), f"{prefix}.global_tempo_scale"
    ) <= 0:
        _fail(f"{prefix}.global_tempo_scale must be positive")
    if not isinstance(row.get("has_pedal"), bool):
        _fail(f"{prefix}.has_pedal must be boolean")
    warnings = _required_list(row.get("warning_flags"), f"{prefix}.warning_flags")
    if any(not isinstance(item, str) or not item for item in warnings):
        _fail(f"{prefix}.warning_flags must contain non-empty strings")
    expected_status = "warning" if warnings else "accepted"
    if row.get("qc_status") != expected_status:
        _fail(f"{prefix}.qc_status does not match warning_flags")
    return shard_path, split, source, note_count, chord_count


def _validate_shard(
    relative: str,
    inventory_entry: Mapping[str, Any],
    rows: list[dict[str, Any]],
    dimensions: tuple[int, int, int, int],
    normalizer_accumulator: TrainOnlyNormalizer,
) -> tuple[int, int, int]:
    path = inventory_entry["path"]
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if len(loaded.files) != len(set(loaded.files)):
                _fail(f"{relative}: NPZ contains duplicate member names")
            if set(loaded.files) != SHARD_ARRAY_NAMES:
                _fail(
                    f"{relative}: NPZ member mismatch: "
                    f"missing={sorted(SHARD_ARRAY_NAMES - set(loaded.files))}, "
                    f"extra={sorted(set(loaded.files) - SHARD_ARRAY_NAMES)}"
                )
            arrays = {name: loaded[name] for name in SHARD_ARRAY_NAMES}
    except DatasetValidationError:
        raise
    except Exception as exc:
        _fail(f"{relative}: cannot load all arrays with allow_pickle=False: {exc}")

    note_features = arrays["note_features"]
    note_targets = arrays["note_targets"]
    note_masks = arrays["note_masks"]
    chord_features = arrays["chord_features"]
    chord_targets = arrays["chord_targets"]
    chord_masks = arrays["chord_masks"]
    note_to_chord = arrays["note_to_chord"]
    note_offsets = arrays["sample_note_offsets"]
    chord_offsets = arrays["sample_chord_offsets"]
    sample_ids = arrays["sample_ids"]

    for name, expected_dtype in SHARD_ARRAY_DTYPES.items():
        array = arrays[name]
        if expected_dtype == "unicode":
            if array.dtype.kind != "U":
                _fail(f"{relative}:{name} must use a Unicode dtype, got {array.dtype}")
        elif str(array.dtype) != expected_dtype:
            _fail(
                f"{relative}:{name} dtype mismatch: "
                f"expected {expected_dtype}, got {array.dtype}"
            )

    if note_features.ndim != 2:
        _fail(f"{relative}: note_features must be two-dimensional")
    if chord_features.ndim != 2:
        _fail(f"{relative}: chord_features must be two-dimensional")
    note_count = note_features.shape[0]
    chord_count = chord_features.shape[0]
    note_feature_dim, note_target_dim, chord_feature_dim, chord_target_dim = dimensions
    _require_array_shape(note_features, (note_count, note_feature_dim), f"{relative}:note_features")
    _require_array_shape(note_targets, (note_count, note_target_dim), f"{relative}:note_targets")
    _require_array_shape(note_masks, (note_count, note_target_dim), f"{relative}:note_masks")
    _require_array_shape(chord_features, (chord_count, chord_feature_dim), f"{relative}:chord_features")
    _require_array_shape(chord_targets, (chord_count, chord_target_dim), f"{relative}:chord_targets")
    _require_array_shape(chord_masks, (chord_count, chord_target_dim), f"{relative}:chord_masks")
    if note_masks.dtype != np.bool_:
        _fail(f"{relative}:note_masks must have bool dtype, got {note_masks.dtype}")
    if chord_masks.dtype != np.bool_:
        _fail(f"{relative}:chord_masks must have bool dtype, got {chord_masks.dtype}")
    if note_count:
        base_note_mask = note_masks[:, 0]
        if not bool(np.all(note_masks[:, :4] == base_note_mask[:, None])):
            _fail(
                f"{relative}: first four note target masks must agree per note"
            )
        if not bool(np.all(note_masks[:, 4] == note_masks[:, 5])):
            _fail(f"{relative}: pedal note target masks must agree per note")
        if bool(np.any(note_masks[:, 4] & ~base_note_mask)):
            _fail(
                f"{relative}: pedal supervision cannot include an interpolated note"
            )
    for name, array in (
        ("note_features", note_features),
        ("note_targets", note_targets),
        ("chord_features", chord_features),
        ("chord_targets", chord_targets),
    ):
        _require_finite(array, f"{relative}:{name}")
    for target_name in ("velocity_0_1", "pedal_at_onset", "pedal_coverage"):
        target_index = NOTE_TARGET_NAMES.index(target_name)
        values = note_targets[:, target_index]
        if bool(np.any((values < 0.0) | (values > 1.0))):
            _fail(f"{relative}:{target_name} values must stay within [0, 1]")

    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in "SU":
        _fail(f"{relative}:sample_ids must be a one-dimensional fixed-width string array")
    sample_count = len(sample_ids)
    _validate_offsets(note_offsets, sample_count, note_count, f"{relative}:sample_note_offsets")
    _validate_offsets(chord_offsets, sample_count, chord_count, f"{relative}:sample_chord_offsets")
    _require_array_shape(note_to_chord, (note_count,), f"{relative}:note_to_chord")
    if note_to_chord.dtype.kind not in "iu":
        _fail(f"{relative}:note_to_chord must use an integer dtype")
    if note_count and (
        chord_count == 0
        or int(note_to_chord.min()) < 0
        or int(note_to_chord.max()) >= chord_count
    ):
        _fail(f"{relative}:note_to_chord contains an out-of-range chord index")
    if note_count and bool(np.any(note_to_chord[1:] < note_to_chord[:-1])):
        _fail(f"{relative}:note_to_chord must be monotonically non-decreasing")
    if chord_count and not np.array_equal(
        np.unique(note_to_chord), np.arange(chord_count, dtype=note_to_chord.dtype)
    ):
        _fail(f"{relative}: every chord must be referenced by at least one note")

    if len(rows) != sample_count:
        _fail(
            f"{relative}: manifest sample count {len(rows)} does not match "
            f"sample_ids count {sample_count}"
        )
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row["sample_index"]
        if index in by_index:
            _fail(f"{relative}: duplicate manifest sample_index {index}")
        by_index[index] = row
    if set(by_index) != set(range(sample_count)):
        _fail(f"{relative}: manifest sample_index values are not contiguous from zero")

    for index in range(sample_count):
        row = by_index[index]
        note_start, note_end = int(note_offsets[index]), int(note_offsets[index + 1])
        chord_start, chord_end = int(chord_offsets[index]), int(chord_offsets[index + 1])
        if str(sample_ids[index]) != row["id"]:
            _fail(f"{relative}: sample_ids[{index}] does not match manifest id")
        expected_values = {
            "note_offset": note_start,
            "chord_offset": chord_start,
            "note_count": note_end - note_start,
            "chord_count": chord_end - chord_start,
            "shard_size_bytes": inventory_entry["size_bytes"],
            "shard_sha256": inventory_entry["sha256"],
        }
        for key, expected in expected_values.items():
            if row.get(key) != expected:
                _fail(
                    f"{relative}: manifest {row['id']!r} {key} mismatch: "
                    f"expected {expected!r}, got {row.get(key)!r}"
                )
        sample_mapping = note_to_chord[note_start:note_end]
        if len(sample_mapping) and (
            chord_start == chord_end
            or int(sample_mapping.min()) < chord_start
            or int(sample_mapping.max()) >= chord_end
        ):
            _fail(f"{relative}: sample {row['id']!r} maps notes outside its chord range")
        expected_chords = np.arange(
            chord_start, chord_end, dtype=note_to_chord.dtype
        )
        if not np.array_equal(np.unique(sample_mapping), expected_chords):
            _fail(
                f"{relative}: sample {row['id']!r} does not reference every chord"
            )
        sample_note_masks = note_masks[note_start:note_end]
        sample_chord_masks = chord_masks[chord_start:chord_end]
        sample_chord_targets = chord_targets[chord_start:chord_end]
        supervised_notes = int(sample_note_masks[:, 0].sum())
        interpolated_notes = len(sample_note_masks) - supervised_notes
        supervised_chords = int(sample_chord_masks[:, 0].sum())
        expected_diagnostics = {
            "interpolated_count": interpolated_notes,
            "supervised_note_count": supervised_notes,
            "supervised_chord_count": supervised_chords,
        }
        for key, expected in expected_diagnostics.items():
            if row.get(key) != expected:
                _fail(
                    f"{relative}: manifest {row['id']!r} {key} mismatch: "
                    f"expected {expected}, got {row.get(key)!r}"
                )
        if not row["has_pedal"] and bool(np.any(sample_note_masks[:, 4])):
            _fail(
                f"{relative}: manifest {row['id']!r} has_pedal=false "
                "but pedal targets are supervised"
            )
        if sample_chord_masks[0, 1]:
            _fail(
                f"{relative}: sample {row['id']!r} first chord cannot supervise log_ioi_ratio"
            )
        if bool(
            np.any(
                sample_chord_masks[:, 0]
                & (np.abs(sample_chord_targets[:, 0]) > 2_000.0)
            )
        ):
            _fail(
                f"{relative}: sample {row['id']!r} supervises an onset_delta_ms "
                "outside the supported bound"
            )

    # Shards are split-specific by path contract.  Updating the fixed 20k-bin
    # histograms once per train shard avoids eight large temporary bincount
    # allocations for every individual sample while producing identical sums.
    if rows and rows[0]["split"] == "train":
        normalizer_accumulator.update(
            PerformanceArrays(
                note_features=note_features,
                note_targets=note_targets,
                note_masks=note_masks,
                chord_features=chord_features,
                chord_targets=chord_targets,
                chord_masks=chord_masks,
                note_to_chord=note_to_chord,
                diagnostics={},
            ),
            "train",
        )
    return sample_count, note_count, chord_count


def _compare_normalizer_value(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            _fail(f"{label} keys do not match the recomputed normalizer")
        for key in expected:
            _compare_normalizer_value(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _fail(f"{label} does not match the recomputed normalizer")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_normalizer_value(
                actual_item, expected_item, f"{label}[{index}]"
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            _fail(f"{label} must be numeric")
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            _fail(f"{label} does not match the recomputed normalizer")
        return
    if actual != expected:
        _fail(f"{label} does not match the recomputed normalizer")


def _validate_normalizer(
    actual: Any, accumulator: TrainOnlyNormalizer
) -> None:
    _compare_normalizer_value(
        _required_object(actual, "normalizer.json"),
        accumulator.summary(),
        "normalizer.json",
    )


def _validate_qc_summary(
    summary: Any,
    *,
    accepted_count: int,
    rejected_count: int,
    split_counts: Counter[str],
    source_counts: Counter[str],
    reason_counts: Counter[str],
    note_total: int,
    chord_total: int,
    piece_overlap: int,
    score_path_overlap: int,
    score_semantic_overlap: int,
    performer_count: int,
    performer_overlap: int,
) -> None:
    value = _required_object(summary, "qc_summary.json")
    expected_scalars = {
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "note_count": note_total,
        "chord_count": chord_total,
    }
    for key, expected in expected_scalars.items():
        actual = _required_nonnegative_int(value.get(key), f"qc_summary.json.{key}")
        if actual != expected:
            _fail(f"qc_summary.json.{key} mismatch: expected {expected}, got {actual}")
    selected = _required_nonnegative_int(value.get("selected_count"), "qc_summary.json.selected_count")
    if selected != accepted_count + rejected_count:
        _fail(
            "qc_summary.json.selected_count must equal accepted_count + rejected_count"
        )
    _validate_count_map(value.get("split_counts"), split_counts, "qc_summary.json.split_counts")
    _validate_count_map(value.get("source_counts"), source_counts, "qc_summary.json.source_counts")
    _validate_count_map(
        value.get("reject_reason_counts"),
        reason_counts,
        "qc_summary.json.reject_reason_counts",
    )
    pass_rate = value.get("pass_rate")
    if isinstance(pass_rate, bool) or not isinstance(pass_rate, (int, float)):
        _fail("qc_summary.json.pass_rate must be numeric")
    expected_pass_rate = accepted_count / selected if selected else 0.0
    if not math.isfinite(float(pass_rate)) or not math.isclose(
        float(pass_rate), expected_pass_rate, rel_tol=0.0, abs_tol=1e-15
    ):
        _fail(
            "qc_summary.json.pass_rate mismatch: "
            f"expected {expected_pass_rate}, got {pass_rate}"
        )

    computed = {
        "piece_group_split_overlap": piece_overlap,
        "score_path_split_overlap": score_path_overlap,
        "score_semantic_split_overlap": score_semantic_overlap,
    }
    for key, actual_overlap in computed.items():
        reported = _required_nonnegative_int(value.get(key), f"qc_summary.json.{key}")
        if reported != 0 or actual_overlap != 0:
            _fail(
                f"{key} must be zero (reported={reported}, recomputed={actual_overlap})"
            )
    reported_performer_overlap = _required_nonnegative_int(
        value.get("performer_split_overlap"),
        "qc_summary.json.performer_split_overlap",
    )
    if reported_performer_overlap != performer_overlap:
        _fail(
            "performer_split_overlap mismatch: "
            f"reported={reported_performer_overlap}, recomputed={performer_overlap}"
        )
    reported_performer_count = _required_nonnegative_int(
        value.get("performer_count"), "qc_summary.json.performer_count"
    )
    if reported_performer_count != performer_count:
        _fail(
            "performer_count mismatch: "
            f"reported={reported_performer_count}, recomputed={performer_count}"
        )


def validate_dataset(dataset_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a completed dataset without changing any file.

    Returns a compact, JSON-serializable report.  The first invariant failure
    raises :class:`DatasetValidationError`.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        _fail(f"dataset root is not a directory: {root}")
    inventory, shard_paths = _validate_inventory(root)
    build_config = _load_json(root / "build_config.json")
    schema = _load_json(root / "schema.json")
    summary = _load_json(root / "qc_summary.json")
    normalizer = _load_json(root / "normalizer.json")
    integrity = _load_json(root / "integrity.json")
    _validate_metadata_contract(
        build_config=build_config,
        schema=schema,
        summary=summary,
        normalizer=normalizer,
        integrity=integrity,
    )
    dimensions = _validate_schema(schema)
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    accepted_ids: set[str] = set()
    piece_splits: dict[str, set[str]] = defaultdict(set)
    score_path_splits: dict[str, set[str]] = defaultdict(set)
    score_semantic_splits: dict[str, set[str]] = defaultdict(set)
    performer_splits: dict[str, set[str]] = defaultdict(set)
    score_path_identity: dict[str, tuple[str, str, str]] = {}
    semantic_piece_group: dict[str, str] = {}
    processed_shards: set[str] = set()
    current_shard: str | None = None
    current_rows: list[dict[str, Any]] = []
    accepted_count = 0
    manifest_notes = 0
    manifest_chords = 0
    shard_sample_total = 0
    shard_note_total = 0
    shard_chord_total = 0
    recomputed_normalizer = TrainOnlyNormalizer()

    def finish_current_shard() -> None:
        nonlocal shard_sample_total, shard_note_total, shard_chord_total
        if current_shard is None:
            return
        if current_shard in processed_shards:
            _fail(f"manifest rows for shard {current_shard!r} are not contiguous")
        sample_count, note_count, chord_count = _validate_shard(
            current_shard,
            inventory[current_shard],
            current_rows,
            dimensions,
            recomputed_normalizer,
        )
        processed_shards.add(current_shard)
        shard_sample_total += sample_count
        shard_note_total += note_count
        shard_chord_total += chord_count

    for row_index, row in enumerate(_iter_jsonl(root / "manifest.jsonl")):
        shard_path, split, source, note_count, chord_count = _validate_manifest_record(
            root, row, row_index, shard_paths
        )
        sample_id = row["id"]
        if sample_id in accepted_ids:
            _fail(f"manifest.jsonl contains duplicate id {sample_id!r}")
        accepted_ids.add(sample_id)
        if current_shard is None:
            current_shard = shard_path
        elif shard_path != current_shard:
            finish_current_shard()
            current_rows = []
            current_shard = shard_path
            if current_shard in processed_shards:
                _fail(f"manifest rows for shard {current_shard!r} are not contiguous")
        current_rows.append(row)

        accepted_count += 1
        manifest_notes += note_count
        manifest_chords += chord_count
        split_counts[split] += 1
        source_counts[source] += 1
        piece_group = row["piece_group"]
        semantic_hash = row["score_semantic_sha256"]
        normalized_score_path = PurePosixPath(
            str(row["score_midi_path"]).strip().replace("\\", "/")
        ).as_posix().casefold()
        piece_splits[piece_group].add(split)
        score_path_splits[normalized_score_path].add(split)
        score_semantic_splits[semantic_hash].add(split)
        performer = str(row["performer"]).strip().casefold()
        if performer:
            performer_splits[performer].add(split)
        identity = (
            row["score_midi_sha256"],
            semantic_hash,
            piece_group,
        )
        previous_identity = score_path_identity.setdefault(
            normalized_score_path, identity
        )
        if previous_identity != identity:
            _fail(
                f"score path {row['score_midi_path']!r} has inconsistent hashes or piece_group"
            )
        previous_piece = semantic_piece_group.setdefault(semantic_hash, piece_group)
        if previous_piece != piece_group:
            _fail("one semantic score hash maps to multiple piece_group values")

    finish_current_shard()
    if not accepted_count:
        _fail("manifest.jsonl contains no accepted samples")
    if processed_shards != shard_paths:
        _fail(
            "manifest/shard coverage mismatch: "
            f"unreferenced shards={sorted(shard_paths - processed_shards)}"
        )
    if shard_sample_total != accepted_count:
        _fail("aggregate shard sample count does not match manifest.jsonl")
    if shard_note_total != manifest_notes or shard_chord_total != manifest_chords:
        _fail("aggregate shard note/chord counts do not match manifest.jsonl")
    if not split_counts.get("train"):
        _fail("accepted dataset contains no train samples")

    rejected_ids: set[str] = set()
    rejected_count = 0
    for row_index, reject in enumerate(_iter_jsonl(root / "rejects.jsonl")):
        prefix = f"rejects.jsonl row {row_index}"
        sample_id = _required_string(reject.get("id"), f"{prefix}.id")
        if sample_id in rejected_ids:
            _fail(f"rejects.jsonl contains duplicate id {sample_id!r}")
        if sample_id in accepted_ids:
            _fail(f"sample id {sample_id!r} appears in both manifest and rejects")
        rejected_ids.add(sample_id)
        reason = _required_string(reject.get("reason_type"), f"{prefix}.reason_type")
        _required_string(reject.get("reason"), f"{prefix}.reason")
        _required_string(
            reject.get("performance_dataset"), f"{prefix}.performance_dataset"
        )
        reject_split = _required_string(reject.get("split"), f"{prefix}.split")
        if reject_split not in {"train", "validation", "test"}:
            _fail(f"{prefix}.split is unsupported: {reject_split!r}")
        reason_counts[reason] += 1
        rejected_count += 1
    config_object = _required_object(build_config, "build_config.json")
    if config_object.get("strict") is True and rejected_count:
        _fail("strict=true is inconsistent with a non-empty rejects.jsonl")

    piece_overlap = sum(len(values) > 1 for values in piece_splits.values())
    score_path_overlap = sum(
        len(values) > 1 for values in score_path_splits.values()
    )
    score_semantic_overlap = sum(
        len(values) > 1 for values in score_semantic_splits.values()
    )
    performer_overlap = sum(
        len(values) > 1 for values in performer_splits.values()
    )

    _validate_qc_summary(
        summary,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        split_counts=split_counts,
        source_counts=source_counts,
        reason_counts=reason_counts,
        note_total=shard_note_total,
        chord_total=shard_chord_total,
        piece_overlap=piece_overlap,
        score_path_overlap=score_path_overlap,
        score_semantic_overlap=score_semantic_overlap,
        performer_count=len(performer_splits),
        performer_overlap=performer_overlap,
    )
    _validate_normalizer(normalizer, recomputed_normalizer)
    return {
        "status": "valid",
        "dataset_root": str(root.resolve()),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "shard_count": len(shard_paths),
        "note_count": shard_note_total,
        "chord_count": shard_chord_total,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation for a completed PianoCoRe preprocessing dataset."
    )
    parser.add_argument("--dataset", required=True, help="preprocessed dataset root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = validate_dataset(args.dataset)
    except DatasetValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DatasetValidationError", "main", "validate_dataset"]
