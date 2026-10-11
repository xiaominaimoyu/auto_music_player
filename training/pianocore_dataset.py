"""Reproducible PianoCoRe-A* dataset builder for expressive-performance work.

Raw dataset files are opened read-only.  The builder keeps the official locked
test partition, creates validation groups from training works only, masks
interpolated notes, and writes deterministic ragged NumPy shards plus JSONL
manifests.  It deliberately does not train a model or change playback code.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from training.midi_features import (
    CHORD_FEATURE_NAMES,
    CHORD_TARGET_NAMES,
    NOTE_FEATURE_NAMES,
    NOTE_TARGET_NAMES,
    MidiFeatureError,
    ParsedMidi,
    PerformanceArrays,
    build_aligned_features,
    parse_midi_bytes,
)


PIPELINE_NAME = "auto_music_player.pianocore_expression"
PIPELINE_VERSION = 2
DEFAULT_SEED = 20260928
ZIP_PREFIX = "PianoCoRe/refined"
REQUIRED_COLUMNS = {
    "id",
    "composer",
    "composition",
    "movement",
    "performance_id",
    "split",
    "tier_a_star",
    "performance_dataset",
    "performer",
    "quality_label",
    "is_duplicate",
    "is_refined",
    "refined_score_midi_path",
    "refined_score_note_count",
    "refined_performance_midi_path",
    "refined_performance_note_count",
    "refined_performance_interpolated_note_count",
    "refined_alignment_path",
    "refined_recall",
}

# Keep only fields used after eligibility, grouping, and split assignment.  The
# upstream CSV contains many additional columns; retaining them for every A*
# row needlessly increases the parent process footprint during a full build and
# also makes each multiprocessing submission more expensive to pickle.
PROCESSING_COLUMNS = (
    "id",
    "composer",
    "composition",
    "movement",
    "performance_id",
    "split",
    "performance_dataset",
    "performer",
    "refined_score_midi_path",
    "refined_score_note_count",
    "refined_performance_midi_path",
    "refined_performance_note_count",
    "refined_performance_interpolated_note_count",
    "refined_alignment_path",
    "refined_recall",
)
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


class DatasetBuildError(RuntimeError):
    """Raised when dataset-level invariants are violated."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _pipeline_provenance() -> dict[str, Any]:
    """Return source/runtime fingerprints needed to reproduce a build."""

    source_files = {
        "training/midi_features.py": Path(__file__).with_name("midi_features.py"),
        "training/pianocore_dataset.py": Path(__file__),
    }
    try:
        mido_version = importlib.metadata.version("mido")
    except importlib.metadata.PackageNotFoundError:
        mido_version = "unknown"
    return {
        "source_sha256": {
            name: sha256_file(path) for name, path in sorted(source_files.items())
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "mido_version": mido_version,
            "zlib_version": zlib.ZLIB_VERSION,
        },
    }


def _as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetBuildError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DatasetBuildError(f"{label} is not finite")
    return number


def _safe_int(value: Any, label: str) -> int:
    number = _safe_float(value, label)
    if int(number) != number:
        raise DatasetBuildError(f"{label} is not an integer: {value!r}")
    return int(number)


def piece_key(row: Mapping[str, Any]) -> str:
    """Work-level grouping key; score_id is not reliable across this corpus."""

    identity = [
        str(row.get("composer") or "").strip(),
        str(row.get("composition") or "").strip(),
        str(row.get("movement") or "").strip(),
    ]
    if not identity[0] or not identity[1]:
        raise DatasetBuildError(f"row {row.get('id')} has no composer/composition")
    return _sha256_bytes(_canonical_json(identity))


def _normalized_score_path(row: Mapping[str, Any]) -> str:
    value = str(row.get("refined_score_midi_path") or "").strip().replace("\\", "/")
    return PurePosixPath(value).as_posix().casefold() if value else ""


def _tick_fraction(tick: int, ticks_per_beat: int) -> tuple[int, int]:
    divisor = math.gcd(abs(int(tick)), int(ticks_per_beat))
    return int(tick) // divisor, int(ticks_per_beat) // divisor


def score_semantic_fingerprint(score: ParsedMidi) -> str:
    """Hash score structure independently of PPQ, tempo, tracks, and encoding.

    Model inputs use beat-relative note geometry and the resulting per-note bar
    phase; raw time-signature events are not inputs by themselves.  Score
    velocity, channel, tempo events, and MIDI serialization are also ignored.
    Canonicalizing the same representation prevents an equivalent score saved
    with a different PPQ or track layout from leaking across dataset splits.
    """

    notes = []
    signature_index = 0
    for note in score.notes:
        while (
            signature_index + 1 < len(score.time_signatures)
            and score.time_signatures[signature_index + 1].tick <= note.start_tick
        ):
            signature_index += 1
        signature = score.time_signatures[signature_index]
        beats_from_signature = Fraction(
            note.start_tick - signature.tick, score.ticks_per_beat
        )
        bar_beats = Fraction(signature.numerator * 4, signature.denominator)
        bar_phase = (beats_from_signature % bar_beats) / bar_beats
        start = _tick_fraction(note.start_tick, score.ticks_per_beat)
        duration = _tick_fraction(
            max(0, note.end_tick - note.start_tick), score.ticks_per_beat
        )
        notes.append(
            (
                start[0],
                start[1],
                duration[0],
                duration[1],
                note.pitch,
                bar_phase.numerator,
                bar_phase.denominator,
            )
        )
    notes.sort()
    return _sha256_bytes(_canonical_json({"notes_with_bar_phase": notes}))


def connected_piece_groups(
    rows: Sequence[Mapping[str, Any]],
    score_semantic_hash_by_path: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Group rows by connected work identity, score path, and score semantics.

    A work can legitimately have multiple refined score files, while malformed
    metadata can also give one score file several work names.  Connected
    components enforce both constraints without choosing one key over the
    other.  Semantic hashes close the remaining same-score/different-path gap.
    """

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    row_nodes: dict[str, list[str]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id:
            raise DatasetBuildError("metadata row has no id")
        raw_score_path = str(row.get("refined_score_midi_path") or "").strip()
        normalized_path = _normalized_score_path(row)
        nodes = [f"piece:{piece_key(row)}"]
        if normalized_path:
            nodes.append(f"path:{normalized_path}")
        if score_semantic_hash_by_path is not None and raw_score_path:
            score_hash = score_semantic_hash_by_path.get(raw_score_path)
            if not score_hash:
                raise DatasetBuildError(
                    f"score semantic hash is missing: {raw_score_path}"
                )
            nodes.append(f"semantic_sha256:{score_hash}")
        for node in nodes[1:]:
            union(nodes[0], node)
        find(nodes[0])
        row_nodes[row_id] = nodes

    component_nodes: dict[str, set[str]] = defaultdict(set)
    for nodes in row_nodes.values():
        root = find(nodes[0])
        component_nodes[root].update(nodes)
    component_ids = {
        root: _sha256_bytes(_canonical_json(sorted(nodes)))
        for root, nodes in component_nodes.items()
    }
    return {
        row_id: component_ids[find(nodes[0])]
        for row_id, nodes in row_nodes.items()
    }


def assign_split(
    row: Mapping[str, Any],
    *,
    validation_fraction: float = 0.10,
    seed: int = DEFAULT_SEED,
    group_key: str | None = None,
) -> str:
    official = str(row.get("split") or "").strip().casefold()
    if official == "test":
        return "test"
    if official != "train":
        raise DatasetBuildError(f"unsupported official split: {official!r}")
    token = f"{seed}:{group_key or piece_key(row)}".encode("ascii")
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return "validation" if bucket < validation_fraction else "train"


def read_metadata(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise DatasetBuildError(
                "metadata is missing required columns: " + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise DatasetBuildError("metadata contains no rows")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise DatasetBuildError("metadata id values are not unique")
    return rows


def validate_metadata_splits(
    rows: Sequence[Mapping[str, Any]],
    score_semantic_hash_by_path: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Prove that official train/test do not share a musical work or score path."""

    memberships: dict[str, set[str]] = defaultdict(set)
    score_path_memberships: dict[str, set[str]] = defaultdict(set)
    score_hash_memberships: dict[str, set[str]] = defaultdict(set)
    performer_memberships: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        official = str(row.get("split") or "").strip().casefold()
        if official not in {"train", "test"}:
            raise DatasetBuildError(f"unsupported official split: {official!r}")
        memberships[piece_key(row)].add(official)
        performer = str(row.get("performer") or "").strip().casefold()
        if performer:
            performer_memberships[performer].add(official)
        score_path = str(row.get("refined_score_midi_path") or "").strip()
        if score_path:
            score_path_memberships[_normalized_score_path(row)].add(official)
            if score_semantic_hash_by_path is not None:
                score_hash = score_semantic_hash_by_path.get(score_path)
                if not score_hash:
                    raise DatasetBuildError(
                        f"score semantic hash is missing: {score_path}"
                    )
                score_hash_memberships[score_hash].add(official)
    leaked_pieces = sorted(key for key, values in memberships.items() if len(values) > 1)
    leaked_score_paths = sorted(
        key for key, values in score_path_memberships.items() if len(values) > 1
    )
    leaked_score_hashes = sorted(
        key for key, values in score_hash_memberships.items() if len(values) > 1
    )
    if leaked_pieces or leaked_score_paths or leaked_score_hashes:
        raise DatasetBuildError(
            "official split leakage detected: "
            f"piece_groups={len(leaked_pieces)} score_paths={len(leaked_score_paths)} "
            f"score_semantic_hashes={len(leaked_score_hashes)}"
        )
    return {
        "piece_group_count": len(memberships),
        "score_path_count": len(score_path_memberships),
        "official_piece_group_overlap": 0,
        "official_score_path_overlap": 0,
        "score_semantic_hash_count": len(score_hash_memberships),
        "official_score_semantic_overlap": 0,
        "performer_count": len(performer_memberships),
        "official_performer_overlap": sum(
            len(splits) > 1 for splits in performer_memberships.values()
        ),
    }


def _eligible_metadata_row(row: Mapping[str, Any]) -> bool:
    return (
        _as_bool(row.get("tier_a_star"))
        and _as_bool(row.get("is_refined"))
        and not _as_bool(row.get("is_duplicate"))
        and str(row.get("quality_label") or "").strip().casefold() == "high quality"
        and bool(str(row.get("refined_score_midi_path") or "").strip())
        and bool(str(row.get("refined_performance_midi_path") or "").strip())
        and bool(str(row.get("refined_alignment_path") or "").strip())
    )


def select_rows(
    rows: Sequence[dict[str, str]],
    *,
    mode: str,
    validation_fraction: float,
    seed: int,
    smoke_per_group: int,
    max_records: int | None,
    score_semantic_hash_by_path: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    # Hold references only while computing connected groups.  Each entry is
    # replaced in place by a compact dict below, avoiding a second full-width
    # copy of the eligible metadata table at peak memory.
    selected = [row for row in rows if _eligible_metadata_row(row)]
    group_keys = connected_piece_groups(selected, score_semantic_hash_by_path)
    for index, row in enumerate(selected):
        piece_group = group_keys[row["id"]]
        split = assign_split(
            row,
            validation_fraction=validation_fraction,
            seed=seed,
            group_key=piece_group,
        )
        selected[index] = {
            **{column: row[column] for column in PROCESSING_COLUMNS},
            "_piece_group": piece_group,
            "_split": split,
            "_score_semantic_sha256": (
                score_semantic_hash_by_path[row["refined_score_midi_path"]]
                if score_semantic_hash_by_path is not None
                else ""
            ),
        }
    ordering = lambda row: (
        row["_split"],
        row["performance_dataset"],
        row["_piece_group"],
        row["id"],
    )
    selected.sort(key=ordering)
    if mode == "smoke":
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in selected:
            grouped[(row["_split"], row["performance_dataset"])].append(row)
        smoke: list[dict[str, str]] = []
        for key in sorted(grouped):
            candidates = sorted(
                grouped[key],
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row['id']}".encode("utf-8")
                ).digest(),
            )
            smoke.extend(candidates[:smoke_per_group])
        selected = sorted(smoke, key=ordering)
    if max_records is not None:
        selected = selected[: max(0, int(max_records))]
    if not selected:
        raise DatasetBuildError("selection contains no eligible rows")
    return selected


class RefinedArchive:
    """Read-only, path-confined view of the refined ZIP package."""

    def __init__(self, archive_path: str | os.PathLike[str]):
        self.path = Path(archive_path).resolve()
        self._zip: zipfile.ZipFile | None = None

    def __enter__(self) -> "RefinedArchive":
        self._zip = zipfile.ZipFile(self.path, "r")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._zip is not None:
            self._zip.close()
        self._zip = None

    @staticmethod
    def _member(relative_path: str) -> str:
        path = PurePosixPath(str(relative_path).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise DatasetBuildError(f"unsafe refined path: {relative_path!r}")
        return str(PurePosixPath(ZIP_PREFIX) / path)

    def read(self, relative_path: str) -> bytes:
        if self._zip is None:
            raise RuntimeError("RefinedArchive must be used as a context manager")
        member = self._member(relative_path)
        try:
            return self._zip.read(member)
        except KeyError as exc:
            raise DatasetBuildError(f"archive member is missing: {member}") from exc


def score_semantic_hashes(
    rows: Sequence[Mapping[str, Any]],
    archive: RefinedArchive,
) -> dict[str, str]:
    """Fingerprint each eligible refined score once for semantic split grouping."""

    paths = sorted(
        {
            str(row.get("refined_score_midi_path") or "").strip()
            for row in rows
            if _eligible_metadata_row(row)
        }
    )
    if not paths or not paths[0]:
        raise DatasetBuildError("eligible metadata contains no refined score paths")
    return {
        path: score_semantic_fingerprint(parse_midi_bytes(archive.read(path)))
        for path in paths
    }


def write_deterministic_npz(path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]) -> str:
    """Write a compressed NPZ with fixed metadata and return its SHA-256."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return sha256_file(output)


@dataclass
class _Histogram:
    low: float
    high: float
    bins: int = 20_000
    counts: np.ndarray = field(init=False)
    count: int = 0
    below: int = 0
    above: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.bins, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.count += len(values)
        self.below += int((values < self.low).sum())
        self.above += int((values > self.high).sum())
        self.total += float(values.sum(dtype=np.float64))
        self.total_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        clipped = np.clip(values, self.low, np.nextafter(self.high, self.low))
        indices = ((clipped - self.low) / (self.high - self.low) * self.bins).astype(np.int64)
        indices = np.clip(indices, 0, self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins)

    def quantile(self, probability: float) -> float | None:
        if not self.count:
            return None
        target = probability * max(0, self.count - 1)
        index = int(np.searchsorted(np.cumsum(self.counts), target + 1, side="left"))
        return self.low + (index + 0.5) * (self.high - self.low) / self.bins

    def summary(self) -> dict[str, Any]:
        q1, median, q3 = self.quantile(0.25), self.quantile(0.5), self.quantile(0.75)
        mean = self.total / self.count if self.count else None
        variance = (
            max(0.0, self.total_sq / self.count - mean * mean)
            if self.count and mean is not None
            else None
        )
        return {
            "count": self.count,
            "histogram_range": [self.low, self.high],
            "below_range": self.below,
            "above_range": self.above,
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
        }


class TrainOnlyNormalizer:
    """Deterministic bounded histograms; update calls are train-only by contract."""

    NOTE_RANGES = {
        "micro_offset_ms": (-500.0, 500.0),
        "log_key_duration_ratio": (-4.0, 4.0),
        "velocity_0_1": (0.0, 1.0),
        "velocity_robust_z": (-8.0, 8.0),
        "pedal_at_onset": (0.0, 1.0),
        "pedal_coverage": (0.0, 1.0),
    }
    CHORD_RANGES = {
        "onset_delta_ms": (-2000.0, 2000.0),
        "log_ioi_ratio": (-4.0, 4.0),
    }

    def __init__(self) -> None:
        self.note = {name: _Histogram(*self.NOTE_RANGES[name]) for name in NOTE_TARGET_NAMES}
        self.chord = {
            name: _Histogram(*self.CHORD_RANGES[name]) for name in CHORD_TARGET_NAMES
        }

    def update(self, arrays: PerformanceArrays, split: str) -> None:
        if split != "train":
            return
        for index, name in enumerate(NOTE_TARGET_NAMES):
            self.note[name].update(arrays.note_targets[arrays.note_masks[:, index], index])
        for index, name in enumerate(CHORD_TARGET_NAMES):
            self.chord[name].update(
                arrays.chord_targets[arrays.chord_masks[:, index], index]
            )

    def summary(self) -> dict[str, Any]:
        return {
            "fit_split": "train",
            "method": "fixed-range deterministic histogram median/IQR",
            "note_targets": {name: value.summary() for name, value in self.note.items()},
            "chord_targets": {name: value.summary() for name, value in self.chord.items()},
        }


@dataclass
class _PendingSample:
    sample_id: str
    arrays: PerformanceArrays
    manifest: dict[str, Any]


class ShardWriter:
    def __init__(
        self,
        output_root: Path,
        split: str,
        source: str,
        *,
        max_samples: int,
        max_notes: int,
    ):
        self.output_root = output_root
        self.split = split
        self.source = source
        self.max_samples = max_samples
        self.max_notes = max_notes
        self._pending: list[_PendingSample] = []
        self._pending_notes = 0
        self._index = 0

    def add(self, sample: _PendingSample) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        note_count = len(sample.arrays.note_features)
        if self._pending and (
            len(self._pending) >= self.max_samples
            or self._pending_notes + note_count > self.max_notes
        ):
            completed.extend(self.flush())
        self._pending.append(sample)
        self._pending_notes += note_count
        return completed

    def flush(self) -> list[dict[str, Any]]:
        if not self._pending:
            return []
        safe_source = source_directory_name(self.source)
        relative = Path("shards") / f"split={self.split}" / f"source={safe_source}" / f"part-{self._index:05d}.npz"
        note_offsets = [0]
        chord_offsets = [0]
        note_features = []
        note_targets = []
        note_masks = []
        chord_features = []
        chord_targets = []
        chord_masks = []
        note_to_chord = []
        sample_ids = []
        manifests = []
        for sample_index, sample in enumerate(self._pending):
            arrays = sample.arrays
            note_features.append(arrays.note_features)
            note_targets.append(arrays.note_targets)
            note_masks.append(arrays.note_masks)
            chord_features.append(arrays.chord_features)
            chord_targets.append(arrays.chord_targets)
            chord_masks.append(arrays.chord_masks)
            note_to_chord.append(arrays.note_to_chord + chord_offsets[-1])
            note_offsets.append(note_offsets[-1] + len(arrays.note_features))
            chord_offsets.append(chord_offsets[-1] + len(arrays.chord_features))
            sample_ids.append(sample.sample_id)
            manifest = dict(sample.manifest)
            manifest.update(
                {
                    "shard_path": relative.as_posix(),
                    "sample_index": sample_index,
                    "note_offset": note_offsets[-2],
                    "chord_offset": chord_offsets[-2],
                }
            )
            manifests.append(manifest)
        max_id_length = max(1, max(len(sample_id) for sample_id in sample_ids))
        arrays_to_write = {
            "chord_features": np.concatenate(chord_features, axis=0),
            "chord_masks": np.concatenate(chord_masks, axis=0),
            "chord_targets": np.concatenate(chord_targets, axis=0),
            "note_features": np.concatenate(note_features, axis=0),
            "note_masks": np.concatenate(note_masks, axis=0),
            "note_targets": np.concatenate(note_targets, axis=0),
            "note_to_chord": np.concatenate(note_to_chord, axis=0),
            "sample_chord_offsets": np.asarray(chord_offsets, dtype=np.int64),
            "sample_ids": np.asarray(sample_ids, dtype=f"<U{max_id_length}"),
            "sample_note_offsets": np.asarray(note_offsets, dtype=np.int64),
        }
        shard_hash = write_deterministic_npz(self.output_root / relative, arrays_to_write)
        shard_size = (self.output_root / relative).stat().st_size
        for manifest in manifests:
            manifest["shard_sha256"] = shard_hash
            manifest["shard_size_bytes"] = shard_size
        self._pending.clear()
        self._pending_notes = 0
        self._index += 1
        return manifests


def source_directory_name(source: str) -> str:
    """Return a readable, collision-resistant directory component."""

    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_") or "unknown"
    suffix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
    return f"{readable}--{suffix}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        for value in values:
            handle.write(_canonical_json(value) + b"\n")


def _load_alignment(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != {"perf_idx", "interpolated"}:
                raise DatasetBuildError(
                    f"unexpected alignment arrays: {sorted(archive.files)}"
                )
            return np.asarray(archive["perf_idx"]), np.asarray(archive["interpolated"])
    except DatasetBuildError:
        raise
    except Exception as exc:
        raise DatasetBuildError(f"alignment NPZ parse failed: {exc}") from exc


def _process_row(
    row: Mapping[str, str],
    source: RefinedArchive,
    score_cache: dict[str, tuple[ParsedMidi, str, int]],
    *,
    max_interpolation_rate: float,
    max_score_cache_entries: int,
) -> _PendingSample:
    score_path = row["refined_score_midi_path"]
    cached_score = score_cache.pop(score_path, None)
    if cached_score is None:
        score_bytes = source.read(score_path)
        cached_score = (
            parse_midi_bytes(score_bytes),
            _sha256_bytes(score_bytes),
            len(score_bytes),
        )
    score_cache[score_path] = cached_score
    while len(score_cache) > max_score_cache_entries:
        score_cache.pop(next(iter(score_cache)))
    score, score_hash, score_size = cached_score
    performance_bytes = source.read(row["refined_performance_midi_path"])
    alignment_bytes = source.read(row["refined_alignment_path"])
    performance = parse_midi_bytes(performance_bytes)
    perf_idx, interpolated = _load_alignment(alignment_bytes)

    expected_score = _safe_int(row["refined_score_note_count"], "refined_score_note_count")
    expected_performance = _safe_int(
        row["refined_performance_note_count"], "refined_performance_note_count"
    )
    expected_interpolated = _safe_int(
        row["refined_performance_interpolated_note_count"],
        "refined_performance_interpolated_note_count",
    )
    if len(score.notes) != expected_score:
        raise DatasetBuildError(
            f"score note count mismatch: parsed={len(score.notes)} metadata={expected_score}"
        )
    if len(performance.notes) != expected_performance:
        raise DatasetBuildError(
            "performance note count mismatch: "
            f"parsed={len(performance.notes)} metadata={expected_performance}"
        )
    if int(np.asarray(interpolated, dtype=np.bool_).sum()) != expected_interpolated:
        raise DatasetBuildError(
            "interpolated note count mismatch: "
            f"npz={int(np.asarray(interpolated, dtype=np.bool_).sum())} "
            f"metadata={expected_interpolated}"
        )
    interpolation_rate = expected_interpolated / max(1, expected_score)
    if interpolation_rate > max_interpolation_rate:
        raise DatasetBuildError(
            f"interpolation rate {interpolation_rate:.6f} exceeds {max_interpolation_rate:.6f}"
        )
    arrays = build_aligned_features(score, performance, perf_idx, interpolated)
    metadata_recall = _safe_float(row["refined_recall"], "refined_recall")
    warning_flags = []
    if interpolation_rate > 0.10:
        warning_flags.append("interpolation_rate_over_10_percent")
    if arrays.diagnostics["micro_offset_abs_p95_ms"] > 100:
        warning_flags.append("large_chord_spread")
    manifest = {
        "id": row["id"],
        "source": "pianocore_a_star_refined",
        "performance_dataset": row["performance_dataset"],
        "performer": str(row.get("performer") or "").strip(),
        "performance_id": row["performance_id"],
        "official_split": str(row["split"]).strip().casefold(),
        "split": row["_split"],
        "piece_group": row["_piece_group"],
        "composer": row["composer"],
        "composition": row["composition"],
        "movement": row["movement"],
        "score_midi_path": score_path,
        "performance_midi_path": row["refined_performance_midi_path"],
        "alignment_path": row["refined_alignment_path"],
        "score_midi_sha256": score_hash,
        "score_semantic_sha256": row["_score_semantic_sha256"],
        "performance_midi_sha256": _sha256_bytes(performance_bytes),
        "alignment_sha256": _sha256_bytes(alignment_bytes),
        "score_midi_size_bytes": score_size,
        "performance_midi_size_bytes": len(performance_bytes),
        "alignment_size_bytes": len(alignment_bytes),
        "refined_recall": metadata_recall,
        "note_count": int(arrays.diagnostics["note_count"]),
        "chord_count": int(arrays.diagnostics["chord_count"]),
        "interpolated_count": int(arrays.diagnostics["interpolated_count"]),
        "interpolation_rate": float(arrays.diagnostics["interpolation_rate"]),
        "supervised_note_count": int(arrays.diagnostics["supervised_note_count"]),
        "supervised_chord_count": int(arrays.diagnostics["supervised_chord_count"]),
        "onset_delta_aux_masked_count": int(
            arrays.diagnostics["onset_delta_aux_masked_count"]
        ),
        "global_tempo_scale": float(arrays.diagnostics["global_tempo_scale"]),
        "has_pedal": bool(arrays.diagnostics["has_pedal"]),
        "warning_flags": warning_flags,
        "qc_status": "warning" if warning_flags else "accepted",
    }
    return _PendingSample(row["id"], arrays, manifest)


@dataclass
class _ProcessOutcome:
    sample: _PendingSample | None = None
    error_type: str | None = None
    error_message: str | None = None


def _safe_process_row(
    row: Mapping[str, str],
    source: RefinedArchive,
    score_cache: dict[str, tuple[ParsedMidi, str, int]],
    *,
    max_interpolation_rate: float,
    max_score_cache_entries: int,
) -> _ProcessOutcome:
    try:
        return _ProcessOutcome(
            sample=_process_row(
                row,
                source,
                score_cache,
                max_interpolation_rate=max_interpolation_rate,
                max_score_cache_entries=max_score_cache_entries,
            )
        )
    except (DatasetBuildError, MidiFeatureError, KeyError, ValueError) as exc:
        return _ProcessOutcome(
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


_WORKER_ARCHIVE: RefinedArchive | None = None
_WORKER_SCORE_CACHE: dict[str, tuple[ParsedMidi, str, int]] = {}
_WORKER_MAX_INTERPOLATION_RATE = 0.20
_WORKER_MAX_SCORE_CACHE_ENTRIES = 16


def _close_worker_archive() -> None:
    global _WORKER_ARCHIVE
    if _WORKER_ARCHIVE is not None:
        _WORKER_ARCHIVE.__exit__(None, None, None)
        _WORKER_ARCHIVE = None


def _initialize_worker(
    archive_path: str,
    max_interpolation_rate: float,
    max_score_cache_entries: int,
) -> None:
    global _WORKER_ARCHIVE
    global _WORKER_SCORE_CACHE
    global _WORKER_MAX_INTERPOLATION_RATE
    global _WORKER_MAX_SCORE_CACHE_ENTRIES
    _WORKER_ARCHIVE = RefinedArchive(archive_path)
    _WORKER_ARCHIVE.__enter__()
    _WORKER_SCORE_CACHE = {}
    _WORKER_MAX_INTERPOLATION_RATE = max_interpolation_rate
    _WORKER_MAX_SCORE_CACHE_ENTRIES = max_score_cache_entries
    atexit.register(_close_worker_archive)


def _worker_process_row(row: Mapping[str, str]) -> _ProcessOutcome:
    if _WORKER_ARCHIVE is None:
        raise RuntimeError("dataset worker archive was not initialized")
    return _safe_process_row(
        row,
        _WORKER_ARCHIVE,
        _WORKER_SCORE_CACHE,
        max_interpolation_rate=_WORKER_MAX_INTERPOLATION_RATE,
        max_score_cache_entries=_WORKER_MAX_SCORE_CACHE_ENTRIES,
    )


def _parallel_outcomes(
    rows: Sequence[dict[str, str]],
    *,
    archive_path: Path,
    workers: int,
    max_interpolation_rate: float,
    max_score_cache_entries: int,
) -> Iterator[tuple[dict[str, str], _ProcessOutcome]]:
    """Process rows concurrently while yielding results in deterministic order."""

    max_pending = max(workers, workers * 2)
    indexed_rows = iter(enumerate(rows))
    pending: dict[int, tuple[dict[str, str], Future[_ProcessOutcome]]] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(
            str(archive_path),
            max_interpolation_rate,
            max_score_cache_entries,
        ),
    ) as executor:
        def fill() -> None:
            while len(pending) < max_pending:
                try:
                    index, row = next(indexed_rows)
                except StopIteration:
                    return
                pending[index] = (row, executor.submit(_worker_process_row, row))

        fill()
        next_index = 0
        while pending:
            row, future = pending.pop(next_index)
            yield row, future.result()
            next_index += 1
            fill()


def build_dataset(
    *,
    metadata_path: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    mode: str = "smoke",
    validation_fraction: float = 0.10,
    seed: int = DEFAULT_SEED,
    smoke_per_group: int = 2,
    max_records: int | None = None,
    max_samples_per_shard: int = 256,
    max_notes_per_shard: int = 250_000,
    max_interpolation_rate: float = 0.20,
    workers: int = 1,
    max_score_cache_entries: int = 16,
    strict: bool = False,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path).resolve()
    archive_path = Path(archive_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise DatasetBuildError(f"output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    incomplete_marker = output_path / "_INCOMPLETE"
    incomplete_marker.write_text("dataset build in progress\n", encoding="utf-8")

    if mode not in {"smoke", "full"}:
        raise DatasetBuildError("mode must be smoke or full")
    if not 0.0 < validation_fraction < 0.5:
        raise DatasetBuildError("validation_fraction must be between 0 and 0.5")
    if smoke_per_group <= 0 or max_samples_per_shard <= 0 or max_notes_per_shard <= 0:
        raise DatasetBuildError("shard and smoke limits must be positive")
    if not 0.0 <= max_interpolation_rate <= 1.0:
        raise DatasetBuildError("max_interpolation_rate must be between 0 and 1")
    if not 1 <= workers <= 32:
        raise DatasetBuildError("workers must be between 1 and 32")
    if max_score_cache_entries <= 0:
        raise DatasetBuildError("max_score_cache_entries must be positive")

    rows = read_metadata(metadata_path)
    eligible_rows = [row for row in rows if _eligible_metadata_row(row)]
    with RefinedArchive(archive_path) as archive:
        score_semantic_hash_by_path = score_semantic_hashes(eligible_rows, archive)
    split_audit = validate_metadata_splits(
        eligible_rows, score_semantic_hash_by_path
    )
    selected = select_rows(
        eligible_rows,
        mode=mode,
        validation_fraction=validation_fraction,
        seed=seed,
        smoke_per_group=smoke_per_group,
        max_records=max_records,
        score_semantic_hash_by_path=score_semantic_hash_by_path,
    )
    # ``selected`` contains compact processing records now; release the full
    # 130k-row metadata table before MIDI parsing and shard buffering begin.
    del eligible_rows, rows
    config = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "mode": mode,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "smoke_per_group": smoke_per_group,
        "max_records": max_records,
        "max_samples_per_shard": max_samples_per_shard,
        "max_notes_per_shard": max_notes_per_shard,
        "max_interpolation_rate": max_interpolation_rate,
        "workers": workers,
        "max_score_cache_entries": max_score_cache_entries,
        "strict": strict,
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_size_bytes": metadata_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "score_semantic_file_count": len(score_semantic_hash_by_path),
        "score_semantic_inventory_sha256": _sha256_bytes(
            _canonical_json(sorted(score_semantic_hash_by_path.items()))
        ),
        "selected_count": len(selected),
        "provenance": _pipeline_provenance(),
    }
    config["config_sha256"] = _sha256_bytes(_canonical_json(config))
    schema = {
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
    }
    _write_json(output_path / "build_config.json", config)
    _write_json(output_path / "schema.json", schema)

    score_cache: dict[str, tuple[ParsedMidi, str, int]] = {}
    writers: dict[tuple[str, str], ShardWriter] = {}
    normalizer = TrainOnlyNormalizer()
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    note_total = 0
    chord_total = 0
    manifest_count = 0
    reject_count = 0
    active_writer_key: tuple[str, str] | None = None
    closed_writer_keys: set[tuple[str, str]] = set()
    final_piece_splits: dict[str, set[str]] = defaultdict(set)
    final_score_path_splits: dict[str, set[str]] = defaultdict(set)
    final_score_semantic_splits: dict[str, set[str]] = defaultdict(set)
    final_performer_splits: dict[str, set[str]] = defaultdict(set)

    manifest_handle = open(output_path / "manifest.jsonl", "wb")
    try:
        rejects_handle = open(output_path / "rejects.jsonl", "wb")
    except Exception:
        manifest_handle.close()
        raise

    def emit_manifests(values: Iterable[Mapping[str, Any]]) -> None:
        nonlocal manifest_count
        for value in values:
            manifest_handle.write(_canonical_json(value) + b"\n")
            manifest_count += 1

    def emit_reject(value: Mapping[str, Any]) -> None:
        nonlocal reject_count
        rejects_handle.write(_canonical_json(value) + b"\n")
        reject_count += 1

    def report_progress(index: int) -> None:
        if index % 100 == 0 or index == len(selected):
            print(
                f"processed={index}/{len(selected)} accepted={sum(source_counts.values())} "
                f"rejected={reject_count} notes={note_total}",
                flush=True,
            )

    def consume(
        outcomes: Iterable[tuple[dict[str, str], _ProcessOutcome]],
    ) -> None:
        nonlocal note_total, chord_total, active_writer_key
        for index, (row, outcome) in enumerate(outcomes, 1):
            sample = outcome.sample
            if sample is None:
                reason = outcome.error_type or "DatasetBuildError"
                message = outcome.error_message or "unknown dataset processing error"
                reason_counts[reason] += 1
                emit_reject(
                    {
                        "id": row.get("id"),
                        "performance_dataset": row.get("performance_dataset"),
                        "split": row.get("_split"),
                        "reason_type": reason,
                        "reason": message,
                    }
                )
                if strict:
                    raise DatasetBuildError(f"{reason}: {message}")
                report_progress(index)
                continue
            key = (row["_split"], row["performance_dataset"])
            if active_writer_key is not None and key != active_writer_key:
                emit_manifests(writers[active_writer_key].flush())
                closed_writer_keys.add(active_writer_key)
            if key in closed_writer_keys:
                raise DatasetBuildError(
                    "selected rows are not contiguous by split and performance_dataset"
                )
            active_writer_key = key
            writer = writers.get(key)
            if writer is None:
                writer = writers[key] = ShardWriter(
                    output_path,
                    key[0],
                    key[1],
                    max_samples=max_samples_per_shard,
                    max_notes=max_notes_per_shard,
                )
            emit_manifests(writer.add(sample))
            normalizer.update(sample.arrays, row["_split"])
            source_counts[row["performance_dataset"]] += 1
            split_counts[row["_split"]] += 1
            note_total += len(sample.arrays.note_features)
            chord_total += len(sample.arrays.chord_features)
            manifest = sample.manifest
            final_piece_splits[manifest["piece_group"]].add(manifest["split"])
            final_score_path_splits[
                _normalized_score_path(
                    {"refined_score_midi_path": manifest["score_midi_path"]}
                )
            ].add(manifest["split"])
            final_score_semantic_splits[manifest["score_semantic_sha256"]].add(
                manifest["split"]
            )
            performer = str(manifest.get("performer") or "").strip().casefold()
            if performer:
                final_performer_splits[performer].add(manifest["split"])
            report_progress(index)

    try:
        if workers == 1:
            with RefinedArchive(archive_path) as archive:
                outcomes = (
                    (
                        row,
                        _safe_process_row(
                            row,
                            archive,
                            score_cache,
                            max_interpolation_rate=max_interpolation_rate,
                            max_score_cache_entries=max_score_cache_entries,
                        ),
                    )
                    for row in selected
                )
                consume(outcomes)
        else:
            consume(
                _parallel_outcomes(
                    selected,
                    archive_path=archive_path,
                    workers=workers,
                    max_interpolation_rate=max_interpolation_rate,
                    max_score_cache_entries=max_score_cache_entries,
                )
            )
        for key in sorted(writers):
            emit_manifests(writers[key].flush())
    finally:
        rejects_handle.close()
        manifest_handle.close()

    if not manifest_count:
        raise DatasetBuildError("all selected samples were rejected; dataset is incomplete")
    if manifest_count != sum(source_counts.values()):
        raise DatasetBuildError(
            "written manifest count does not match accepted sample count"
        )
    if not split_counts.get("train"):
        raise DatasetBuildError("accepted samples contain no train split; normalizer is undefined")
    normalizer_summary = normalizer.summary()
    _write_json(output_path / "normalizer.json", normalizer_summary)

    final_leaks = sum(len(splits) > 1 for splits in final_piece_splits.values())
    final_path_leaks = sum(
        len(splits) > 1 for splits in final_score_path_splits.values()
    )
    final_semantic_leaks = sum(
        len(splits) > 1 for splits in final_score_semantic_splits.values()
    )
    if final_leaks or final_path_leaks or final_semantic_leaks:
        raise DatasetBuildError(
            "generated split leakage detected: "
            f"piece_groups={final_leaks} score_paths={final_path_leaks} "
            f"score_semantic_hashes={final_semantic_leaks}"
        )

    accepted = manifest_count
    summary = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": config["config_sha256"],
        "selected_count": len(selected),
        "accepted_count": accepted,
        "rejected_count": reject_count,
        "pass_rate": accepted / len(selected),
        "note_count": note_total,
        "chord_count": chord_total,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "piece_group_split_overlap": final_leaks,
        "score_path_split_overlap": final_path_leaks,
        "score_semantic_split_overlap": final_semantic_leaks,
        "performer_count": len(final_performer_splits),
        "performer_split_overlap": sum(
            len(splits) > 1 for splits in final_performer_splits.values()
        ),
        "split_audit": split_audit,
        "selected_score_path_count": len(
            {row["refined_score_midi_path"] for row in selected}
        ),
        "score_cache_capacity_per_process": max_score_cache_entries,
        "normalizer_fit_split": "train",
    }
    _write_json(output_path / "qc_summary.json", summary)

    inventory_paths = [
        output_path / "build_config.json",
        output_path / "schema.json",
        output_path / "manifest.jsonl",
        output_path / "rejects.jsonl",
        output_path / "normalizer.json",
        output_path / "qc_summary.json",
        *(path for path in output_path.rglob("*.npz") if path.is_file()),
    ]
    integrity = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "files": [
            {
                "path": path.relative_to(output_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(inventory_paths, key=lambda item: item.relative_to(output_path).as_posix())
        ],
    }
    _write_json(output_path / "integrity.json", integrity)
    integrity_hash = sha256_file(output_path / "integrity.json")
    incomplete_marker.unlink(missing_ok=True)
    (output_path / "_SUCCESS").write_text(
        integrity_hash + "\n", encoding="ascii"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic expressive-performance shards from PianoCoRe-A*."
    )
    parser.add_argument("--metadata", required=True, help="PianoCoRe A* metadata CSV")
    parser.add_argument("--archive", required=True, help="PianoCoRe refined ZIP")
    parser.add_argument("--output", required=True, help="new or empty output directory")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke-per-group", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-samples-per-shard", type=int, default=256)
    parser.add_argument("--max-notes-per-shard", type=int, default=250_000)
    parser.add_argument("--max-interpolation-rate", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-score-cache-entries", type=int, default=16)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = build_dataset(
            metadata_path=args.metadata,
            archive_path=args.archive,
            output_path=args.output,
            mode=args.mode,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            smoke_per_group=args.smoke_per_group,
            max_records=args.max_records,
            max_samples_per_shard=args.max_samples_per_shard,
            max_notes_per_shard=args.max_notes_per_shard,
            max_interpolation_rate=args.max_interpolation_rate,
            workers=args.workers,
            max_score_cache_entries=args.max_score_cache_entries,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"dataset build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DatasetBuildError",
    "PIPELINE_NAME",
    "PIPELINE_VERSION",
    "RefinedArchive",
    "TrainOnlyNormalizer",
    "assign_split",
    "build_dataset",
    "connected_piece_groups",
    "piece_key",
    "read_metadata",
    "score_semantic_fingerprint",
    "score_semantic_hashes",
    "select_rows",
    "sha256_file",
    "source_directory_name",
    "validate_metadata_splits",
    "write_deterministic_npz",
]
