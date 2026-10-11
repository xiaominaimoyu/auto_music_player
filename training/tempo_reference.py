"""Immutable score-timing references for selected train/validation samples.

The sidecar restores MIDI seconds and score tempo conditions without changing
the v2 shards. Performance-derived calibration is evaluation-only; consumers
must not use it as a model input. Real-bar drift is measured only at recorded
bar boundaries, across native valid IOIs, and never across partial bars.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import mido
import numpy as np

from training.midi_features import ParsedMidi, build_aligned_features, parse_midi_bytes
from training.pianocore_dataset import RefinedArchive, _load_alignment, write_deterministic_npz


class TimingReferenceError(ValueError):
    """The sidecar or its source binding cannot be trusted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise TimingReferenceError(f"unsafe sidecar/source path: {relative!r}")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise TimingReferenceError(f"path escapes source root: {relative!r}")
    return resolved


def _bar_coordinates(score: ParsedMidi, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A genuine meter change starts a new bar. If it occurs mid-bar, the
    # preceding partial bar is recorded and excluded from full-bar metrics.
    signatures = []
    for event in score.time_signatures:
        if signatures and (event.numerator, event.denominator) == (
            signatures[-1].numerator, signatures[-1].denominator
        ):
            continue
        signatures.append(event)
    starts, bars, lengths, partial = [], [], [], []
    for event in signatures:
        onset = event.tick / score.ticks_per_beat
        length = event.numerator * 4.0 / event.denominator
        bar = 0
        if starts:
            elapsed = (onset - starts[-1]) / lengths[-1]
            whole = math.floor(elapsed + 1e-10)
            has_partial = not math.isclose(elapsed, round(elapsed), abs_tol=1e-9)
            if has_partial:
                partial.append(bars[-1] + whole)
            bar = bars[-1] + whole + int(has_partial)
        starts.append(onset)
        bars.append(bar)
        lengths.append(length)
    segment = np.maximum(0, np.searchsorted(starts, beats, side="right") - 1)
    elapsed = (beats - np.asarray(starts)[segment]) / np.asarray(lengths)[segment]
    whole = np.floor(elapsed + 1e-10).astype(np.int64)
    phase = elapsed - whole
    phase[np.abs(phase) < 1e-9] = 0.0
    return np.asarray(bars, dtype=np.int64)[segment] + whole, phase, np.asarray(partial, dtype=np.int64)


def _tempo_events(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    midi = mido.MidiFile(file=io.BytesIO(data))
    ticks, values = [0], [500_000]
    absolute = 0
    for message in mido.merge_tracks(midi.tracks):
        absolute += int(message.time)
        if message.type == "set_tempo":
            if ticks[-1] == absolute:
                values[-1] = int(message.tempo)
            else:
                ticks.append(absolute)
                values.append(int(message.tempo))
    return np.asarray(ticks, dtype=np.int64), np.asarray(values, dtype=np.int64)


def score_timing_reference(score_data: bytes) -> dict[str, np.ndarray]:
    """Derive score-only conditions, also usable for original local A/B scores.

    Bytes are required because ParsedMidi deliberately omits tempo events.
    Local chord tempo uses the integrated preceding score IOI per beat, with
    instantaneous MIDI tempo at the first chord. Note tempo uses integrated
    score key duration per beat. Both ratios share the median chord baseline.
    """
    score = parse_midi_bytes(score_data)
    ticks = sorted({note.start_tick for note in score.notes})
    tick_to_chord = {tick: index for index, tick in enumerate(ticks)}
    mapping = np.array([tick_to_chord[note.start_tick] for note in score.notes], dtype=np.int32)
    chord_count = len(ticks)
    first = np.array([np.flatnonzero(mapping == index)[0] for index in range(chord_count)])
    seconds = np.array([score.notes[index].start_s for index in first], dtype=np.float64)
    beats = np.array([score.notes[index].onset_beats for index in first], dtype=np.float64)
    duration_seconds = np.array([note.duration_s for note in score.notes], dtype=np.float64)
    duration_beats = np.array([note.duration_beats for note in score.notes], dtype=np.float64)
    tempo_ticks, tempos = _tempo_events(score_data)
    chord_ticks = np.array([score.notes[index].start_tick for index in first])
    onset_seconds_per_beat = tempos[np.searchsorted(tempo_ticks, chord_ticks, side="right") - 1] / 1_000_000.0
    interval_seconds_per_beat = onset_seconds_per_beat.copy()
    if chord_count > 1:
        interval_seconds_per_beat[1:] = np.diff(seconds) / np.diff(beats)
    baseline = float(np.median(interval_seconds_per_beat))
    duration_seconds_per_beat = np.divide(
        duration_seconds, duration_beats,
        out=onset_seconds_per_beat[mapping].copy(), where=duration_beats > 0,
    )
    bar_index, bar_phase, partial = _bar_coordinates(score, beats)
    return {
        "chord_score_seconds": seconds,
        "chord_score_beats": beats,
        "chord_bar_index": bar_index,
        "chord_bar_phase": bar_phase,
        "partial_bar_indices": partial,
        "note_score_duration_seconds": duration_seconds,
        "note_score_duration_beats": duration_beats,
        "note_to_chord": mapping,
        "score_seconds_per_beat_reference": np.asarray(baseline, dtype=np.float64),
        "chord_tempo_log_ratio": np.log(interval_seconds_per_beat / baseline).astype(np.float32),
        "note_duration_tempo_log_ratio": np.log(duration_seconds_per_beat / baseline).astype(np.float32),
        "score_tempo_ticks": tempo_ticks,
        "score_tempo_microseconds": tempos,
        "score_ticks_per_beat": np.asarray(score.ticks_per_beat, dtype=np.int64),
        "meter_ticks": np.array([event.tick for event in score.time_signatures], dtype=np.int64),
        "meter_numerators": np.array([event.numerator for event in score.time_signatures], dtype=np.int64),
        "meter_denominators": np.array([event.denominator for event in score.time_signatures], dtype=np.int64),
    }


def make_timing_reference(score_data: bytes, performance_data: bytes,
                          perf_idx: np.ndarray, interpolated: np.ndarray) -> dict[str, np.ndarray]:
    """Derive one reference in the exact note/chord order used by v2."""
    score = parse_midi_bytes(score_data)
    performance = parse_midi_bytes(performance_data)
    arrays = build_aligned_features(score, performance, perf_idx, interpolated)
    reference = score_timing_reference(score_data)
    mapping = arrays.note_to_chord
    performance_seconds, real_ratio = [], []
    for index in range(len(arrays.chord_features)):
        members = np.flatnonzero(mapping == index)
        real = members[~interpolated[members]]
        chosen = real if len(real) else members
        performance_seconds.append(float(np.median([
            performance.notes[int(perf_idx[member])].start_s for member in chosen
        ])))
        real_ratio.append(len(real) / len(members))
    reference.update({
        "chord_performance_seconds": np.asarray(performance_seconds, dtype=np.float64),
        "chord_trusted": np.asarray(real_ratio) >= 0.5,
        "chord_native_masks": arrays.chord_masks,
        "note_native_masks": arrays.note_masks,
        "chord_log_ioi_ratio": arrays.chord_targets[:, 1],
        "note_log_key_duration_ratio": arrays.note_targets[:, 1],
        "global_tempo_scale": np.asarray(arrays.diagnostics["global_tempo_scale"], dtype=np.float64),
        "tempo_intercept_seconds": np.asarray(arrays.diagnostics["tempo_intercept_s"], dtype=np.float64),
    })
    return reference


def _validate_reference(reference: Mapping[str, np.ndarray]) -> None:
    chord_count = len(reference["chord_score_seconds"])
    note_count = len(reference["note_score_duration_seconds"])
    for name in ("chord_score_beats", "chord_bar_index", "chord_bar_phase",
                 "chord_performance_seconds", "chord_trusted", "chord_log_ioi_ratio", "chord_tempo_log_ratio"):
        if reference[name].shape != (chord_count,):
            raise TimingReferenceError(f"wrong chord reference shape: {name}")
    for name in ("note_score_duration_beats", "note_to_chord", "note_log_key_duration_ratio",
                 "note_duration_tempo_log_ratio"):
        if reference[name].shape != (note_count,):
            raise TimingReferenceError(f"wrong note reference shape: {name}")
    if reference["chord_native_masks"].shape != (chord_count, 2) or reference["note_native_masks"].shape != (note_count, 6):
        raise TimingReferenceError("wrong native mask shapes")
    if not chord_count or not note_count or any(not np.isfinite(value).all() for value in reference.values()):
        raise TimingReferenceError("empty or nonfinite timing reference")
    if np.any(np.diff(reference["chord_score_seconds"]) <= 0) or np.any(np.diff(reference["chord_score_beats"]) <= 0):
        raise TimingReferenceError("score chord times must be strictly increasing")
    if np.any(reference["note_to_chord"] < 0) or np.any(reference["note_to_chord"] >= chord_count):
        raise TimingReferenceError("note_to_chord must use sample-local indices")
    for name in ("global_tempo_scale", "score_seconds_per_beat_reference"):
        if reference[name].shape != () or float(reference[name]) <= 0:
            raise TimingReferenceError(f"invalid positive scalar: {name}")


def _read_selected_rows(dataset: Path, id_files: Sequence[Path]) -> list[dict[str, Any]]:
    wanted: set[str] = set()
    for path in id_files:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
        if not values:
            raise TimingReferenceError(f"empty sample ID file: {path}")
        wanted.update(values)
    if not wanted:
        raise TimingReferenceError("at least one sample ID file is required")
    rows = []
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["id"]) in wanted:
                if row.get("split") not in {"train", "validation"} or row.get("official_split") == "test":
                    raise TimingReferenceError(f"test/unknown split forbidden: {row['id']}")
                rows.append(row)
    if len(rows) != len(wanted) or {str(row["id"]) for row in rows} != wanted:
        raise TimingReferenceError("sample IDs missing or duplicated in source manifest")
    return sorted(rows, key=lambda row: str(row["id"]))


def build_timing_reference(*, dataset_root: str | Path, archive_path: str | Path,
                           sample_id_files: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    dataset, archive_path, output = Path(dataset_root).resolve(), Path(archive_path).resolve(), Path(output_dir).resolve()
    id_files = [Path(path).resolve() for path in sample_id_files]
    rows = _read_selected_rows(dataset, id_files)
    if output.exists() and any(output.iterdir()):
        raise TimingReferenceError("output directory must be new or empty")
    # Bind every selected v2 shard and source member before deriving labels.
    shard_hashes: dict[str, str] = {}
    for row in rows:
        relative = str(row["shard_path"])
        if relative not in shard_hashes:
            shard_hashes[relative] = _sha256(_safe_path(dataset, relative))
        if shard_hashes[relative] != row["shard_sha256"]:
            raise TimingReferenceError(f"source shard hash mismatch: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "_INCOMPLETE").write_text("timing reference build in progress\n", encoding="utf-8")
    index = []
    loaded_path, loaded_arrays = None, None
    with RefinedArchive(archive_path) as archive:
        for row in rows:
            source_bytes = {}
            for prefix in ("score_midi", "performance_midi", "alignment"):
                value = archive.read(str(row[f"{prefix}_path"]))
                if hashlib.sha256(value).hexdigest() != row[f"{prefix}_sha256"]:
                    raise TimingReferenceError(f"source member hash mismatch: {row['id']} {prefix}")
                source_bytes[prefix] = value
            perf_idx, interpolated = _load_alignment(source_bytes["alignment"])
            reference = make_timing_reference(source_bytes["score_midi"], source_bytes["performance_midi"], perf_idx, interpolated)
            _validate_reference(reference)
            if loaded_path != row["shard_path"]:
                loaded_path = row["shard_path"]
                with np.load(_safe_path(dataset, loaded_path), allow_pickle=False) as loaded:
                    loaded_arrays = {name: loaded[name] for name in loaded.files}
            position = int(row["sample_index"])
            assert loaded_arrays is not None
            if str(loaded_arrays["sample_ids"][position]) != str(row["id"]):
                raise TimingReferenceError(f"source shard sample index mismatch: {row['id']}")
            n0, n1 = map(int, loaded_arrays["sample_note_offsets"][position:position + 2])
            c0, c1 = map(int, loaded_arrays["sample_chord_offsets"][position:position + 2])
            checks = (
                (reference["note_to_chord"], loaded_arrays["note_to_chord"][n0:n1] - c0),
                (reference["note_native_masks"], loaded_arrays["note_masks"][n0:n1]),
                (reference["chord_native_masks"], loaded_arrays["chord_masks"][c0:c1]),
                (reference["chord_log_ioi_ratio"], loaded_arrays["chord_targets"][c0:c1, 1]),
                (reference["note_log_key_duration_ratio"], loaded_arrays["note_targets"][n0:n1, 1]),
            )
            if (n0, c0, n1 - n0, c1 - c0) != (int(row["note_offset"]), int(row["chord_offset"]), int(row["note_count"]), int(row["chord_count"])) or not all(np.array_equal(a, b) for a, b in checks):
                raise TimingReferenceError(f"native v2 index/labels mismatch: {row['id']}")
            relative = f"samples/{hashlib.sha256(str(row['id']).encode()).hexdigest()}.npz"
            destination = output / relative
            destination.parent.mkdir(exist_ok=True)
            digest = write_deterministic_npz(destination, reference)
            index.append({"id": str(row["id"]), "split": row["split"], "piece_group": row["piece_group"],
                          "performance_dataset": row["performance_dataset"], "path": relative, "sha256": digest,
                          "note_count": n1 - n0, "chord_count": c1 - c0,
                          "source_row_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
                          "source_shard_path": row["shard_path"], "source_shard_sha256": row["shard_sha256"],
                          "source_sample_index": position, "note_offset": n0, "chord_offset": c0,
                          "score_midi_sha256": row["score_midi_sha256"],
                          "tempo_event_count": len(reference["score_tempo_ticks"]),
                          "tempo_distinct_count": len(np.unique(reference["score_tempo_microseconds"])),
                          "partial_bar_count": len(reference["partial_bar_indices"])})
    summary = {
        "schema_version": 1, "status": "complete", "kind": "pianocore_timing_reference",
        "dataset_root": str(dataset), "dataset_manifest_sha256": _sha256(dataset / "manifest.jsonl"),
        "archive_path": str(archive_path), "source_integrity": "selected ZIP member and selected v2 shard SHA256 verified",
        "sample_id_files": [{"path": str(path), "sha256": _sha256(path)} for path in id_files],
        "sample_count": len(index), "split_counts": dict(Counter(row["split"] for row in index)),
        "meter_policy": "real meter changes start new bars; redundant changes ignored; partial bars excluded from drift",
        "calibration_policy": "global scale/intercept and performance seconds are evaluation-only, never model inputs",
        "samples": index,
    }
    (output / "summary.json").write_bytes(_canonical(summary) + b"\n")
    (output / "_SUCCESS").write_text(_sha256(output / "summary.json") + "\n", encoding="utf-8")
    (output / "_INCOMPLETE").unlink()
    return summary


def load_timing_reference(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Verify the complete sidecar, its selected source binding, and arrays."""
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    summary_path = root / "summary.json"
    if (root / "_INCOMPLETE").exists() or not (root / "_SUCCESS").is_file() or (root / "_SUCCESS").read_text().strip() != _sha256(summary_path):
        raise TimingReferenceError("timing reference incomplete or summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1 or summary.get("status") != "complete" or summary.get("kind") != "pianocore_timing_reference":
        raise TimingReferenceError("unsupported timing reference schema/status")
    dataset = Path(summary["dataset_root"])
    if _sha256(dataset / "manifest.jsonl") != summary["dataset_manifest_sha256"]:
        raise TimingReferenceError("source dataset manifest hash mismatch")
    index = summary["samples"]
    wanted = {str(row["id"]): row for row in index}
    if len(wanted) != len(index) or len(index) != summary["sample_count"]:
        raise TimingReferenceError("duplicate sample IDs or sample count mismatch")
    found = set()
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in wanted:
                if row.get("split") not in {"train", "validation"} or row.get("official_split") == "test":
                    raise TimingReferenceError("test reference forbidden")
                if hashlib.sha256(_canonical(row)).hexdigest() != wanted[sample_id]["source_row_sha256"]:
                    raise TimingReferenceError("source selected-row binding mismatch")
                found.add(sample_id)
    if found != set(wanted):
        raise TimingReferenceError("selected source IDs missing")
    result, verified_shards = {}, {}
    for sample_id, row in wanted.items():
        relative = row["source_shard_path"]
        if relative not in verified_shards:
            verified_shards[relative] = _sha256(_safe_path(dataset, relative))
        if verified_shards[relative] != row["source_shard_sha256"]:
            raise TimingReferenceError("source shard hash mismatch")
        sample_path = _safe_path(root, row["path"])
        if _sha256(sample_path) != row["sha256"]:
            raise TimingReferenceError(f"reference sample hash mismatch: {sample_id}")
        with np.load(sample_path, allow_pickle=False) as loaded:
            reference = {name: loaded[name] for name in loaded.files}
        # The v2 deterministic NPZ writer makes all arrays contiguous, which
        # promotes scalar arrays to length one. Restore the public scalar API.
        for name in ("global_tempo_scale", "tempo_intercept_seconds",
                     "score_seconds_per_beat_reference", "score_ticks_per_beat"):
            if reference[name].shape == (1,):
                reference[name] = reference[name].reshape(())
        _validate_reference(reference)
        if len(reference["chord_score_seconds"]) != row["chord_count"] or len(reference["note_score_duration_seconds"]) != row["note_count"]:
            raise TimingReferenceError("reference count differs from bound source index")
        result[sample_id] = reference
    return result


def reconstruct_ioi_seconds(reference: Mapping[str, np.ndarray], predicted_log_ioi: np.ndarray,
                            *, calibration: str = "oracle_global", fixed_scale: float | None = None) -> np.ndarray:
    prediction = np.asarray(predicted_log_ioi, dtype=np.float64)
    if prediction.shape != reference["chord_score_seconds"].shape or not np.isfinite(prediction).all():
        raise TimingReferenceError("invalid IOI prediction shape/values")
    if calibration == "oracle_global":
        scale = float(reference["global_tempo_scale"])
    elif calibration == "fixed_score" and fixed_scale is not None:
        scale = float(fixed_scale)
    else:
        raise TimingReferenceError("calibration must be oracle_global or fixed_score with explicit fixed_scale")
    if not math.isfinite(scale) or scale <= 0:
        raise TimingReferenceError("calibration scale must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        reconstructed = np.diff(reference["chord_score_seconds"], prepend=reference["chord_score_seconds"][0]) * scale * np.exp(prediction)
    if not np.isfinite(reconstructed).all() or np.any(reconstructed[1:] <= 0):
        raise TimingReferenceError("reconstructed IOIs must be finite and positive")
    return reconstructed


def evaluate_timing_prediction(reference: Mapping[str, np.ndarray], predicted_log_ioi: np.ndarray,
                               *, calibration: str = "oracle_global", fixed_scale: float | None = None,
                               horizons: Sequence[int] = (1, 4, 16)) -> dict[str, Any]:
    """Evaluate one sample; bar windows require exact and fully trusted bounds."""
    predicted = reconstruct_ioi_seconds(reference, predicted_log_ioi, calibration=calibration, fixed_scale=fixed_scale)
    actual = np.diff(reference["chord_performance_seconds"], prepend=reference["chord_performance_seconds"][0])
    valid = reference["chord_native_masks"][:, 1].astype(bool)
    error_ms = (predicted - actual) * 1000.0
    trusted_indices = np.flatnonzero(reference["chord_trusted"])
    anchor = int(trusted_indices[0]) if len(trusted_indices) else None
    # Absolute onset is aligned only at the first trusted chord. Invalid
    # intermediate IOIs permanently exclude subsequent accumulated onsets.
    onset_values = []
    if anchor is not None:
        for index in range(anchor + 1, len(predicted)):
            if not valid[index]:
                break
            onset_values.append(abs(float(error_ms[anchor + 1:index + 1].sum())))
    boundaries = {int(reference["chord_bar_index"][index]): index
                  for index in np.flatnonzero(np.isclose(reference["chord_bar_phase"], 0.0, atol=1e-9))}
    partial = set(map(int, reference["partial_bar_indices"]))
    drift = {}
    for horizon in horizons:
        if int(horizon) != horizon or horizon <= 0:
            raise TimingReferenceError("bar horizons must be positive integers")
        values, skipped = [], 0
        for bar, start in boundaries.items():
            end = boundaries.get(bar + int(horizon))
            if end is None:
                skipped += 1
                continue
            if any(value in partial for value in range(bar, bar + int(horizon))) or not bool(reference["chord_trusted"][start]) or not valid[start + 1:end + 1].all():
                skipped += 1
                continue
            values.append(abs(float(error_ms[start + 1:end + 1].sum())))
        drift[str(horizon)] = {"window_count": len(values), "skipped_window_count": skipped,
                               "mae_ms": float(np.mean(values)) if values else None,
                               "p95_ms": float(np.quantile(values, 0.95)) if values else None,
                               "absolute_error_ms": values}
    return {"calibration": calibration, "uses_performance_calibration": calibration == "oracle_global",
            "deployment_evidence": False,
            "first_trusted_chord": anchor, "valid_ioi_count": int(valid.sum()),
            "ioi_mae_ms": float(np.abs(error_ms[valid]).mean()) if valid.any() else None,
            "aligned_onset_count": len(onset_values),
            "aligned_onset_mae_ms": float(np.mean(onset_values)) if onset_values else None,
            "bar_drift": drift}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_timing_reference(dataset_root=args.dataset_root, archive_path=args.archive,
                                    sample_id_files=args.sample_id_file, output_dir=args.output_dir)
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"], "split_counts": result["split_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
