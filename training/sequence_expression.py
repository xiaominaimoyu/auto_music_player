"""Matched current-chord MLP and small offline GRU expression models.

Windows are cut inside complete performances.  Only their disjoint cores
contribute labels; context is score-only and padding is never supervised.
The optional timing reference exposes an explicit allowlist of score-derived
features, never performance timing, global fitted scale, or supervision flags.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from training.bar_supervision import BarSpans, build_bar_spans, validate_horizons
from training.bar_windows import DEFAULT_MAX_WINDOW_CHORDS, bar_boundary_indices, plan_bar_windows
from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.shape_features import SHAPE_FEATURE_NAMES, derive_shape_features
from training.train_expression import TrainingRunError, _selected_shard_paths


@dataclass(frozen=True)
class SequenceSample:
    sample_id: str
    note_features: np.ndarray
    chord_features: np.ndarray
    note_targets: np.ndarray
    chord_targets: np.ndarray
    note_masks: np.ndarray
    chord_masks: np.ndarray
    note_to_chord: np.ndarray
    # Pure score-only supervision geometry, separate from model channels.
    # Optional to preserve historical no-tempo samples/checkpoints.
    score_ioi_seconds: np.ndarray | None = None
    # Score-only true-bar geometry. Optional so the historical fixed-128 path,
    # its checkpoints, and its no-bar audits stay byte-identical. The planner
    # reads only these arrays plus ``score_ioi_seconds``: never a target, mask,
    # native mask, trust flag, or performance clock.
    bar_span_rows: np.ndarray | None = None
    bar_span_horizons: np.ndarray | None = None
    bar_boundary_indices: np.ndarray | None = None
    bar_requested_horizons: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class SequenceWindow:
    sample: SequenceSample
    start: int
    end: int
    core_start: int
    core_end: int
    # Sample-level span rows whose complete endpoints live inside this core.
    # Halo, padding, and uncovered spans never appear here.
    bar_span_rows: np.ndarray | None = None
    bar_span_horizons: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        """Layout equality.

        The generated dataclass comparison cannot be used: numpy fields make ``==``
        raise on ambiguity.  Samples are matched by id because a window's meaning is
        its position in a sample, not a copy of its feature arrays.
        """
        if not isinstance(other, SequenceWindow):
            return NotImplemented
        return (self.sample.sample_id == other.sample.sample_id and self.start == other.start
                and self.end == other.end and self.core_start == other.core_start
                and self.core_end == other.core_end
                and _same_rows(self.bar_span_rows, other.bar_span_rows)
                and _same_rows(self.bar_span_horizons, other.bar_span_horizons))

    def __hash__(self) -> int:
        return hash((self.sample.sample_id, self.start, self.end, self.core_start, self.core_end))


def _same_rows(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return np.array_equal(left, right)


BAR_WEIGHT_KEYS = ("bar_elapsed_weight", "bar_tempo_weight", "bar_local_weight")
# ``true_bar_objective`` names its losses by what they measure, while the trainer
# names its knobs by what the user tunes; this map is the only place they meet.
BAR_WEIGHT_TO_COMPONENT = {name: name.removesuffix("_weight") for name in BAR_WEIGHT_KEYS}
# Which note-target column carries the per-key duration ratio, resolved from the shipped
# name list rather than a hardcoded index so a schema change cannot pass silently.
DURATION_TARGET_COLUMN = NOTE_TARGET_NAMES.index("log_key_duration_ratio")


def bar_weight_total(config: Mapping[str, Any]) -> float:
    return sum(float(config.get(name, 0.0)) for name in BAR_WEIGHT_KEYS)


def checkpoint_preparation_flags(config: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    """The input/target transforms a checkpoint was trained under, read back out of its config.

    Sample preparation must match the checkpoint exactly: a shape-feature run carries three
    extra input columns, and feeding it legacy-width samples fails on a broadcast error
    instead of producing a number.  Returns ``(duration_centering, shape_features,
    relative_position)``.
    """
    return (bool(config.get("duration_center_per_sample", False)),
            bool(config.get("note_shape_features", False)),
            bool(config.get("relative_position", True)))


def prepare_sample(
    sample_id: str,
    arrays: Mapping[str, np.ndarray],
    *,
    timing_reference: Mapping[str, Any] | None = None,
    relative_position: bool = True,
    duration_centering: bool = False,
    shape_features: bool = False,
) -> SequenceSample:
    """Prepare one sample whose note_to_chord is already sample-local."""
    note = np.asarray(arrays["note_features"], dtype=np.float32).copy()
    chord = np.asarray(arrays["chord_features"], dtype=np.float32).copy()
    mapping = np.asarray(arrays["note_to_chord"], dtype=np.int64)
    if note.ndim != 2 or chord.ndim != 2 or not len(chord):
        raise TrainingRunError(f"{sample_id}: empty or malformed feature arrays")
    if mapping.shape != (len(note),) or np.any(mapping < 0) or np.any(mapping >= len(chord)):
        raise TrainingRunError(f"{sample_id}: note_to_chord escapes sample")
    # derived while beats are still in score units: the shape columns need absolute
    # distances, and ``relative_position`` below rewrites column 3 in place
    derived_shape = derive_shape_features(note, chord[:, 0]) if shape_features else None
    if relative_position:
        lo, hi = float(chord[0, 0]), float(chord[-1, 0])
        span = max(hi - lo, 1e-6)
        note[:, 3] = (note[:, 3] - lo) / span
        chord[:, 0] = (chord[:, 0] - lo) / span
    if derived_shape is not None:
        # appended before the timing columns so train and inference agree on column order
        # from the same code path, without either of them hardcoding an index
        note = np.column_stack((note, derived_shape)).astype(np.float32)
    score_ioi_seconds = None
    if timing_reference is not None:
        baseline = float(np.asarray(timing_reference["score_seconds_per_beat_reference"]).item())
        if not np.isfinite(baseline) or baseline <= 0:
            raise TrainingRunError(f"{sample_id}: invalid score tempo reference")
        chord_tempo = np.asarray(timing_reference["chord_tempo_log_ratio"], dtype=np.float32)
        note_tempo = np.asarray(timing_reference["note_duration_tempo_log_ratio"], dtype=np.float32)
        if chord_tempo.shape != (len(chord),) or note_tempo.shape != (len(note),):
            raise TrainingRunError(f"{sample_id}: timing reference count mismatch")
        if "note_to_chord" in timing_reference and not np.array_equal(
            mapping, np.asarray(timing_reference["note_to_chord"], dtype=np.int64)
        ):
            raise TrainingRunError(f"{sample_id}: timing reference note mapping mismatch")
        if "chord_score_seconds" in timing_reference:
            score_seconds = np.asarray(timing_reference["chord_score_seconds"], dtype=np.float64)
            if score_seconds.shape != (len(chord),) or not np.isfinite(score_seconds).all():
                raise TrainingRunError(f"{sample_id}: invalid score seconds reference")
            score_ioi_seconds = np.diff(score_seconds, prepend=score_seconds[0])
        else:
            # The feature stores log1p(previous IOI in beats); this fallback
            # is only physical seconds when a pure score tempo is provided.
            score_ioi_seconds = np.expm1(chord[:, 8].astype(np.float64)) * baseline * np.exp(chord_tempo.astype(np.float64))
            score_ioi_seconds[0] = 0.0
            # Historical callers can supply only tempo-feature stubs. Do not
            # invent physical geometry for them; cumulative use fails later.
            if not np.isfinite(score_ioi_seconds).all() or np.any(score_ioi_seconds[1:] <= 0):
                score_ioi_seconds = None
        # This is intentionally the only score-timing feature allowlist.
        note = np.column_stack((note, np.full(len(note), np.log(baseline)), note_tempo)).astype(np.float32)
        chord = np.column_stack((chord, np.full(len(chord), np.log(baseline)), chord_tempo)).astype(np.float32)
    if "score_ioi_seconds" in arrays:
        explicit = np.asarray(arrays["score_ioi_seconds"], dtype=np.float64)
        if explicit.shape != (len(chord),):
            raise TrainingRunError(f"{sample_id}: invalid pure score IOI seconds shape")
        if score_ioi_seconds is not None and not np.allclose(explicit, score_ioi_seconds, rtol=1e-5, atol=1e-8):
            raise TrainingRunError(f"{sample_id}: explicit score IOI differs from timing reference")
        score_ioi_seconds = explicit.copy()
    if score_ioi_seconds is not None:
        if (score_ioi_seconds.shape != (len(chord),) or not np.isfinite(score_ioi_seconds).all()
                or score_ioi_seconds[0] != 0 or np.any(score_ioi_seconds[1:] <= 0)):
            raise TrainingRunError(f"{sample_id}: invalid pure score IOI seconds")
    if not np.isfinite(note).all() or not np.isfinite(chord).all():
        raise TrainingRunError(f"{sample_id}: non-finite score inputs")
    note_targets = np.asarray(arrays["note_targets"], dtype=np.float32)
    chord_targets = np.asarray(arrays["chord_targets"], dtype=np.float32)
    note_masks = np.asarray(arrays["note_masks"], dtype=bool)
    chord_masks = np.asarray(arrays["chord_masks"], dtype=bool)
    for label, target, mask, count, width in (
        ("note", note_targets, note_masks, len(note), len(NOTE_TARGET_NAMES)),
        ("chord", chord_targets, chord_masks, len(chord), len(CHORD_TARGET_NAMES)),
    ):
        if target.shape != (count, width) or mask.shape != target.shape or not np.isfinite(target).all():
            raise TrainingRunError(f"{sample_id}: invalid {label} labels")
        ref_name = f"{label}_native_masks"
        if timing_reference is not None and ref_name in timing_reference and not np.array_equal(
            mask, np.asarray(timing_reference[ref_name], dtype=bool)
        ):
            raise TrainingRunError(f"{sample_id}: timing reference native masks mismatch")
    if duration_centering:
        # Take each recording's duration *level* out of the target so the model only has to
        # reproduce the within-recording profile.  The centre is the median of that
        # recording's own supervised rows, which is a label-side statistic: it never enters
        # the inputs, and inference on a fresh performance simply has no centre to remove.
        supervised = note_masks[:, DURATION_TARGET_COLUMN]
        if supervised.any():
            centre = float(np.median(note_targets[supervised, DURATION_TARGET_COLUMN]))
            note_targets = note_targets.copy()
            note_targets[:, DURATION_TARGET_COLUMN] -= centre
    return SequenceSample(sample_id, note, chord, note_targets, chord_targets, note_masks, chord_masks, mapping,
                          score_ioi_seconds)


def load_selected_samples(
    dataset_root: Path,
    split: str,
    sample_ids: frozenset[str],
    *,
    timing_reference: Mapping[str, Mapping[str, Any]] | None = None,
    relative_position: bool = True,
    duration_centering: bool = False,
    shape_features: bool = False,
) -> list[SequenceSample]:
    """Load only approved performances and remap their ragged indices."""
    paths = _selected_shard_paths(dataset_root, split, sample_ids)
    samples: list[SequenceSample] = []
    found: set[str] = set()
    for path in paths or ():
        with np.load(path, allow_pickle=False) as shard:
            stored_ids = shard["sample_ids"]
            note_offsets = shard["sample_note_offsets"]
            chord_offsets = shard["sample_chord_offsets"]
            # Decompress each selected shard once, including the mapping omitted
            # by the historical flat-row training loader.
            arrays = {name: shard[name] for name in (
                "note_features", "note_targets", "note_masks", "chord_features",
                "chord_targets", "chord_masks", "note_to_chord",
            )}
        for index, stored_id in enumerate(stored_ids):
            sample_id = str(stored_id)
            if sample_id not in sample_ids:
                continue
            if sample_id in found:
                raise TrainingRunError(f"duplicate selected sample: {sample_id}")
            found.add(sample_id)
            ns, ne = int(note_offsets[index]), int(note_offsets[index + 1])
            cs, ce = int(chord_offsets[index]), int(chord_offsets[index + 1])
            selected = {
                name: value[ns:ne] if name.startswith("note") else value[cs:ce]
                for name, value in arrays.items()
            }
            selected["note_to_chord"] = arrays["note_to_chord"][ns:ne].astype(np.int64) - cs
            reference = None
            if timing_reference is not None:
                if sample_id not in timing_reference:
                    raise TrainingRunError(f"missing timing reference sample: {sample_id}")
                reference = timing_reference[sample_id]
            samples.append(prepare_sample(sample_id, selected, timing_reference=reference,
                                          relative_position=relative_position,
                                          duration_centering=duration_centering,
                                          shape_features=shape_features))
    if found != set(sample_ids):
        raise TrainingRunError("selected sample loading did not cover all approved IDs")
    return sorted(samples, key=lambda sample: sample.sample_id)


def make_windows(sample: SequenceSample, core_size: int = 128, context_size: int = 16) -> Iterator[SequenceWindow]:
    if core_size <= 0 or context_size < 0:
        raise TrainingRunError("core_size must be positive and context_size nonnegative")
    count = len(sample.chord_features)
    for start in range(0, count, core_size):
        end = min(count, start + core_size)
        yield SequenceWindow(sample, max(0, start - context_size), min(count, end + context_size), start, end)


def attach_bar_geometry(sample: SequenceSample, score_reference: Mapping[str, Any], *,
                        horizons: Sequence[int] = (1, 4, 16)) -> SequenceSample:
    """Derive score-only true-bar geometry; targets, masks, and trust are not read."""
    selected = validate_horizons(horizons)
    if sample.score_ioi_seconds is None:
        raise TrainingRunError(f"{sample.sample_id}: bar geometry requires pure score IOI seconds")
    spans = build_bar_spans(score_reference, selected)
    if spans.chord_count != len(sample.chord_features):
        raise TrainingRunError(f"{sample.sample_id}: bar geometry covers {spans.chord_count} chords, "
                               f"sample has {len(sample.chord_features)}")
    if not np.allclose(spans.score_ioi_seconds, sample.score_ioi_seconds, rtol=1e-6, atol=1e-9):
        raise TrainingRunError(f"{sample.sample_id}: bar geometry score clock differs from sample IOI seconds")
    boundaries = bar_boundary_indices(score_reference)
    rows = np.stack((spans.starts, spans.ends), axis=1) if len(spans.starts) else np.zeros((0, 2), dtype=np.int64)
    return replace(sample,
                   bar_span_rows=np.ascontiguousarray(rows, dtype=np.int64),
                   bar_span_horizons=np.ascontiguousarray(spans.horizons, dtype=np.int64),
                   bar_boundary_indices=np.ascontiguousarray(boundaries, dtype=np.int64),
                   bar_requested_horizons=np.ascontiguousarray(selected, dtype=np.int64))


def _stored_bar_spans(sample: SequenceSample) -> BarSpans:
    """Rebuild the planner's span view from stored score-only arrays.

    ``partial_bar_count`` and the two exclusion maps are diagnostic only: the plan
    consumes chord count, endpoints, and horizons, and ``build_bar_spans`` already
    applied partial-bar and missing-endpoint exclusions before these arrays were stored.
    """
    rows = np.asarray(sample.bar_span_rows, dtype=np.int64).reshape(-1, 2)
    return BarSpans(len(sample.chord_features), rows[:, 0], rows[:, 1],
                    np.asarray(sample.bar_span_horizons, dtype=np.int64), sample.score_ioi_seconds,
                    tuple(int(value) for value in np.asarray(sample.bar_requested_horizons, dtype=np.int64)),
                    int(np.asarray(sample.bar_boundary_indices, dtype=np.int64).size), 0, {}, {})


def make_windows_from_config(sample: SequenceSample, config: Mapping[str, Any]) -> list[SequenceWindow]:
    """Cut windows by the configured policy; inference must use the same policy."""
    policy = str(config.get("window_policy", "fixed"))
    if policy == "fixed":
        return list(make_windows(sample, int(config.get("core_size", 128)), int(config.get("context_size", 16))))
    if policy != "bar":
        raise TrainingRunError(f"unknown window policy: {policy}")
    if sample.bar_span_rows is None or sample.bar_boundary_indices is None or sample.bar_requested_horizons is None:
        raise TrainingRunError(f"{sample.sample_id}: bar window policy requires score-only bar geometry")
    spans = _stored_bar_spans(sample)
    planned = plan_bar_windows(spans.chord_count, sample.bar_boundary_indices, spans,
                               int(config["context_size"]),
                               int(config.get("max_window_chords") or DEFAULT_MAX_WINDOW_CHORDS))
    windows: list[SequenceWindow] = []
    for window in planned:
        contained = (spans.starts >= window.core_start) & (spans.ends < window.core_end)
        rows = np.stack((spans.starts[contained], spans.ends[contained]), axis=1) if contained.any() \
            else np.zeros((0, 2), dtype=np.int64)
        windows.append(SequenceWindow(sample, window.start, window.end, window.core_start, window.core_end,
                                      np.ascontiguousarray(rows, dtype=np.int64), spans.horizons[contained]))
    return windows


def shuffle_training_targets(sample: SequenceSample, rng: np.random.Generator) -> SequenceSample:
    """Negative control: targets and masks travel together within one work."""
    ni = rng.permutation(len(sample.note_targets))
    ci = rng.permutation(len(sample.chord_targets))
    return SequenceSample(sample.sample_id, sample.note_features, sample.chord_features,
                          sample.note_targets[ni], sample.chord_targets[ci],
                          sample.note_masks[ni], sample.chord_masks[ci], sample.note_to_chord,
                          sample.score_ioi_seconds, sample.bar_span_rows, sample.bar_span_horizons,
                          sample.bar_boundary_indices, sample.bar_requested_horizons)


def fit_input_normalizers(samples: Sequence[SequenceSample]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for kind in ("note", "chord"):
        count = 0
        sums = squares = None
        for sample in samples:
            values = np.asarray(getattr(sample, f"{kind}_features"), dtype=np.float64)
            if sums is None:
                sums = np.zeros(values.shape[1], dtype=np.float64)
                squares = np.zeros(values.shape[1], dtype=np.float64)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
            count += len(values)
        if not count or sums is None or squares is None:
            raise TrainingRunError(f"empty train inputs: {kind}")
        mean = sums / count
        scale = np.sqrt(np.maximum(squares / count - mean * mean, 1e-8))
        result[f"{kind}_input_means"] = mean.tolist()
        result[f"{kind}_input_scales"] = scale.tolist()
    return result


def collate_windows(windows: Sequence[SequenceWindow], config: Mapping[str, Any]) -> dict[str, Any]:
    if not windows:
        raise TrainingRunError("cannot collate an empty window batch")
    lengths = [window.end - window.start for window in windows]
    width = max(lengths)
    chord_features = np.zeros((len(windows), width, windows[0].sample.chord_features.shape[1]), dtype=np.float32)
    chord_targets = np.zeros((len(windows), width, len(CHORD_TARGET_NAMES)), dtype=np.float32)
    chord_masks = np.zeros_like(chord_targets, dtype=bool)
    cumulative_enabled = float(config.get("cumulative_weight", 0.0)) > 0
    bar_enabled = bar_weight_total(config) > 0
    interval_seconds_needed = cumulative_enabled or bar_enabled
    score_ioi_seconds = np.zeros((len(windows), width), dtype=np.float64) if interval_seconds_needed else None
    cumulative_ioi_mask = np.zeros((len(windows), width), dtype=bool) if cumulative_enabled else None
    bar_ioi_mask = np.zeros((len(windows), width), dtype=bool) if bar_enabled else None
    span_capacity = max((0 if window.bar_span_rows is None else len(window.bar_span_rows) for window in windows),
                        default=0) if bar_enabled else 0
    bar_span_rows = np.zeros((len(windows), span_capacity, 2), dtype=np.int64) if bar_enabled else None
    bar_span_horizons = np.zeros((len(windows), span_capacity), dtype=np.int64) if bar_enabled else None
    bar_span_candidates = np.zeros((len(windows), span_capacity), dtype=bool) if bar_enabled else None
    bar_span_mask = np.zeros((len(windows), span_capacity), dtype=bool) if bar_enabled else None
    note_features, note_targets, note_masks, note_mapping = [], [], [], []
    records: list[tuple[str, np.ndarray, int, int]] = []
    for index, window in enumerate(windows):
        sample = window.sample
        length = lengths[index]
        cstart, cend = window.core_start - window.start, window.core_end - window.start
        chord_features[index, :length] = sample.chord_features[window.start:window.end]
        chord_targets[index, :length] = sample.chord_targets[window.start:window.end]
        chord_masks[index, cstart:cend] = sample.chord_masks[window.core_start:window.core_end]
        if interval_seconds_needed:
            if sample.score_ioi_seconds is None:
                raise TrainingRunError(f"{sample.sample_id}: requires pure score IOI seconds")
            score_ioi_seconds[index, :length] = sample.score_ioi_seconds[window.start:window.end]
        if cumulative_enabled:
            # The interval at core_start reaches into the preceding core.
            # Exclude it, even if the predecessor happens to be in the halo.
            cumulative_ioi_mask[index, cstart + 1:cend] = sample.chord_masks[window.core_start + 1:window.core_end, 1]
        if bar_enabled:
            if window.bar_span_rows is None:
                raise TrainingRunError(f"{sample.sample_id}: bar supervision requires planned bar windows")
            rows = np.asarray(window.bar_span_rows, dtype=np.int64).reshape(-1, 2)
            if len(rows):
                # Spans are stored sample-level and become window-local here; the
                # planner already guarantees both endpoints sit inside this core.
                bar_span_rows[index, :len(rows)] = rows - window.start
                bar_span_horizons[index, :len(rows)] = np.asarray(window.bar_span_horizons, dtype=np.int64)
                bar_span_candidates[index, :len(rows)] = True
                # A complete span supervises only when every interval row inside it
                # is native-valid. Eligibility never changes the window layout.
                bar_span_mask[index, :len(rows)] = [bool(sample.chord_masks[start + 1:end + 1, 1].all())
                                                    for start, end in rows]
            bar_ioi_mask[index, cstart + 1:cend] = sample.chord_masks[window.core_start + 1:window.core_end, 1]
        ni = np.flatnonzero((sample.note_to_chord >= window.core_start) & (sample.note_to_chord < window.core_end))
        note_features.append(sample.note_features[ni])
        note_targets.append(sample.note_targets[ni])
        note_masks.append(sample.note_masks[ni])
        note_mapping.append(sample.note_to_chord[ni] - window.start + index * width)
        records.append((sample.sample_id, ni, cstart, cend))
    nf = np.concatenate(note_features, axis=0)
    cm = np.asarray(config["chord_input_means"], dtype=np.float32)
    cs = np.asarray(config["chord_input_scales"], dtype=np.float32)
    nm = np.asarray(config["note_input_means"], dtype=np.float32)
    ns = np.asarray(config["note_input_scales"], dtype=np.float32)
    batch = {
        "chord_features": torch.from_numpy((chord_features - cm) / cs),
        "note_features": torch.from_numpy((nf - nm) / ns),
        "note_targets": torch.from_numpy(np.concatenate(note_targets, axis=0)),
        "note_masks": torch.from_numpy(np.concatenate(note_masks, axis=0)),
        "chord_targets": torch.from_numpy(chord_targets),
        "chord_masks": torch.from_numpy(chord_masks),
        "note_to_chord": torch.from_numpy(np.concatenate(note_mapping, axis=0)),
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "records": records,
    }
    if cumulative_enabled:
        batch["score_ioi_seconds"] = torch.from_numpy(score_ioi_seconds)
        batch["cumulative_ioi_mask"] = torch.from_numpy(cumulative_ioi_mask)
    if bar_enabled:
        batch["score_ioi_seconds"] = torch.from_numpy(score_ioi_seconds)
        batch["bar_span_rows"] = torch.from_numpy(bar_span_rows)
        batch["bar_span_horizons"] = torch.from_numpy(bar_span_horizons)
        batch["bar_span_candidates"] = torch.from_numpy(bar_span_candidates)
        batch["bar_span_mask"] = torch.from_numpy(bar_span_mask)
        batch["bar_ioi_mask"] = torch.from_numpy(bar_ioi_mask)
    return batch


class SequenceExpressionModel(nn.Module):
    """Both models consume precisely the same current score conditions."""

    def __init__(self, *, architecture: str, note_feature_dim: int, chord_feature_dim: int,
                 hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if architecture not in {"mlp", "gru"} or hidden_dim <= 0 or not 0 <= dropout < 1:
            raise TrainingRunError("invalid sequence model configuration")
        self.architecture = architecture
        if architecture == "gru":
            self.chord_encoder = nn.GRU(chord_feature_dim, hidden_dim, batch_first=True, bidirectional=True)
            context_dim = hidden_dim * 2
        else:
            self.chord_encoder = nn.Identity()
            context_dim = chord_feature_dim
        self.chord_head = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, len(CHORD_TARGET_NAMES)))
        self.note_head = nn.Sequential(nn.Linear(note_feature_dim + context_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
                                       nn.GELU(), nn.Linear(hidden_dim, len(NOTE_TARGET_NAMES)))

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        chords = batch["chord_features"]
        if self.architecture == "gru":
            packed = pack_padded_sequence(chords, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.chord_encoder(packed)
            context, _ = pad_packed_sequence(encoded, batch_first=True, total_length=chords.shape[1])
        else:
            context = chords
        mapped = context.reshape(-1, context.shape[-1])[batch["note_to_chord"]]
        notes = torch.cat((batch["note_features"], mapped), dim=-1)
        return self.note_head(notes), self.chord_head(context)


def make_model(config: Mapping[str, Any]) -> SequenceExpressionModel:
    return SequenceExpressionModel(architecture=str(config["architecture"]),
                                   note_feature_dim=len(config["note_input_means"]),
                                   chord_feature_dim=len(config["chord_input_means"]),
                                   hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]))


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device) if isinstance(value, Tensor) and name != "lengths" else value
            for name, value in batch.items()}


def predict_sample(model: SequenceExpressionModel, arrays: SequenceSample | Mapping[str, np.ndarray], *,
                   config: Mapping[str, Any], timing_reference: Mapping[str, Any] | None = None,
                   device: str | torch.device = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Predict complete sample, writing each disjoint core exactly once.

    ``arrays`` mapping uses original v2 fields and sample-local note_to_chord.
    ``SequenceSample`` is already prepared with the same score conditions.
    Outputs are raw target units, without clipping or performance anchors.
    """
    if isinstance(arrays, SequenceSample):
        sample = arrays
    else:
        # inference must rebuild inputs exactly the way the checkpoint was trained, so both
        # target/input transforms are read back out of the saved config rather than defaulted
        sample = prepare_sample(str(config.get("sample_id", "sample")), arrays,
                                timing_reference=timing_reference,
                                relative_position=bool(config.get("relative_position", True)),
                                duration_centering=bool(config.get("duration_center_per_sample", False)),
                                shape_features=bool(config.get("note_shape_features", False)))
    selected_device = torch.device(device)
    model.eval()
    note_output = np.empty_like(sample.note_targets)
    chord_output = np.empty_like(sample.chord_targets)
    note_count = np.zeros(len(note_output), dtype=np.int32)
    chord_count = np.zeros(len(chord_output), dtype=np.int32)
    note_mean = np.asarray(config["note_target_means"], dtype=np.float32)
    note_scale = np.asarray(config["note_target_scales"], dtype=np.float32)
    chord_mean = np.asarray(config["chord_target_means"], dtype=np.float32)
    chord_scale = np.asarray(config["chord_target_scales"], dtype=np.float32)
    # Training-only supervision is not an inference dependency, but the window
    # policy is: inference with policy=bar must not silently fall back to 128.
    inference_config = dict(config, **{name: 0.0 for name in (*BAR_WEIGHT_KEYS, "cumulative_weight")})
    with torch.inference_mode():
        for window in make_windows_from_config(sample, config):
            batch = collate_windows([window], inference_config)
            note, chord = model(to_device(batch, selected_device))
            _, ni, cs, ce = batch["records"][0]
            note_output[ni] = note.cpu().numpy() * note_scale + note_mean
            chord_output[window.core_start:window.core_end] = chord.cpu().numpy()[0, cs:ce] * chord_scale + chord_mean
            note_count[ni] += 1
            chord_count[window.core_start:window.core_end] += 1
    if np.any(note_count != 1) or np.any(chord_count != 1):
        raise TrainingRunError("prediction cores did not cover each event exactly once")
    if not np.isfinite(note_output).all() or not np.isfinite(chord_output).all():
        raise TrainingRunError("sequence prediction produced non-finite outputs")
    return note_output, chord_output
