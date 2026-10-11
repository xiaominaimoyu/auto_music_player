"""Load and apply auditable per-sample expression quality sidecars.

The sidecar is deliberately applied after the dataset builder's native
per-note/per-chord masks.  A quality decision can remove supervision or lower
its loss weight, but it cannot turn an originally invalid target back on.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES


class QualityMaskError(ValueError):
    """Raised when a quality sidecar is malformed or incompatible."""


TARGET_NAMES = NOTE_TARGET_NAMES + CHORD_TARGET_NAMES


@dataclass(frozen=True)
class QualityMaskData:
    decisions: dict[str, dict[str, Any]]
    path: str
    sha256: str
    policy_version: str
    row_count: int
    reviewed_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _identity_decision(row: Mapping[str, Any]) -> bool:
    masks = row.get("target_mask")
    weights = row.get("target_weight")
    return (
        row.get("reviewed") is False
        and isinstance(masks, dict)
        and isinstance(weights, dict)
        and all(masks.get(name) is True for name in TARGET_NAMES)
        and all(float(weights.get(name, -1.0)) == 1.0 for name in TARGET_NAMES)
    )


def load_quality_mask(path: str | Path) -> QualityMaskData:
    """Load a JSONL sidecar, retaining only non-identity decisions in memory."""

    mask_path = Path(path).resolve()
    if not mask_path.is_file():
        raise QualityMaskError(f"quality mask does not exist: {mask_path}")
    decisions: dict[str, dict[str, Any]] = {}
    policy_version: str | None = None
    row_count = 0
    reviewed_count = 0
    seen_ids: set[str] = set()
    with mask_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QualityMaskError(
                    f"quality mask line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise QualityMaskError(f"quality mask line {line_number} is not an object")
            sample_id = row.get("sample_id")
            if not sample_id:
                raise QualityMaskError(f"quality mask line {line_number} has no sample_id")
            sample_id = str(sample_id)
            if sample_id in seen_ids:
                raise QualityMaskError(f"duplicate quality mask sample_id: {sample_id}")
            seen_ids.add(sample_id)
            row_policy = row.get("policy_version")
            if not isinstance(row_policy, str) or not row_policy:
                raise QualityMaskError(
                    f"quality mask line {line_number} has no policy_version"
                )
            if policy_version is None:
                policy_version = row_policy
            elif row_policy != policy_version:
                raise QualityMaskError(
                    f"quality mask mixes policy versions: {policy_version!r} and {row_policy!r}"
                )
            masks = row.get("target_mask")
            weights = row.get("target_weight")
            if not isinstance(masks, dict) or set(masks) != set(TARGET_NAMES):
                raise QualityMaskError(
                    f"quality mask line {line_number} has incompatible target_mask keys"
                )
            if not isinstance(weights, dict) or set(weights) != set(TARGET_NAMES):
                raise QualityMaskError(
                    f"quality mask line {line_number} has incompatible target_weight keys"
                )
            if any(not isinstance(masks[name], bool) for name in TARGET_NAMES):
                raise QualityMaskError(
                    f"quality mask line {line_number} contains a non-boolean target mask"
                )
            normalized_weights: dict[str, float] = {}
            for name in TARGET_NAMES:
                try:
                    value = float(weights[name])
                except (TypeError, ValueError) as exc:
                    raise QualityMaskError(
                        f"quality mask line {line_number} has invalid weight for {name}"
                    ) from exc
                if not math.isfinite(value) or value < 0.0:
                    raise QualityMaskError(
                        f"quality mask line {line_number} has invalid weight for {name}"
                    )
                normalized_weights[name] = value
            reviewed = row.get("reviewed")
            if not isinstance(reviewed, bool):
                raise QualityMaskError(
                    f"quality mask line {line_number} has invalid reviewed flag"
                )
            if reviewed:
                reviewed_count += 1
            if _identity_decision(row):
                continue
            decisions[sample_id] = {
                "target_mask": {name: bool(masks[name]) for name in TARGET_NAMES},
                "target_weight": normalized_weights,
                "policy_version": row_policy,
            }
    if policy_version is None:
        raise QualityMaskError("quality mask is empty")
    return QualityMaskData(
        decisions=decisions,
        path=str(mask_path),
        sha256=_sha256(mask_path),
        policy_version=policy_version,
        row_count=row_count,
        reviewed_count=reviewed_count,
    )


def apply_quality_mask(
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
    decisions: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native masks plus sidecar masks and multiplicative weights."""

    if kind == "note":
        mask_name = "note_masks"
        offset_name = "sample_note_offsets"
        target_names = NOTE_TARGET_NAMES
    elif kind == "chord":
        mask_name = "chord_masks"
        offset_name = "sample_chord_offsets"
        target_names = CHORD_TARGET_NAMES
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(kind)
    native_masks = np.asarray(arrays[mask_name], dtype=np.bool_)
    masks = native_masks.copy()
    quality_weights = np.ones(masks.shape, dtype=np.float32)
    if not decisions:
        return masks, quality_weights
    sample_ids = np.asarray(arrays["sample_ids"])
    offsets = np.asarray(arrays[offset_name], dtype=np.int64)
    if len(offsets) != len(sample_ids) + 1 or int(offsets[-1]) != len(masks):
        raise QualityMaskError(
            f"{kind} sample offsets do not cover the shard arrays"
        )
    for sample_index, raw_id in enumerate(sample_ids):
        decision = decisions.get(_sample_id(raw_id))
        if decision is None:
            continue
        start, end = int(offsets[sample_index]), int(offsets[sample_index + 1])
        target_masks = decision["target_mask"]
        target_weights = decision["target_weight"]
        for target_index, target_name in enumerate(target_names):
            masks[start:end, target_index] &= bool(target_masks[target_name])
            quality_weights[start:end, target_index] *= float(
                target_weights[target_name]
            )
    return masks, quality_weights


__all__ = [
    "QualityMaskData",
    "QualityMaskError",
    "TARGET_NAMES",
    "apply_quality_mask",
    "load_quality_mask",
]
