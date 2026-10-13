"""The duration shape a piece *shares* across its performers, used as a training target.

Why this exists.  Five training-side levers and two feature rounds all left the duration channel
where it was, and the explanation that survived is about the *target*, not the inputs: every
recording is supervised against its own performance, which is the shared musical shape plus one
performer's irreducible noise.  The cross-performance audit measured that noise directly -- the
per-note reproducibility ceiling sits well above the model's error -- so a model trained on one
recording at a time is being asked to regress noise, and the shape only becomes visible after
averaging across performers.  This module builds exactly that averaged target: for a piece with
several accepted recordings, each recording's median-centred duration label is averaged with its
peers' at the same note position, and every recording of the piece is then supervised against
what they agree on.

Unit discipline.  The profile is defined on **median-centred** labels, so a run that uses it must
also centre its targets (``--duration-centering``); mixing the two units silently moves the
per-recording level back into a target that was built to forget it.  That is why the trainer
refuses the combination rather than accepting it.

Eligibility, and what happens to the rest.  Positions are matched by index, so a piece group
counts only when its recordings have identical note counts and identical score geometry (pitch,
onset beats and written duration).  A recording whose group has no usable peer keeps its own
centred label and is reported as ``shared=False`` instead of being dropped -- dropping them would
quietly change how much data the arm sees, which is exactly the kind of confound this line has
been careful about.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import collections
from typing import Mapping, Sequence

import numpy as np

from training.midi_features import NOTE_TARGET_NAMES

DURATION_COLUMN = NOTE_TARGET_NAMES.index("log_key_duration_ratio")
# pitch, onset beats, written duration: the score geometry that must match for two recordings
# to be the same piece of music rather than two different alignments of it
GEOMETRY_COLUMNS = (0, 3, 4)


@dataclass(frozen=True)
class SharedTarget:
    """The vector a recording should be trained to predict, and how it was obtained."""

    values: np.ndarray
    group_size: int
    shared: bool
    reason: str = ""


def _centred_duration(sample) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(sample.note_targets[:, DURATION_COLUMN], dtype=np.float64)
    supervised = np.asarray(sample.note_masks[:, DURATION_COLUMN], dtype=bool)
    if not np.any(supervised):
        raise ValueError(f"{sample.sample_id}: no supervised duration rows")
    return targets, supervised


def _same_score(samples: Sequence) -> bool:
    length = len(samples[0].note_features)
    geometry = np.asarray(samples[0].note_features[:, GEOMETRY_COLUMNS], dtype=np.float64)
    for sample in samples[1:]:
        if len(sample.note_features) != length:
            return False
        if not np.array_equal(np.asarray(sample.note_features[:, GEOMETRY_COLUMNS], dtype=np.float64),
                              geometry):
            return False
    return True


def shared_duration_targets(samples: Sequence, membership: Mapping[str, str]) -> dict[str, SharedTarget]:
    """Average each piece's median-centred duration labels across its recordings.

    Every position is averaged only over the recordings that actually supervise it, so a masked
    row in one performance cannot poison its peers.  A position no recording supervises keeps
    each recording's own value, and a group whose recordings are not the same score (or has only
    one member present) falls back recording by recording with the reason recorded.
    """
    by_group: dict[str, list] = collections.defaultdict(list)
    result: dict[str, SharedTarget] = {}
    for sample in samples:
        group = membership.get(sample.sample_id)
        if group is None:
            values, _ = _centred_duration(sample)
            result[sample.sample_id] = SharedTarget(values, 1, False, "no piece group in this split")
            continue
        by_group[group].append(sample)
    for group, members in sorted(by_group.items()):
        if len(members) < 2:
            for sample in members:
                values, _ = _centred_duration(sample)
                result[sample.sample_id] = SharedTarget(values, len(members), False, "alone in this split")
            continue
        if not _same_score(members):
            for sample in members:
                values, _ = _centred_duration(sample)
                result[sample.sample_id] = SharedTarget(values, len(members), False,
                                                        "score geometry differs within the group")
            continue
        labels = []
        masks = []
        for sample in members:
            values, supervised = _centred_duration(sample)
            labels.append(np.where(supervised, values, np.nan))
            masks.append(supervised)
        stacked = np.vstack(labels)
        supervised_rows = np.vstack(masks)
        counts = supervised_rows.sum(axis=0)
        with np.errstate(invalid="ignore"):
            profile = np.where(counts > 0, np.nansum(stacked, axis=0) / np.maximum(counts, 1), np.nan)
        for sample in members:
            values, _unsupervised = _centred_duration(sample)
            filled = np.where(np.isfinite(profile), profile, values)
            result[sample.sample_id] = SharedTarget(filled, len(members), True)
    return result


def apply_shared_targets(samples: Sequence, targets: Mapping[str, SharedTarget]):
    """Return copies of ``samples`` whose duration column predicts the shared shape instead."""
    from dataclasses import replace

    updated = []
    for sample in samples:
        entry = targets.get(sample.sample_id)
        if entry is None or not entry.shared:
            updated.append(sample)
            continue
        note_targets = np.array(sample.note_targets, dtype=np.float32, copy=True)
        note_targets[:, DURATION_COLUMN] = np.asarray(entry.values, dtype=np.float32)
        if not np.isfinite(note_targets[:, DURATION_COLUMN]).all():
            raise ValueError(f"{sample.sample_id}: shared profile produced a non-finite target")
        updated.append(replace(sample, note_targets=note_targets))
    return updated


def profile_coverage(targets: Mapping[str, SharedTarget]) -> dict[str, object]:
    """How much of the split actually got a shared target, for the run record."""
    sizes = [entry.group_size for entry in targets.values()]
    shared = [entry for entry in targets.values() if entry.shared]
    reasons: collections.Counter = collections.Counter(entry.reason for entry in targets.values()
                                                       if not entry.shared and entry.reason)
    return {"recordings": len(targets), "shared_recordings": len(shared),
            "share_shared": float(len(shared) / len(targets)) if targets else 0.0,
            "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
            "fallback_reasons": dict(reasons)}


__all__ = ["DURATION_COLUMN", "GEOMETRY_COLUMNS", "SharedTarget", "apply_shared_targets",
           "profile_coverage", "shared_duration_targets"]
