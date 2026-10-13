"""Which recordings of one score can be matched position by position.

Both the cross-performance reproducibility probe and the shared-profile training target need the
same thing: groups of accepted performances of one score whose note and chord positions line up
exactly, so that "the shape the performers agree on" is an average over matching indices rather
than a second alignment pass nobody has written.

This used to live inside ``tools/audit_cross_performance_reproducibility.py``.  It moved here
when training started consuming the same grouping, because ``training`` must not import from
``tools`` and a second implementation of eligibility rules is how two tools end up measuring
different things while reporting the same name.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path


def score_groups(manifest_path: Path, split: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Groups of two or more accepted performances of one score with identical lengths.

    Identical chord *and* note counts are required because positions are matched index by
    index; a group whose performances disagree on length cannot be aligned without a second
    alignment pass, which this probe deliberately does not attempt.  ``qc_status`` is read as
    accepted when it is blank, ``accepted`` or ``ok``: the builder left the field unset for
    the earliest shards, and those are the corpus.
    """
    groups: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != split:
                continue
            if str(row.get("qc_status")) not in {"", "accepted", "ok"}:
                continue
            group = str(row.get("piece_group") or "")
            if not group:
                continue
            groups[group].append({"id": str(row["id"]), "performer": str(row.get("performer") or ""),
                                  "chord_count": int(row.get("chord_count") or 0),
                                  "note_count": int(row.get("note_count") or 0)})
    eligible = []
    for group, rows in groups.items():
        if len(rows) < 2:
            continue
        if len({int(row["chord_count"]) for row in rows}) != 1:
            continue
        if len({int(row["note_count"]) for row in rows}) != 1:
            continue
        eligible.append((group, sorted(rows, key=lambda row: row["id"])))
    return sorted(eligible, key=lambda item: item[0])


def select_performances(rows: list[dict[str, object]], cap: int) -> list[str]:
    """Take at most ``cap`` performances, no two from the same ``performer``.

    Deterministic: for each performer the lowest sample id wins, and the performer buckets are
    sorted, so the same manifest always yields the same selection.  An empty ``performer``
    collapses into one bucket, which is conservative -- several unnamed recordings are treated
    as a single possible person rather than as independent evidence.  Every pair produced
    downstream is therefore two different performers by construction.
    """
    buckets: dict[str, str] = {}
    for row in sorted(rows, key=lambda row: (str(row["performer"]), str(row["id"]))):
        buckets.setdefault(str(row["performer"]), str(row["id"]))
    return [buckets[performer] for performer in sorted(buckets)[:cap]]


def group_membership(rows_by_group: list[tuple[str, list[dict[str, object]]]],
                     wanted: frozenset[str] | set[str]) -> dict[str, str]:
    """``sample id -> piece group`` for the ids actually being used.

    Restricting to ``wanted`` matters at training time: a piece group may span splits, and a
    profile must average only over recordings the model is allowed to see.
    """
    membership: dict[str, str] = {}
    for group, rows in rows_by_group:
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in wanted:
                membership[sample_id] = group
    return membership
