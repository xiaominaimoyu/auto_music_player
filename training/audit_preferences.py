"""Audit manually returned offline answers without changing the originals.

The first collection stage is evaluation-only: training_ready stays false even
after answers arrive.  Inferential intervals are withheld for fewer than five
participants or five works, and otherwise remain exploratory two-way cluster
bootstrap intervals rather than independent-vote binomial intervals.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from training.preference_bundle import PreferenceError, canonical_json, file_hash, json_hash, write_json
from training.preference_dataset import load_response, validate_bundle


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    offset = (len(values) - 1) * probability
    lower = int(offset)
    fraction = offset - lower
    return values[lower] * (1.0 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction


def preference_statistics(rows: list[dict[str, Any]], *, bootstrap: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Return equal-group descriptive metrics and crossed-cluster uncertainty.

    Repeated excerpts first collapse into participant x work cells.  The cell
    mean gives each observed pair equal weight; participant/work macro means are
    also reported so unequal response counts are visible.
    """
    judged = [row for row in rows if row["candidate_score"] is not None]
    decisive = [row for row in judged if row["outcome"] in {"candidate", "baseline"}]
    counts = Counter(row["outcome"] or "no_selection" for row in rows)
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in judged:
        cells[(row["participant_id"], row["piece_group"])].append(row["candidate_score"])
    cell_values = {key: float(_mean(value)) for key, value in cells.items()}
    participant_values: dict[str, list[float]] = defaultdict(list)
    piece_values: dict[str, list[float]] = defaultdict(list)
    for (participant, piece), value in cell_values.items():
        participant_values[participant].append(value)
        piece_values[piece].append(value)
    by_participant = {key: _mean(value) for key, value in sorted(participant_values.items())}
    by_piece = {key: _mean(value) for key, value in sorted(piece_values.items())}
    participants, pieces = sorted(participant_values), sorted(piece_values)
    interval = None
    bootstrap_draws = 0
    interval_reason = "insufficient independent clusters: require at least 5 participants and 5 works"
    if len(participants) >= 5 and len(pieces) >= 5 and bootstrap >= 100:
        rng = random.Random(seed)
        draws = []
        for _ in range(bootstrap):
            participant_weights = Counter(rng.choices(participants, k=len(participants)))
            piece_weights = Counter(rng.choices(pieces, k=len(pieces)))
            numerator = 0.0
            denominator = 0
            for (participant, piece), value in cell_values.items():
                weight = participant_weights[participant] * piece_weights[piece]
                numerator += weight * value
                denominator += weight
            if denominator:
                draws.append(numerator / denominator)
        bootstrap_draws = len(draws)
        if len(draws) >= 100:
            interval = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
            interval_reason = "exploratory participant-and-work crossed bootstrap of equal observed cells; not a powered acceptance test"
    elif len(participants) >= 5 and len(pieces) >= 5:
        interval_reason = "bootstrap disabled or fewer than 100 requested replicates"
    by_position = {}
    for position in sorted({row["position"] for row in rows}):
        selected = [row for row in rows if row["position"] == position]
        by_position[str(position)] = {"count": len(selected), "candidate_score": _mean(row["candidate_score"] for row in selected if row["candidate_score"] is not None)}
    repeats = defaultdict(list)
    for row in judged:
        repeated_key = (row["participant_id"], row["piece_group"], *sorted([row["a_sha256"], row["b_sha256"]]))
        repeats[repeated_key].append(row["outcome"])
    repeated = [values for values in repeats.values() if len(values) > 1]
    return {
        "counts": {key: counts[key] for key in ("candidate", "baseline", "tie", "skip", "no_selection")},
        "judged_count": len(judged), "decisive_count": len(decisive),
        "judging_participant_count": len(participants), "judged_piece_group_count": len(pieces),
        "candidate_win_fraction_decisive": _mean(1.0 if row["outcome"] == "candidate" else 0.0 for row in decisive),
        "candidate_score_with_half_ties": _mean(row["candidate_score"] for row in judged),
        "participant_equal_candidate_score": _mean(by_participant.values()),
        "piece_equal_candidate_score": _mean(by_piece.values()),
        "participant_work_cell_equal_score": _mean(cell_values.values()),
        "participant_scores": by_participant, "piece_scores": by_piece,
        "crossed_cluster_ci95": interval, "bootstrap_draws": bootstrap_draws,
        "interval_interpretation": interval_reason,
        "position_bias": {
            "a_choice_fraction_decisive": _mean(1.0 if row["choice"] == "a" else 0.0 for row in decisive),
            "candidate_shown_a_fraction": _mean(1.0 if row["candidate_at"] == "a" else 0.0 for row in rows),
            "candidate_score_when_shown_a": _mean(row["candidate_score"] for row in judged if row["candidate_at"] == "a"),
            "candidate_score_when_shown_b": _mean(row["candidate_score"] for row in judged if row["candidate_at"] == "b"),
            "by_presentation_position": by_position,
        },
        "repeat_consistency": {"repeated_pair_count": len(repeated), "all_choices_agree_count": sum(len(set(values)) == 1 for values in repeated), "used_for_automatic_exclusion": False},
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json(row) + b"\n")


def audit_responses(*, bundle_root: str | Path, responses_root: str | Path, output: str | Path, bootstrap: int = 1000, seed: int = 42) -> dict[str, Any]:
    if not 0 <= bootstrap <= 10000:
        raise PreferenceError("bootstrap must be between 0 and 10000")
    bundle = validate_bundle(bundle_root)
    source = Path(responses_root).resolve()
    destination = Path(output).resolve()
    if not source.is_dir():
        raise PreferenceError("responses directory does not exist")
    if destination == source or source in destination.parents or destination in source.parents:
        raise PreferenceError("audit output and response directory must be separate, non-nested paths")
    if destination == bundle["root"] or destination in bundle["root"].parents or (bundle["root"] / "public") == destination or (bundle["root"] / "public") in destination.parents:
        raise PreferenceError("audit output must not overwrite or enter the public bundle")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise PreferenceError("audit output must be absent or empty")
    files = sorted(source.glob("*.json"))
    inventory = []
    validated = []
    rejects = []
    excluded = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            rejects.append({"source_file": path.name, "reason": "symlink/non-file responses are not accepted"})
            continue
        inventory.append({"source_file": path.name, "sha256": file_hash(path), "size_bytes": path.stat().st_size})
        try:
            result = load_response(path, bundle=bundle)
        except (OSError, PreferenceError) as exc:
            rejects.append({"source_file": path.name, "source_sha256": file_hash(path), "reason": str(exc)})
            continue
        if result["synthetic"]:
            excluded.append({"source_file": path.name, "reason": "synthetic_response_never_human", "response_id": result["response_id"]})
        else:
            validated.append(result)
    # Reject *all* conflicting versions; file order must never decide the winner.
    by_response = defaultdict(list)
    by_session = defaultdict(list)
    for result in validated:
        by_response[result["response_id"]].append(result)
        by_session[result["session_id"]].append(result)
    conflicted_ids = {key for key, values in by_response.items() if len({value["source_sha256"] for value in values}) > 1}
    for values in by_session.values():
        if len({value["participant_id"] for value in values}) > 1:
            conflicted_ids.update(value["response_id"] for value in values)
    unique = []
    seen_response_ids = set()
    for result in validated:
        reason = "conflicting_response_or_session_identity" if result["response_id"] in conflicted_ids else "duplicate_response" if result["response_id"] in seen_response_ids else None
        if reason:
            excluded.append({"source_file": result["source_file"], "response_id": result["response_id"], "reason": reason})
        else:
            seen_response_ids.add(result["response_id"])
            unique.append(result)
    grouped_rows = defaultdict(list)
    for result in unique:
        for row in result["rows"]:
            grouped_rows[(row["participant_id"], row["bundle_id"], row["trial_id"])].append({**row, "source_file": result["source_file"], "source_sha256": result["source_sha256"]})
    accepted = []
    trial_conflict_count = 0
    for key, rows in sorted(grouped_rows.items()):
        if len({row["outcome"] for row in rows}) > 1:
            trial_conflict_count += 1
            for row in rows:
                excluded.append({"source_file": row["source_file"], "response_id": row["response_id"], "participant_id": key[0], "trial_id": key[2], "reason": "conflicting_repeat_answer"})
            continue
        # Exact semantic repeat from a new session receives one vote, not two.
        accepted.append(rows[0])
        for row in rows[1:]:
            excluded.append({"source_file": row["source_file"], "response_id": row["response_id"], "participant_id": key[0], "trial_id": key[2], "reason": "duplicate_participant_trial_answer"})
    statistics = preference_statistics(accepted, bootstrap=bootstrap, seed=seed)
    summary = {
        "schema": "auto_music_player.preference_audit", "schema_version": 1,
        "status": "complete", "study_id": bundle["manifest"]["study_id"], "study_version": bundle["manifest"]["study_version"],
        "bundle_id": bundle["manifest"]["bundle_id"], "manifest_sha256": bundle["manifest"]["manifest_sha256"],
        "input_inventory_sha256": json_hash(inventory), "scanned_response_files": len(files), "invalid_response_files": len(rejects),
        "excluded_counts": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "conflicting_response_id_count": len(conflicted_ids), "conflicting_participant_trial_count": trial_conflict_count,
        "human_participant_count": len({row["participant_id"] for row in accepted}), "human_trial_records": len(accepted),
        "human_preference_count": statistics["judged_count"], "has_human_preferences": bool(statistics["judged_count"]),
        "training_ready": False,
        "training_readiness_reason": "evaluation-only stage; requires real responses, independent participant/work split review and a separate training decision",
        "identity_and_listening_are_self_reported": True,
        "synthetic_never_included_in_human_statistics": True,
        "statistics": statistics,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "_INCOMPLETE").write_text("preference audit in progress\n", encoding="utf-8")
    _write_jsonl(destination / "response_events.jsonl", accepted)
    _write_jsonl(destination / "preference_manifest.jsonl", (row for row in accepted if row["candidate_score"] is not None))
    _write_jsonl(destination / "rejects.jsonl", rejects)
    _write_jsonl(destination / "excluded.jsonl", excluded)
    write_json(destination / "input_inventory.json", inventory)
    write_json(destination / "summary.json", summary)
    artifacts = ["response_events.jsonl", "preference_manifest.jsonl", "rejects.jsonl", "excluded.jsonl", "input_inventory.json", "summary.json"]
    write_json(destination / "integrity.json", {"files": [{"path": name, "sha256": file_hash(destination / name)} for name in artifacts]})
    (destination / "_INCOMPLETE").unlink()
    (destination / "_SUCCESS").write_text(file_hash(destination / "integrity.json") + "\n", encoding="ascii")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--responses", required=True, help="read-only directory of returned answer JSONs")
    parser.add_argument("--output", required=True, help="new derived audit directory")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(audit_responses(bundle_root=args.bundle_root, responses_root=args.responses, output=args.output, bootstrap=args.bootstrap, seed=args.seed), ensure_ascii=False))
    except (OSError, PreferenceError) as exc:
        print(f"preference audit failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
