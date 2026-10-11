"""Pre-register a fresh source/work-balanced development selection.

Selection uses metadata and a seeded hash ordering, never model errors or MIDI
targets.  It respects the existing splits, excludes prior works globally, and
opens only selected train/validation shards for read-only structural checking.
The result is development data, not a new untouched test or blind holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from training.midi_features import (
    CHORD_FEATURE_NAMES, CHORD_TARGET_NAMES, NOTE_FEATURE_NAMES, NOTE_TARGET_NAMES,
)
from training.validate_pianocore_dataset import (
    DatasetValidationError, SHA256_PATTERN, _canonical_json, _iter_jsonl,
    _load_json, _safe_relative_path, _sha256_file, _validate_manifest_record,
    _validate_shard,
)


SOURCES = ("ASAP", "ATEPP", "Aria-MIDI", "GiantMIDI-Piano", "PERiScoPe")
DEFAULT_SEED = 20261006
IDENTITY_FIELDS = ("piece_group", "score_semantic_sha256", "score_midi_sha256")


class SelectionError(RuntimeError):
    """A selection or its source binding failed closed."""


class SelectionCapacityError(SelectionError):
    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__("insufficient globally distinct works: " + json.dumps(report, sort_keys=True))


def _rank(seed: int, *parts: str) -> str:
    return hashlib.sha256(_canonical_json([seed, *parts])).hexdigest()


def _global_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Connected components across piece, semantic and raw score identities."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row in rows:
        nodes = [f"{field}:{row[field]}" for field in IDENTITY_FIELDS]
        for node in nodes[1:]:
            left, right = find(nodes[0]), find(node)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return {str(row["id"]): find(f"piece_group:{row['piece_group']}") for row in rows}


def plan_selection(
    rows: Sequence[Mapping[str, Any]], excluded_ids: Iterable[str], *,
    train_per_source: int = 100, validation_per_source: int = 20,
    sources: Sequence[str] = SOURCES, seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic bipartite matching, one performance per global work.

    A shared work can count for exactly one source, not once per source.  An
    augmenting path prevents greedy allocation from starving scarce sources.
    """
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                   for n in (train_per_source, validation_per_source))):
        raise SelectionError("seed must be an integer and source quotas must be positive integers")
    sources = tuple(sources)
    if not sources or len(sources) != len(set(sources)) or any(not s for s in sources):
        raise SelectionError("sources must be unique non-empty names")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SelectionError("duplicate manifest IDs")
    for row in rows:
        if row["official_split"] == "test" and row["split"] != "test":
            raise SelectionError("official test record moved into a development split")
        if row["official_split"] not in {"train", "test"} or row["split"] not in {"train", "validation", "test"}:
            raise SelectionError("unsupported source split")
        if row["official_split"] == "train" and row["split"] == "test":
            raise SelectionError("official train record moved into test")
    excluded = set(excluded_ids)
    if excluded - by_id.keys():
        raise SelectionError("excluded IDs absent from manifest: " + ", ".join(sorted(excluded - by_id.keys())[:5]))
    group_by_id = _global_groups(rows)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_by_group[group_by_id[str(row["id"])]].add(str(row["split"]))
    if any(len(splits) != 1 for splits in split_by_group.values()):
        raise SelectionError("global score/work identity crosses original dataset splits")
    blocked_groups = {group_by_id[sample_id] for sample_id in excluded}
    prior_identities = {field: {by_id[sample_id][field] for sample_id in excluded}
                        for field in IDENTITY_FIELDS}
    exclusion_reasons: Counter[str] = Counter()
    candidates: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    excluded_rows = 0
    for row in rows:
        group = group_by_id[str(row["id"])]
        if group in blocked_groups:
            excluded_rows += 1
            direct_reasons = [field for field in IDENTITY_FIELDS if row[field] in prior_identities[field]]
            exclusion_reasons.update(direct_reasons or ["transitive_connected_identity_only"])
            if str(row["id"]) in excluded:
                exclusion_reasons["prior_sample_id"] += 1
            continue
        if row["official_split"] == "train" and row["split"] in {"train", "validation"} and row["performance_dataset"] in sources:
            candidates[(str(row["split"]), str(row["performance_dataset"]))][group].append(row)
    report: dict[str, Any] = {
        "seed": seed, "sources": list(sources),
        "requested_per_source": {"train": train_per_source, "validation": validation_per_source},
        "available_global_work_counts": {}, "selected_source_counts": {},
        "excluded_id_count": len(excluded), "excluded_global_work_count": len(blocked_groups),
        "excluded_manifest_record_count": excluded_rows,
        "exclusion_reason_counts_nonexclusive": dict(sorted(exclusion_reasons.items())),
        "official_test_metadata_record_count": sum(row["official_split"] == "test" for row in rows),
        "official_test_midi_or_label_files_opened": 0,
        "selection_policy": "seeded hash + source/work maximum matching; one performance per global work",
        "selection_uses_model_errors_or_targets": False,
        "development_only": True, "untouched_blind_holdout": False,
        "exclusion_identity_fields": list(IDENTITY_FIELDS),
        "metadata_only_representativeness": {
            "available": ["note_count", "chord_count", "mean_notes_per_chord_proxy"],
            "unavailable": ["score_pitch_range", "local_event_density", "tempo_change_count", "cumulative_drift"],
            "policy": "report available distributions; no quality/error-dependent stratification",
        },
    }
    chosen: list[dict[str, Any]] = []
    capacity_failed = False
    for split, quota in (("train", train_per_source), ("validation", validation_per_source)):
        available = {source: len(candidates[(split, source)]) for source in sources}
        ordered = {source: sorted(candidates[(split, source)], key=lambda group: (_rank(seed, split, source, group), group)) for source in sources}
        owners: dict[str, str] = {}

        def augment(source: str, visited_groups: set[str], visited_sources: set[str]) -> bool:
            if source in visited_sources:
                return False
            visited_sources.add(source)
            for group in ordered[source]:
                if group in visited_groups or owners.get(group) == source:
                    continue
                visited_groups.add(group)
                owner = owners.get(group)
                if owner is None or augment(owner, visited_groups, visited_sources):
                    owners[group] = source
                    return True
            return False

        # Allocate scarce sources first; matching can subsequently reassign.
        source_order = sorted(sources, key=lambda source: (available[source], source))
        for _ in range(quota):
            for source in source_order:
                if not augment(source, set(), set()):
                    capacity_failed = True
        counts = Counter(owners.values())
        report["available_global_work_counts"][split] = available
        report["selected_source_counts"][split] = {source: counts[source] for source in sources}
        for group, source in sorted(owners.items()):
            row = min(candidates[(split, source)][group], key=lambda value: (_rank(seed, split, source, group, str(value["id"])), str(value["id"])))
            chosen.append(dict(row))
    if capacity_failed:
        report["status"] = "insufficient_capacity"
        raise SelectionCapacityError(report)
    report["status"] = "planned"
    report["selected_count"] = len(chosen)
    report["selected_global_work_count"] = len({group_by_id[row["id"]] for row in chosen})
    report["selected_identity_overlap_train_validation"] = 0
    report["selected_identity_overlap_prior_selection"] = 0
    return sorted(chosen, key=lambda row: (row["split"], row["performance_dataset"], row["id"])), report


def _source_bindings(root: Path, cache_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if (root / "_INCOMPLETE").exists():
        raise SelectionError("source dataset is incomplete")
    integrity_path = root / "integrity.json"
    if (root / "_SUCCESS").read_text(encoding="ascii").strip() != _sha256_file(integrity_path):
        raise SelectionError("source dataset _SUCCESS/integrity binding mismatch")
    inventory = _load_json(integrity_path)
    if inventory.get("pipeline_version") != 2 or inventory.get("pipeline") != "auto_music_player.pianocore_expression":
        raise SelectionError("unsupported dataset inventory")
    entries: dict[str, dict[str, Any]] = {}
    for entry in inventory["files"]:
        relative, _ = _safe_relative_path(root, entry["path"], "inventory path")
        if relative in entries:
            raise SelectionError("duplicate inventory path")
        entries[relative] = entry
    cache = _load_json(cache_path)
    validation = cache.get("validation_report", {})
    if (validation.get("status") != "valid"
            or Path(str(validation.get("dataset_root"))).resolve() != root
            or cache.get("dataset_success_sha256") != _sha256_file(root / "_SUCCESS")):
        raise SelectionError("validation cache does not bind to this accepted dataset")
    bindings = {"dataset_root": str(root), "validation_cache": str(cache_path),
                "validation_cache_sha256": _sha256_file(cache_path),
                "dataset_success_sha256": _sha256_file(root / "_SUCCESS"),
                "dataset_integrity_sha256": _sha256_file(integrity_path), "metadata_files": {}}
    for relative in ("manifest.jsonl", "schema.json", "build_config.json", "normalizer.json", "qc_summary.json", "rejects.jsonl"):
        entry = entries.get(relative)
        path = root / relative
        if entry is None or path.stat().st_size != entry["size_bytes"] or _sha256_file(path) != entry["sha256"]:
            raise SelectionError(f"source metadata binding mismatch: {relative}")
        bindings["metadata_files"][relative] = entry
    return bindings, entries


class _ReadOnlyAccumulator:
    def update(self, *args: Any, **kwargs: Any) -> None:
        # Existing normalizer remains unchanged; selected shard verification is
        # structural, not a second fit of the full training normalizer.
        pass


def build_selection(
    dataset_root: str | Path, output: str | Path | None, *, validation_cache: str | Path,
    excluded_id_files: Sequence[str | Path], train_per_source: int = 100,
    validation_per_source: int = 20, sources: Sequence[str] = SOURCES,
    seed: int = DEFAULT_SEED, inspect_only: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    cache_path = Path(validation_cache).resolve()
    target = Path(output).resolve() if output is not None else None
    if not inspect_only and (target is None or target.exists()):
        raise SelectionError("output must be a new, non-existing directory")
    if target is not None and (target == root or root in target.parents):
        raise SelectionError("selection output must be outside the source dataset")
    bindings, inventory = _source_bindings(root, cache_path)
    rows = list(_iter_jsonl(root / "manifest.jsonl"))
    shard_paths = {path for path in inventory if path.endswith(".npz")}
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        _validate_manifest_record(root, row, index, shard_paths)
        entry = inventory[row["shard_path"]]
        if row["shard_sha256"] != entry["sha256"] or row["shard_size_bytes"] != entry["size_bytes"]:
            raise SelectionError("manifest/shard inventory binding mismatch")
        by_shard[row["shard_path"]].append(row)
    validation = _load_json(cache_path)["validation_report"]
    if validation.get("accepted_count") != len(rows):
        raise SelectionError("validation cache/manifest accepted count mismatch")
    excluded: set[str] = set()
    exclusions = []
    if not excluded_id_files:
        raise SelectionError("at least one explicit prior-selection ID file is required")
    for raw_path in excluded_id_files:
        path = Path(raw_path).resolve()
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise SelectionError(f"duplicate exclusion IDs in {path}")
        excluded.update(ids)
        exclusions.append({"path": str(path), "sha256": _sha256_file(path), "id_count": len(ids)})
    selected, report = plan_selection(rows, excluded, train_per_source=train_per_source,
        validation_per_source=validation_per_source, sources=sources, seed=seed)
    report["bindings"] = bindings
    report["prior_selection_files"] = exclusions
    report["implementation_sha256"] = {
        "build_development_selection.py": _sha256_file(Path(__file__)),
        "validate_pianocore_dataset.py": _sha256_file(Path(__file__).with_name("validate_pianocore_dataset.py")),
    }
    if inspect_only:
        return report
    assert target is not None
    target.mkdir(parents=True, exist_ok=False)
    (target / "_INCOMPLETE").write_text("selected train/validation shard verification in progress\n", encoding="utf-8")
    shard_checks = []
    dimensions = (len(NOTE_FEATURE_NAMES), len(NOTE_TARGET_NAMES), len(CHORD_FEATURE_NAMES), len(CHORD_TARGET_NAMES))
    selected_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_shard[row["shard_path"]].append(row)
    for relative, selected_rows in sorted(selected_by_shard.items()):
        if any(row["official_split"] != "train" or row["split"] not in {"train", "validation"} for row in by_shard[relative]):
            raise SelectionError("refusing to open a shard containing official test data")
        _, path = _safe_relative_path(root, relative, "selected shard")
        entry = inventory[relative]
        if path.stat().st_size != entry["size_bytes"] or _sha256_file(path) != entry["sha256"]:
            raise SelectionError(f"selected shard binding mismatch: {relative}")
        counts = _validate_shard(relative, {**entry, "path": path}, by_shard[relative], dimensions, _ReadOnlyAccumulator())
        shard_checks.append({**entry, "shard_sample_count": counts[0],
                             "selected_sample_count": len(selected_rows),
                             "selected_ids": sorted(row["id"] for row in selected_rows)})
    report["status"] = "verified"
    report["verified_selected_shards"] = shard_checks
    report["selected_note_count"] = sum(row["note_count"] for row in selected)
    report["selected_chord_count"] = sum(row["chord_count"] for row in selected)
    distributions: dict[str, Any] = {}
    for split in ("train", "validation"):
        distributions[split] = {}
        for source in sources:
            subset = [row for row in selected if row["split"] == split and row["performance_dataset"] == source]
            distributions[split][source] = {key: {"min": float(min(values)), "median": float(np.median(values)), "max": float(max(values))}
                for key, values in (("note_count", [row["note_count"] for row in subset]),
                                    ("chord_count", [row["chord_count"] for row in subset]),
                                    ("mean_notes_per_chord_proxy", [row["note_count"] / row["chord_count"] for row in subset]))}
    report["metadata_distributions"] = distributions
    for split in ("train", "validation"):
        ids = sorted(row["id"] for row in selected if row["split"] == split)
        (target / f"{split}_sample_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (target / "selected_manifest.jsonl").write_text("".join(_canonical_json(row).decode("utf-8") + "\n" for row in selected), encoding="utf-8")
    (target / "selection.json").write_bytes(_canonical_json(report) + b"\n")
    # Recheck immutable metadata/exclusions immediately before the seal.
    final_bindings, _ = _source_bindings(root, cache_path)
    if final_bindings != bindings:
        raise SelectionError("source metadata/cache bindings changed during selection")
    for entry in exclusions:
        if _sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise SelectionError("prior-selection exclusion file changed during selection")
    files = [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
             for path in sorted(target.iterdir()) if path.name != "_INCOMPLETE"]
    (target / "integrity.json").write_bytes(_canonical_json({"pipeline": "auto_music_player.development_selection", "schema_version": 1, "files": files}) + b"\n")
    (target / "_INCOMPLETE").unlink()
    (target / "_SUCCESS").write_text(_sha256_file(target / "integrity.json") + "\n", encoding="ascii")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--exclude-id-file", action="append", required=True)
    parser.add_argument("--train-per-source", type=int, default=100)
    parser.add_argument("--validation-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_selection(args.dataset, args.output, validation_cache=args.validation_cache,
            excluded_id_files=args.exclude_id_file, train_per_source=args.train_per_source,
            validation_per_source=args.validation_per_source, seed=args.seed, inspect_only=args.inspect_only)
    except (SelectionError, DatasetValidationError, OSError, ValueError, KeyError) as exc:
        print(f"development selection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
