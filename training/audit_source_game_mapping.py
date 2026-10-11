"""Bind native source-to-game diagnostics to a completed production-route audit.

All inputs are read-only. Only train/validation references are accepted; rejected
imports remain in the denominator. This command never starts playback or trains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from core.midi_reader import read_midi_source
from core.profile import load_profiles
from core.source_score import AdaptOptions
from training import audit_game_expressibility as game
from training.pianocore_dataset import RefinedArchive, write_deterministic_npz
from training.source_game_mapping import build_sample_game_mapping
from training.tempo_reference import _canonical, _safe_path, _sha256, load_timing_reference


_ROOT = Path(__file__).resolve().parents[1]
_CASE_FIELDS = ("id", "split", "performance_dataset", "piece_group", "style", "score_midi_path",
                "score_midi_sha256", "status")
_STATUSES = {"complete", "import_rejected", "transport_rejected"}


def _root(path: str | Path) -> Path:
    value = Path(path).resolve()
    return value.parent if value.is_file() else value


def _member_path(value: str) -> str:
    """ZIP members are relative POSIX paths, never filesystem destinations."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("unsafe source ZIP member path")
    path = PurePosixPath(value)
    # PianoCoRe legitimately uses colons in work titles. These names are
    # looked up inside the ZIP, never extracted as Windows filesystem paths.
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or PureWindowsPath(value).drive):
        raise ValueError("unsafe source ZIP member path")
    return value


def _implementation_bindings(profiles_dir: Path, config_path: Path) -> dict[str, str]:
    bindings = game._implementation_bindings(profiles_dir, config_path)
    for path in (Path(__file__), _ROOT / "training/source_game_mapping.py"):
        bindings[str(path.resolve())] = _sha256(path)
    return bindings


def _load_game_audit(root: Path, *, reference_root: Path, reference_sha256: str,
                     reference_rows: Mapping[str, dict], profiles_dir: Path,
                     config_path: Path) -> tuple[dict, list[dict]]:
    summary_path = root / "summary.json"
    marker = root / "_SUCCESS"
    if ((root / "_INCOMPLETE").exists() or not marker.is_file()
            or marker.read_text(encoding="ascii").strip() != _sha256(summary_path)):
        raise ValueError("game audit incomplete or summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("schema_version") != 1 or summary.get("status") != "complete"
            or summary.get("kind") != "game_expressibility_audit"):
        raise ValueError("unsupported game audit schema/status")
    if (Path(summary["timing_reference_root"]).resolve() != reference_root
            or summary["timing_reference_summary_sha256"] != reference_sha256):
        raise ValueError("game audit timing reference binding differs")
    if summary["implementation_and_config_sha256"] != game._implementation_bindings(profiles_dir, config_path):
        raise ValueError("game audit implementation/config binding differs from current files")
    # Dataclass humanization ranges are tuples; JSON persists them as lists.
    if _canonical(summary["playback_settings"]) != _canonical(asdict(game.load_playback_settings(config_path))):
        raise ValueError("game audit playback settings differ from bound config")
    styles = summary["styles"]
    if (not styles or len(styles) != len(set(styles))
            or not set(styles).issubset({"piano", "original", "preserve"})):
        raise ValueError("invalid game audit styles")
    expected = {(sample_id, style) for sample_id in reference_rows for style in styles}
    rows, found, paths = [], set(), set()
    for entry in summary["samples"]:
        key = (entry["id"], entry["style"])
        if key not in expected or key in found or entry["status"] not in _STATUSES:
            raise ValueError("game audit selection/status differs or has duplicates")
        found.add(key)
        reference = reference_rows[entry["id"]]
        for field in ("split", "performance_dataset", "piece_group", "score_midi_sha256"):
            if entry[field] != reference[field]:
                raise ValueError(f"game audit selected source binding differs: {field}")
        if entry["split"] not in {"train", "validation"}:
            raise ValueError("test mapping audit forbidden")
        _member_path(entry["score_midi_path"])
        path = _safe_path(root, entry["path"])
        if path in paths or _sha256(path) != entry["sha256"]:
            raise ValueError("game audit case duplicate path or hash mismatch")
        paths.add(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        if any(row[field] != entry[field] for field in _CASE_FIELDS):
            raise ValueError("game audit case metadata differs from summary")
        if row["status"] == "complete":
            if row["audit"]["options"] != asdict(AdaptOptions(style=row["style"])):
                raise ValueError("game audit importer options differ from supported options")
            projected = {(p["profile_id"], p["scenario_id"], p["baseline"]) for p in row["audit"]["projection"]}
            if not {(profile, "free_play", "mechanical") for profile in ("default", "delta_force_harmonica")} <= projected:
                raise ValueError("game audit lacks required free-play identity cases")
        elif not isinstance(row.get("error"), str) or not row["error"] or "audit" in row:
            raise ValueError("game audit rejected case lacks an error or has a projection")
        rows.append({**row, "game_case_path": entry["path"], "game_case_sha256": entry["sha256"]})
    if (found != expected or summary["sample_count"] != len(reference_rows)
            or summary["sample_style_count"] != len(rows)
            or summary["status_counts"] != dict(Counter(row["status"] for row in rows))):
        raise ValueError("game audit selection/count/denominator mismatch")
    expected_files = paths | {summary_path, marker}
    if {p.resolve() for p in root.rglob("*") if p.is_file()} != expected_files:
        raise ValueError("game audit contains unbound files")
    return summary, rows


def _check_mapping_against_game(mapping: dict, case: dict) -> None:
    """Do not silently recompute a different importer or profile projection."""
    summary, audit = mapping["summary"], case["audit"]
    if summary["options"] != audit["options"]:
        raise ValueError("mapping/game importer options differ")
    if set(summary["profiles"]) != {"default", "delta_force_harmonica"}:
        raise ValueError("mapping lacks required game profiles")
    for stage, field in (("source", "source_to_import"), ("import", "source_to_import"),
                         ("prepared", "source_to_transport")):
        count_key = "source_voice_count" if stage == "source" else "projected_voice_count"
        if summary["stages"][stage]["voice_count"] != audit[field][count_key]:
            raise ValueError(f"mapping/game voice counts differ: {stage}")
    for profile_id, values in summary["profiles"].items():
        projections = [row for row in audit["projection"] if row["profile_id"] == profile_id
                       and row["scenario_id"] == "free_play" and row["baseline"] == "mechanical"]
        if len(projections) != 1 or values["voice_count"] != projections[0]["logical_voice_count"]:
            raise ValueError("mapping/game profile voice counts differ")


def _aggregate(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["performance_dataset"], row["style"])].append(row)
    result = []
    for (split, source, style), values in sorted(groups.items()):
        stages, profiles = {}, {}
        complete = [row for row in values if row["status"] == "complete"]
        for row in complete:
            for name, metrics in row["mapping"]["stages"].items():
                destination = stages.setdefault(name, {"voice_count": 0, "strict_voice_count": 0,
                                                       "unknown_voice_count": 0, "reason_counts": {}})
                for field in ("voice_count", "strict_voice_count", "unknown_voice_count"):
                    destination[field] += metrics[field]
                for reason, count in metrics["reason_counts"].items():
                    destination["reason_counts"][reason] = destination["reason_counts"].get(reason, 0) + count
            for name, metrics in row["mapping"]["profiles"].items():
                destination = profiles.setdefault(name, {})
                for field, count in metrics.items():
                    if isinstance(count, int) and not isinstance(count, bool):
                        destination[field] = destination.get(field, 0) + count
        result.append({"split": split, "performance_dataset": source, "style": style,
            "sample_style_count": len(values), "status_counts": dict(Counter(row["status"] for row in values)),
            "completed_sample_style_count": len(complete), "stages": stages, "profiles": profiles})
    return {"scope": "diagnostic candidates among complete cases; all refused imports remain in denominators",
            "groups": result}


def audit_source_game_mapping(*, timing_reference: str | Path, game_audit: str | Path,
                              output_dir: str | Path, profiles_dir: str | Path = _ROOT / "profiles",
                              config_path: str | Path = _ROOT / "config.yaml") -> dict[str, Any]:
    reference_root, game_root, output = _root(timing_reference), _root(game_audit), Path(output_dir).resolve()
    profiles_dir, config_path = Path(profiles_dir).resolve(), Path(config_path).resolve()
    reference_sha256 = _sha256(reference_root / "summary.json")
    references = load_timing_reference(reference_root)
    reference_summary = json.loads((reference_root / "summary.json").read_text(encoding="utf-8"))
    if _sha256(reference_root / "summary.json") != reference_sha256:
        raise ValueError("timing reference changed during verification")
    reference_rows = {row["id"]: row for row in reference_summary["samples"]}
    game_summary, cases = _load_game_audit(game_root, reference_root=reference_root,
        reference_sha256=reference_sha256, reference_rows=reference_rows, profiles_dir=profiles_dir, config_path=config_path)
    game_sha256 = _sha256(game_root / "summary.json")
    bindings = _implementation_bindings(profiles_dir, config_path)
    dataset = Path(reference_summary["dataset_root"]).resolve()
    archive_path = Path(reference_summary["archive_path"]).resolve()
    protected_roots = [reference_root, game_root, dataset, archive_path.parent]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected_roots):
        raise ValueError("output must be isolated from bound inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty")
    source_rows = {}
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] not in references:
                continue
            if row["id"] in source_rows or row["split"] not in {"train", "validation"} or row.get("official_split") == "test":
                raise ValueError("duplicate source ID or test mapping audit forbidden")
            _member_path(row["score_midi_path"])
            source_rows[row["id"]] = row
    if set(source_rows) != set(references):
        raise ValueError("selected source IDs missing")
    for case in cases:
        row = source_rows[case["id"]]
        if any(case[key] != row[key] for key in _CASE_FIELDS if key not in {"style", "status"}):
            raise ValueError("game audit original source metadata differs")
    profiles = [profile for profile in load_profiles(str(profiles_dir))
                if profile.id in {"default", "delta_force_harmonica"}]
    if len(profiles) != 2 or {p.id for p in profiles} != {"default", "delta_force_harmonica"}:
        raise ValueError("mapping requires the bound default and Delta profiles")
    settings = game.load_playback_settings(config_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "_INCOMPLETE").write_text("source-to-game mapping audit in progress\n", encoding="ascii")
    rows, source_digests = [], {}
    with RefinedArchive(archive_path) as archive, tempfile.TemporaryDirectory(prefix="amp_mapping_") as temporary:
        original = Path(temporary) / "bound_score.mid"
        for position, case in enumerate(sorted(cases, key=lambda row: (row["id"], row["style"]))):
            score_bytes = archive.read(case["score_midi_path"])
            if hashlib.sha256(score_bytes).hexdigest() != case["score_midi_sha256"]:
                raise ValueError("original score ZIP member hash mismatch")
            source_digests[case["score_midi_path"]] = case["score_midi_sha256"]
            row = {key: case[key] for key in _CASE_FIELDS}
            row.update(game_case_path=case["game_case_path"], game_case_sha256=case["game_case_sha256"])
            stem = hashlib.sha256((case["id"] + ":" + case["style"]).encode()).hexdigest()
            if case["status"] == "complete":
                original.write_bytes(score_bytes)
                parsed = read_midi_source(original)
                mapping = build_sample_game_mapping(score_bytes=score_bytes, production_read_result=parsed,
                    timing_reference=references[case["id"]], options=AdaptOptions(style=case["style"]),
                    profiles=profiles, playback_settings=settings)
                _check_mapping_against_game(mapping, case)
                arrays_path = f"samples/{stem}.npz"
                (output / "samples").mkdir(exist_ok=True)
                row.update(mapping=mapping["summary"], arrays_path=arrays_path,
                           arrays_sha256=write_deterministic_npz(output / arrays_path, mapping["arrays"]))
            else:
                row["error"] = case["error"]
                if "transport_errors" in case:
                    row["transport_errors"] = case["transport_errors"]
            relative = f"samples/{stem}.json"
            (output / "samples").mkdir(exist_ok=True)
            (output / relative).write_bytes(_canonical(row) + b"\n")
            rows.append({**row, "path": relative, "sha256": _sha256(output / relative)})
            if (position + 1) % 25 == 0 or position + 1 == len(cases):
                print(f"source_game_mapping={position + 1}/{len(cases)}", flush=True)
    # Reopen the ZIP so a replacement during processing cannot pass using an
    # already-open file handle. References recheck selected source shards too.
    with RefinedArchive(archive_path) as archive:
        for member, digest in source_digests.items():
            if hashlib.sha256(archive.read(member)).hexdigest() != digest:
                raise ValueError("original score ZIP member changed during audit")
    load_timing_reference(reference_root)
    if _sha256(reference_root / "summary.json") != reference_sha256:
        raise ValueError("timing reference changed during audit")
    _load_game_audit(game_root, reference_root=reference_root, reference_sha256=reference_sha256,
                     reference_rows=reference_rows, profiles_dir=profiles_dir, config_path=config_path)
    if _sha256(game_root / "summary.json") != game_sha256:
        raise ValueError("game audit changed during mapping")
    if _implementation_bindings(profiles_dir, config_path) != bindings:
        raise ValueError("implementation/config changed during mapping audit")
    for row in rows:
        if _sha256(_safe_path(output, row["path"])) != row["sha256"]:
            raise ValueError("mapping case changed during audit")
        if row["status"] == "complete" and (
                _sha256(_safe_path(output, row["arrays_path"])) != row["arrays_sha256"]):
            raise ValueError("mapping arrays changed during audit")
    summary = {"schema_version": 1, "kind": "source_game_mapping_audit", "status": "complete",
        "timing_reference_root": str(reference_root), "timing_reference_summary_sha256": reference_sha256,
        "game_audit_root": str(game_root), "game_audit_summary_sha256": game_sha256,
        "implementation_and_config_sha256": bindings, "playback_settings": asdict(settings),
        "sample_count": len(references), "sample_style_count": len(rows), "styles": game_summary["styles"],
        "status_counts": dict(Counter(row["status"] for row in rows)), "aggregate": _aggregate(rows),
        "samples": [{key: row[key] for key in (*_CASE_FIELDS, "path", "sha256")} for row in rows],
        "policy": "free_play logical mechanical identity; candidate masks are diagnostic only; native arrays unchanged",
        "acceptance": {"native_masks_modified": False, "training_applied": False, "game_input_sent": False,
                       "production_approved": False}}
    (output / "summary.json").write_bytes(_canonical(summary) + b"\n")
    (output / "_SUCCESS").write_text(_sha256(output / "summary.json") + "\n", encoding="ascii")
    (output / "_INCOMPLETE").unlink()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-reference", required=True)
    parser.add_argument("--game-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles-dir", default=str(_ROOT / "profiles"))
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    args = parser.parse_args(argv)
    result = audit_source_game_mapping(timing_reference=args.timing_reference, game_audit=args.game_audit,
        output_dir=args.output_dir, profiles_dir=args.profiles_dir, config_path=args.config)
    print(json.dumps({"sample_count": result["sample_count"], "status_counts": result["status_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
