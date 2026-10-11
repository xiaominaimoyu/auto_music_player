"""Validate local opt-in performance captures and emit a safe manifest.

The source ``performance_captures`` tree is never modified.  Completed JSONL
files are validated by :mod:`training.capture_dataset`; invalid samples are
listed separately and never enter the accepted manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training.capture_dataset import (
    CaptureValidationError,
    iter_completed_captures,
    load_capture,
)


PIPELINE_NAME = "auto_music_player.local_capture_manifest"
PIPELINE_VERSION = 1


class CaptureManifestError(RuntimeError):
    """Raised when the manifest destination or invocation is unsafe."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for value in values:
            handle.write(_canonical_json(value) + b"\n")


def build_capture_manifest(
    *,
    capture_root: str | Path,
    output_path: str | Path,
    require_training_eligible: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate all direct ``completed/*.jsonl`` files without changing them."""

    root = Path(capture_root).resolve()
    output = Path(output_path).resolve()
    if not root.is_dir():
        raise CaptureManifestError(f"capture root does not exist: {root}")
    if output.exists() and any(output.iterdir()):
        raise CaptureManifestError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    incomplete = output / "_INCOMPLETE"
    incomplete.write_text("capture manifest build in progress\n", encoding="utf-8")

    paths = list(iter_completed_captures(root))
    inventory = [
        {
            "capture_file": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    input_inventory_sha256 = hashlib.sha256(_canonical_json(inventory)).hexdigest()

    accepted: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    piece_counts: Counter[str] = Counter()
    note_total = 0
    duration_total = 0.0
    for path, source_record in zip(paths, inventory):
        try:
            sample = load_capture(
                path,
                require_training_eligible=require_training_eligible,
            )
        except CaptureValidationError as exc:
            rejects.append(
                {
                    "capture_file": source_record["capture_file"],
                    "capture_sha256": source_record["sha256"],
                    "reason_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
            if strict:
                raise
            continue

        duration_s = max(
            (note.onset_s + note.duration_s for note in sample.notes),
            default=0.0,
        )
        note_count = len(sample.notes)
        piece_counts[sample.piece_group] += 1
        note_total += note_count
        duration_total += duration_s
        accepted.append(
            {
                "capture_file": source_record["capture_file"],
                "capture_sha256": source_record["sha256"],
                "capture_size_bytes": source_record["size_bytes"],
                "session_id": sample.session_id,
                "piece_group": sample.piece_group,
                "mapping_hash": sample.mapping_hash,
                "bpm": sample.bpm,
                "quantize": sample.quantize,
                "note_count": note_count,
                "duration_s": duration_s,
                "training_eligible": sample.training_eligible,
                "quality": dict(sample.quality),
            }
        )

    accepted.sort(key=lambda row: (row["piece_group"], row["session_id"], row["capture_file"]))
    rejects.sort(key=lambda row: row["capture_file"])
    source_files = {
        "training/capture_dataset.py": Path(__file__).with_name("capture_dataset.py"),
        "training/local_capture_manifest.py": Path(__file__),
    }
    config = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "require_training_eligible": require_training_eligible,
        "strict": strict,
        "capture_file_count_scanned": len(paths),
        "input_inventory_sha256": input_inventory_sha256,
        "source_sha256": {
            name: _sha256_file(path) for name, path in sorted(source_files.items())
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
    }
    config["config_sha256"] = hashlib.sha256(_canonical_json(config)).hexdigest()
    summary = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "config_sha256": config["config_sha256"],
        "scanned_count": len(paths),
        "accepted_count": len(accepted),
        "rejected_count": len(rejects),
        "piece_group_count": len(piece_counts),
        "piece_repeat_count": sum(count > 1 for count in piece_counts.values()),
        "note_count": note_total,
        "duration_s": duration_total,
        "training_ready": bool(accepted),
    }

    _write_json(output / "build_config.json", config)
    _write_jsonl(output / "manifest.jsonl", accepted)
    _write_jsonl(output / "rejects.jsonl", rejects)
    _write_json(output / "summary.json", summary)
    no_samples_marker = output / "_NO_ACCEPTED_SAMPLES"
    if not accepted:
        no_samples_marker.write_text(
            "manifest audit completed, but no accepted training samples were found\n",
            encoding="utf-8",
        )
    artifact_paths = [
        output / "build_config.json",
        output / "manifest.jsonl",
        output / "rejects.jsonl",
        output / "summary.json",
        *([no_samples_marker] if no_samples_marker.is_file() else []),
    ]
    integrity = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    _write_json(output / "integrity.json", integrity)
    incomplete.unlink(missing_ok=True)
    (output / "_SUCCESS").write_text(
        _sha256_file(output / "integrity.json") + "\n",
        encoding="ascii",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate opt-in local performance captures and build a safe manifest."
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-ineligible",
        action="store_true",
        help="include integrity-valid captures that quality gates marked ineligible",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = build_capture_manifest(
            capture_root=args.capture_root,
            output_path=args.output,
            require_training_eligible=not args.allow_ineligible,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"capture manifest build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CaptureManifestError",
    "PIPELINE_NAME",
    "PIPELINE_VERSION",
    "build_capture_manifest",
]
