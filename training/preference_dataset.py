"""Strict read-only validation of offline listening bundles and human answer files.

Hashes bind an answer to researcher-held audio and trial definitions; they are
not signatures, identity proof or proof that the participant actually listened.
Synthetic bundles/answers always remain synthetic after validation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from training.preference_bundle import (
    BUNDLE_SCHEMA, CONSENT_PURPOSE, MAX_TRIALS, PRIVATE_SCHEMA, RESPONSE_SCHEMA,
    SCHEMA_VERSION, PreferenceError, file_hash, json_hash, read_json,
)


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_ID = re.compile(r"[0-9a-f]{32}\Z")


def _keys(value: Any, names: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise PreferenceError(f"{label}: missing or unexpected fields")
    return value


def _uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PreferenceError(f"{label}: expected a random UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise PreferenceError(f"{label}: invalid UUID") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise PreferenceError(f"{label}: expected a canonical lower-case UUID v4")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise PreferenceError(f"{label}: invalid SHA-256")
    return value


def _opaque_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value):
        raise PreferenceError(f"{label}: invalid opaque ID")
    return value


def _public_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise PreferenceError("public file path must use a safe relative POSIX path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or ":" in relative or not parts.parts:
        raise PreferenceError("unsafe public file path")
    path = root.joinpath(*parts.parts)
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise PreferenceError(f"missing or unsafe public file: {relative}")
    return path


def validate_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Validate the research copy, including every public asset and the private map."""
    root = Path(bundle_root).resolve()
    if (root / "_INCOMPLETE").exists() or not (root / "_SUCCESS").is_file():
        raise PreferenceError("bundle is not a completed research artifact")
    summary_path = root / "summary.json"
    if (root / "_SUCCESS").read_text(encoding="ascii").strip() != file_hash(summary_path):
        raise PreferenceError("bundle summary hash mismatch")
    summary = read_json(summary_path)
    private_path = root / "private_manifest.json"
    if summary.get("private_manifest_sha256") != file_hash(private_path):
        raise PreferenceError("private manifest hash mismatch")
    if summary.get("public_zip_sha256") != file_hash(root / "listening_bundle.zip"):
        raise PreferenceError("public ZIP hash mismatch")
    private = _keys(read_json(private_path), {"schema", "schema_version", "study_id", "study_version", "consent_version", "bundle_id", "manifest_sha256", "synthetic", "render_config", "public_files", "trials"}, "private manifest")
    if private["schema"] != PRIVATE_SCHEMA or type(private["schema_version"]) is not int or private["schema_version"] != SCHEMA_VERSION:
        raise PreferenceError("unsupported private manifest schema")
    public_root = root / "public"
    files = private["public_files"]
    if not isinstance(files, list) or not files:
        raise PreferenceError("missing public file inventory")
    inventory = {}
    for item in files:
        _keys(item, {"path", "sha256", "size_bytes"}, "public file inventory item")
        path = _public_file(public_root, item["path"])
        if item["path"] in inventory:
            raise PreferenceError("duplicate public file path")
        if type(item["size_bytes"]) is not int or path.stat().st_size != item["size_bytes"] or file_hash(path) != _hash(item["sha256"], "public file hash"):
            raise PreferenceError(f"public file hash/size mismatch: {item['path']}")
        inventory[item["path"]] = item
    actual_files = {path.relative_to(public_root).as_posix() for path in public_root.rglob("*") if path.is_file()}
    if actual_files != set(inventory):
        raise PreferenceError("unlisted or missing files in public bundle")
    manifest = _keys(read_json(public_root / "manifest.json"), {"schema", "schema_version", "study_id", "study_version", "consent_version", "bundle_id", "synthetic", "render_config_sha256", "trials", "manifest_sha256"}, "public manifest")
    if manifest["schema"] != BUNDLE_SCHEMA or type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise PreferenceError("unsupported public manifest schema")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if json_hash(body) != _hash(manifest["manifest_sha256"], "public manifest hash"):
        raise PreferenceError("public manifest content hash mismatch")
    for name in ("bundle_id", "study_id", "study_version", "consent_version", "synthetic", "manifest_sha256"):
        if manifest[name] != private[name]:
            raise PreferenceError(f"public/private {name} mismatch")
    _uuid(manifest["bundle_id"], "bundle_id")
    if type(manifest["synthetic"]) is not bool:
        raise PreferenceError("synthetic must be a boolean")
    if json_hash(private["render_config"]) != manifest["render_config_sha256"]:
        raise PreferenceError("render configuration hash mismatch")
    if not isinstance(manifest["trials"], list) or not 1 <= len(manifest["trials"]) <= MAX_TRIALS:
        raise PreferenceError("invalid trial count")
    if not isinstance(private["trials"], list):
        raise PreferenceError("private trials must be a list")
    private_trials = {}
    for item in private["trials"]:
        _keys(item, {"trial_id", "piece_group", "source_split", "excerpt_id", "candidates"}, "private trial")
        trial_id = _opaque_id(item["trial_id"], "private trial ID")
        if trial_id in private_trials or item["source_split"] not in {"train", "validation", "local_dev"}:
            raise PreferenceError("duplicate private trial or forbidden source split")
        if not isinstance(item["piece_group"], str) or not item["piece_group"] or not isinstance(item["excerpt_id"], str) or not item["excerpt_id"]:
            raise PreferenceError("private trial is missing its piece/excerpt group")
        if not isinstance(item["candidates"], list) or len(item["candidates"]) != 2:
            raise PreferenceError("private trial must map two candidates")
        for candidate in item["candidates"]:
            _keys(candidate, {"asset_id", "candidate_id", "role", "source_sha256"}, "private candidate")
            _hash(candidate["source_sha256"], "source WAV hash")
            if not isinstance(candidate["candidate_id"], str) or not candidate["candidate_id"]:
                raise PreferenceError("empty private candidate ID")
        if {candidate["role"] for candidate in item["candidates"]} != {"baseline", "candidate"}:
            raise PreferenceError("private trial must map one baseline and one candidate")
        private_trials[trial_id] = item
    combined = {}
    asset_ids = set()
    expected_paths = {"index.html", "manifest.json"}
    for trial in manifest["trials"]:
        _keys(trial, {"trial_id", "assets", "trial_sha256"}, "public trial")
        trial_id = _opaque_id(trial["trial_id"], "trial_id")
        if trial_id in combined or trial_id not in private_trials:
            raise PreferenceError("duplicate/unknown public trial")
        if json_hash({key: value for key, value in trial.items() if key != "trial_sha256"}) != _hash(trial["trial_sha256"], "trial hash"):
            raise PreferenceError("trial content hash mismatch")
        if not isinstance(trial["assets"], list) or len(trial["assets"]) != 2:
            raise PreferenceError("trial must contain exactly two WAV assets")
        mapped = {candidate["asset_id"]: candidate for candidate in private_trials[trial_id]["candidates"]}
        if len(mapped) != 2 or set(mapped) != {asset.get("asset_id") for asset in trial["assets"]}:
            raise PreferenceError("candidate/asset identity mismatch")
        for asset in trial["assets"]:
            _keys(asset, {"asset_id", "path", "sha256", "duration_s", "sample_rate", "channels", "sample_width"}, "public asset")
            asset_id = _opaque_id(asset["asset_id"], "asset_id")
            if asset_id in asset_ids or asset["path"] != f"assets/{asset_id}.wav":
                raise PreferenceError("duplicate asset or non-opaque asset path")
            asset_ids.add(asset_id)
            expected_paths.add(asset["path"])
            if asset["path"] not in inventory or asset["sha256"] != inventory[asset["path"]]["sha256"]:
                raise PreferenceError("asset hash differs from file inventory")
        combined[trial_id] = {"public": trial, "private": private_trials[trial_id], "candidates": mapped}
    if set(combined) != set(private_trials) or expected_paths != set(inventory):
        raise PreferenceError("unreferenced private trial or public file")
    return {"root": root, "manifest": manifest, "private": private, "trials": combined}


def load_response(path: str | Path, *, bundle_root: str | Path | None = None, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read one answer without changing it; reject unknown/modified/unsolicited fields."""
    if bundle is None:
        if bundle_root is None:
            raise PreferenceError("a verified research bundle is required")
        bundle = validate_bundle(bundle_root)
    path = Path(path)
    response = _keys(read_json(path), {"schema", "schema_version", "study_id", "study_version", "bundle_id", "manifest_sha256", "consent_version", "participant_id", "session_id", "response_id", "synthetic", "consent", "responses"}, "response")
    if response["schema"] != RESPONSE_SCHEMA or type(response["schema_version"]) is not int or response["schema_version"] != SCHEMA_VERSION:
        raise PreferenceError("unsupported response schema")
    manifest = bundle["manifest"]
    for name in ("study_id", "study_version", "bundle_id", "manifest_sha256", "consent_version"):
        if response[name] != manifest[name]:
            raise PreferenceError(f"response {name} does not match this research bundle")
    for name in ("participant_id", "session_id", "response_id"):
        _uuid(response[name], name)
    if type(response["synthetic"]) is not bool:
        raise PreferenceError("response synthetic must be a boolean")
    consent = _keys(response["consent"], {"accepted", "version", "purpose"}, "consent")
    if consent["accepted"] is not True or consent["version"] != manifest["consent_version"] or consent["purpose"] != CONSENT_PURPOSE:
        raise PreferenceError("explicit sharing/research consent is missing or mismatched")
    responses = response["responses"]
    if not isinstance(responses, list) or len(responses) != len(bundle["trials"]):
        raise PreferenceError("answer must contain every trial, retaining null for unanswered items")
    seen = set()
    rows = []
    for position, answer in enumerate(responses, 1):
        _keys(answer, {"trial_id", "trial_sha256", "position", "a_asset_id", "a_sha256", "b_asset_id", "b_sha256", "choice", "a_listened", "b_listened"}, "answer trial")
        trial_id = answer["trial_id"]
        if not isinstance(trial_id, str) or trial_id not in bundle["trials"] or trial_id in seen:
            raise PreferenceError("unknown or duplicate trial in answer")
        seen.add(trial_id)
        if type(answer["position"]) is not int or answer["position"] != position:
            raise PreferenceError("answer trial positions are not contiguous presentation order")
        trial = bundle["trials"][trial_id]
        if answer["trial_sha256"] != trial["public"]["trial_sha256"]:
            raise PreferenceError("answer trial hash mismatch")
        assets = {asset["asset_id"]: asset for asset in trial["public"]["assets"]}
        for side in ("a", "b"):
            asset_id = answer[f"{side}_asset_id"]
            if not isinstance(asset_id, str) or asset_id not in assets or answer[f"{side}_sha256"] != assets[asset_id]["sha256"]:
                raise PreferenceError("unknown/modified audio asset in answer")
            if type(answer[f"{side}_listened"]) is not bool:
                raise PreferenceError("playback completion markers must be boolean")
        if answer["a_asset_id"] == answer["b_asset_id"]:
            raise PreferenceError("A and B must reference different assets")
        choice = answer["choice"]
        if choice is not None and (not isinstance(choice, str) or choice not in {"a", "b", "tie", "skip"}):
            raise PreferenceError("invalid choice; use a/b/tie/skip/null")
        if choice in {"a", "b", "tie"} and not (answer["a_listened"] and answer["b_listened"]):
            raise PreferenceError("a preference requires both playback completion markers")
        winner = trial["candidates"][answer[f"{choice}_asset_id"]]["role"] if choice in {"a", "b"} else choice
        rows.append({**answer, "participant_id": response["participant_id"], "session_id": response["session_id"], "response_id": response["response_id"], "bundle_id": response["bundle_id"], "piece_group": trial["private"]["piece_group"], "source_split": trial["private"]["source_split"], "excerpt_id": trial["private"]["excerpt_id"], "candidate_at": "a" if trial["candidates"][answer["a_asset_id"]]["role"] == "candidate" else "b", "outcome": winner, "candidate_score": 1.0 if winner == "candidate" else 0.0 if winner == "baseline" else 0.5 if winner == "tie" else None, "candidate_ids": {candidate["role"]: candidate["candidate_id"] for candidate in trial["candidates"].values()}})
    return {"response_id": response["response_id"], "session_id": response["session_id"], "participant_id": response["participant_id"], "bundle_id": response["bundle_id"], "source_file": path.name, "source_sha256": file_hash(path), "synthetic": bool(manifest["synthetic"] or response["synthetic"]), "rows": rows, "identity_and_listening_are_self_reported": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--response", help="optional answer JSON; omit to verify only bundle files/hashes")
    args = parser.parse_args(argv)
    try:
        bundle = validate_bundle(args.bundle_root)
        result = {"status": "valid", "bundle_id": bundle["manifest"]["bundle_id"], "trial_count": len(bundle["trials"])}
        if args.response:
            response = load_response(args.response, bundle=bundle)
            result.update(response_id=response["response_id"], synthetic=response["synthetic"], row_count=len(response["rows"]))
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, PreferenceError) as exc:
        print(f"preference validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
