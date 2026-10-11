"""Build a completely offline, blinded listening bundle from paired PCM WAVs.

The public ZIP never contains candidate identities, source splits or model paths.
The research mapping stays in ``private_manifest.json`` outside that ZIP.  A
bundle is an evaluation instrument, not evidence of a human preference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import uuid
import wave
import zipfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
BUNDLE_SCHEMA = "auto_music_player.preference_bundle"
PRIVATE_SCHEMA = "auto_music_player.preference_private"
RESPONSE_SCHEMA = "auto_music_player.preference_responses"
CONSENT_PURPOSE = "research_preference_evaluation"
MAX_TRIALS = 200
MAX_WAV_BYTES = 64 * 1024 * 1024


class PreferenceError(ValueError):
    """An invalid bundle, response or output request."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> Any:
    if path.stat().st_size > max_bytes:
        raise PreferenceError(f"JSON file exceeds {max_bytes} bytes: {path.name}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PreferenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise PreferenceError(f"non-finite JSON number: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreferenceError(f"invalid UTF-8 JSON: {path.name}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise PreferenceError(f"{label} must be a non-empty string of at most 256 characters")
    return value


def _wave_info(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise PreferenceError(f"candidate must be a WAV file: {path}")
    if path.stat().st_size > MAX_WAV_BYTES:
        raise PreferenceError(f"WAV exceeds {MAX_WAV_BYTES} bytes: {path.name}")
    try:
        with wave.open(str(path), "rb") as audio:
            channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            if audio.getcomptype() != "NONE" or channels not in (1, 2) or width != 2:
                raise PreferenceError("listening WAV must use browser-compatible 16-bit mono/stereo PCM")
            if not 8000 <= rate <= 192000 or not 0 < frames / rate <= 300:
                raise PreferenceError("WAV sample rate or duration is outside supported bounds (maximum 300 seconds)")
            if len(audio.readframes(frames)) != frames * channels * width:
                raise PreferenceError(f"truncated PCM payload: {path.name}")
    except (wave.Error, EOFError) as exc:
        raise PreferenceError(f"invalid PCM WAV: {path.name}: {exc}") from exc
    return {"duration_s": frames / rate, "sample_rate": rate, "channels": channels, "sample_width": width}


def _copy_anonymous_pcm(source: Path, destination: Path) -> None:
    """Keep PCM samples, stripping optional RIFF metadata that could unblind A/B."""
    with wave.open(str(source), "rb") as original:
        with wave.open(str(destination), "wb") as anonymous:
            anonymous.setnchannels(original.getnchannels())
            anonymous.setsampwidth(original.getsampwidth())
            anonymous.setframerate(original.getframerate())
            anonymous.writeframes(original.readframes(original.getnframes()))


def build_bundle(*, spec_path: str | Path, output: str | Path, seed: int = 42, synthetic: bool = False) -> dict[str, Any]:
    """Create a new research directory and a separate shareable offline ZIP.

    ``spec.trials`` entries contain piece_group, source_split, excerpt_id and
    baseline/candidate objects with candidate_id and wav_path.  Paths are resolved
    relative to the specification, never to the process working directory.
    """
    spec_path = Path(spec_path).resolve()
    destination = Path(output).resolve()
    spec = read_json(spec_path)
    if not isinstance(spec, dict):
        raise PreferenceError("spec must be a JSON object")
    identifiers = {name: _text(spec.get(name), name) for name in ("study_id", "study_version", "consent_version")}
    render_config = spec.get("render_config")
    if not isinstance(render_config, dict) or not render_config:
        raise PreferenceError("render_config must describe the shared renderer/instrument/settings")
    render_hash = json_hash(render_config)
    trials = spec.get("trials")
    if not isinstance(trials, list) or not 1 <= len(trials) <= MAX_TRIALS:
        raise PreferenceError(f"spec must contain 1..{MAX_TRIALS} trials")
    checked = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise PreferenceError("each trial must be an object")
        piece_group = _text(trial.get("piece_group"), "piece_group")
        excerpt_id = _text(trial.get("excerpt_id"), "excerpt_id")
        source_split = trial.get("source_split")
        if source_split not in {"train", "validation", "local_dev"}:
            raise PreferenceError("preference collection accepts train/validation/local_dev only, never locked test")
        candidates = []
        for role in ("baseline", "candidate"):
            candidate = trial.get(role)
            if not isinstance(candidate, dict):
                raise PreferenceError(f"trial {index}: missing {role}")
            candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
            source = Path(_text(candidate.get("wav_path"), "wav_path"))
            source = (spec_path.parent / source).resolve() if not source.is_absolute() else source.resolve()
            if destination == source or destination in source.parents:
                raise PreferenceError("input WAV must be outside the output directory")
            candidates.append({"role": role, "candidate_id": candidate_id, "source": source, "info": _wave_info(source), "sha256": file_hash(source)})
        if candidates[0]["candidate_id"] == candidates[1]["candidate_id"]:
            raise PreferenceError("baseline and candidate must have distinct candidate_id values")
        for field in ("sample_rate", "channels", "sample_width"):
            if candidates[0]["info"][field] != candidates[1]["info"][field]:
                raise PreferenceError(f"paired WAV {field} differs; render both with one configuration")
        checked.append({"piece_group": piece_group, "source_split": source_split, "excerpt_id": excerpt_id, "candidates": candidates})
    for role in ("baseline", "candidate"):
        identities = {candidate["candidate_id"] for trial in checked for candidate in trial["candidates"] if candidate["role"] == role}
        if len(identities) != 1:
            raise PreferenceError("one bundle must compare one fixed baseline/candidate pair; use separate bundles for different models")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise PreferenceError(f"output must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "_INCOMPLETE").write_text("offline listening bundle build in progress\n", encoding="utf-8")
    public = destination / "public"
    (public / "assets").mkdir(parents=True)
    (destination / "responses").mkdir()
    rng = random.Random(seed)
    rng.shuffle(checked)
    public_trials, private_trials = [], []
    for trial in checked:
        trial_id = uuid.uuid4().hex
        rng.shuffle(trial["candidates"])
        assets, mappings = [], []
        for candidate in trial["candidates"]:
            asset_id = uuid.uuid4().hex
            relative = f"assets/{asset_id}.wav"
            _copy_anonymous_pcm(candidate["source"], public / relative)
            if file_hash(candidate["source"]) != candidate["sha256"]:
                raise PreferenceError("WAV changed during bundle creation")
            assets.append({"asset_id": asset_id, "path": relative, "sha256": file_hash(public / relative), **candidate["info"]})
            mappings.append({"asset_id": asset_id, "candidate_id": candidate["candidate_id"], "role": candidate["role"], "source_sha256": candidate["sha256"]})
        public_trial = {"trial_id": trial_id, "assets": assets}
        public_trial["trial_sha256"] = json_hash(public_trial)
        public_trials.append(public_trial)
        private_trials.append({"trial_id": trial_id, "piece_group": trial["piece_group"], "source_split": trial["source_split"], "excerpt_id": trial["excerpt_id"], "candidates": mappings})
    manifest = {"schema": BUNDLE_SCHEMA, "schema_version": SCHEMA_VERSION, **identifiers, "bundle_id": str(uuid.uuid4()), "synthetic": bool(synthetic), "render_config_sha256": render_hash, "trials": public_trials}
    manifest["manifest_sha256"] = json_hash(manifest)
    write_json(public / "manifest.json", manifest)
    template = Path(__file__).with_name("templates") / "preference.html"
    embedded = canonical_json(manifest).decode("utf-8").replace("<", "\\u003c").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    public.joinpath("index.html").write_text(template.read_text(encoding="utf-8").replace("__BUNDLE_MANIFEST__", embedded), encoding="utf-8")
    public_files = [{"path": path.relative_to(public).as_posix(), "sha256": file_hash(path), "size_bytes": path.stat().st_size} for path in sorted(public.rglob("*")) if path.is_file()]
    private = {"schema": PRIVATE_SCHEMA, "schema_version": SCHEMA_VERSION, **identifiers, "bundle_id": manifest["bundle_id"], "manifest_sha256": manifest["manifest_sha256"], "synthetic": bool(synthetic), "render_config": render_config, "public_files": public_files, "trials": private_trials}
    write_json(destination / "private_manifest.json", private)
    zip_path = destination / "listening_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in public_files:
            archive.write(public / item["path"], arcname=item["path"])
    summary = {"schema_version": 1, "status": "complete", "bundle_id": manifest["bundle_id"], "trial_count": len(public_trials), "synthetic": bool(synthetic), "human_response_count": 0, "training_ready": False, "public_zip": str(zip_path), "public_zip_sha256": file_hash(zip_path), "private_manifest_sha256": file_hash(destination / "private_manifest.json"), "manifest_sha256": manifest["manifest_sha256"], "renderer_configuration_is_declared_not_independently_verified": True}
    write_json(destination / "summary.json", summary)
    (destination / "_INCOMPLETE").unlink()
    (destination / "_SUCCESS").write_text(file_hash(destination / "summary.json") + "\n", encoding="ascii")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="private paired-WAV specification JSON")
    parser.add_argument("--output", required=True, help="new research directory; share only listening_bundle.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="mark all answers from this test bundle as non-human/test-only")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_bundle(spec_path=args.spec, output=args.output, seed=args.seed, synthetic=args.synthetic), ensure_ascii=False))
    except (OSError, PreferenceError) as exc:
        print(f"bundle build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
