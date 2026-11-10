#!/usr/bin/env python3
"""Generate and verify the signed update manifest used by phase-one updates.

The private Ed25519 key must live outside the repository.  The client embeds
only the raw public key (base64) and verifies canonical UTF-8 JSON locally.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.update_checker import (  # noqa: E402
    RELEASE_PUBLIC_KEY,
    UpdateManifest,
    _canonical_json,
    verify_manifest,
)
from core.database import ScoreDB  # noqa: E402
from tools.release_check import runtime_version  # noqa: E402


def _cryptography():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise SystemExit("cryptography is required: python -m pip install cryptography") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def generate_key(args) -> int:
    serialization, Ed25519PrivateKey, _ = _cryptography()
    private_path = Path(args.private_key).expanduser().resolve()
    if private_path.exists() and not args.force:
        raise SystemExit(f"private key already exists: {private_path}")
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    _write_atomic(private_path, private_bytes)
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        pass
    public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    public_b64 = base64.b64encode(public_raw).decode("ascii")
    if args.public_key_file:
        _write_atomic(
            Path(args.public_key_file).expanduser().resolve(),
            (public_b64 + "\n").encode("ascii"),
        )
    print(public_b64)
    return 0


def _load_private_key(path: str):
    serialization, Ed25519PrivateKey, _ = _cryptography()
    key = serialization.load_pem_private_key(
        Path(path).expanduser().resolve().read_bytes(),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit("private key is not Ed25519")
    return key


def sign_manifest(args) -> int:
    manifest_path = Path(args.manifest).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("manifest must be a JSON object")
    value.pop("signature", None)
    # Reuse the client's structural validation before signing.
    UpdateManifest.from_dict({**value, "signature": "unsigned"})
    private_key = _load_private_key(args.private_key)
    value["signature"] = base64.b64encode(
        private_key.sign(_canonical_json(value))
    ).decode("ascii")
    verify_manifest(UpdateManifest.from_dict(value), RELEASE_PUBLIC_KEY)
    output = Path(args.output).resolve() if args.output else manifest_path
    _write_atomic(
        output,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(output)
    return 0


def prepare_manifest(args) -> int:
    """Use the final artifact and runtime metadata to avoid hand-copied release hashes."""
    version, sequence = runtime_version(PROJECT_ROOT)
    if (PROJECT_ROOT / "version.txt").read_text(encoding="utf-8-sig").strip() != version:
        raise ValueError("请先同步 version.txt 与 core/version.py")
    artifact = Path(args.artifact).resolve(strict=True)
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("--artifact 必须是 Windows EXE")
        stream.seek(0)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    value = {
        "schema_version": 1,
        "version": version,
        "sequence": sequence,
        "min_supported": args.min_supported,
        "urls": args.url,
        "size": artifact.stat().st_size,
        "sha256": digest.hexdigest(),
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(days=365)).isoformat().replace("+00:00", "Z"),
        "db_schema_version": ScoreDB.SCHEMA_VERSION,
        "release_notes": Path(args.notes).read_text(encoding="utf-8") if args.notes else f"v{version}",
    }
    UpdateManifest.from_dict({**value, "signature": "unsigned"})
    private_key = _load_private_key(args.private_key)
    value["signature"] = base64.b64encode(private_key.sign(_canonical_json(value))).decode("ascii")
    verify_manifest(UpdateManifest.from_dict(value), RELEASE_PUBLIC_KEY)
    output = Path(args.output).resolve()
    _write_atomic(output, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(output)
    return 0


def _read_public_key(args) -> str:
    if args.public_key_file:
        return Path(args.public_key_file).expanduser().resolve().read_text(
            encoding="ascii"
        ).strip()
    return str(args.public_key or "").strip()


def verify(args) -> int:
    path = Path(args.manifest).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    manifest = UpdateManifest.from_dict(value)
    verify_manifest(manifest, _read_public_key(args))
    print(
        f"verified version={manifest.version} sequence={manifest.sequence} "
        f"sha256={manifest.sha256}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    key = commands.add_parser("generate-key", help="create an offline Ed25519 key")
    key.add_argument("--private-key", required=True)
    key.add_argument("--public-key-file")
    key.add_argument("--force", action="store_true")
    key.set_defaults(func=generate_key)

    sign = commands.add_parser("sign", help="sign a manifest JSON file")
    sign.add_argument("manifest")
    sign.add_argument("--private-key", required=True)
    sign.add_argument("--output")
    sign.set_defaults(func=sign_manifest)

    prepare = commands.add_parser("prepare", help="generate and sign a manifest from the final desktop EXE")
    prepare.add_argument("--artifact", required=True)
    prepare.add_argument("--private-key", required=True)
    prepare.add_argument("--url", action="append", required=True)
    prepare.add_argument("--min-supported", default="1.4.2")
    prepare.add_argument("--notes", help="release notes file")
    prepare.add_argument("--output", default=str(PROJECT_ROOT / "updates" / "stable.json"))
    prepare.set_defaults(func=prepare_manifest)

    check = commands.add_parser("verify", help="verify a signed manifest")
    check.add_argument("manifest")
    group = check.add_mutually_exclusive_group(required=True)
    group.add_argument("--public-key")
    group.add_argument("--public-key-file")
    check.set_defaults(func=verify)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
