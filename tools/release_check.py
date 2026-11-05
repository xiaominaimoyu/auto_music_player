"""Check desktop version metadata before build and the signed EXE before release."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.database import ScoreDB  # noqa: E402
from core.update_checker import (  # noqa: E402
    RELEASE_PUBLIC_KEY, UpdateManifest, _check_expiry,
    check_download_compatibility, verify_manifest,
)


def runtime_version(root):
    values = {}
    source = Path(root) / "core" / "version.py"
    for node in ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source)).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {"APP_VERSION", "UPDATE_SEQUENCE"}:
                    values[name.id] = ast.literal_eval(node.value)
    version = values.get("APP_VERSION")
    sequence = values.get("UPDATE_SEQUENCE")
    if not isinstance(version, str) or type(sequence) is not int or sequence <= 0:
        raise ValueError("core/version.py 必须提供 APP_VERSION 和正整数 UPDATE_SEQUENCE")
    return version, sequence


def check_versions(root):
    root = Path(root)
    version, sequence = runtime_version(root)
    text_version = (root / "version.txt").read_text(encoding="utf-8-sig").strip()
    payload = json.loads((root / "updates" / "stable.json").read_text(encoding="utf-8"))
    manifest = UpdateManifest.from_dict(payload)
    if text_version != version or manifest.version != version or manifest.sequence != sequence:
        raise ValueError(
            f"发布版本不一致。version.txt={text_version}，core/version.py={version}/{sequence}，"
            f"updates/stable.json={manifest.version}/{manifest.sequence}。请同步三处版本后重试。"
        )
    if manifest.db_schema_version != ScoreDB.SCHEMA_VERSION:
        raise ValueError("updates/stable.json 的 db_schema_version 与当前数据库版本不一致")
    check_download_compatibility(manifest, version)
    return manifest


def check_artifact(root, artifact):
    manifest = check_versions(root)
    verify_manifest(manifest, RELEASE_PUBLIC_KEY)
    _check_expiry(manifest)
    path = Path(artifact)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != manifest.size or digest.hexdigest() != manifest.sha256:
        raise ValueError("EXE 大小或 SHA-256 与签名清单不一致，请用最终发布文件重新生成并签署清单")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", help="verify the final EXE against the signed manifest")
    args = parser.parse_args(argv)
    try:
        manifest = check_artifact(PROJECT_ROOT, args.artifact) if args.artifact else check_versions(PROJECT_ROOT)
    except (ValueError, OSError, SyntaxError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"verified desktop version={manifest.version} sequence={manifest.sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
