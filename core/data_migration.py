"""Store 首次启动的数据迁移。

迁移只复制，不删除来源文件；冲突文件保留为 ``*.legacy-*``，避免把用户
已有乐谱、原始 MIDI/图片或配置静默覆盖。成功后写入版本化 marker，使迁移
幂等且可审计。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.app_identity import stable_app_root


MIGRATION_ID = "store-v1"


@dataclass(frozen=True)
class MigrationResult:
    status: str
    source: str | None
    copied_files: int
    skipped_files: int
    conflict_files: int
    marker_path: str
    backup_path: str | None = None
    error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path):
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            yield path


def _copy_file_preserving(source: Path, destination: Path, stamp: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
        return "copied"
    try:
        if source.stat().st_size == destination.stat().st_size and _sha256(source) == _sha256(destination):
            return "skipped"
    except OSError:
        pass
    conflict = destination.with_name(f"{destination.name}.legacy-{stamp}")
    index = 1
    while conflict.exists():
        conflict = destination.with_name(f"{destination.name}.legacy-{stamp}-{index}")
        index += 1
    temporary = conflict.with_name(conflict.name + f".tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, conflict)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return "conflict"


def _write_marker(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _default_roots() -> tuple[list[Path], list[Path]]:
    stable = Path(stable_app_root())
    data_dirs = [stable / "data"]
    profile_dirs = [stable / "profiles"]
    raw_data = str(os.environ.get("AUTOMUSIC_LEGACY_DATA_DIRS") or "").strip()
    raw_profiles = str(os.environ.get("AUTOMUSIC_LEGACY_PROFILE_DIRS") or "").strip()
    if raw_data:
        data_dirs.extend(Path(item) for item in raw_data.split(os.pathsep) if item)
    if raw_profiles:
        profile_dirs.extend(Path(item) for item in raw_profiles.split(os.pathsep) if item)
    return data_dirs, profile_dirs


def migrate_store_data(
    target_data_dir: str,
    *,
    candidate_data_dirs: list[str | os.PathLike[str]] | None = None,
    candidate_profile_dirs: list[str | os.PathLike[str]] | None = None,
) -> MigrationResult:
    """将旧安装的 data/profiles 合并到 Store LocalState。

    ``target_data_dir`` 通常是 ``<LocalState>\\data``。默认只探测当前安装版
    已约定的 ``%LOCALAPPDATA%\\AutoMusicPlayer\\data``，未知便携目录可通过
    ``AUTOMUSIC_LEGACY_DATA_DIRS`` 或显式参数提供，不会扫描整个磁盘。
    """

    target = Path(target_data_dir).expanduser().resolve()
    target_root = target.parent
    marker = target_root / "migration" / f"{MIGRATION_ID}.json"
    if marker.is_file():
        return MigrationResult(
            "already-complete", None, 0, 0, 0, str(marker)
        )

    default_data, default_profiles = _default_roots()
    data_sources = default_data if candidate_data_dirs is None else candidate_data_dirs
    profile_sources = (
        default_profiles
        if candidate_profile_dirs is None
        else candidate_profile_dirs
    )
    data_candidates = [Path(item).expanduser().resolve() for item in data_sources]
    profile_candidates = [Path(item).expanduser().resolve() for item in profile_sources]
    data_candidates = list(dict.fromkeys(data_candidates))
    profile_candidates = list(dict.fromkeys(profile_candidates))

    source_data = next(
        (path for path in data_candidates if path.is_dir() and any(_iter_files(path))),
        None,
    )
    source_profiles = next(
        (path for path in profile_candidates if path.is_dir() and any(_iter_files(path))),
        None,
    )
    if source_data is None and source_profiles is None:
        return MigrationResult("no-source", None, 0, 0, 0, str(marker))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = None
    copied = skipped = conflicts = 0
    try:
        if any(_iter_files(target)):
            backup = target_root / "migration" / "backups" / f"pre-{MIGRATION_ID}-{stamp}"
            shutil.copytree(target, backup, dirs_exist_ok=True)
            backup_path = str(backup)

        jobs: list[tuple[Path, Path]] = []
        if source_data is not None:
            jobs.append((source_data, target))
        if source_profiles is not None:
            jobs.append((source_profiles, target / "profiles"))
        for source, destination_root in jobs:
            for source_file in _iter_files(source):
                relative = source_file.relative_to(source)
                result = _copy_file_preserving(
                    source_file, destination_root / relative, stamp
                )
                if result == "copied":
                    copied += 1
                elif result == "skipped":
                    skipped += 1
                else:
                    conflicts += 1

        payload = {
            "migration_id": MIGRATION_ID,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "sources": [str(path) for path in (source_data, source_profiles) if path],
            "target_data_dir": str(target),
            "copied_files": copied,
            "skipped_files": skipped,
            "conflict_files": conflicts,
            "backup_path": backup_path,
        }
        _write_marker(marker, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        return MigrationResult(
            "error",
            str(source_data or source_profiles),
            copied,
            skipped,
            conflicts,
            str(marker),
            backup_path,
            str(exc),
        )
    return MigrationResult(
        "migrated",
        str(source_data or source_profiles),
        copied,
        skipped,
        conflicts,
        str(marker),
        backup_path,
    )


__all__ = ["MIGRATION_ID", "MigrationResult", "migrate_store_data"]
