"""Import a read-only snapshot of an earlier SQLite library into the active library."""

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory

from core.score_model import require_valid


SCORE_COLUMNS = "name, source_file, source_type, raw_text, notes_json, bpm_default, created_at"
SOURCE_COLUMNS = "schema_version, source_kind, original_path, stored_path, sha256, metadata_json, created_at"
PREF_COLUMNS = "profile_id, settings_json, updated_at"


@dataclass
class RecoveryResult:
    added: int = 0
    skipped: int = 0
    backup: str | None = None


def discover_legacy_libraries(target_path, data_dirs, *, db_file="scores.db"):
    """Probe only known data roots, before creating the destination database."""
    target = Path(target_path).resolve()
    if target.exists():
        return []
    found = []
    for root in data_dirs:
        candidate = (Path(root) / db_file).resolve()
        if candidate != target and candidate.is_file() and candidate not in found:
            found.append(candidate)
    return found


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_attachment(source_root, db, source_row, staging, created):
    version, kind, original, stored, digest, metadata, timestamp = source_row
    if not stored:
        return source_row
    archived = (source_root / stored.replace("\\", "/")).resolve()
    if not archived.is_relative_to(source_root / "sources"):
        raise ValueError("旧曲库内的附件路径超出了 sources 目录")
    if not archived.is_file():
        raise ValueError(f"缺少来源附件 {stored}。请将旧 sources 目录与旧数据库放在一起后重试。")
    if archived.stat().st_size > db.MAX_SOURCE_COPY_BYTES:
        raise ValueError(f"来源附件超过 500 MB：{stored}")
    target_dir = Path(db.sources_dir)
    if target_dir.resolve() != Path(db.data_dir).resolve() / "sources":
        raise ValueError("当前曲库的 sources 目录指向了外部位置")
    target_dir.mkdir(exist_ok=True)
    staged = staging / "attachment"
    shutil.copyfile(archived, staged)
    actual = _digest(staged)
    if digest and actual != digest:
        raise ValueError(f"旧来源附件的 SHA-256 与记录不一致：{stored}")
    target = target_dir / (actual + archived.suffix.lower())
    if target.is_symlink():
        raise ValueError("当前曲库的附件指向了外部位置")
    if target.exists():
        if _digest(target) != actual:
            raise ValueError(f"当前曲库存在不同内容的同名附件：{target.name}")
    else:
        os.replace(staged, target)
        created.append(target)
    return version, kind, original, os.path.relpath(target, db.data_dir), actual, metadata, timestamp


def _related(conn, score_id, tables):
    source = None
    preferences = []
    if "score_sources" in tables:
        source = conn.execute(
            f"SELECT {SOURCE_COLUMNS} FROM score_sources WHERE score_id=?", (score_id,)
        ).fetchone()
    if "score_preferences" in tables:
        preferences = conn.execute(
            f"SELECT {PREF_COLUMNS} FROM score_preferences WHERE score_id=? ORDER BY profile_id",
            (score_id,),
        ).fetchall()
    return tuple(source) if source else None, [tuple(row) for row in preferences]


def recover_library(db, old_path):
    """Preserve existing songs and both sets of preferences even when old IDs collide."""
    path = Path(old_path).resolve(strict=True)
    if path.samefile(db.db_path):
        raise ValueError("所选文件就是当前使用的曲库")
    result = RecoveryResult()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as old:
        old.execute("PRAGMA trusted_schema=OFF")
        old.execute("BEGIN")
        tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "scores" not in tables:
            raise ValueError("所选文件没有 AutoMusicPlayer 的 scores 表")
        if old.execute("PRAGMA user_version").fetchone()[0] > db.SCHEMA_VERSION:
            raise ValueError("旧曲库由更高版本的程序创建，请先更新程序")
        songs = old.execute(f"SELECT id, {SCORE_COLUMNS} FROM scores ORDER BY id")
        if not old.execute("SELECT 1 FROM scores LIMIT 1").fetchone():
            return result
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result.backup = db.backup_to(f"{db.db_path}.pre-recovery.{stamp}.bak")
        current = db.conn
        with TemporaryDirectory(dir=db.data_dir) as temp:
            created = []
            current.execute("BEGIN IMMEDIATE")
            try:
                current.execute("""
                    CREATE TABLE IF NOT EXISTS recovered_scores (
                        fingerprint TEXT PRIMARY KEY,
                        score_id INTEGER NOT NULL REFERENCES scores(id) ON DELETE CASCADE
                    )
                """)
                for old_id, *values in songs:
                    source, prefs = _related(old, old_id, tables)
                    payload = json.dumps([old_id, values, source, prefs], ensure_ascii=False)
                    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                    if current.execute("SELECT 1 FROM recovered_scores WHERE fingerprint=?", (key,)).fetchone():
                        result.skipped += 1
                        continue
                    require_valid(json.loads(values[4]), bpm=values[5])
                    if source:
                        source = _restore_attachment(path.parent, db, source, Path(temp), created)
                    existing = current.execute(f"SELECT {SCORE_COLUMNS} FROM scores WHERE id=?", (old_id,)).fetchone()
                    if existing and list(existing) == values and _related(
                        current, old_id, {"score_sources", "score_preferences"}
                    ) == (source, prefs):
                        new_id = old_id
                        result.skipped += 1
                    else:
                        new_id = current.execute(
                            f"INSERT INTO scores ({SCORE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)", values
                        ).lastrowid
                        if source:
                            current.execute(
                                f"INSERT INTO score_sources (score_id, {SOURCE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (new_id, *source),
                            )
                        current.executemany(
                            f"INSERT INTO score_preferences (score_id, {PREF_COLUMNS}) VALUES (?, ?, ?, ?)",
                            [(new_id, *item) for item in prefs],
                        )
                        result.added += 1
                    current.execute("INSERT INTO recovered_scores VALUES (?, ?)", (key, new_id))
                current.commit()
            except BaseException:
                try:
                    for attachment in created:
                        attachment.unlink(missing_ok=True)
                finally:
                    current.rollback()
                raise
    return result
