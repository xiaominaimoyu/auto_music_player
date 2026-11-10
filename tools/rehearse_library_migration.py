"""Exercise migration and recovery using temporary copies of an old scores.db."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.database import ScoreDB  # noqa: E402
from core.library_recovery import recover_library  # noqa: E402


def rehearse_migration(source):
    source = Path(source).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="amp-migration-") as temporary:
        root = Path(temporary)
        snapshot = root / "snapshot.db"
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
            original.execute("PRAGMA trusted_schema=OFF")
            with closing(sqlite3.connect(snapshot)) as copy:
                original.backup(copy)
        with ScoreDB(snapshot) as migrated:
            migrated_count = len(migrated.list_scores())
            backed_up = migrated.pre_migration_backup_path is not None
            if backed_up:
                with closing(sqlite3.connect(migrated.pre_migration_backup_path)) as backup:
                    if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("迁移备份完整性检查失败")
        with ScoreDB(root / "recovered.db") as recovered:
            result = recover_library(recovered, source)
            again = recover_library(recovered, source)
            if (len(recovered.list_scores()) != migrated_count or again.added != 0
                    or again.skipped != migrated_count):
                raise ValueError("迁移或重复恢复后的曲目数不一致")
            if recovered.conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("恢复曲库完整性检查失败")
        return {"schema_version": ScoreDB.SCHEMA_VERSION, "songs": migrated_count,
                "migration_backup_verified": backed_up, "recovered": result.added,
                "retry_skipped": again.skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="旧 data 目录内的 scores.db，保留同级 sources 目录")
    args = parser.parse_args(argv)
    try:
        result = rehearse_migration(args.source)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
