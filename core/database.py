"""乐谱数据库(SQLite)。"""

import json
import os
import sqlite3
from datetime import datetime

from core.score_model import require_valid


class ScoreDB:
    """乐谱库:一份乐谱记录一次,重复演奏无需重新上传。"""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_file TEXT,
                source_type TEXT,
                raw_text TEXT,
                notes_json TEXT NOT NULL,
                bpm_default INTEGER NOT NULL DEFAULT 100,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS score_preferences (
                score_id INTEGER NOT NULL,
                profile_id TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (score_id, profile_id),
                FOREIGN KEY (score_id) REFERENCES scores(id) ON DELETE CASCADE
            )
            """
        )
        self.conn.commit()

    def add_score(self, name, notes, raw_text="", source_file="", source_type="", bpm_default=100):
        require_valid(notes, bpm=bpm_default)
        cur = self.conn.execute(
            "INSERT INTO scores (name, source_file, source_type, raw_text, notes_json, bpm_default, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                source_file,
                source_type,
                raw_text,
                json.dumps(notes, ensure_ascii=False),
                bpm_default,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_scores(self, records):
        """Atomically add multiple already-recognized scores.

        Every record is validated before the transaction starts.  A malformed
        song therefore cannot leave half of an imported JSON library behind.
        """
        prepared = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"批量导入第 {index + 1} 项必须是对象")
            notes = record.get("notes")
            bpm = record.get("bpm_default", 100)
            require_valid(notes, bpm=bpm)
            prepared.append(
                (
                    str(record.get("name") or f"导入乐谱 {index + 1}"),
                    str(record.get("source_file") or ""),
                    str(record.get("source_type") or ""),
                    str(record.get("raw_text") or ""),
                    json.dumps(notes, ensure_ascii=False),
                    int(bpm),
                    datetime.now().isoformat(timespec="seconds"),
                )
            )
        ids = []
        with self.conn:
            for values in prepared:
                cur = self.conn.execute(
                    "INSERT INTO scores (name, source_file, source_type, raw_text, notes_json, bpm_default, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
                ids.append(cur.lastrowid)
        return ids

    def update_score(self, score_id, name, notes, raw_text="", bpm_default=100):
        require_valid(notes, bpm=bpm_default)
        self.conn.execute(
            "UPDATE scores SET name=?, raw_text=?, notes_json=?, bpm_default=? WHERE id=?",
            (name, raw_text, json.dumps(notes, ensure_ascii=False), bpm_default, score_id),
        )
        self.conn.commit()

    def list_scores(self):
        rows = self.conn.execute(
            "SELECT id, name, source_file, source_type, bpm_default, created_at "
            "FROM scores ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_score(self, score_id):
        row = self.conn.execute("SELECT * FROM scores WHERE id=?", (score_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["notes"] = json.loads(d.pop("notes_json"))
        return d

    def delete_score(self, score_id):
        self.conn.execute("DELETE FROM scores WHERE id=?", (score_id,))
        self.conn.commit()

    def get_score_preferences(self, score_id, profile_id):
        """Return per-score/per-profile UI and transport preferences.

        The payload is deliberately a versioned JSON object.  It keeps the
        stable ``scores`` schema untouched while allowing transport and
        practice controls to evolve independently.
        """
        row = self.conn.execute(
            "SELECT settings_json FROM score_preferences WHERE score_id=? AND profile_id=?",
            (int(score_id), str(profile_id)),
        ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(row["settings_json"])
        except (TypeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def save_score_preferences(self, score_id, profile_id, settings):
        if not isinstance(settings, dict):
            raise ValueError("乐谱偏好必须是对象")
        # Ensure invalid/non-serializable values fail before opening a write
        # transaction, rather than leaving a partially updated preference.
        payload = json.dumps(settings, ensure_ascii=False, sort_keys=True)
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            """
            INSERT INTO score_preferences(score_id, profile_id, settings_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(score_id, profile_id) DO UPDATE SET
                settings_json=excluded.settings_json,
                updated_at=excluded.updated_at
            """,
            (int(score_id), str(profile_id), payload, now),
        )
        self.conn.commit()
