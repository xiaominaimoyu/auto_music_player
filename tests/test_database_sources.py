import hashlib
import os
import sqlite3
import tempfile
import unittest

from core.database import ScoreDB


class DatabaseSourceTests(unittest.TestCase):
    def test_schema_backup_and_source_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            db = ScoreDB(os.path.join(temp, "scores.db"))
            source = os.path.join(temp, "input.mid")
            payload = b"midi-source-fixture"
            with open(source, "wb") as stream:
                stream.write(payload)
            archive = db.archive_source_file(source, source_kind="midi")
            score_id = db.add_score(
                "fixture",
                [{"notes": ["mid_1"], "dur": 1.0}],
                source_file=source,
                source_type="import",
                source_meta={
                    **archive,
                    "metadata": {"format": "midi-source", "manual_confirmation_required": True},
                },
            )
            saved = db.get_score_source(score_id)
            self.assertEqual(saved["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(saved["metadata"]["format"], "midi-source")
            self.assertEqual(db.conn.execute("PRAGMA user_version").fetchone()[0], 2)
            backup = db.backup_to(os.path.join(temp, "backup.db"))
            self.assertTrue(os.path.isfile(backup))
            check = sqlite3.connect(backup)
            try:
                self.assertEqual(check.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 1)
            finally:
                check.close()
            db.close()


if __name__ == "__main__":
    unittest.main()
