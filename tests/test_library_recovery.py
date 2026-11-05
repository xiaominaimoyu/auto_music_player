import os
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from core.database import ScoreDB
from core.library_recovery import recover_library


NOTES = [{"notes": ["mid_1"], "dur": 1.0}]


class LibraryRecoveryTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enter_tempdir())
        self.db = ScoreDB(root / "current" / "scores.db")
        self.addCleanup(self.db.close)
        self.current_id = self.db.add_score("current", NOTES)
        self.old_path = root / "old" / "scores.db"
        with ScoreDB(self.old_path) as old:
            midi = root / "original.mid"
            midi.write_bytes(b"source file")
            source = old.archive_source_file(midi)
            self.stored = source["stored_path"]
            old_id = old.add_score("old", NOTES, raw_text="1", source_meta=source)
            self.assertEqual(old_id, self.current_id)
            old.save_score_preferences(old_id, "delta", {"transpose": 1})

    def enter_tempdir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def test_copy_preserves_both_libraries_and_retry(self):
        original = self.old_path.read_bytes()
        result = recover_library(self.db, self.old_path)
        self.assertEqual((result.added, result.skipped), (1, 0))
        self.assertEqual(self.db.get_score(self.current_id)["name"], "current")
        new_id = next(row["id"] for row in self.db.list_scores() if row["name"] == "old")
        song = self.db.get_score(new_id)
        self.assertEqual((song["notes"], song["raw_text"]), (NOTES, "1"))
        self.assertEqual(self.db.get_score_preferences(new_id, "delta"), {"transpose": 1})
        self.assertEqual((Path(self.db.data_dir) / song["source"]["stored_path"]).read_bytes(), b"source file")
        with closing(sqlite3.connect(result.backup)) as backup:
            self.assertEqual(backup.execute("SELECT name FROM scores").fetchall(), [("current",)])
        self.db.update_score(new_id, "renamed locally", NOTES)
        again = recover_library(self.db, self.old_path)
        self.assertEqual((again.added, again.skipped), (0, 1))
        self.assertEqual(self.db.get_score(new_id)["name"], "renamed locally")
        self.assertEqual(self.old_path.read_bytes(), original)

    def test_partial_failure_rolls_back_rows_and_new_attachments(self):
        with ScoreDB(self.old_path) as old:
            old.add_score("bad", NOTES, source_meta={"stored_path": "sources/missing.mid"})
        with self.assertRaisesRegex(ValueError, "缺少来源附件"):
            recover_library(self.db, self.old_path)
        self.assertEqual([row["name"] for row in self.db.list_scores()], ["current"])
        self.assertFalse((Path(self.db.data_dir) / self.stored).exists())

    def test_reject_external_source_and_self(self):
        with ScoreDB(self.old_path) as old:
            old.conn.execute("UPDATE score_sources SET stored_path='../outside.mid'")
            old.conn.commit()
        with self.assertRaisesRegex(ValueError, "超出"):
            recover_library(self.db, self.old_path)
        with self.assertRaisesRegex(ValueError, "当前使用"):
            recover_library(self.db, self.db.db_path)

    def test_schema_without_sidecars(self):
        with closing(sqlite3.connect(self.old_path)) as old:
            old.execute("DROP TABLE score_sources")
            old.execute("DROP TABLE score_preferences")
            old.execute("PRAGMA user_version=0")
            old.commit()
        self.assertEqual(recover_library(self.db, self.old_path).added, 1)

    def test_gui_selection_requires_confirmation_and_refreshes_existing_table(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication, QPushButton
        from gui.library_tab import AppDialog, QFileDialog, LibraryTab

        app = QApplication.instance() or QApplication([])
        tab = LibraryTab(self.db)
        try:
            self.assertIn(self.db.data_dir, tab.location_label.text())
            button = next(b for b in tab.findChildren(QPushButton) if b.text() == "找回旧曲库")
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(self.old_path), "")), \
                 patch.object(AppDialog, "confirm", side_effect=[False, True]), \
                 patch.object(AppDialog, "show_info"):
                button.click()
                self.assertEqual(tab.table.rowCount(), 1)
                button.click()
                self.assertEqual(tab.table.rowCount(), 2)
        finally:
            tab.close()
            app.processEvents()
