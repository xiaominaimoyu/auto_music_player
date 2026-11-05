"""Startup recovery requires confirmation and never upgrades a future library."""

import os
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import main
from core.database import ScoreDB
from core.keymap import KeyMap
from core.library_recovery import discover_legacy_libraries
from core.player import Player
from gui.main_window import MainWindow

_app = None


class StartupLibraryTests(unittest.TestCase):
    def setUp(self):
        global _app
        _app = QApplication.instance() or QApplication([])
        self.app = _app
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old_path = self.root / "LocalAppData" / "data" / "scores.db"
        with ScoreDB(self.old_path) as old:
            old.add_score("旧谱", [{"notes": ["mid_1"], "dur": 1.0}], raw_text="1")
        self.target = self.root / "portable" / "data" / "scores.db"

    def window(self, db, **options):
        cfg = main.load_config()
        cfg["update"]["enabled"] = False
        keymap = KeyMap(cfg["keymap"])
        player = Player(keymap, driver=object())
        with patch("gui.main_window.pk.GlobalHotKeys"):
            win = MainWindow(cfg, db, keymap, player, event_log_dir=str(self.root / "logs"), **options)
        self.addCleanup(win.deleteLater)
        self.addCleanup(win._cleanup_on_quit)
        return win

    def test_known_other_location_is_detected_without_writing(self):
        original = self.old_path.read_bytes()
        paths = discover_legacy_libraries(self.target, [self.target.parent, self.old_path.parent])
        self.assertEqual(paths, [self.old_path])
        self.assertFalse(self.target.exists())
        self.assertEqual(self.old_path.read_bytes(), original)

    def test_existing_destination_suppresses_startup_prompt(self):
        with ScoreDB(self.target):
            pass
        self.assertEqual(discover_legacy_libraries(self.target, [self.old_path.parent]), [])

    def test_startup_confirmation_decline_and_accept_preserve_original(self):
        original = self.old_path.read_bytes()
        for confirm in (False, True):
            with self.subTest(confirm=confirm):
                target = self.root / str(confirm) / "scores.db"
                db = ScoreDB(target)
                self.addCleanup(db.close)
                with patch("gui.library_tab.AppDialog.confirm", return_value=confirm) as prompt, \
                     patch("gui.library_tab.AppDialog.show_info"), \
                     patch("gui.library_tab.AppDialog.show_error") as error:
                    win = self.window(db, legacy_libraries=[self.old_path])
                    self.app.processEvents()
                prompt.assert_called_once()
                self.assertEqual(prompt.call_args.args[1], "发现旧曲库")
                error.assert_not_called()
                self.assertEqual(len(db.list_scores()), int(confirm))
                self.assertEqual(win.library_tab.table.rowCount(), int(confirm))
                self.assertEqual(self.old_path.read_bytes(), original)

    def test_migration_backup_is_visible_and_readable(self):
        with closing(sqlite3.connect(self.old_path)) as conn:
            conn.execute("PRAGMA user_version=1")
        db = ScoreDB(self.old_path)
        self.addCleanup(db.close)
        win = self.window(db)
        self.assertTrue(Path(db.pre_migration_backup_path).is_file())
        self.assertIn(db.pre_migration_backup_path, win.status_label.text())
        with closing(sqlite3.connect(db.pre_migration_backup_path)) as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_newer_schema_is_rejected_without_mutating_database(self):
        with closing(sqlite3.connect(self.old_path)) as conn:
            conn.execute("PRAGMA user_version=99")
        original = self.old_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "更高版本"):
            ScoreDB(self.old_path)
        self.assertEqual(self.old_path.read_bytes(), original)

    def test_migration_rehearsal_preserves_original_and_restores_attachments(self):
        from tools.rehearse_library_migration import rehearse_migration

        with ScoreDB(self.old_path) as old:
            source = self.root / "sample.mid"
            source.write_bytes(b"sample source")
            old.add_score("带来源", [{"notes": ["mid_2"], "dur": 0.5}],
                          source_meta=old.archive_source_file(source))
            old.conn.execute("PRAGMA user_version=1")
            old.conn.commit()
        original = self.old_path.read_bytes()
        result = rehearse_migration(self.old_path)
        self.assertEqual(result["songs"], 2)
        self.assertEqual(result["retry_skipped"], 2)
        self.assertTrue(result["migration_backup_verified"])
        self.assertEqual(self.old_path.read_bytes(), original)
