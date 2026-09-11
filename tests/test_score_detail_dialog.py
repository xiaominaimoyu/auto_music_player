"""乐谱详情的编辑与歌词持久化测试。"""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from core.lyrics import align_lyrics
from core.recognizer import build_recognition_text, parse_recognition_response
from gui.score_detail_dialog import ScoreDetailDialog


class TestScoreDetailDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = ScoreDB(os.path.join(self.directory.name, "scores.db"))
        notes = [
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": ["mid_2"], "dur": 1.0},
            {"notes": ["mid_3"], "dur": 1.0},
        ]
        raw = build_recognition_text("原曲", "1 2 3", ["你", "好", "呀"])
        self.score_id = self.db.add_score("原曲", notes, raw_text=raw, bpm_default=100)
        self.dialog = ScoreDetailDialog(None, self.db, self.db.get_score(self.score_id))

    def tearDown(self):
        self.dialog.stop_preview()
        self.dialog.deleteLater()
        self.db.conn.close()
        self.directory.cleanup()

    def test_save_updates_name_score_lyrics_and_bpm(self):
        self.dialog.name_edit.setText("新曲名")
        self.dialog.jianpu_edit.setPlainText("1 0_ 3\n5")
        self.dialog.lyrics_edit.setPlainText("新 _ 歌 词")
        self.dialog.bpm_slider.setValue(132)
        self.dialog._save()

        saved = self.db.get_score(self.score_id)
        result = parse_recognition_response(saved["raw_text"])
        self.assertEqual(saved["name"], "新曲名")
        self.assertEqual(saved["bpm_default"], 132)
        self.assertEqual(len(saved["notes"]), 4)
        self.assertEqual(align_lyrics(result.jianpu_text, result.lyrics_lines), ["新", "", "歌", "词"])


if __name__ == "__main__":
    unittest.main()
