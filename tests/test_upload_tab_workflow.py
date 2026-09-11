"""上传识别页新流程的组件级测试。"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.upload_tab import UploadTab
from gui.widgets import AppDialog


class FakeDB:
    def __init__(self):
        self.added = []

    def add_score(self, **kwargs):
        self.added.append(kwargs)
        return len(self.added)


class TestUploadWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.db = FakeDB()
        self.tab = UploadTab(self.db, recognizer=None, keymap=None)

    def tearDown(self):
        self.tab.stop_preview()
        self.tab.deleteLater()
        self.app.processEvents()

    def test_missing_provider_routes_to_settings_instead_of_stub(self):
        requested = []
        self.tab.settings_requested.connect(lambda: requested.append(True))
        self.tab._selected_path = "score.png"
        self.tab._source_type = "image"
        with patch.object(AppDialog, "_popup", return_value="去模型设置"):
            self.tab._recognize()
        self.assertEqual(requested, [True])
        self.assertEqual(self.tab.raw_text.toPlainText(), "")

    def test_recognition_fills_title_and_requires_preview(self):
        self.tab._on_recognized("TITLE:\n小星星\nJIANPU:\n1 1 5 5 6 6 5-")
        self.assertEqual(self.tab.name_edit.text(), "小星星")
        self.assertEqual(self.tab.raw_text.toPlainText(), "1 1 5 5 6 6 5-")
        self.tab._parse()
        self.assertTrue(self.tab.preview_btn.isEnabled())
        self.assertFalse(self.tab.save_btn.isEnabled())

    def test_unknown_title_gets_random_editable_name(self):
        self.tab._on_recognized("TITLE: UNKNOWN\nJIANPU: 1 2 3")
        self.assertRegex(self.tab.name_edit.text(), r"^未命名乐谱-[0-9A-F]{4}$")
        self.assertTrue(self.tab.name_edit.isEnabled())

    def test_result_is_structured_and_raw_protocol_is_collapsed(self):
        self.tab._on_recognized(
            "TITLE: 琵琶曲\nJIANPU:\n6 5 3\n3 2 1\n"
            "LYRICS:\n人 间 琴\n悠 扬 曲\n"
            "RHYTHM: 原谱没有明确休止符，停顿按分行推断"
        )
        self.assertEqual(self.tab.name_edit.text(), "琵琶曲")
        self.assertEqual(self.tab.table.rowCount(), 6)
        self.assertEqual(self.tab.table.item(0, 1).text(), "人")
        self.assertEqual(self.tab.table.item(4, 1).text(), "扬")
        self.assertTrue(self.tab.raw_text.isHidden())
        self.assertTrue(self.tab.rhythm_panel.isVisibleTo(self.tab))
        self.assertIn("0 个休止", self.tab.result_summary.text())
        self.assertTrue(self.tab.add_rests_btn.isVisibleTo(self.tab.rhythm_panel))

    def test_add_line_rests_updates_score_and_table(self):
        self.tab._on_recognized(
            "TITLE: 琵琶曲\nJIANPU:\n6 5 3\n3 2 1\nLYRICS:\n人 间 琴\n悠 扬 曲"
        )
        self.tab._add_line_rests()
        self.assertEqual(self.tab.raw_text.toPlainText(), "6 5 3 0_\n3 2 1")
        self.assertEqual(self.tab.table.rowCount(), 7)
        self.assertEqual(self.tab.table.item(3, 0).text(), "(休止)")
        self.assertEqual(self.tab.table.item(3, 1).text(), "")
        self.assertEqual(self.tab.table.item(4, 1).text(), "悠")
        self.assertEqual(self.tab.table.item(3, 2).text(), "0.5")

    def test_insert_rest_after_clicked_note_updates_source_and_table(self):
        self.tab._on_recognized("TITLE: 琵琶曲\nJIANPU:\n6 5 3 5 3\n6 5 6")
        self.tab._insert_rest_at_row(4, before=False)
        self.assertEqual(self.tab.raw_text.toPlainText(), "6 5 3 5 3 0_\n6 5 6")
        self.assertEqual(self.tab.table.rowCount(), 9)
        self.assertEqual(self.tab.table.item(5, 0).text(), "(休止)")
        self.assertFalse(self.tab.save_btn.isEnabled())

    def test_delete_clicked_note_updates_source_and_lyrics(self):
        self.tab._on_recognized("TITLE: 测试\nJIANPU: 1 2 3\nLYRICS: 你 好 呀")
        self.tab._delete_event_at_row(1)
        self.assertEqual(self.tab.raw_text.toPlainText(), "1 3")
        self.assertEqual(self.tab.table.rowCount(), 2)
        self.assertEqual(self.tab.table.item(0, 1).text(), "你")
        self.assertEqual(self.tab.table.item(1, 1).text(), "呀")

    def test_bpm_is_slider_only_from_zero_to_300(self):
        self.assertEqual(self.tab.bpm_slider.minimum(), 0)
        self.assertEqual(self.tab.bpm_slider.maximum(), 300)
        self.tab.bpm_slider.setValue(287)
        self.assertEqual(self.tab.bpm_slider.value(), 287)
        self.tab.bpm_slider.setValue(0)
        self.assertFalse(self.tab.preview_btn.isEnabled())
        self.assertIn("无法试听", self.tab.bpm_hint.text())

    def test_preview_unlocks_save_and_table_change_locks_it_again(self):
        self.tab._on_recognized("TITLE: 测试曲\nJIANPU: 1 2 3")
        self.tab._parse()
        with patch("winsound.PlaySound"):
            self.tab._preview()
        self.assertTrue(self.tab.save_btn.isEnabled())
        self.assertTrue(self.tab._preview_ready)
        self.tab.table.item(0, 2).setText("2.0")
        self.assertFalse(self.tab.save_btn.isEnabled())
        self.assertFalse(self.tab._preview_ready)

    def test_semitone_is_visible_and_preserved_by_review_table(self):
        self.tab._on_recognized("TITLE: 半音测试\nJIANPU: 1# 2")
        self.assertEqual(self.tab.table.columnCount(), 4)
        self.assertEqual(self.tab.table.item(0, 3).text(), "#")
        self.assertEqual(self.tab._table_to_notes()[0]["semitone"], 1)

    def test_confirmed_preview_saves_recognized_name_and_notes(self):
        self.tab._on_recognized("TITLE: 茉莉花\nJIANPU: 3 3 5-")
        self.tab._parse()
        with patch("winsound.PlaySound"):
            self.tab._preview()
            with patch.object(AppDialog, "show_success"):
                self.tab._save()
        self.assertEqual(len(self.db.added), 1)
        self.assertEqual(self.db.added[0]["name"], "茉莉花")
        self.assertEqual(len(self.db.added[0]["notes"]), 3)
        self.assertFalse(self.tab.save_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
