import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from gui.score_editor_dialog import ScoreEditorDialog


def ensure_qapp():
    global _app
    _app = QApplication.instance() or QApplication(sys.argv)
    return _app


class TestScoreEditorDialog(unittest.TestCase):
    def setUp(self):
        ensure_qapp()
        self.dialog = ScoreEditorDialog(
            {
                "name": "原曲",
                "bpm_default": 120,
                "notes": [
                    {"notes": ["mid_1"], "dur": 1.0},
                    {"notes": ["mid_2"], "dur": 0.5, "semitone": 1},
                ],
            }
        )

    def tearDown(self):
        self.dialog.deleteLater()

    def test_load_edit_and_extract_values(self):
        self.dialog.name_edit.setText("新曲")
        self.dialog.bpm_spin.setValue(144)
        self.dialog.table.item(0, 0).setText("mid_1,mid_3")
        values = self.dialog.values()
        self.assertEqual(values["name"], "新曲")
        self.assertEqual(values["bpm_default"], 144)
        self.assertEqual(values["notes"][0]["notes"], ["mid_1", "mid_3"])
        self.assertEqual(values["notes"][1]["semitone"], 1)

    def test_add_rest_undo_and_redo(self):
        self.dialog._add_rest()
        self.assertEqual(self.dialog.table.rowCount(), 3)
        self.dialog._undo()
        self.assertEqual(self.dialog.table.rowCount(), 2)
        self.dialog._redo()
        self.assertEqual(self.dialog.table.rowCount(), 3)
        self.assertEqual(self.dialog.notes()[-1], {"notes": [], "dur": 1.0})

    def test_invalid_duration_is_rejected_before_database_write(self):
        self.dialog.table.item(0, 1).setText("bad")
        with self.assertRaises(ValueError):
            self.dialog.notes()

    def test_unknown_semitone_marker_is_rejected(self):
        self.dialog.table.item(0, 2).setText("sharp")
        with self.assertRaisesRegex(ValueError, "半音"):
            self.dialog.notes()


if __name__ == "__main__":
    unittest.main()
