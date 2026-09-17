import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.source_score import SourceNote, SourceSong
from gui.import_dialog import MidiImportDialog


def ensure_app():
    return QApplication.instance() or QApplication([])


class TestMidiImportDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = ensure_app()

    def setUp(self):
        self.song = SourceSong(
            "多轨",
            [
                SourceNote(0, 1, 48, 0),
                SourceNote(0, 1, 60, 1),
                SourceNote(1, 2, 62, 1),
            ],
            {0: "伴奏", 1: "主旋律"},
            2,
            120,
        )
        self.dialog = MidiImportDialog(self.song)

    def tearDown(self):
        self.dialog.deleteLater()

    def test_defaults_are_safe_and_recommended(self):
        options = self.dialog.options()
        self.assertEqual(options.track, "auto")
        self.assertEqual(options.style, "piano")
        self.assertTrue(options.fold_octaves)
        self.assertEqual(options.bpm, 120)

    def test_user_choices_are_returned_without_side_effects(self):
        track_index = self.dialog.track_combo.findData(1)
        self.dialog.track_combo.setCurrentIndex(track_index)
        self.dialog.style_combo.setCurrentIndex(
            self.dialog.style_combo.findData("preserve")
        )
        self.dialog.transpose_spin.setValue(-3)
        self.dialog.fold_checkbox.setChecked(False)
        options = self.dialog.options()
        self.assertEqual(options.track, 1)
        self.assertEqual(options.style, "preserve")
        self.assertEqual(options.transpose, -3)
        self.assertFalse(options.fold_octaves)


if __name__ == "__main__":
    unittest.main()
