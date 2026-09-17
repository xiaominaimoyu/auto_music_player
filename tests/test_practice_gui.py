import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from core.keymap import KeyMap
from core.player import Player
from gui.player_tab import PlayerTab
from gui.upload_tab import UploadTab


MAPPING = {
    "high": ["Q", "W", "E", "R", "T", "Y", "U"],
    "mid": ["A", "S", "D", "F", "G", "H", "J"],
    "low": ["Z", "X", "C", "V", "B", "N", "M"],
}


class FakePreviewPlayer(QObject):
    finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def play(self, *args, **kwargs):
        return True

    def stop(self):
        pass


class FakeDriver:
    def __init__(self):
        self.events = []

    def press_chord(self, keys):
        self.events.append(("press_chord", tuple(keys)))

    def release_chord(self, keys):
        self.events.append(("release_chord", tuple(keys)))


def ensure_qapp():
    global _app
    _app = QApplication.instance() or QApplication(sys.argv)
    return _app


def test_upload_step_recording_appends_editable_notes():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        tab = UploadTab(db, preview_player=FakePreviewPlayer())
        try:
            tab.record_octave_combo.setCurrentIndex(1)  # high
            tab.record_dur_combo.setCurrentIndex(1)     # 1/2 beat
            tab.record_semitone_btn.setChecked(True)
            tab._append_recorded_note(2)
            tab._append_recorded_rest()

            notes = tab._table_to_notes()
            assert notes == [
                {"notes": ["high_2"], "dur": 0.5, "semitone": 1},
                {"notes": [], "dur": 0.5},
            ]

            tab._insert_rest_at_selection(before=True)
            assert tab.table.rowCount() == 3
            tab._delete_last_table_row()
            assert tab.table.rowCount() == 2
        finally:
            tab.deleteLater()
            db.conn.close()


def test_player_practice_mode_does_not_start_real_player():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        notes = [
            {"notes": ["mid_1"], "dur": 0.25},
            {"notes": ["mid_2"], "dur": 0.25},
        ]
        db.add_score("练习曲", notes, bpm_default=120)
        driver = FakeDriver()
        player = Player(KeyMap(MAPPING), driver=driver)
        tab = PlayerTab(
            db,
            player,
            {"practice_input": {"enabled": False}},
            preview_player=FakePreviewPlayer(),
        )
        try:
            tab.refresh()
            tab.mode_combo.setCurrentIndex(1)
            tab._play()

            assert tab._practice_active
            assert not player.is_playing
            assert driver.events == []
            assert "乐谱轨 1/2" in tab.score_rail_label.text()
            assert "输入轨: A" in tab.input_rail_label.text()

            tab._on_practice_input(("S",), ())
            assert tab.pos_label.text() == "0 / 2"
            assert "错误 1" in tab.progress_state.text()
            assert driver.events == []

            tab._practice_tick()
            assert tab.pos_label.text() == "1 / 2"

            tab._stop()
            assert not tab._practice_active
            assert tab.play_btn.text() == "继续练习"

            tab._play()
            tab._practice_tick()
            assert tab.progress_state.text().startswith("练习完成")
            assert driver.events == []

            tab._play()
            tab.practice_stop_requested.emit()
            assert not tab._practice_active
            assert driver.events == []
        finally:
            tab._practice_timer.stop()
            tab.deleteLater()
            db.conn.close()
