import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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


class FakeListener:
    def __init__(self, **callbacks):
        self.callbacks = callbacks
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


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


def test_live_recording_opt_in_persists_privacy_bounded_training_sample():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        tab = UploadTab(
            db,
            preview_player=FakePreviewPlayer(),
            keymap=KeyMap(MAPPING),
        )
        try:
            tab.capture_training_check.setChecked(True)
            with (
                patch("pynput.keyboard.Listener", FakeListener),
                patch("pynput.mouse.Listener", FakeListener),
            ):
                tab._start_live_recording()
                assert tab._live_recorder is not None
                assert tab._live_capture is not None
                assert not tab.capture_training_check.isEnabled()

                assert tab._live_recorder.press("A")
                assert tab._live_recorder.release("A")
                tab._stop_live_recording()

            completed = list(
                (Path(temp_dir) / "performance_captures" / "completed").glob(
                    "*.jsonl"
                )
            )
            assert len(completed) == 1
            records = [
                json.loads(line)
                for line in completed[0].read_text(encoding="utf-8").splitlines()
            ]
            events = [record["event"] for record in records if record["record"] == "event"]
            assert [event["type"] for event in events] == ["note_down", "note_up"]
            assert all(
                set(event)
                <= {
                    "type",
                    "press_id",
                    "relative_ns",
                    "note_id",
                    "semitone",
                    "modifier",
                    "synthetic_close",
                }
                for event in events
            )
            assert {event["modifier"] for event in events} == {"legacy"}
            assert records[-1]["status"] == "completed"
            assert records[-1]["captured_count"] == 1
            assert records[-1]["canonical_notes"] == tab._table_to_notes()
            assert tab.table.rowCount() == 1
            assert tab.capture_training_check.isEnabled()
            assert tab.live_discard_btn.isEnabled() is False
        finally:
            tab.shutdown()
            tab.deleteLater()
            db.conn.close()


def test_live_recording_discard_never_completes_or_updates_score_table():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        tab = UploadTab(
            db,
            preview_player=FakePreviewPlayer(),
            keymap=KeyMap(MAPPING),
        )
        try:
            tab.capture_training_check.setChecked(True)
            with (
                patch("pynput.keyboard.Listener", FakeListener),
                patch("pynput.mouse.Listener", FakeListener),
            ):
                tab._start_live_recording()
                assert tab._live_recorder.press("A")
                tab._discard_live_recording()

            root = Path(temp_dir) / "performance_captures"
            assert not list((root / "completed").glob("*.jsonl"))
            rejected = list((root / "rejected").glob("*.jsonl"))
            assert len(rejected) == 1
            end = json.loads(rejected[0].read_text(encoding="utf-8").splitlines()[-1])
            assert end["status"] == "rejected"
            assert end["reason"] == "user_discarded"
            assert tab.table.rowCount() == 0
        finally:
            tab.shutdown()
            tab.deleteLater()
            db.conn.close()


def test_live_recording_is_blocked_while_playback_is_active():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        tab = UploadTab(
            db,
            preview_player=FakePreviewPlayer(),
            keymap=KeyMap(MAPPING),
            playback_active=lambda: True,
        )
        try:
            with patch("gui.upload_tab.AppDialog.show_warning") as warning:
                tab._start_live_recording()
            assert tab._live_recorder is None
            assert tab._live_capture is None
            warning.assert_called_once()
            assert "停止演奏" in warning.call_args.args[2]
        finally:
            tab.shutdown()
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
