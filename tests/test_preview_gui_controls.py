"""试听按钮 GUI 接线测试:校对表格和演奏页都不触发真实输入。"""

import os
import sys
import tempfile
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from core.event_player import EventPlayer
from core.keymap import KeyMap
from core.player import Player
from core.profile import load_profiles
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

    def __init__(self):
        super().__init__()
        self.calls = []
        self.is_playing = False

    def play(self, notes, bpm=100, gap_ms=0, score_name="", *, program=0):
        self.calls.append(
            {"notes": notes, "bpm": bpm, "gap_ms": gap_ms, "score_name": score_name,
             "program": program}
        )
        self.is_playing = True
        return True

    def stop(self):
        self.is_playing = False

    def play_events(self, events, *, program=0, duration_ms=0):
        self.calls.append({"events": events, "program": program, "duration_ms": duration_ms})
        self.is_playing = True
        return True


class FakeDriver:
    def press_chord(self, _keys):
        pass

    def release_chord(self, _keys):
        pass


def ensure_qapp():
    global _app
    _app = QApplication.instance() or QApplication(sys.argv)
    return _app


def test_upload_review_table_can_preview_current_unsaved_notes():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        preview = FakePreviewPlayer()
        tab = UploadTab(db, preview_player=preview)
        try:
            tab.raw_text.setPlainText("1 2 0 3#")
            tab._parse()
            assert tab.preview_btn.isEnabled()
            tab.preview_instrument_combo.setCurrentText("竖琴")
            mixed = {"notes": ["mid_1", "mid_3"], "semitones": [1, 0], "dur": 1.0}
            tab._append_table_item(mixed)

            tab._toggle_preview()

            assert len(preview.calls) == 1
            assert preview.calls[0]["notes"][3]["semitone"] == 1
            assert preview.calls[0]["notes"][-1] == mixed
            assert preview.calls[0]["notes"][2]["notes"] == []
            assert preview.calls[0]["program"] == 46
            assert not tab.preview_instrument_combo.isEnabled()
            assert tab.preview_btn.text() == "停止试听"
        finally:
            tab._stop_preview()
            tab.deleteLater()
            db.conn.close()


def test_player_page_preview_uses_selected_saved_score():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as temp_dir:
        db = ScoreDB(os.path.join(temp_dir, "scores.db"))
        notes = [{"notes": ["mid_1"], "dur": 1.0}]
        db.add_score("测试曲", notes, bpm_default=120)
        preview = FakePreviewPlayer()
        player = Player(KeyMap(MAPPING), driver=FakeDriver())
        tab = PlayerTab(db, player, {}, preview_player=preview)
        try:
            tab.refresh()
            assert tab.preview_btn.isEnabled()

            tab._toggle_preview()

            assert preview.calls[0]["notes"] == notes
            assert preview.calls[0]["bpm"] == 120
            assert tab.preview_btn.text() == "停止试听"
        finally:
            tab._stop_preview()
            tab.deleteLater()
            db.conn.close()


def test_game_preview_from_page_uses_profile_limits_and_disables_selectors(monkeypatch):
    ensure_qapp()
    monkeypatch.setattr("gui.player_tab.get_event_logger", lambda: None)
    profiles = [replace(profile, preview_instrument="") for profile in
                load_profiles(os.path.join(os.path.dirname(__file__), "..", "profiles"))]
    for profile_id, expected, program in (("default", [60, 76, 50], 46),
                                          ("delta_force_harmonica", [76, 50], 22)):
        with tempfile.TemporaryDirectory() as directory:
            db = ScoreDB(os.path.join(directory, "scores.db"))
            notes = [{"notes": ["mid_1", "high_3"], "dur": 1.0, "semitones": [1, 0]},
                     {"notes": ["low_2"], "dur": 1.0, "semitone": 1}]
            db.add_score("游戏模拟", notes, bpm_default=120)
            preview = FakePreviewPlayer()
            driver = FakeDriver()
            player = Player(KeyMap(MAPPING), driver=driver)
            profile = next(item for item in profiles if item.id == profile_id)
            tab = PlayerTab(db, player, {"humanize": {"enabled": False}},
                            profile=profile, profiles=profiles, event_player=EventPlayer(driver),
                            preview_player=preview)
            try:
                tab.refresh()
                tab.preview_mode_combo.setCurrentIndex(tab.preview_mode_combo.findData("game"))
                tab._toggle_preview()
                call = preview.calls[0]
                assert [pitch for _at, pitch, down in call["events"] if down] == expected
                assert call["program"] == program
                assert call["duration_ms"] >= 875
                assert "降级" in tab.preview_status.text()
                assert not tab.preview_mode_combo.isEnabled()
                assert not tab.preview_instrument_combo.isEnabled()
                tab._stop_preview()
                assert tab.preview_instrument_combo.isEnabled()
                other = "delta_force_harmonica" if profile_id == "default" else "default"
                tab.profile_combo.setCurrentIndex(tab.profile_combo.findData(other))
                assert tab.preview_instrument_combo.currentData() == (22 if other != "default" else 46)
            finally:
                tab.shutdown()
                tab.deleteLater()
                db.close()


def test_original_preview_keeps_short_notes_and_per_note_accidentals():
    ensure_qapp()
    with tempfile.TemporaryDirectory() as directory:
        db = ScoreDB(os.path.join(directory, "scores.db"))
        notes = [{"notes": ["mid_1", "mid_3"], "dur": 0.01, "semitones": [0, 1]},
                 {"notes": ["mid_5"], "dur": 1.0}]
        db.add_score("校对", notes, source_file="old.mid", source_type="import")
        preview = FakePreviewPlayer()
        tab = PlayerTab(db, Player(KeyMap(MAPPING), driver=FakeDriver()), {}, preview_player=preview)
        try:
            tab.refresh()
            tab._toggle_preview()
            assert preview.calls[0]["notes"] == notes
        finally:
            tab.shutdown()
            tab.deleteLater()
            db.close()
