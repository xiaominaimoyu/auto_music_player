import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from core.keymap import KeyMap
from core.player import Player
from core.profile import GameProfile
from core.windows_reliability import assess_elevation
from gui.player_tab import PlayerTab


MAPPING = {
    "high": list("QWERTYU"),
    "mid": list("ASDFGHJ"),
    "low": list("ZXCVBNM"),
}


class FakeDriver:
    def __init__(self):
        self.events = []

    def press_chord(self, keys):
        self.events.append(("press", tuple(keys)))

    def release_chord(self, keys):
        self.events.append(("release", tuple(keys)))


class FakePreview(QObject):
    finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def play(self, *_args, **_kwargs):
        return True

    def stop(self):
        pass


class FakeEventPlayer(QObject):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool)
    paused = pyqtSignal(int, int)
    aborted = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_playing = False
        self.last_log_path = ""
        self.last_log_error = None
        self.guard = None

    def set_dispatch_guard(self, guard):
        self.guard = guard

    def play(self, *_args, **_kwargs):
        pass

    def stop(self):
        pass


class FakeWatcher:
    def __init__(self, current):
        self.current = current

    @staticmethod
    def is_available():
        return True

    def capture_current(self):
        return dict(self.current) if self.current is not None else None


def ensure_qapp():
    global _app
    _app = QApplication.instance() or QApplication(sys.argv)
    return _app


class TestPlayerTabTransport(unittest.TestCase):
    def setUp(self):
        ensure_qapp()
        self.temp = tempfile.TemporaryDirectory()
        self.db = ScoreDB(os.path.join(self.temp.name, "scores.db"))
        self.score_id = self.db.add_score(
            "片段测试",
            [
                {"notes": ["mid_1"], "dur": 1.0},
                {"notes": ["mid_2"], "dur": 0.5},
                {"notes": ["mid_3"], "dur": 1.0},
                {"notes": ["mid_4"], "dur": 2.0},
            ],
            bpm_default=120,
        )
        self.driver = FakeDriver()
        self.player = Player(KeyMap(MAPPING), driver=self.driver)

    def tearDown(self):
        self.player.shutdown()
        self.db.conn.close()
        self.temp.cleanup()

    def make_tab(self, **kwargs):
        tab = PlayerTab(
            self.db,
            self.player,
            {"practice_input": {"enabled": False}},
            preview_player=FakePreview(),
            **kwargs,
        )
        tab.refresh()
        return tab

    def test_segment_seek_and_bpm_persist_without_starting_player(self):
        tab = self.make_tab()
        try:
            tab.segment_start_spin.setValue(2)
            tab.segment_end_spin.setValue(3)
            tab.bpm_spin.setValue(144)
            self.assertEqual(len(tab._active_notes), 2)
            tab.seek_slider.setValue(1)
            tab._on_seek_released()
            self.assertEqual(tab._paused_done, 1)
            self.assertEqual(tab.play_btn.text(), "继续演奏")
            self.assertFalse(self.player.is_playing)
            self.assertEqual(self.driver.events, [])
            saved = self.db.get_score_preferences(self.score_id, "legacy")
            self.assertEqual(saved["segment"], [1, 3])
            self.assertEqual(saved["bpm"], 144)
        finally:
            tab.shutdown()
            tab.deleteLater()

    def test_event_profile_transpose_stays_above_compiler_path(self):
        profile = GameProfile(
            id="delta",
            name="Delta",
            pitch_keys=("Z", "X", "C", "V", "B", "N", "M"),
            modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"},
            legacy_keymap=None,
        )
        event_player = FakeEventPlayer()
        tab = self.make_tab(
            profile=profile, profiles=[profile], event_player=event_player
        )
        try:
            self.assertTrue(tab.transpose_spin.isEnabled())
            tab.segment_end_spin.setValue(1)
            tab.transpose_spin.setValue(1)
            self.assertEqual(
                tab._active_notes,
                [{"notes": ["mid_1"], "dur": 1.0, "semitone": 1}],
            )
            self.assertIsNotNone(event_player.guard)
            self.assertEqual(self.driver.events, [])
        finally:
            tab.shutdown()
            tab.deleteLater()

    def test_midi_playback_removes_timing_fragments_and_keeps_source_timeline(self):
        midi_id = self.db.add_score(
            "MIDI 碎片回归",
            [
                {"notes": [], "dur": 0.01},
                {"notes": ["mid_1"], "dur": 0.5},
                {"notes": ["mid_2"], "dur": 0.02},
                {"notes": ["mid_3"], "dur": 0.5},
            ],
            source_file="C:/music/test.mid",
            source_type="import",
            bpm_default=120,
        )
        tab = self.make_tab()
        try:
            tab.select_score(midi_id)
            self.assertEqual(len(tab._active_notes), 2)
            tab._play()
            self.assertEqual(tab._playback_gap_ms, 0.0)
            self.assertIsNone(tab._playback_humanize)
            self.assertGreater(tab._transport_degradation_count, 0)
            tab._countdown_timer.stop()
        finally:
            tab.shutdown()
            tab.deleteLater()

    @mock.patch(
        "gui.player_tab.inspect_target_elevation",
        return_value=assess_elevation(False, False),
    )
    def test_preflight_locks_target_and_dispatch_guard_fails_closed(self, _inspect):
        tab = self.make_tab()
        try:
            watcher = FakeWatcher({"hwnd": 12345, "title": "Game"})
            tab._watcher = watcher
            self.assertTrue(tab._preflight_target())
            self.assertEqual(tab._dispatch_allowed(), (True, ""))
            watcher.current = {"hwnd": 99999, "title": "Other"}
            allowed, reason = tab._dispatch_allowed()
            self.assertFalse(allowed)
            self.assertIn("失去", reason)
        finally:
            tab.shutdown()
            tab.deleteLater()

    @mock.patch("gui.player_tab.AppDialog.show_warning")
    def test_preflight_never_accepts_own_window_even_with_title_override(self, warning):
        tab = self.make_tab()
        try:
            tab._target_title = "Auto"
            tab._watcher = FakeWatcher({"hwnd": 12345, "title": "Auto Music Player"})
            tab._own_hwnd = lambda: 12345
            self.assertFalse(tab._preflight_target())
            self.assertIn("未切换", tab.progress_state.text())
            warning.assert_called_once()
        finally:
            tab.shutdown()
            tab.deleteLater()

    @mock.patch("gui.player_tab.AppDialog.show_warning")
    def test_preflight_blocks_delta_window_with_default_21_key_profile(self, warning):
        profile = GameProfile(
            id="default",
            name="默认(鸣潮 / 原神)",
            legacy_keymap=MAPPING,
        )
        tab = self.make_tab(profile=profile, profiles=[profile])
        try:
            tab._watcher = FakeWatcher({"hwnd": 12345, "title": "三角洲行动"})
            self.assertFalse(tab._preflight_target())
            self.assertIn("档位", tab.progress_state.text())
            self.assertIn("三角洲", warning.call_args.args[2])
        finally:
            tab.shutdown()
            tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
