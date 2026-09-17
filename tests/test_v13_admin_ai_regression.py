"""v1.3 管理员 + 外部 AI 乐谱回归测试。

测试只使用临时数据库、临时日志目录和 FakeDriver，不发送真实键鼠输入。
"""

import json
import os
import sys
import threading
import time
from datetime import datetime as RealDateTime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
import pytest

import core.event_logger as event_logger_mod
from core.database import ScoreDB
from core.event_logger import EventLogger
from core.event_player import EventPlayer
from core.parser import AI_MISSING_SEPARATOR_REASON, parse_jianpu
from core.profile import load_profiles
from core.prompt import JIANPU_PROMPT
from gui.player_tab import build_event_plan
from gui.upload_tab import UploadTab
from gui.widgets import AppDialog
from main import resolve_data_dir


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(REPO_ROOT, "profiles")
PLAN_KW = dict(
    bpm=300,
    settle_ms=1.0,
    release_settle_ms=1.0,
    hold_ratio=0.01,
    gap_ms=0.0,
)


class FakeDriver:
    def __init__(self, fail_key=None):
        self.events = []
        self.lock = threading.Lock()
        self.fail_key = fail_key

    def _append(self, kind, value):
        with self.lock:
            self.events.append((kind, value))

    def press_key(self, key):
        if key == self.fail_key:
            raise OSError("模拟 SendInput 失败")
        self._append("press_key", key)

    def release_key(self, key):
        self._append("release_key", key)

    def press_mouse(self, button):
        self._append("press_mouse", button)

    def release_mouse(self, button):
        self._append("release_mouse", button)

    def panic_release(self, keys, mouse):
        self._append("panic_release", (tuple(keys), tuple(mouse)))


class FakePreviewPlayer(QObject):
    finished = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_playing = False

    def play(self, *_args, **_kwargs):
        self.is_playing = True
        return True

    def stop(self):
        self.is_playing = False


def _qapp():
    global _APP
    _APP = QApplication.instance() or QApplication(sys.argv)
    return _APP


def _delta_profile():
    return next(
        profile
        for profile in load_profiles(PROFILES_DIR)
        if profile.id == "delta_force_harmonica"
    )


def _wait_idle(player, timeout=5.0):
    deadline = time.time() + timeout
    while player.is_playing and time.time() < deadline:
        time.sleep(0.005)
    assert not player.is_playing


def _wait_driver_action(driver, action, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with driver.lock:
            if any(kind == action for kind, _value in driver.events):
                return
        time.sleep(0.005)
    raise TimeoutError(f"未观察到驱动动作: {action}")


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ai_unicode_sharp_matches_manual_ascii_and_reaches_delta_modifier():
    manual = parse_jianpu("1# 2, 3'")
    ai = parse_jianpu("1♯ 2, 3'")
    assert ai == manual

    variants, errors = parse_jianpu("1♯ ♯2 3＃ ＃4", collect=True)
    assert errors == []
    assert len(variants) == 4
    assert all(note.get("semitone") == 1 for note in variants)

    plan, degradations = build_event_plan(ai, _delta_profile(), "free_play", **PLAN_KW)
    mouse_down = [
        event.key
        for event in plan.events
        if event.device == "mouse" and event.action == "down"
    ]
    assert mouse_down == ["middle", "left", "right"]
    assert degradations == []


def test_ai_strict_mode_reports_joined_notes_without_changing_legacy_mode():
    samples = ("7_1", "17_1", "7_7_1222")
    for sample in samples:
        legacy = parse_jianpu(sample)
        strict_notes, errors = parse_jianpu(sample, collect=True, strict_ai=True)
        assert strict_notes == legacy
        assert any(error.reason == AI_MISSING_SEPARATOR_REASON for error in errors)

    for compact in ("12345", "12 34", "1#2", "1,2"):
        _notes, errors = parse_jianpu(compact, collect=True, strict_ai=True)
        assert not any(error.reason == AI_MISSING_SEPARATOR_REASON for error in errors)


def test_ai_prompt_defines_keyboard_octaves_and_forbids_underscore_as_low_note():
    assert "Q/W/E/R/T/Y/U 分别是高音" in JIANPU_PROMPT
    assert "A/S/D/F/G/H/J 分别是中音" in JIANPU_PROMPT
    assert "Z/X/C/V/B/N/M 分别是低音" in JIANPU_PROMPT
    assert "下划线 _ 只表示时值" in JIANPU_PROMPT
    assert "7_1" in JIANPU_PROMPT
    assert "禁止输出 1'' 或 1,," in JIANPU_PROMPT
    assert "整个连续乐句统一移高或移低八度" in JIANPU_PROMPT


def test_upload_ai_ambiguity_clears_previous_table_and_blocks_save_path(
    tmp_path, monkeypatch
):
    _qapp()
    db = ScoreDB(str(tmp_path / "scores.db"))
    tab = UploadTab(db, preview_player=FakePreviewPlayer())
    warnings = []
    monkeypatch.setattr(
        AppDialog,
        "show_warning",
        lambda *_args, **_kwargs: warnings.append((_args, _kwargs)),
    )
    try:
        tab.raw_text.setPlainText("1 2 3")
        tab._parse()
        assert tab.table.rowCount() == 3

        tab.raw_text.setPlainText("17_1")
        tab._parse()
        assert tab.table.rowCount() == 0
        assert not tab.preview_btn.isEnabled()
        assert warnings
        assert "未进入校对表格" in tab.parse_status.text()
    finally:
        tab.deleteLater()
        db.conn.close()


def test_event_logger_exclusive_names_preserve_existing_history(tmp_path, monkeypatch):
    class FixedDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 14, 12, 34, 56, 123456, tzinfo=tz)

    monkeypatch.setattr(event_logger_mod, "datetime", FixedDateTime)
    historical = tmp_path / "20260914_123456_同名曲.jsonl"
    historical.write_text("historical-sentinel\n", encoding="utf-8")

    logger = EventLogger(tmp_path)
    paths = []
    for _ in range(2):
        logger.start_session("同名曲", bpm=100, note_count=1)
        paths.append(logger.log_path)
        logger.end_session(completed=1, total=1)

    assert historical.read_text(encoding="utf-8") == "historical-sentinel\n"
    assert paths[0] != paths[1]
    assert all(path is not None and path.exists() for path in paths)
    assert len(list(tmp_path.glob("*.jsonl"))) == 3


def test_delta_event_player_writes_complete_trace_and_note_logs(tmp_path):
    notes = parse_jianpu("1♯ 2, 3'")
    for note in notes:
        note["dur"] = 0.001
    plan, _ = build_event_plan(notes, _delta_profile(), "free_play", **PLAN_KW)
    driver = FakeDriver()
    logger = EventLogger(tmp_path / "play_logs")
    player = EventPlayer(driver, logger=logger)

    player.play(
        plan.events,
        plan.interrupt_mode,
        score_name="AI 回归样本",
        source_total=len(notes),
        source_start_index=0,
        source_notes=notes,
        bpm=300,
        trace_context={"input_origin": "external_ai_paste", "profile_id": "delta_force_harmonica"},
    )
    _wait_idle(player)

    path = logger.log_path
    assert path is not None and path.exists()
    events = _read_events(path)
    assert events[0]["type"] == "session_start"
    assert events[-1]["type"] == "session_end"
    assert events[-1]["completed"] == len(notes)
    assert events[-1]["stopped_early"] is False
    assert len([event for event in events if event["type"] == "note"]) == len(notes)
    traces = [event for event in events if event["type"] == "env"]
    assert any(event["details"].get("stage") == "event_plan_received" for event in traces)
    modifier_down = {
        event["details"]["key"]
        for event in traces
        if event["details"].get("stage") == "event_dispatched"
        and event["details"].get("device") == "mouse"
        and event["details"].get("action") == "down"
    }
    assert modifier_down == {"middle", "left", "right"}


def test_delta_event_player_closes_log_on_driver_error(tmp_path):
    notes = [{"notes": ["mid_1"], "dur": 0.001}]
    plan, _ = build_event_plan(notes, _delta_profile(), "free_play", **PLAN_KW)
    logger = EventLogger(tmp_path / "play_logs")
    player = EventPlayer(FakeDriver(fail_key="Z"), logger=logger)

    player.play(
        plan.events,
        score_name="失败样本",
        source_total=1,
        source_start_index=0,
        source_notes=notes,
        bpm=300,
    )
    _wait_idle(player)

    events = _read_events(logger.log_path)
    assert events[-1]["type"] == "session_end"
    assert events[-1]["stopped_early"] is True
    assert "模拟 SendInput 失败" in events[-1]["error"]
    failed_notes = [event for event in events if event["type"] == "note"]
    assert len(failed_notes) == 1
    assert failed_notes[0]["success"] is False
    assert logger.is_active is False


@pytest.mark.parametrize(
    ("interrupt", "expected_error"),
    (("pause", None), ("abort", "焦点丢失")),
)
def test_delta_event_player_closes_log_on_pause_and_abort(
    tmp_path, interrupt, expected_error
):
    notes = [{"notes": ["low_1"], "dur": 10.0}]
    plan, _ = build_event_plan(notes, _delta_profile(), "free_play", **PLAN_KW)
    logger = EventLogger(tmp_path / interrupt / "play_logs")
    driver = FakeDriver()
    player = EventPlayer(driver, logger=logger)
    player.play(
        plan.events,
        interrupt_mode="pause",
        score_name=f"{interrupt} 样本",
        source_total=1,
        source_start_index=0,
        source_notes=notes,
        bpm=300,
    )
    _wait_driver_action(driver, "press_key")
    if interrupt == "abort":
        player.abort("焦点丢失")
    else:
        player.stop()
    _wait_idle(player)

    events = _read_events(logger.log_path)
    footer = events[-1]
    assert footer["type"] == "session_end"
    assert footer["stopped_early"] is True
    assert footer["completed"] == 0
    assert footer["error"] == expected_error
    assert logger.is_active is False


def test_all_rest_event_session_logs_complete_source_total(tmp_path):
    notes = [{"notes": [], "dur": 0.25} for _ in range(3)]
    plan, _ = build_event_plan(notes, _delta_profile(), "free_play", **PLAN_KW)
    assert plan.events == []
    logger = EventLogger(tmp_path / "all-rest" / "play_logs")
    player = EventPlayer(FakeDriver(), logger=logger)
    player.play(
        plan.events,
        score_name="全休止",
        source_total=len(notes),
        source_start_index=0,
        source_notes=notes,
        bpm=300,
    )
    _wait_idle(player)

    footer = _read_events(logger.log_path)[-1]
    assert footer["type"] == "session_end"
    assert footer["completed"] == 3
    assert footer["total"] == 3
    assert footer["stopped_early"] is False


def test_event_logger_resolves_relative_directory_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = EventLogger("relative/play_logs")
    expected = (tmp_path / "relative" / "play_logs").resolve()
    assert logger.log_dir == expected
    monkeypatch.chdir(tmp_path.parent)
    assert logger.log_dir == expected


def test_data_dir_is_resolved_from_config_location_not_admin_start_cwd(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "installed app"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_data_dir(str(config_path), "data") == str(
        (config_dir / "data").resolve()
    )
