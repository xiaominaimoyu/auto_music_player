"""模拟试听:编译事件/21 键时间轴 → MIDI 起音序列,全程不触碰真实输入。"""

from core.compiler import InputEvent
from core.game_preview import (
    compiled_preview_duration,
    compiled_preview_events,
    direct_preview_events,
)
from core.humanize import HumanizeParams
from core.keymap import KeyMap
from core.preview_player import PREVIEW_INSTRUMENTS
from core.profile import GameProfile
from core.timing import TimingConfig

MAPPING = {
    "high": ["Q", "W", "E", "R", "T", "Y", "U"],
    "mid": ["A", "S", "D", "F", "G", "H", "J"],
    "low": ["Z", "X", "C", "V", "B", "N", "M"],
}


def _profile(**overrides):
    values = dict(
        id="test",
        name="测试档位",
        pitch_keys=("Z", "X", "C", "V", "B", "N", "M"),
        modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"},
        pitch_direct_overrides={},
    )
    values.update(overrides)
    return GameProfile(**values)


def test_compiled_events_apply_modifier_shifts_and_releases():
    profile = _profile()
    events = [
        InputEvent(0.0, "mouse", "left", "down", 0),
        InputEvent(10.0, "kb", "Z", "down", 0),
        InputEvent(410.0, "kb", "Z", "up", 0, source_end=True),
        InputEvent(430.0, "mouse", "left", "up", 0),
        InputEvent(430.0, "mouse", "right", "down", 1),
        InputEvent(440.0, "kb", "Z", "down", 1),
        InputEvent(640.0, "kb", "Z", "up", 1, source_end=True),
        InputEvent(660.0, "mouse", "right", "up", 1),
    ]

    timeline = compiled_preview_events(events, profile)

    # mid_1 = 60;按住左键降八度 → 48,松开后改按右键升八度 → 72
    assert timeline == [(10.0, 48, True), (410.0, 48, False), (440.0, 72, True), (640.0, 72, False)]
    assert compiled_preview_duration(events) == 660.0


def test_compiled_events_use_direct_key_override_and_skip_submit_key():
    profile = _profile(pitch_direct_overrides={"high_1": ","})
    events = [
        InputEvent(0.0, "kb", ",", "down", 0),
        InputEvent(100.0, "kb", ",", "up", 0, source_end=True),
        InputEvent(200.0, "kb", "Q", "down", None),   # 场景收尾的提交键
        InputEvent(280.0, "kb", "Q", "up", None),
    ]

    # 直达键不受修饰态影响;source_index=None 的提交键不发声
    assert compiled_preview_events(events, profile) == [(0.0, 72, True), (100.0, 72, False)]


def test_compiled_events_ignore_unknown_pitch_key_instead_of_raising():
    profile = _profile(pitch_keys=("Z", "X", "C", "V", "B", "N", "M"))
    events = [
        InputEvent(0.0, "kb", "P", "down", 0),
        InputEvent(50.0, "kb", "P", "up", 0, source_end=True),
    ]

    assert compiled_preview_events(events, profile) == []


def test_compiled_duration_falls_back_to_caller_tail():
    assert compiled_preview_duration([]) == 0.0
    assert compiled_preview_duration([InputEvent(120.0, "kb", "Z", "down", 0)], timeline_ms=500.0) == 500.0


def test_direct_timeline_follows_slots_hold_ratio_and_mapping():
    notes = [{"notes": ["mid_1"], "dur": 1.0}, {"notes": ["mid_3"], "dur": 1.0}]

    events, duration = direct_preview_events(
        notes, 120, KeyMap(MAPPING),
        hold_ratio=1.0, gap_ms=0.0, timing=TimingConfig(), humanize=None, seed=None,
    )

    # bpm 120 → 每拍 500ms;两个四分音符 → 起音 0 / 500,总时长 1000
    assert events == [(0.0, 60, True), (500.0, 60, False), (500.0, 64, True), (1000.0, 64, False)]
    assert duration == 1000.0


def test_direct_timeline_strums_chord_and_orders_next_onset_after_release():
    notes = [{"notes": ["mid_1", "mid_3"], "dur": 1.0}, {"notes": ["mid_5"], "dur": 1.0}]
    humanize = HumanizeParams(jitter_ms=0.0, breath_ms=0.0, strum_ms=12.0)

    events, duration = direct_preview_events(
        notes, 120, KeyMap(MAPPING),
        hold_ratio=1.0, gap_ms=0.0, timing=TimingConfig(), humanize=humanize, seed=1,
    )

    downs = [(at, pitch) for at, pitch, is_down in events if is_down]
    # 强拍重音默认把首拍提前 4ms;第一个和弦偶数位正序扫弦,12ms 均分两个音 → 间隔 12ms
    assert downs[:2] == [(4.0, 60), (16.0, 64)]
    assert downs[-1][1] == 67
    # 扫弦把第二个音推到 16ms,它的释放也顺延;下一音必须在其完全松开之后起音
    first_chord_release = max(at for at, _pitch, is_down in events if not is_down and at < 500.0)
    assert downs[-1][0] > first_chord_release
    # 槽位终点(1000ms)大于所有事件,时长以谱面为准
    assert duration == 1000.0


def test_direct_timeline_skips_notes_without_keys():
    notes = [{"notes": ["mid_8"], "dur": 1.0}, {"notes": [], "dur": 1.0}]

    events, _duration = direct_preview_events(
        notes, 120, KeyMap(MAPPING),
        hold_ratio=1.0, gap_ms=0.0, timing=TimingConfig(), humanize=None, seed=None,
    )

    assert events == []


def test_preview_instrument_names_match_profiles():
    # 档位里的 preview_instrument 必须能对上界面音色表,否则静默回落到首选
    assert "竖琴" in PREVIEW_INSTRUMENTS
    assert "口琴" in PREVIEW_INSTRUMENTS
    assert PREVIEW_INSTRUMENTS["竖琴"] == 46
    assert PREVIEW_INSTRUMENTS["口琴"] == 22
