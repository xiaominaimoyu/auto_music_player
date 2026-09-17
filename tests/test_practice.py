import pytest

from core.keymap import KeyMap
from core.practice import (
    PracticeSession,
    StepRecorder,
    build_dual_rail_hint,
    build_practice_cues,
    format_score_note,
)
from core.profile import GameProfile
from core.score_model import ScoreValidationError


MAPPING = {
    "high": ["Q", "W", "E", "R", "T", "Y", "U"],
    "mid": ["A", "S", "D", "F", "G", "H", "J"],
    "low": ["Z", "X", "C", "V", "B", "N", "M"],
}


def test_format_score_note_keeps_rest_and_semitone():
    assert format_score_note({"notes": [], "dur": 0.5}) == "休止 · 0.5 拍"
    assert format_score_note({"notes": ["mid_3"], "dur": 1, "semitone": 1}) == "mid_3# · 1 拍"


def test_dual_rail_legacy_hint_uses_keymap_and_warns_for_semitone():
    hint = build_dual_rail_hint(
        [{"notes": ["mid_3"], "dur": 1, "semitone": 1}],
        0,
        keymap=KeyMap(MAPPING),
    )
    assert hint.score_text == "乐谱轨 1/1: mid_3# · 1 拍"
    assert hint.input_text.startswith("输入轨: D")
    assert "半音标记不会由旧 21 键档位发送" in hint.input_text


def test_dual_rail_event_hint_uses_profile_modifier():
    profile = GameProfile(
        id="delta",
        name="Delta",
        pitch_keys=("Z", "X", "C", "V", "B", "N", "M"),
        modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"},
        legacy_keymap=None,
    )
    hint = build_dual_rail_hint(
        [{"notes": ["high_1"], "dur": 1}],
        0,
        profile=profile,
        use_event_path=True,
    )
    assert "鼠标:right" in hint.input_text
    assert "键盘:Z" in hint.input_text


def test_step_recorder_appends_valid_rows_and_deletes_last():
    recorder = StepRecorder()
    recorder.append_note("mid_1", 0.5)
    recorder.append_rest(1.0)
    assert recorder.items == [
        {"notes": ["mid_1"], "dur": 0.5},
        {"notes": [], "dur": 1.0},
    ]
    assert recorder.delete_last() == {"notes": [], "dur": 1.0}
    assert recorder.items == [{"notes": ["mid_1"], "dur": 0.5}]


def test_step_recorder_rejects_invalid_note():
    recorder = StepRecorder()
    with pytest.raises(ScoreValidationError):
        recorder.append_note("bad_1", 1.0)


def test_practice_session_only_advances_on_correct_key_and_skips_rest():
    cues = build_practice_cues(
        [
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": [], "dur": 0.5},
            {"notes": ["mid_2"], "dur": 1.0},
        ],
        keymap=KeyMap(MAPPING),
    )
    session = PracticeSession(cues)
    wrong = session.submit(["S"])
    assert not wrong.correct
    assert wrong.index == 0
    assert wrong.misses == 1

    hit = session.submit(["a"])
    assert hit.correct
    assert hit.index == 2
    assert hit.advanced == 2
    assert hit.hits == 1
    assert not hit.completed

    completed = session.submit(["S"])
    assert completed.correct
    assert completed.completed
    assert completed.index == 3


def test_practice_strict_modifier_can_be_relaxed_without_dispatching_input():
    profile = GameProfile(
        id="delta",
        name="Delta",
        pitch_keys=("Z", "X", "C", "V", "B", "N", "M"),
        modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"},
        legacy_keymap=None,
    )
    cues = build_practice_cues(
        [{"notes": ["high_1"], "dur": 1.0}],
        profile=profile,
        use_event_path=True,
    )
    strict = PracticeSession(cues)
    assert not strict.submit(["z"], [], strict_modifiers=True).correct
    relaxed = PracticeSession(cues)
    assert relaxed.submit(["z"], [], strict_modifiers=False).completed
