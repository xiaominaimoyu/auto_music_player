"""Practice-mode helpers: dual-rail hints and step recording.

The module is deliberately GUI-free. Playback and real input stay in
Player/EventPlayer; practice helpers only describe what would happen and build
normal score-storage rows for later validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core import ir as ir_mod
from core.compiler import compile_score
from core.score_model import require_valid


@dataclass(frozen=True)
class DualRailHint:
    index: int
    total: int
    score_text: str
    input_text: str
    degradation_text: str = ""


@dataclass(frozen=True)
class PracticeCue:
    index: int
    keyboard: frozenset[str]
    mouse: frozenset[str]
    score_text: str
    input_text: str
    degradation_text: str = ""

    @property
    def requires_input(self) -> bool:
        return bool(self.keyboard or self.mouse)


@dataclass(frozen=True)
class PracticeResult:
    correct: bool
    advanced: int
    index: int
    total: int
    hits: int
    misses: int
    completed: bool
    message: str


def _fmt_beats(value) -> str:
    text = f"{float(value):g}"
    return f"{text} 拍"


def format_score_note(item: dict) -> str:
    dur = _fmt_beats(item.get("dur", 0))
    notes = list(item.get("notes") or [])
    if not notes:
        return f"休止 · {dur}"
    suffix = "#" if int(item.get("semitone", 0) or 0) else ""
    return f"{' + '.join(n + suffix for n in notes)} · {dur}"


def _legacy_input_hint(item: dict, keymap) -> str:
    notes = list(item.get("notes") or [])
    if not notes:
        return "不发送输入"
    keys = []
    for note_id in notes:
        key = keymap.key_for(note_id) if keymap is not None else None
        keys.append(key or "?")
    hint = " + ".join(keys)
    if int(item.get("semitone", 0) or 0):
        hint += " · 半音标记不会由旧 21 键档位发送"
    return hint


def _event_input_hint(
    item: dict,
    profile,
    *,
    bpm: int,
    settle_ms: float,
    release_settle_ms: float,
    hold_ratio: float,
    gap_ms: float,
) -> tuple[str, str]:
    elements = ir_mod.from_storage([item])
    params = profile.build_compile_params(
        bpm=bpm,
        settle_ms=settle_ms,
        release_settle_ms=release_settle_ms,
        hold_ratio=hold_ratio,
        max_hold_ms=None,
        gap_ms=gap_ms,
    )
    result = compile_score(elements, params)
    downs = [e for e in result.events if e.action == "down"]
    if not downs:
        input_text = "不发送输入"
    else:
        parts = []
        for e in downs:
            prefix = "鼠标" if e.device == "mouse" else "键盘"
            parts.append(f"{prefix}:{e.key}")
        input_text = " + ".join(parts)
    degradation_text = "; ".join(str(d) for d in result.degradations)
    return input_text, degradation_text


def build_dual_rail_hint(
    notes: Sequence[dict],
    index: int,
    *,
    keymap=None,
    profile=None,
    use_event_path: bool = False,
    bpm: int = 100,
    settle_ms: float = 30.0,
    release_settle_ms: float = 20.0,
    hold_ratio: float = 0.75,
    gap_ms: float = 20.0,
) -> DualRailHint:
    total = len(notes)
    if total <= 0:
        return DualRailHint(0, 0, "乐谱轨: 无乐谱", "输入轨: 无输入")
    safe_index = max(0, min(int(index), total))
    if safe_index >= total:
        return DualRailHint(total, total, "乐谱轨: 已完成", "输入轨: 已完成")

    item = notes[safe_index]
    score_text = f"乐谱轨 {safe_index + 1}/{total}: {format_score_note(item)}"
    if use_event_path and profile is not None:
        input_text, degradation_text = _event_input_hint(
            item,
            profile,
            bpm=bpm,
            settle_ms=settle_ms,
            release_settle_ms=release_settle_ms,
            hold_ratio=hold_ratio,
            gap_ms=gap_ms,
        )
    else:
        input_text = _legacy_input_hint(item, keymap)
        degradation_text = ""
    return DualRailHint(
        safe_index,
        total,
        score_text,
        f"输入轨: {input_text}",
        degradation_text,
    )


def _normalize_keyboard(value) -> str:
    text = str(value or "").strip()
    return text.upper() if len(text) == 1 else text.lower()


def _normalize_mouse(value) -> str:
    return str(value or "").strip().lower()


def build_practice_cues(
    notes: Sequence[dict],
    *,
    keymap=None,
    profile=None,
    use_event_path: bool = False,
    bpm: int = 100,
    settle_ms: float = 30.0,
    release_settle_ms: float = 20.0,
    hold_ratio: float = 0.75,
    gap_ms: float = 20.0,
) -> list[PracticeCue]:
    """Compile expected user inputs without invoking either playback engine."""

    require_valid(list(notes), bpm=bpm)
    cues = []
    for index, item in enumerate(notes):
        hint = build_dual_rail_hint(
            notes,
            index,
            keymap=keymap,
            profile=profile,
            use_event_path=use_event_path,
            bpm=bpm,
            settle_ms=settle_ms,
            release_settle_ms=release_settle_ms,
            hold_ratio=hold_ratio,
            gap_ms=gap_ms,
        )
        keyboard = set()
        mouse = set()
        if item.get("notes"):
            if use_event_path and profile is not None:
                elements = ir_mod.from_storage([item])
                params = profile.build_compile_params(
                    bpm=bpm,
                    settle_ms=settle_ms,
                    release_settle_ms=release_settle_ms,
                    hold_ratio=hold_ratio,
                    max_hold_ms=None,
                    gap_ms=gap_ms,
                )
                compiled = compile_score(elements, params)
                for event in compiled.events:
                    if event.action != "down":
                        continue
                    if event.device == "kb":
                        keyboard.add(_normalize_keyboard(event.key))
                    elif event.device == "mouse":
                        mouse.add(_normalize_mouse(event.key))
            else:
                for note_id in item.get("notes") or []:
                    key = keymap.key_for(note_id) if keymap is not None else None
                    if key:
                        keyboard.add(_normalize_keyboard(key))
        cues.append(
            PracticeCue(
                index=index,
                keyboard=frozenset(keyboard),
                mouse=frozenset(mouse),
                score_text=hint.score_text,
                input_text=hint.input_text,
                degradation_text=hint.degradation_text,
            )
        )
    return cues


class PracticeSession:
    """Correct-input-driven practice cursor with hit/miss accounting.

    Rest or fully degraded rows advance automatically. The class is pure and
    deliberately has no dependency on KeyboardDriver, Player or EventPlayer.
    """

    def __init__(self, cues: Sequence[PracticeCue], start_index: int = 0):
        self.cues = list(cues)
        if not 0 <= int(start_index) <= len(self.cues):
            raise ValueError("练习起点超出乐谱范围")
        self.index = int(start_index)
        self.hits = 0
        self.misses = 0
        self._skip_non_input()

    @property
    def total(self) -> int:
        return len(self.cues)

    @property
    def completed(self) -> bool:
        return self.index >= self.total

    @property
    def current(self) -> PracticeCue | None:
        return None if self.completed else self.cues[self.index]

    def _skip_non_input(self) -> int:
        start = self.index
        while self.index < self.total and not self.cues[self.index].requires_input:
            self.index += 1
        return self.index - start

    def submit(self, keyboard, mouse=(), *, strict_modifiers: bool = True) -> PracticeResult:
        if self.completed:
            return PracticeResult(
                False, 0, self.index, self.total, self.hits, self.misses, True, "练习已完成"
            )
        cue = self.cues[self.index]
        actual_keyboard = frozenset(_normalize_keyboard(key) for key in keyboard if key)
        actual_mouse = frozenset(_normalize_mouse(button) for button in mouse if button)
        key_ok = actual_keyboard == cue.keyboard
        mouse_ok = actual_mouse == cue.mouse if strict_modifiers else True
        if not (key_ok and mouse_ok):
            self.misses += 1
            return PracticeResult(
                False,
                0,
                self.index,
                self.total,
                self.hits,
                self.misses,
                False,
                "按键或修饰键与当前提示不一致",
            )

        self.hits += 1
        self.index += 1
        advanced = 1 + self._skip_non_input()
        return PracticeResult(
            True,
            advanced,
            self.index,
            self.total,
            self.hits,
            self.misses,
            self.completed,
            "命中" if not self.completed else "练习完成",
        )


class StepRecorder:
    """Build score rows by tapping note/rest buttons at a selected duration."""

    def __init__(self, items: Sequence[dict] | None = None):
        self._items = [dict(item) for item in (items or [])]
        if self._items:
            require_valid(self._items)

    @property
    def items(self) -> list[dict]:
        return [dict(item) for item in self._items]

    def append_note(self, note_id: str, dur: float = 1.0, *, semitone: int = 0):
        item = {"notes": [note_id], "dur": float(dur)}
        if semitone:
            item["semitone"] = 1
        require_valid([item])
        self._items.append(item)
        return dict(item)

    def append_rest(self, dur: float = 1.0):
        item = {"notes": [], "dur": float(dur)}
        require_valid([item])
        self._items.append(item)
        return dict(item)

    def delete_last(self):
        if not self._items:
            return None
        return self._items.pop()

    def clear(self):
        self._items.clear()
