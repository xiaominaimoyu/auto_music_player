"""内部音符 IR(中间表示)。

与持久层 `note_id` 的分工(决策 D1「schema 冻结、内部 IR 不冻结」):

    持久层  "high|mid|low_1~7"   与安卓端共享,冻结不动,semitone 不落盘
    本 IR   pitch + octave + semitone 三个独立维度,semitone 从第一天就存在

这样 V2 放开持久层正则时,只需改 `from_storage` / `to_storage` 两处,
编译器与 Player 一行不动——避免同时改 schema + 编译器 + Player 三层。
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

OCTAVE_TO_NAME = {-1: "low", 0: "mid", 1: "high"}
NAME_TO_OCTAVE = {v: k for k, v in OCTAVE_TO_NAME.items()}

MIN_PITCH, MAX_PITCH = 1, 7
MAX_DUR_BEATS = 16.0


@dataclass(frozen=True)
class IRNote:
    """单个音符。

    pitch    1-7(简谱音级,不绑定具体音名——物理定调未知,见调研 §6 待确认 1)
    octave   -1 低八度 / 0 自然 / +1 高八度
    semitone 0 自然 / 1 升半音(鼠标中键)
    dur      时值(拍)
    """

    pitch: int = 1
    octave: int = 0
    semitone: int = 0
    dur: float = 1.0

    def __post_init__(self):
        if not isinstance(self.pitch, int) or isinstance(self.pitch, bool):
            raise ValueError(f"pitch 必须是整数: {self.pitch!r}")
        if not MIN_PITCH <= self.pitch <= MAX_PITCH:
            raise ValueError(f"pitch 必须在 {MIN_PITCH}-{MAX_PITCH}: {self.pitch}")
        if self.octave not in OCTAVE_TO_NAME:
            raise ValueError(f"octave 必须是 -1/0/1: {self.octave}")
        if self.semitone not in (0, 1):
            raise ValueError(f"semitone 必须是 0/1: {self.semitone}")
        if not isinstance(self.dur, (int, float)) or isinstance(self.dur, bool):
            raise ValueError(f"dur 必须是数字: {self.dur!r}")
        if not 0 < float(self.dur) <= MAX_DUR_BEATS:
            raise ValueError(f"dur 必须在 (0, {MAX_DUR_BEATS}]: {self.dur}")

    @property
    def note_id(self) -> str:
        """持久层用的音符 id——**不含 semitone**(D1)。"""
        return f"{OCTAVE_TO_NAME[self.octave]}_{self.pitch}"


@dataclass(frozen=True)
class IRChord:
    """和弦:若干音同时按下,共享一个时值。"""
    notes: tuple = ()
    dur: float = 1.0

    def __post_init__(self):
        if not self.notes:
            raise ValueError("和弦至少包含一个音符")
        if not 0 < float(self.dur) <= MAX_DUR_BEATS:
            raise ValueError(f"dur 必须在 (0, {MAX_DUR_BEATS}]: {self.dur}")


@dataclass(frozen=True)
class IRRest:
    """休止。"""
    dur: float = 1.0

    def __post_init__(self):
        if not 0 < float(self.dur) <= MAX_DUR_BEATS:
            raise ValueError(f"dur 必须在 (0, {MAX_DUR_BEATS}]: {self.dur}")


def _note_from_id(note_id: str, dur: float, semitone: int) -> IRNote:
    if "_" not in note_id:
        raise ValueError(f"无效音符 id: {note_id!r}")
    name, _, num = note_id.partition("_")
    if name not in NAME_TO_OCTAVE:
        raise ValueError(f"无效八度 '{name}'(应为 high/mid/low): {note_id!r}")
    try:
        pitch = int(num)
    except ValueError:
        raise ValueError(f"无效音级 '{num}': {note_id!r}") from None
    return IRNote(pitch=pitch, octave=NAME_TO_OCTAVE[name], semitone=semitone, dur=dur)


def from_storage(items: Sequence[dict], semitones: Iterable[int] | None = None) -> list:
    """存储格式 → IR。

    items     [{"notes": ["mid_1", ...], "dur": 拍数}, ...],空 notes 为休止
    semitones 与 items 等长的 0/1 半音提示(D8 注入通道)。
              None 表示全部为自然音;长度不一致直接报错——绝不静默错位回填。
    """
    if semitones is not None:
        semitones = [int(s) for s in semitones]
        if len(semitones) != len(items):
            raise ValueError(
                f"半音提示长度({len(semitones)})与音符数({len(items)})不一致,拒绝注入"
            )
    out = []
    for i, item in enumerate(items):
        dur = float(item["dur"])
        ids = list(item.get("notes") or [])
        semi = semitones[i] if semitones is not None else 0
        if not ids:
            out.append(IRRest(dur))
        elif len(ids) == 1:
            out.append(_note_from_id(ids[0], dur, semi))
        else:
            out.append(IRChord(tuple(_note_from_id(n, dur, semi) for n in ids), dur))
    return out


def to_storage(elements: Sequence) -> list:
    """IR → 存储格式。semitone **不落盘**(D1)。"""
    out = []
    for el in elements:
        if isinstance(el, IRRest):
            out.append({"notes": [], "dur": el.dur})
        elif isinstance(el, IRNote):
            out.append({"notes": [el.note_id], "dur": el.dur})
        elif isinstance(el, IRChord):
            out.append({"notes": [n.note_id for n in el.notes], "dur": el.dur})
        else:
            raise TypeError(f"未知 IR 元素类型: {type(el)!r}")
    return out
