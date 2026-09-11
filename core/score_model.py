"""统一曲谱数据模型与 Schema 校验器。

存储格式(与数据库 notes_json 及安卓端一致,向后兼容):
    [{"notes": ["mid_1", ...], "dur": 拍数}, ...]
    - notes 空列表 = 休止;dur 以拍为单位(四分音符 = 1 拍)

类型化模型:
    Note(音高 1-7, 八度 high/mid/low, 时值拍)
    Chord(若干 Note 同时按下, 时值)
    Rest(休止)
    Score(名称, BPM, 音符序列)
附点与时值后缀已折算进 dur(如 0.75 = 附点八分),不单独存储。

校验器:入库前对存储格式数据做结构/取值校验,
错误带音符序号(第 N 个)与可读原因,保证非法数据无法写入乐谱库。
"""

import math
import re
from dataclasses import dataclass, field

_NOTE_ID_RE = re.compile(r"^(high|mid|low)_([1-7])$")
MIN_BPM, MAX_BPM = 1, 300
MAX_DUR_BEATS = 16.0


@dataclass(frozen=True)
class Note:
    num: int        # 音高 1-7
    octave: str     # high / mid / low
    dur: float = 1.0
    semitone: int = 0   # 0 自然 / 1 升半音

    @property
    def note_id(self) -> str:
        return f"{self.octave}_{self.num}"


@dataclass(frozen=True)
class Chord:
    notes: tuple
    dur: float = 1.0


@dataclass(frozen=True)
class Rest:
    dur: float = 1.0


@dataclass
class Score:
    name: str
    bpm: int = 100
    elements: list = field(default_factory=list)  # list[Note | Chord | Rest]

    @classmethod
    def from_storage(cls, name: str, bpm: int, items: list) -> "Score":
        """从存储格式构建;音符 id 非法时抛 ValueError。"""
        elements = []
        for item in items:
            dur = float(item["dur"])
            semitone = int(item.get("semitone", 0))
            notes = tuple(_note_from_id(nid, dur, semitone) for nid in item.get("notes") or [])
            if not notes:
                elements.append(Rest(dur))
            elif len(notes) == 1:
                elements.append(notes[0])
            else:
                elements.append(Chord(notes, dur))
        return cls(name=name, bpm=int(bpm), elements=elements)

    def to_storage(self) -> list:
        out = []
        for el in self.elements:
            if isinstance(el, Rest):
                out.append({"notes": [], "dur": el.dur})
            elif isinstance(el, Note):
                item = {"notes": [el.note_id], "dur": el.dur}
                if el.semitone:
                    item["semitone"] = el.semitone
                out.append(item)
            elif isinstance(el, Chord):
                item = {"notes": [n.note_id for n in el.notes], "dur": el.dur}
                if any(n.semitone for n in el.notes):
                    item["semitone"] = 1
                out.append(item)
            else:
                raise TypeError(f"未知音符类型: {type(el)!r}")
        return out

    @property
    def total_beats(self) -> float:
        return sum(el.dur for el in self.elements)


def _note_from_id(note_id: str, dur: float, semitone: int = 0) -> Note:
    m = _NOTE_ID_RE.match(note_id)
    if not m:
        raise ValueError(f"无效音符: {note_id}(应为 high/mid/low_1~7)")
    return Note(num=int(m.group(2)), octave=m.group(1), dur=dur, semitone=semitone)


@dataclass
class ValidationError:
    index: int      # 音符序号(从 0 计);-1 = 曲目级错误(如 BPM)
    message: str

    def __str__(self):
        return self.message if self.index < 0 else f"音符 {self.index + 1}: {self.message}"


class ScoreValidationError(ValueError):
    """校验失败。errors 为全部错误;message 为可读摘要(前 5 条)。"""

    def __init__(self, errors):
        self.errors = errors
        summary = "\n".join(str(e) for e in errors[:5])
        if len(errors) > 5:
            summary += f"\n... 共 {len(errors)} 个错误"
        super().__init__(summary)


def validate_notes(notes, *, bpm=None) -> list:
    """校验存储格式曲谱数据,返回全部错误(空列表 = 合法)。"""
    if not isinstance(notes, list):
        return [ValidationError(-1, "曲谱数据必须为列表")]
    errors = []
    for i, item in enumerate(notes):
        errors.extend(_validate_item(i, item))
    if bpm is not None and (
        isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not MIN_BPM <= bpm <= MAX_BPM
    ):
        errors.append(ValidationError(-1, f"BPM 必须在 {MIN_BPM}-{MAX_BPM} 之间"))
    return errors


def require_valid(notes, *, bpm=None):
    """校验失败时抛 ScoreValidationError(用于入库强制校验)。"""
    errors = validate_notes(notes, bpm=bpm)
    if errors:
        raise ScoreValidationError(errors)


def _validate_item(i: int, item) -> list:
    if not isinstance(item, dict):
        return [ValidationError(i, "音符数据必须为对象")]
    errors = []
    ids = item.get("notes")
    if isinstance(ids, list):
        for nid in ids:
            if not isinstance(nid, str) or not _NOTE_ID_RE.match(nid):
                errors.append(ValidationError(i, f"音符无效 '{nid}'(应为 high/mid/low_1~7)"))
    else:
        errors.append(ValidationError(i, "notes 必须为列表(空列表 = 休止)"))
    if "dur" not in item:
        errors.append(ValidationError(i, "缺少时值字段 dur"))
        return errors
    dur = item["dur"]
    if isinstance(dur, bool) or not isinstance(dur, (int, float)):
        errors.append(ValidationError(i, "时值不是有效数字"))
        return errors
    if not math.isfinite(dur):
        errors.append(ValidationError(i, "时值必须为有限数字"))
    elif dur <= 0:
        errors.append(ValidationError(i, "时值必须大于 0"))
    elif dur > MAX_DUR_BEATS:
        errors.append(ValidationError(i, f"时值超出上限(最大 {MAX_DUR_BEATS:g} 拍)"))
    if "semitone" in item:
        semi = item["semitone"]
        if isinstance(semi, bool) or not isinstance(semi, (int, float)) or int(semi) not in (0, 1):
            errors.append(ValidationError(i, "半音标记必须为 0 或 1"))
    return errors
