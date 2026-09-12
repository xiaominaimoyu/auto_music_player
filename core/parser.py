"""简谱解析器:规范化简谱文本 -> 结构化音符序列。

输出协议(与大模型 prompt 一致):
- 音高:1-7 中音;数字+' 高音;数字+, 低音;0 休止
- 时值:无后缀=四分音符(1拍);_ = 八分(0.5拍), __ = 十六分(0.25拍);
  - = 二分(2拍), -- = 全音符(4拍);附点用 . 或 · 跟在时值符号后,时值 ×1.5
- 和弦:[音1 音2 ...]时值后缀,如 [1' 3' 5']- ;和弦内部只写音高
- 小节线 | ‖ 、调号行(1=C)、歌词行自动忽略

输出: [{ "notes": ["high_1"], "dur": 0.5 }, ...]
dur 单位为拍;休止符 notes 为空列表。

两种模式:
- 宽容模式(默认):无效记号静默跳过,与历史行为一致
- collect=True:额外返回错误列表,记录每个无效记号的行号、原文与原因,
  供识别校对页定位问题;音符输出与宽容模式完全一致
同一行内记号按从左到右顺序输出,和弦与单音混排时保持真实顺序。
"""

import re
from dataclasses import dataclass

_PITCH_SUFFIX = {"'": "high", ",": "low"}
_CHORD_RE = re.compile(r"[\[\(]([^\]\)]+)[\]\)]([_\-.·]*)")
_NOTE_RE = re.compile(r"[0-7](?:'|,|\.|·|_|-|#)*")
_TUNE_LINE_RE = re.compile(r"^\s*1\s*=\s*[A-Ga-g]")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class ParseError:
    line: int      # 行号,从 1 起
    token: str     # 原文片段
    reason: str

    def __str__(self):
        return f"第 {self.line} 行:记号 '{self.token}'——{self.reason}"


def _split_pitch_dur(suffix: str):
    """把数字后的修饰符串拆成(音高, 时值部分, 半音标记)。

    . 双语义判定规则:
    - . 单独出现(如 5.) → 低音(三角洲下点约定),rest 清空,dur=1.0
    - . 后跟时值符号(如 5._ 5.-) → 附点,rest 透传给 _parse_dur
    - · 中文圆点始终作为附点符号
    - 已有八度标记时(如 1'.) 消费 . 但保留原八度

    # 可出现在八度标记前或后或 . 之后(如 1#, 1'#, 1,#, 5.#);
    多个 # 不叠加,semitone 恒为 1。
    """
    pitch = "mid"
    semitone = 0
    rest = suffix
    has_octave = False
    if rest.startswith("#"):
        semitone = 1
        rest = rest[1:]
    if rest.startswith(("'", ",")):
        pitch = _PITCH_SUFFIX[rest[0]]
        rest = rest[1:]
        has_octave = True
    if "#" in rest:
        semitone = 1
        rest = rest.replace("#", "")
    if rest == ".":
        if not has_octave:
            pitch = "low"
        rest = ""
    return pitch, rest, semitone


def _parse_dur(rest: str) -> float:
    dur = 1.0
    dotted = False
    for ch in rest:
        if ch == "_":
            dur *= 0.5
        elif ch == "-":
            dur *= 2.0
        elif ch in ".·":
            dotted = True
    if dotted:
        dur *= 1.5
    return dur


def _note_id(num: int, pitch: str) -> str:
    return f"{pitch}_{num}"


def _build_single(token: str):
    """返回 (音符, 问题)。0 一律为休止;休止带八度记号记为问题,不影响输出。"""
    num = int(token[0])
    pitch, rest, semitone = _split_pitch_dur(token[1:])
    problem = None
    if num == 0:
        if token[1:2] in ("'", ","):
            problem = "休止符 0 不应带八度记号"
        return {"notes": [], "dur": _parse_dur(token[1:])}, problem
    dur = _parse_dur(rest)
    item = {"notes": [_note_id(num, pitch)], "dur": dur}
    if semitone:
        item["semitone"] = 1
    return item, problem


def _build_chord(inner: str, suffix: str):
    """返回 (和弦, 无效内部记号列表);空和弦由调用方丢弃。"""
    dur = _parse_dur(suffix)
    note_ids = []
    invalid = []
    has_semitone = False
    for part in re.split(r"[,\s]+", inner.strip()):
        if not part:
            continue
        m = _NOTE_RE.match(part)
        if not m:
            invalid.append(part)
            continue
        token = m.group(0)
        num = int(token[0])
        if num == 0 or not 1 <= num <= 7:
            invalid.append(part)
            continue
        pitch, _, semitone = _split_pitch_dur(token[1:])
        note_ids.append(_note_id(num, pitch))
        if semitone:
            has_semitone = True
    item = {"notes": note_ids, "dur": dur}
    if has_semitone:
        item["semitone"] = 1
    return item, invalid


def _is_lyric_line(line: str) -> bool:
    """歌词行:中文字符明显多于数字,跳过。"""
    chinese = len(_CHINESE_RE.findall(line))
    digits = len(re.findall(r"[0-7]", line))
    return chinese > digits


def _iter_content_lines(text: str):
    """枚举 (行号, 内容行);跳过空行、调号行与歌词行。"""
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if _TUNE_LINE_RE.match(line) or _is_lyric_line(line):
            continue
        line = line.replace("|", " ").replace("‖", " ")
        yield line_no, line


def _parse_line(line: str):
    """解析单行,返回 (音符序列, 问题列表)。问题为 (token, 原因)。"""
    notes = []
    problems = []
    consumed = bytearray(len(line))
    tokens = []  # (起始位置, 原文, 音符 dict 或 None, 问题或 None)

    for cm in _CHORD_RE.finditer(line):
        for j in range(cm.start(), cm.end()):
            consumed[j] = 1
        chord, invalid = _build_chord(cm.group(1), cm.group(2))
        for part in invalid:
            problems.append((part, "和弦内含无效音符"))
        tokens.append((cm.start(), cm.group(0), chord if chord["notes"] else None, None))

    for m in _NOTE_RE.finditer(line):
        if any(consumed[m.start():m.end()]):
            continue  # 和弦跨度内的音符字符
        note, problem = _build_single(m.group(0))
        tokens.append((m.start(), m.group(0), note, problem))
        for j in range(m.start(), m.end()):
            consumed[j] = 1

    tokens.sort(key=lambda t: t[0])
    for _, token, note, problem in tokens:
        if note is not None:
            notes.append(note)
        if problem:
            problems.append((token, problem))

    # 未被任何记号消耗的非空白字符 = 无法识别的内容
    run_start = None
    for j in range(len(line) + 1):
        ch = line[j] if j < len(line) else " "
        if j < len(line) and not consumed[j] and not ch.isspace():
            if run_start is None:
                run_start = j
        elif run_start is not None:
            fragment = line[run_start:j]
            if any(c in "\u2018\u2019\uff0c" for c in fragment):
                problems.append((fragment, "中文标点不被识别为八度标记，请使用英文 ' 或 ,"))
            else:
                problems.append((fragment, "无法识别的内容"))
            run_start = None
    return notes, problems


def parse_jianpu(text: str, *, collect: bool = False):
    """解析规范化简谱文本。

    collect=False(默认):返回音符序列,无效记号静默跳过(宽容,兼容旧行为)。
    collect=True:返回 (音符序列, 错误列表),错误含行号/原文/原因。
    """
    result = []
    errors = []
    for line_no, line in _iter_content_lines(text):
        notes, problems = _parse_line(line)
        result.extend(notes)
        if collect:
            errors.extend(ParseError(line_no, token, reason) for token, reason in problems)
    return (result, errors) if collect else result
