"""对规范化简谱执行按解析音符索引定位的文本编辑。"""

import re

_CHORD_RE = re.compile(r"[\[\(]([^\]\)]+)[\]\)]([_\-.·]*)")
_NOTE_RE = re.compile(r"[0-7](?:'|,|\.|·|_|-)*")
_TUNE_LINE_RE = re.compile(r"^\s*1\s*=\s*[A-Ga-g]")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_lyric_line(line: str) -> bool:
    return len(_CHINESE_RE.findall(line)) > len(re.findall(r"[0-7]", line))


def _event_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        if line.strip() and not _TUNE_LINE_RE.match(line.strip()) and not _is_lyric_line(line.strip()):
            occupied = bytearray(len(line))
            local = []
            for match in _CHORD_RE.finditer(line):
                occupied[match.start():match.end()] = b"\x01" * (match.end() - match.start())
                local.append((match.start(), match.end()))
            for match in _NOTE_RE.finditer(line):
                if not any(occupied[match.start():match.end()]):
                    local.append((match.start(), match.end()))
            ranges.extend((offset + start, offset + end) for start, end in sorted(local))
        offset += len(raw)
    return ranges


def insert_rest_at_event(text: str, event_index: int, *, before: bool, rest_token: str = "0_") -> str:
    """在第 ``event_index`` 个音符前/后插入休止，并保留原有换行。"""
    value = str(text or "")
    ranges = _event_ranges(value)
    if not 0 <= event_index < len(ranges):
        return value
    start, end = ranges[event_index]
    position = start if before else end
    left_space = " " if position > 0 and not value[position - 1].isspace() else ""
    right_space = " " if position < len(value) and not value[position].isspace() else ""
    return value[:position] + left_space + rest_token + right_space + value[position:]


def delete_event(text: str, event_index: int) -> str:
    """删除指定音符事件，并清理相邻的水平空白。"""
    value = str(text or "")
    ranges = _event_ranges(value)
    if not 0 <= event_index < len(ranges):
        return value
    start, end = ranges[event_index]
    delete_start, delete_end = start, end
    while delete_end < len(value) and value[delete_end] in " \t":
        delete_end += 1
    if delete_end == end:
        while delete_start > 0 and value[delete_start - 1] in " \t":
            delete_start -= 1
    return value[:delete_start] + value[delete_end:]
