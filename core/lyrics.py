"""把 AI 返回的逐行歌词安全对齐到解析后的音符事件。"""

import re

from core.parser import parse_jianpu

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokens(line: str) -> list[str]:
    tokens = [part for part in re.split(r"\s+", str(line or "").strip()) if part]
    if len(tokens) == 1 and len(_CHINESE_RE.findall(tokens[0])) > 1:
        tokens = list(tokens[0])
    return tokens


def align_lyrics(jianpu: str, lyrics_lines) -> list[str]:
    """逐谱行对齐；数量不匹配的行全部留空，避免歌词串位。"""
    score_lines = [line for line in str(jianpu or "").splitlines() if line.strip()]
    source_lines = list(lyrics_lines or ())
    all_events = [parse_jianpu(line) for line in score_lines]

    # 应用保存的结构化文本使用一条全局歌词行；数量完全匹配时可安全恢复。
    if len(source_lines) == 1 and len(score_lines) > 1:
        tokens = _tokens(source_lines[0])
        flat_events = [event for events in all_events for event in events]
        sounding_count = sum(bool(event["notes"]) for event in flat_events)
        if len(tokens) == len(flat_events):
            return ["" if token == "_" else token for token in tokens]
        if len(tokens) == sounding_count:
            token_index = 0
            aligned = []
            for event in flat_events:
                if event["notes"]:
                    token = tokens[token_index]
                    aligned.append("" if token == "_" else token)
                    token_index += 1
                else:
                    aligned.append("")
            return aligned

    aligned = []
    for index, score_line in enumerate(score_lines):
        events = all_events[index]
        tokens = _tokens(source_lines[index] if index < len(source_lines) else "")
        sounding_count = sum(bool(event["notes"]) for event in events)
        if len(tokens) == len(events):
            aligned.extend("" if token == "_" else token for token in tokens)
        elif len(tokens) == sounding_count:
            token_index = 0
            for event in events:
                if event["notes"]:
                    token = tokens[token_index]
                    aligned.append("" if token == "_" else token)
                    token_index += 1
                else:
                    aligned.append("")
        else:
            aligned.extend("" for _ in events)
    return aligned
