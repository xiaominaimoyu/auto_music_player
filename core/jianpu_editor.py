"""简谱编辑器:提供简谱文本的编辑功能。

当前支持:
- insert_rest:在指定事件位置插入休止符(间隔)

实现参考自 APK 中的 JianpuEditor.kt
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


# 正则表达式定义(与 APK 中一致)
_CHORD_RE = re.compile(r"[\[\(]([^\]\)]+)[\]\)]([_\-.·]*)")
_NOTE_RE = re.compile(r"[0-7](?:'|,|\.|·|_|-|#)*")
_TUNE_LINE_RE = re.compile(r"^\s*1\s*=\s*[A-Ga-g]")
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class JianpuEditor:
    """简谱编辑器,提供简谱文本的编辑功能。"""

    def event_ranges(self, text: str) -> List[Tuple[int, int]]:
        """获取简谱文本中所有事件(音符、和弦)的范围。

        Args:
            text: 简谱文本

        Returns:
            事件范围列表,每个元素为 (start, end) 表示事件在文本中的起止位置
        """
        result: List[Tuple[int, int]] = []
        line_start = 0

        # 按换行符分割文本,逐行处理
        for line_with_break in re.split(r"(?<=\n)", text):
            line = line_with_break.rstrip("\n").rstrip("\r")
            trimmed = line.strip()

            # 跳过空行、调号行、歌词行
            if not trimmed:
                continue
            if _TUNE_LINE_RE.match(trimmed):
                continue
            if self._is_lyric_line(trimmed):
                continue

            # 标记已占用的位置
            occupied = [False] * len(line)
            local: List[Tuple[int, int]] = []

            # 匹配和弦,标记其占用的位置
            for match in _CHORD_RE.finditer(line):
                start, end = match.start(), match.end() - 1  # 转换为包含结束位置
                for i in range(start, end + 1):
                    if i < len(occupied):
                        occupied[i] = True
                local.append((start, end))

            # 匹配音符,获取音符的范围(排除和弦内部的音符)
            for match in _NOTE_RE.finditer(line):
                start, end = match.start(), match.end() - 1  # 转换为包含结束位置
                # 检查是否在和弦范围内
                if any(s <= start <= e for s, e in local):
                    continue
                local.append((start, end))

            # 按起始位置排序
            local.sort(key=lambda r: r[0])

            # 将本地范围转换为全局范围
            for start, end in local:
                result.append((line_start + start, line_start + end))

            line_start += len(line_with_break)

        return result

    def _is_lyric_line(self, line: str) -> bool:
        """判断是否为歌词行:中文字符明显多于数字。"""
        chinese = len(_CHINESE_RE.findall(line))
        digits = len(re.findall(r"[0-7]", line))
        return chinese > digits

    def insert_rest(
        self,
        text: str,
        event_index: int,
        before: bool = True,
        rest_token: str = "0_",
    ) -> str:
        """在简谱文本的指定事件位置插入休止符(间隔)。

        Args:
            text: 原始简谱文本
            event_index: 要插入位置的事件索引(音符或休止符的索引)
            before: 是否在指定事件之前插入(True=之前, False=之后)
            rest_token: 要插入的休止符令牌(默认为 "0_", 表示四分休止符)

        Returns:
            插入休止符后的新简谱文本
        """
        ranges = self.event_ranges(text)

        # 检查 event_index 是否有效
        if not (0 <= event_index < len(ranges)):
            return text

        # 获取指定事件的范围
        start, end = ranges[event_index]

        # 确定插入位置
        if before:
            position = start
        else:
            position = end + 1

        # 检查插入位置左右是否需要空格
        left_needs_space = position > 0 and not text[position - 1].isspace()
        right_needs_space = (
            (position < len(text) and not text[position].isspace())
            if position < len(text)
            else False
        )

        # 构建插入字符串
        insertion = ""
        if left_needs_space:
            insertion += " "
        insertion += rest_token
        if right_needs_space:
            insertion += " "

        # 插入休止符
        return text[:position] + insertion + text[position:]

    def delete_event(self, text: str, event_index: int) -> str:
        """删除简谱文本中指定的音符事件,并清理相邻的水平空白。

        Args:
            text: 原始简谱文本
            event_index: 要删除的事件索引

        Returns:
            删除音符后的新简谱文本
        """
        ranges = self.event_ranges(text)

        # 检查 event_index 是否有效
        if not (0 <= event_index < len(ranges)):
            return text

        # 获取指定事件的范围
        start, end = ranges[event_index]
        end += 1  # 转为不包含结束位置
        original_end = end

        # 尝试向右清理空格/制表符
        while end < len(text) and text[end] in " \t":
            end += 1

        # 如果右侧没有清理到空白,则向左清理
        if end == original_end:
            while start > 0 and text[start - 1] in " \t":
                start -= 1

        # 删除事件和相邻空白
        return text[:start] + text[end:]


# 便捷函数
def insert_rest(
    text: str,
    event_index: int,
    before: bool = True,
    rest_token: str = "0_",
) -> str:
    """在简谱文本的指定事件位置插入休止符(间隔)。

    这是 JianpuEditor.insert_rest 的便捷封装。

    Args:
        text: 原始简谱文本
        event_index: 要插入位置的事件索引
        before: 是否在指定事件之前插入
        rest_token: 要插入的休止符令牌

    Returns:
        插入休止符后的新简谱文本
    """
    editor = JianpuEditor()
    return editor.insert_rest(text, event_index, before, rest_token)


def delete_event(text: str, event_index: int) -> str:
    """删除简谱文本中指定的音符事件。

    这是 JianpuEditor.delete_event 的便捷封装。

    Args:
        text: 原始简谱文本
        event_index: 要删除的事件索引

    Returns:
        删除音符后的新简谱文本
    """
    editor = JianpuEditor()
    return editor.delete_event(text, event_index)
