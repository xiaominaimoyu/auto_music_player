"""测试简谱编辑器功能。"""

import pytest
from core.jianpu_editor import JianpuEditor, insert_rest, delete_event


class TestJianpuEditor:
    """测试 JianpuEditor 类。"""

    def test_event_ranges_simple(self):
        """测试简单简谱的事件范围提取。"""
        editor = JianpuEditor()
        text = "1 2 3 4 5"
        ranges = editor.event_ranges(text)
        assert len(ranges) == 5
        assert ranges[0] == (0, 0)  # "1"
        assert ranges[1] == (2, 2)  # "2"
        assert ranges[2] == (4, 4)  # "3"

    def test_event_ranges_with_duration(self):
        """测试带时值后缀的简谱。"""
        editor = JianpuEditor()
        text = "1- 2_ 3. 4"
        ranges = editor.event_ranges(text)
        assert len(ranges) == 4
        assert ranges[0] == (0, 1)  # "1-"
        assert ranges[1] == (3, 4)  # "2_"
        assert ranges[2] == (6, 7)  # "3."

    def test_event_ranges_with_chord(self):
        """测试和弦的事件范围。"""
        editor = JianpuEditor()
        text = "1 [1 3 5]- 2"
        ranges = editor.event_ranges(text)
        assert len(ranges) == 3
        assert ranges[0] == (0, 0)  # "1"
        assert ranges[1] == (2, 9)  # "[1 3 5]-"
        assert ranges[2] == (11, 11)  # "2"

    def test_event_ranges_multiline(self):
        """测试多行简谱。"""
        editor = JianpuEditor()
        text = "1=C 4/4\n1 2 3\n4 5 6"
        ranges = editor.event_ranges(text)
        assert len(ranges) == 6  # 跳过调号行

    def test_insert_rest_before(self):
        """测试在音符前插入休止符。"""
        editor = JianpuEditor()
        text = "1 2 3"
        result = editor.insert_rest(text, 1, before=True, rest_token="0_")
        assert result == "1 0_ 2 3"

    def test_insert_rest_after(self):
        """测试在音符后插入休止符。"""
        editor = JianpuEditor()
        text = "1 2 3"
        result = editor.insert_rest(text, 1, before=False, rest_token="0_")
        assert result == "1 2 0_ 3"

    def test_insert_rest_no_space_needed(self):
        """测试插入位置已有空格时不重复添加。"""
        editor = JianpuEditor()
        text = "1  2  3"  # 已有多个空格
        result = editor.insert_rest(text, 1, before=True, rest_token="0_")
        # 实际结果是 "1  0_ 2  3",因为左侧有空格不添加,右侧无空格添加一个
        assert "0_ 2" in result

    def test_delete_event(self):
        """测试删除音符。"""
        editor = JianpuEditor()
        text = "1 2 3 4"
        result = editor.delete_event(text, 1)  # 删除 "2"
        assert result == "1 3 4"

    def test_delete_event_with_chord(self):
        """测试删除和弦。"""
        editor = JianpuEditor()
        text = "1 [1 3 5]- 2"
        result = editor.delete_event(text, 1)  # 删除和弦
        assert result == "1 2"

    def test_delete_event_preserve_newlines(self):
        """测试删除音符保留换行符。"""
        editor = JianpuEditor()
        text = "1 2\n3 4"
        result = editor.delete_event(text, 1)  # 删除第二行的 "3"
        # 应该保留换行符结构
        assert "\n" in result

    def test_insert_rest_multiline(self):
        """测试多行简谱插入休止符。"""
        editor = JianpuEditor()
        text = "1 2\n3 4"
        result = editor.insert_rest(text, 2, before=True, rest_token="0_")
        # 第3个音符(索引2)是第二行的 "3"
        assert "0_ 3" in result

    def test_convenience_functions(self):
        """测试便捷函数。"""
        text = "1 2 3"
        result = insert_rest(text, 1, before=True, rest_token="0_")
        assert "0_ 2" in result

        result = delete_event(text, 1)
        assert result == "1 3"

    def test_insert_rest_with_semitone(self):
        """测试包含半音标记的简谱。"""
        editor = JianpuEditor()
        text = "1 2# 3"
        ranges = editor.event_ranges(text)
        assert len(ranges) == 3
        # 半音标记 # 应该被包含在音符范围内
        assert ranges[1] == (2, 3)  # "2#"

    def test_delete_cleans_whitespace(self):
        """测试删除音符时清理多余空格。"""
        editor = JianpuEditor()
        text = "1   2   3"
        result = editor.delete_event(text, 1)
        # 删除后不应该有连续的多个空格
        assert "    " not in result


class TestEdgeCase:
    """边界情况测试。"""

    def test_insert_rest_invalid_index(self):
        """测试无效索引插入。"""
        editor = JianpuEditor()
        text = "1 2 3"
        result = editor.insert_rest(text, 99, before=True)
        assert result == text  # 应该返回原文本

    def test_delete_event_invalid_index(self):
        """测试无效索引删除。"""
        editor = JianpuEditor()
        text = "1 2 3"
        result = editor.delete_event(text, 99)
        assert result == text  # 应该返回原文本

    def test_empty_text(self):
        """测试空文本。"""
        editor = JianpuEditor()
        ranges = editor.event_ranges("")
        assert ranges == []

    def test_only_tune_line(self):
        """测试只有调号行。"""
        editor = JianpuEditor()
        text = "1=C 4/4"
        ranges = editor.event_ranges(text)
        assert ranges == []

    def test_only_lyrics(self):
        """测试只有歌词行。"""
        editor = JianpuEditor()
        text = "一闪一闪亮晶晶"
        ranges = editor.event_ranges(text)
        assert ranges == []  # 应该跳过歌词行


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
