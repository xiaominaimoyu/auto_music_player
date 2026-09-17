"""演奏日志前端展示测试(按 T1–T22 编号组织,覆盖点见 design.md §3.1)。

T1      M1  常量模块导入与关键常量校验(无 Qt)
T2–T4   M2  六类事件行格式化 / 字段缺失占位 / 边界值(无 Qt)
T5–T9   M3  空目录 / 列表与异常标记 / 旧版日志 / 损坏文件 / 半行截断(sync_mode)
T10–T11 M3  删除:确认取消与执行 / 部分失败(sync_mode)
T12–T13 M3  导出:成功与失败 / 多选约束(sync_mode)
T14     M3  自动清理:过期清理提示 / 无过期无提示(sync_mode)
T15     M3  worker 异常注入 → 错误提示不崩溃(sync_mode)
T16–T19 M4  完整会话摘要与时间线 / 头尾缺失 / 类型过滤 / 分批渲染
T20     M4  back_requested 信号 + 返回后列表状态保持联动
T21     M5  真实 MainWindow 导航接线(NAV_ITEMS/stack/默认页/_go_play)

GUI 用例需 QApplication:全量回归请使用 tests/_run_offscreen_tests.py
(设置 QT_QPA_PLATFORM=offscreen 后 discover)。
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_app = None


def ensure_qapp():
    global _app
    if _app is None:
        from PyQt6.QtWidgets import QApplication

        _app = QApplication.instance() or QApplication([])
    return _app


# ==================== 假日志构造工具 ====================


def write_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def make_session_events(note_count=3, with_end=True, with_start=True):
    events = []
    if with_start:
        events.append(
            {
                "type": "session_start",
                "timestamp": 1789300000.0,
                "session_id": "abcd1234efgh5678",
                "score_name": "测试曲",
                "bpm": 120,
                "note_count": note_count,
                "started_at": "2026-09-13T10:00:00",
            }
        )
    for i in range(note_count):
        events.append(
            {
                "type": "note",
                "timestamp": 1789300000.0 + i,
                "index": i,
                "notes": ["C4"],
                "keys": ["A"],
                "deviation_ms": -12.3,
                "success": True,
            }
        )
    if with_end:
        events.append(
            {
                "type": "session_end",
                "timestamp": 1789300000.0 + note_count,
                "completed": note_count,
                "total": note_count,
                "stopped_early": False,
                "events_count": len(events) + 1,
                "ended_at": "2026-09-13T10:01:00",
            }
        )
    return events


class _TempDirsMixin:
    """tempfile 临时目录隔离,不污染真实 data/。"""

    def setUp_dirs(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_dir = os.path.join(self._tmp.name, "play_logs")
        self.export_dir = os.path.join(self._tmp.name, "exports")
        os.makedirs(self.log_dir, exist_ok=True)


class _DialogStubMixin:
    """AppDialog 全静态方法打桩(模态对话框会阻塞测试);记录调用供断言。"""

    def stub_dialogs(self):
        self._calls = []
        self._confirm_result = False

        def _mk(name):
            return staticmethod(
                lambda parent, title, message, _n=name: self._calls.append((_n, title, message))
            )

        confirm_stub = staticmethod(
            lambda parent, title, message: self._confirm_result
        )
        for name, stub in (
            ("show_info", _mk("info")),
            ("show_success", _mk("success")),
            ("show_warning", _mk("warning")),
            ("show_error", _mk("error")),
            ("confirm", confirm_stub),
        ):
            patcher = mock.patch.object(
                __import__("gui.widgets", fromlist=["AppDialog"]).AppDialog, name, stub
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def calls_of(self, name):
        return [c for c in self._calls if c[0] == name]


# ==================== T1:文案常量组 ====================


class TestT1LogTexts(unittest.TestCase):
    def test_module_import_and_constants(self):
        from gui import log_texts as tx

        self.assertEqual(tx.NAV_LOG_TEXT, "演奏记录")
        self.assertTrue(tx.LOG_PAGE_TITLE)
        self.assertTrue(tx.LOG_PAGE_SUBTITLE)
        self.assertEqual(
            tx.LOG_TABLE_HEADERS,
            ["曲名", "开始时间", "时长", "完成度", "异常", "大小"],
        )
        self.assertEqual(tx.LOG_EMPTY_TEXT, "暂无演奏记录")
        self.assertTrue(tx.LOG_LOADING_TEXT)
        for btn in (
            tx.BTN_VIEW_DETAIL_TEXT,
            tx.BTN_EXPORT_TEXT,
            tx.BTN_DELETE_TEXT,
            tx.BTN_BACK_TEXT,
            tx.BTN_LOAD_MORE_TEXT,
        ):
            self.assertTrue(btn)
        self.assertEqual(tx.EXPORT_FORMAT_LABEL, "导出格式")
        self.assertEqual(
            [v for v, _d in tx.EXPORT_FORMAT_ITEMS], ["jsonl", "csv", "markdown"]
        )
        self.assertEqual(tx.EXPORT_ALL_FILTER_TEXT, "全部")
        self.assertTrue(tx.DELETE_CONFIRM_TITLE)
        self.assertIn("{n}", tx.DELETE_CONFIRM_BODY)
        self.assertIn("{n}", tx.DELETE_DONE_TEXT)
        self.assertIn("{n}", tx.DELETE_PARTIAL_TEXT)
        self.assertIn("{m}", tx.DELETE_PARTIAL_TEXT)
        self.assertIn("{path}", tx.EXPORT_SUCCESS_TEXT)
        self.assertTrue(tx.EXPORT_FAIL_TEXT)
        self.assertTrue(tx.EXPORT_SELECT_ONE_TEXT)
        self.assertIn("{n}", tx.CLEANUP_DONE_TEXT)
        self.assertIn("{m}", tx.CLEANUP_DONE_TEXT)
        self.assertEqual(tx.PLACEHOLDER_UNKNOWN, "未知")

    def test_event_type_labels_exact_six(self):
        from gui import log_texts as tx

        self.assertEqual(
            set(tx.EVENT_TYPE_LABELS.keys()),
            {"note", "focus", "control", "external_input", "modifier_anomaly", "env"},
        )
        for label in tx.EVENT_TYPE_LABELS.values():
            self.assertTrue(label)

    def test_event_row_templates_cover_all(self):
        from gui import log_texts as tx

        for key in (
            "note",
            "focus",
            "control",
            "external_input",
            "modifier_anomaly",
            "env",
            "session_start",
            "session_end",
        ):
            self.assertIn(key, tx.EVENT_ROW_TEMPLATES)


# ==================== T2–T4:格式化纯函数 ====================


class TestT2EventRows(unittest.TestCase):
    def test_six_event_rows_and_session_lines(self):
        from gui.log_formatting import format_event_row, format_row_time

        ts = 1789300000.5
        t = format_row_time(ts)
        cases = [
            (
                {"type": "note", "timestamp": ts, "notes": ["C4"], "keys": ["A"],
                 "deviation_ms": -12.3, "success": True},
                f"{t} [音符演奏] C4 / 键A 偏差-12.3ms 成功",
            ),
            (
                {"type": "focus", "timestamp": ts, "event": "lost", "window_title": "游戏窗口"},
                f"{t} [焦点变化] 失焦：游戏窗口",
            ),
            (
                {"type": "focus", "timestamp": ts, "event": "regained"},
                f"{t} [焦点变化] 回焦",
            ),
            (
                {"type": "control", "timestamp": ts, "action": "start", "params": {"bpm": 120}},
                f"{t} [控制操作] start {{'bpm': 120}}",
            ),
            (
                {"type": "control", "timestamp": ts, "action": "stop", "params": {}},
                f"{t} [控制操作] stop",
            ),
            (
                {"type": "external_input", "timestamp": ts, "vk_code": 65, "source": "user"},
                f"{t} [外部输入] VK 65（来源：用户）",
            ),
            (
                {"type": "external_input", "timestamp": ts, "vk_code": 88, "source": "injected"},
                f"{t} [外部输入] VK 88（来源：注入）",
            ),
            (
                {"type": "modifier_anomaly", "timestamp": ts, "modifier": "LButton",
                 "expected": False, "actual": True},
                f"{t} [修饰键异常] LButton 预期=False 实际=True",
            ),
            (
                {"type": "env", "timestamp": ts, "event": "lock_screen", "details": {"a": 1}},
                f"{t} [环境事件] lock_screen {{'a': 1}}",
            ),
            (
                {"type": "env", "timestamp": ts, "event": "show_desktop", "details": {}},
                f"{t} [环境事件] show_desktop",
            ),
        ]
        for event, expected in cases:
            self.assertEqual(format_event_row(event), expected)

        row_start = format_event_row(
            {"type": "session_start", "timestamp": ts, "score_name": "测试曲",
             "bpm": 120, "note_count": 3}
        )
        self.assertEqual(row_start, f"{t} [会话开始] 测试曲（BPM 120，音符 3）")
        row_end = format_event_row(
            {"type": "session_end", "timestamp": ts, "completed": 3, "total": 3,
             "events_count": 5}
        )
        self.assertEqual(row_end, f"{t} [会话结束] 完成 3/3，事件 5")


class TestT3MissingFields(unittest.TestCase):
    def test_missing_fields_yield_placeholder_no_raise(self):
        from gui.log_formatting import format_event_row

        self.assertTrue(format_event_row({}).endswith("[未知事件]"))
        self.assertIn("[未知事件]", format_event_row({"type": "mystery", "timestamp": 0}))
        self.assertIn("未知", format_event_row({"type": "note", "timestamp": 0}))
        self.assertIn("未知", format_event_row({"type": "focus", "timestamp": 0}))
        self.assertIn("未知", format_event_row({"type": "control", "timestamp": 0}))
        self.assertIn("未知", format_event_row({"type": "external_input", "timestamp": 0}))
        self.assertIn("未知", format_event_row({"type": "modifier_anomaly", "timestamp": 0}))
        self.assertIn("未知", format_event_row({"type": "env", "timestamp": 0}))
        # 非 dict 输入不抛出
        self.assertIn("[未知事件]", format_event_row(None))
        self.assertIn("[未知事件]", format_event_row("garbage"))

    def test_empty_lists_use_placeholder(self):
        from gui.log_formatting import format_event_row

        row = format_event_row(
            {"type": "note", "timestamp": 1789300000.0, "notes": [], "keys": [],
             "deviation_ms": 0.0, "success": True}
        )
        self.assertIn("未知", row)


class TestT4FormattingBounds(unittest.TestCase):
    def test_row_time_bounds(self):
        import re

        from gui.log_formatting import format_row_time

        self.assertRegex(format_row_time(0), r"^\d{2}:\d{2}:\d{2}$")
        self.assertEqual(format_row_time(None), "未知")
        self.assertEqual(format_row_time("abc"), "未知")

    def test_file_size_bounds(self):
        from gui.log_formatting import format_file_size

        self.assertEqual(format_file_size(0), "0 B")
        self.assertEqual(format_file_size(1023), "1023 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_file_size(1536), "1.5 KB")
        self.assertEqual(format_file_size(-5), "未知")
        self.assertEqual(format_file_size(None), "未知")

    def test_completion_and_anomaly(self):
        from gui.log_formatting import format_anomaly_flag, format_completion

        self.assertEqual(format_completion(18, 20), "18/20")
        self.assertEqual(format_completion(0, 0), "未知")
        self.assertEqual(format_anomaly_flag(0), "")
        self.assertEqual(format_anomaly_flag(3), "⚠ 3")
        self.assertEqual(format_anomaly_flag(None), "")

    def test_summary_pairs(self):
        from core.log_export import get_session_info
        from gui.log_formatting import format_summary_pairs

        info = get_session_info(make_session_events(3))
        pairs = dict(format_summary_pairs(info))
        self.assertEqual(pairs["曲名"], "测试曲")
        self.assertEqual(pairs["会话 ID"], "abcd1234efgh5678")
        self.assertEqual(pairs["BPM"], "120")
        self.assertEqual(pairs["开始时间"], "2026-09-13T10:00:00")
        self.assertEqual(pairs["结束时间"], "2026-09-13T10:01:00")
        self.assertEqual(pairs["完成度"], "3/3")
        self.assertEqual(pairs["提前停止"], "否")
        self.assertTrue(pairs["总事件数"])

    def test_summary_pairs_placeholders(self):
        from gui.log_formatting import format_summary_pairs

        pairs = dict(format_summary_pairs({}))
        self.assertEqual(pairs["曲名"], "未知")
        self.assertEqual(pairs["完成度"], "未知")
        self.assertEqual(pairs["总事件数"], "未知")


# ==================== T5–T9:记录列表(sync_mode) ====================


class TestT5EmptyDirs(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def test_empty_and_missing_dirs_show_empty_state(self):
        ensure_qapp()

        from gui.log_tab import PlayLogTab
        from gui.log_texts import LOG_EMPTY_TEXT

        self.setUp_dirs()
        self.stub_dialogs()
        tab = PlayLogTab(log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True)
        tab.refresh_now()
        self.assertEqual(self.calls_of("error"), [])
        self.assertEqual(tab.empty_label.text(), LOG_EMPTY_TEXT)

    def test_missing_dir_shows_empty_state(self):
        ensure_qapp()
        from gui.log_tab import PlayLogTab
        from gui.log_texts import LOG_EMPTY_TEXT

        self.setUp_dirs()
        self.stub_dialogs()
        missing = os.path.join(self._tmp.name, "no_such_dir")
        tab = PlayLogTab(log_dir=missing, export_dir=self.export_dir, sync_mode=True)
        tab.refresh_now()
        self.assertEqual(self.calls_of("error"), [])
        self.assertEqual(tab.empty_label.text(), LOG_EMPTY_TEXT)


class TestT6ListRows(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def _make_tab(self):
        from gui.log_tab import PlayLogTab

        return PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )

    def test_three_logs_desc_order_and_anomaly(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        now = time.time()
        paths = []
        # 3 条日志:异常计数 0 / 2 / 1;mtime 递增 → 列表倒序
        specs = [
            ("a.jsonl", make_session_events(2), 0),
            ("b.jsonl", make_session_events(2) + [
                {"type": "focus", "timestamp": 1, "event": "lost", "window_title": "w"},
                {"type": "focus", "timestamp": 2, "event": "regained"},
            ], 1),
            ("c.jsonl", make_session_events(2) + [
                {"type": "env", "timestamp": 3, "event": "lock_screen", "details": {}},
            ], 2),
        ]
        for i, (name, events, _anomalies) in enumerate(specs):
            p = os.path.join(self.log_dir, name)
            write_jsonl(p, events)
            os.utime(p, (now - 100 + i, now - 100 + i))
            paths.append(p)
        tab = self._make_tab()
        tab.refresh_now()
        table = tab._table
        self.assertEqual(table.rowCount(), 3)
        # 倒序:最新修改(c, mtime 最大)在首行
        self.assertEqual(table.item(0, 0).text(), "测试曲")
        self.assertEqual(table.item(0, 4).text(), "⚠ 1")
        self.assertEqual(table.item(1, 4).text(), "⚠ 2")
        self.assertEqual(table.item(2, 4).text(), "-")

    def test_note_only_legacy_log(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        p = os.path.join(self.log_dir, "legacy.jsonl")
        write_jsonl(p, [
            {"type": "note", "timestamp": 1, "index": 0, "notes": ["1"], "keys": ["D"],
             "deviation_ms": 0.0, "success": True},
        ])
        tab = self._make_tab()
        tab.refresh_now()
        self.assertEqual(tab._table.rowCount(), 1)
        self.assertEqual(tab._table.item(0, 0).text(), "未知曲目")

    def test_corrupted_first_line_stays_listed(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        bad = os.path.join(self.log_dir, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("this is not json\n")
        good = os.path.join(self.log_dir, "good.jsonl")
        write_jsonl(good, make_session_events(2))
        tab = self._make_tab()
        tab.refresh_now()
        self.assertEqual(self.calls_of("error"), [])
        self.assertEqual(tab._table.rowCount(), 2)
        texts = {tab._table.item(r, 0).text() for r in range(2)}
        self.assertIn("测试曲", texts)
        self.assertIn("未知曲目", texts)

    def test_half_truncated_file_readable_head(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        p = os.path.join(self.log_dir, "partial.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(make_session_events(1)[0], ensure_ascii=False) + "\n")
            f.write('{"type": "no')  # 半行截断(演奏中写入)
        tab = self._make_tab()
        tab.refresh_now()
        self.assertEqual(self.calls_of("error"), [])
        self.assertEqual(tab._table.rowCount(), 1)
        self.assertEqual(tab._table.item(0, 0).text(), "测试曲")


# ==================== T10–T11:删除 ====================


class TestT10Delete(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def _prepare(self, count=2):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from gui.log_tab import PlayLogTab

        tab = PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )
        for i in range(count):
            write_jsonl(os.path.join(self.log_dir, f"s{i}.jsonl"), make_session_events(2))
        tab.refresh_now()
        return tab

    def _check_all(self, tab):
        from PyQt6.QtCore import Qt

        for r in range(tab._table.rowCount()):
            tab._table.item(r, 0).setCheckState(Qt.CheckState.Checked)

    def test_cancel_deletes_nothing(self):
        tab = self._prepare(2)
        self._check_all(tab)
        self._confirm_result = False
        tab._delete_selected()
        self.assertEqual(len(os.listdir(self.log_dir)), 2)
        self.assertEqual(self.calls_of("success"), [])

    def test_confirm_deletes_and_notifies(self):
        from gui.log_texts import DELETE_DONE_TEXT

        tab = self._prepare(2)
        self._check_all(tab)
        self._confirm_result = True
        tab._delete_selected()
        self.assertEqual(len(os.listdir(self.log_dir)), 0)
        success_calls = self.calls_of("success")
        self.assertEqual(len(success_calls), 1)
        self.assertIn(DELETE_DONE_TEXT.format(n=2), success_calls[0][2])


class TestT11DeletePartial(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def test_partial_failure_reports_honestly(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from PyQt6.QtCore import Qt

        from gui.log_tab import PlayLogTab
        from gui.log_texts import DELETE_PARTIAL_TEXT

        tab = PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )
        p1 = os.path.join(self.log_dir, "s1.jsonl")
        p2 = os.path.join(self.log_dir, "s2.jsonl")
        write_jsonl(p1, make_session_events(2))
        write_jsonl(p2, make_session_events(2))
        tab.refresh_now()
        for r in range(tab._table.rowCount()):
            tab._table.item(r, 0).setCheckState(Qt.CheckState.Checked)
        # 预占 p1 句柄(Windows 上 unlink 打开中的文件必失败)
        holder = open(p1, "a", encoding="utf-8")
        try:
            self._confirm_result = True
            tab._delete_selected()
            warning_calls = self.calls_of("warning")
            self.assertEqual(len(warning_calls), 1)
            self.assertIn(DELETE_PARTIAL_TEXT.format(n=1, m=1), warning_calls[0][2])
            self.assertFalse(os.path.exists(p2))
        finally:
            holder.close()


# ==================== T12–T13:导出 ====================


class TestT12Export(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def setUp(self):
        self.setUp_dirs()
        self.stub_dialogs()

    def _prepare(self):
        ensure_qapp()
        from PyQt6.QtCore import Qt

        from gui.log_tab import PlayLogTab

        tab = PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )
        self._src = os.path.join(self.log_dir, "s.jsonl")
        write_jsonl(self._src, make_session_events(2))
        tab.refresh_now()
        tab._table.item(0, 0).setCheckState(Qt.CheckState.Checked)
        return tab

    def test_export_success_shows_path(self):
        from gui.log_texts import EXPORT_SUCCESS_TEXT

        tab = self._prepare()
        tab._ask_export_format = lambda: "jsonl"
        tab._export_selected()
        info_calls = self.calls_of("info")
        self.assertEqual(len(info_calls), 1)
        expected = os.path.join(self.export_dir, "s.jsonl")
        self.assertTrue(os.path.exists(expected))
        self.assertIn(EXPORT_SUCCESS_TEXT.format(path=expected), info_calls[0][2])

    def test_export_fail_when_source_missing(self):
        from gui.log_texts import EXPORT_FAIL_TEXT

        tab = self._prepare()
        tab._ask_export_format = lambda: "jsonl"
        os.remove(self._src)  # 外部删除源文件(列表尚未刷新,勾选仍在)
        tab._export_selected()
        error_calls = self.calls_of("error")
        self.assertEqual(len(error_calls), 1)
        self.assertIn(EXPORT_FAIL_TEXT, error_calls[0][2])


class TestT13ExportMultiSelect(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def test_multi_select_export_rejected(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from PyQt6.QtCore import Qt

        from gui.log_tab import PlayLogTab
        from gui.log_texts import EXPORT_SELECT_ONE_TEXT

        tab = PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )
        for i in range(2):
            write_jsonl(os.path.join(self.log_dir, f"s{i}.jsonl"), make_session_events(2))
        tab.refresh_now()
        for r in range(tab._table.rowCount()):
            tab._table.item(r, 0).setCheckState(Qt.CheckState.Checked)
        export_format_asked = []
        tab._ask_export_format = lambda: export_format_asked.append(1) or "jsonl"
        tab._export_selected()
        info_calls = self.calls_of("info")
        self.assertEqual(len(info_calls), 1)
        self.assertIn(EXPORT_SELECT_ONE_TEXT, info_calls[0][2])
        self.assertEqual(export_format_asked, [])
        self.assertEqual(os.listdir(self.export_dir), [])


# ==================== T14:自动清理 ====================


class TestT14Cleanup(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def _make_tab(self):
        from gui.log_tab import PlayLogTab

        return PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )

    def test_expired_log_cleaned_with_notice(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from gui.log_texts import CLEANUP_DONE_TEXT

        old = os.path.join(self.log_dir, "old.jsonl")
        write_jsonl(old, make_session_events(2))
        stale = time.time() - 8 * 86400
        os.utime(old, (stale, stale))
        tab = self._make_tab()
        tab.refresh_now()
        self.assertFalse(os.path.exists(old))
        info_calls = self.calls_of("info")
        self.assertEqual(len(info_calls), 1)
        self.assertIn(CLEANUP_DONE_TEXT.format(n=1, m=0), info_calls[0][2])

    def test_no_expired_no_notice(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from gui.log_texts import CLEANUP_DONE_TITLE

        fresh = os.path.join(self.log_dir, "fresh.jsonl")
        write_jsonl(fresh, make_session_events(2))
        tab = self._make_tab()
        tab.refresh_now()
        self.assertTrue(os.path.exists(fresh))
        self.assertEqual(self.calls_of("info"), [])
        self.assertEqual(self.calls_of(CLEANUP_DONE_TITLE), [])


# ==================== T15:worker 异常注入 ====================


class TestT15WorkerError(_TempDirsMixin, _DialogStubMixin, unittest.TestCase):
    def test_worker_exception_shows_error_not_crash(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from gui.log_tab import PlayLogTab
        from gui.log_texts import LOAD_ERROR_TEXT

        tab = PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )
        tab._manager.list_logs = mock.Mock(side_effect=RuntimeError("boom"))
        tab.refresh_now()
        error_calls = self.calls_of("error")
        self.assertEqual(len(error_calls), 1)
        self.assertIn(LOAD_ERROR_TEXT, error_calls[0][2])


# ==================== T16–T20:详情视图 ====================


class _DetailBase(_TempDirsMixin, unittest.TestCase):
    def _make_detail(self):
        ensure_qapp()
        self.setUp_dirs()
        from gui.log_detail_view import SessionDetailView

        return SessionDetailView(sync_mode=True, log_dir=self.log_dir)

    @staticmethod
    def visible_rows(detail):
        from PyQt6.QtCore import Qt

        return [
            detail.timeline.item(i)
            for i in range(detail.timeline.count())
            if not detail.timeline.item(i).isHidden()
        ]


class TestT16FullSession(_DetailBase, unittest.TestCase):
    def test_summary_and_timeline_match_file(self):
        detail = self._make_detail()
        events = make_session_events(3)
        path = os.path.join(self.log_dir, "full.jsonl")
        write_jsonl(path, events)
        detail.load_session(path)
        # 摘要与文件一致
        pairs = {}
        grid = detail.summary_grid
        for i in range(grid.count()):
            widget = grid.itemAt(i).widget()
            if widget is not None:
                pairs.setdefault(i, []).append(widget.text())
        texts = [grid.itemAt(i).widget().text() for i in range(grid.count())]
        self.assertIn("测试曲", texts)
        self.assertIn("abcd1234efgh5678", texts)
        self.assertIn("120", texts)
        self.assertIn("2026-09-13T10:00:00", texts)
        self.assertIn("2026-09-13T10:01:00", texts)
        self.assertIn("3/3", texts)
        self.assertIn("否", texts)
        # 时间线行数 = 事件数,按文件(时间)正序
        self.assertEqual(detail.timeline.count(), len(events))
        self.assertIn("会话开始", detail.timeline.item(0).text())
        self.assertIn("会话结束", detail.timeline.item(detail.timeline.count() - 1).text())


class TestT17MissingHeadTail(_DetailBase, unittest.TestCase):
    def test_missing_tail_placeholder(self):
        from gui.log_texts import PLACEHOLDER_UNKNOWN

        detail = self._make_detail()
        events = make_session_events(2, with_end=False)
        path = os.path.join(self.log_dir, "no_tail.jsonl")
        write_jsonl(path, events)
        detail.load_session(path)
        texts = [
            detail.summary_grid.itemAt(i).widget().text()
            for i in range(detail.summary_grid.count())
        ]
        self.assertIn("测试曲", texts)
        self.assertIn(PLACEHOLDER_UNKNOWN, texts)  # 结束时间/完成度占位
        self.assertEqual(detail.timeline.count(), len(events))

    def test_missing_head_placeholder(self):
        detail = self._make_detail()
        events = make_session_events(2, with_start=False)
        path = os.path.join(self.log_dir, "no_head.jsonl")
        write_jsonl(path, events)
        detail.load_session(path)
        texts = [
            detail.summary_grid.itemAt(i).widget().text()
            for i in range(detail.summary_grid.count())
        ]
        self.assertIn("未知曲目", texts)
        self.assertEqual(detail.timeline.count(), len(events))


class TestT18TypeFilter(_DetailBase, unittest.TestCase):
    def test_filter_and_restore_all(self):
        detail = self._make_detail()
        events = make_session_events(3)[:-1] + [  # 去掉 session_end
            {"type": "focus", "timestamp": 1789300009.0, "event": "regained"}
        ]
        path = os.path.join(self.log_dir, "mix.jsonl")
        write_jsonl(path, events)
        detail.load_session(path)
        self.assertEqual(detail.timeline.count(), 5)
        detail.set_type_filter("note")
        self.assertEqual(len(self.visible_rows(detail)), 3)
        detail.set_type_filter("focus")
        self.assertEqual(len(self.visible_rows(detail)), 1)
        detail.set_type_filter(None)
        self.assertEqual(len(self.visible_rows(detail)), 5)
        detail.set_type_filter("all")
        self.assertEqual(len(self.visible_rows(detail)), 5)


class TestT19ChunkedRender(_DetailBase, unittest.TestCase):
    def test_load_more_batches_to_full(self):
        from gui.log_detail_view import LOG_RENDER_CHUNK_SIZE

        detail = self._make_detail()
        total = LOG_RENDER_CHUNK_SIZE * 2 + 102
        note_count = total - 1  # session_start + note_count = total 行
        events = make_session_events(note_count, with_start=True, with_end=False)
        path = os.path.join(self.log_dir, "big.jsonl")
        write_jsonl(path, events)
        detail.load_session(path)
        self.assertEqual(detail.timeline.count(), LOG_RENDER_CHUNK_SIZE)
        self.assertTrue(detail.load_more_btn.isVisibleTo(detail))
        detail.load_more_btn.click()
        self.assertEqual(detail.timeline.count(), LOG_RENDER_CHUNK_SIZE * 2)
        detail.load_more_btn.click()
        self.assertEqual(detail.timeline.count(), total)
        self.assertFalse(detail.load_more_btn.isVisibleTo(detail))


class TestT20BackSignal(_DetailBase, _DialogStubMixin, unittest.TestCase):
    def test_back_requested_signal(self):
        detail = self._make_detail()
        fired = []
        detail.back_requested.connect(lambda: fired.append(1))
        detail.back_btn.click()
        self.assertEqual(fired, [1])

    def test_back_keeps_list_state(self):
        ensure_qapp()
        self.setUp_dirs()
        self.stub_dialogs()
        from PyQt6.QtCore import Qt

        from gui.log_tab import PlayLogTab

        tab = PlayLogTab(
            log_dir=self.log_dir, export_dir=self.export_dir, sync_mode=True
        )
        for i in range(3):
            write_jsonl(os.path.join(self.log_dir, f"s{i}.jsonl"), make_session_events(2))
        tab.refresh_now()
        # 勾选第 1 行(列表控件状态)
        tab._table.item(1, 0).setCheckState(Qt.CheckState.Checked)
        selected_row_before = tab._table.currentRow()
        # 进入详情再返回
        tab._open_row_detail(0)
        self.assertEqual(tab._page_stack.currentIndex(), 1)
        tab._detail.back_requested.emit()
        self.assertEqual(tab._page_stack.currentIndex(), 0)
        # 列表控件未重建:行数与勾选状态保持(spec 5.2.1.5)
        self.assertEqual(tab._table.rowCount(), 3)
        self.assertEqual(
            tab._table.item(1, 0).checkState(), Qt.CheckState.Checked
        )


# ==================== T21:主窗口导航接线 ====================


class TestT21Navigation(_TempDirsMixin, unittest.TestCase):
    def test_main_window_navigation_wiring(self):
        ensure_qapp()
        self.setUp_dirs()
        import main as m
        from gui.log_texts import NAV_LOG_TEXT
        from gui.main_window import NAV_ITEMS

        self.assertEqual(len(NAV_ITEMS), 4)
        self.assertEqual(NAV_ITEMS[-1][0], NAV_LOG_TEXT)

        db_path = os.path.join(self._tmp.name, "t21.db")
        cfg = m.load_config()
        db = m.ScoreDB(db_path)
        keymap = m.KeyMap(cfg["keymap"])
        player = m.Player(keymap)
        win = None
        try:
            win = m.MainWindow(cfg, db, keymap, player)
            self.assertEqual(win.stack.count(), 4)
            self.assertEqual(win.nav.currentRow(), 0)
            # _go_play 仍落演奏控制页(index 2,select_score 打桩避免依赖乐谱数据)
            with mock.patch.object(win.player_tab, "select_score", lambda score_id: None):
                win._go_play(1)
            self.assertEqual(win.nav.currentRow(), 2)
            self.assertEqual(win.stack.currentIndex(), 2)
        finally:
            if win is not None:
                win._cleanup_on_quit()
                win.deleteLater()
            db.conn.close()


if __name__ == "__main__":
    unittest.main()
