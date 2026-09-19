"""上传识别页:复制提示词 -> 外部 AI 识别 -> 粘贴简谱 -> 解析 -> 校对表格 -> 保存入库。

识别只保留复制粘贴这一条途径:本程序不内置任何在线大模型调用,
用户把内置提示词粘贴到任意外部 AI 工具并附上乐谱图片,把返回的简谱
粘贴回来解析即可,全程无需 API Key。
"""

import sqlite3
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.parser import AI_MISSING_SEPARATOR_REASON, parse_jianpu
from core.practice import StepRecorder
from core.prompt import JIANPU_PROMPT
from core.preview_player import PreviewPlayer
from core.recording import PerformanceRecorder, PhysicalNoteResolver
from core.score_model import validate_notes
from core.jianpu_editor import insert_rest, delete_event
from gui.widgets import AppDialog, BottomResizableCard


def _step_row(step: str, title: str, subtitle: str = ""):
    row = QHBoxLayout()
    row.setSpacing(10)
    badge = QLabel(step)
    badge.setObjectName("StepBadge")
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    row.addWidget(badge)
    texts = QVBoxLayout()
    texts.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("SectionTitle")
    texts.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("SectionSubtitle")
        texts.addWidget(s)
    row.addLayout(texts, 1)
    return row


class PromptCard(QFrame):
    """提示词卡片:展示提示词摘要 + 复制按钮。"""

    def __init__(self):
        super().__init__()
        self.setObjectName("SectionCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)
        lay.addLayout(
            _step_row(
                "1",
                "复制识别提示词",
                "把提示词粘贴到任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi),并附上乐谱图片一起发送",
            )
        )
        self.prompt_view = QPlainTextEdit()
        self.prompt_view.setPlainText(JIANPU_PROMPT)
        self.prompt_view.setReadOnly(True)
        self.prompt_view.setFixedHeight(120)
        self.prompt_view.setCursor(Qt.CursorShape.IBeamCursor)
        lay.addWidget(self.prompt_view)
        btn_row = QHBoxLayout()
        self.copy_btn = QPushButton("复制提示词")
        self.copy_btn.setObjectName("BtnPrimary")
        self.copy_btn.clicked.connect(self._copy)
        btn_row.addWidget(self.copy_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

    def _copy(self):
        QApplication.clipboard().setText(JIANPU_PROMPT)
        AppDialog.show_info(
            self,
            "提示词已复制",
            "已复制到剪贴板。\n\n使用方法:\n"
            "1. 粘贴到任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi)\n"
            "2. 附上乐谱图片一起发送\n"
            "3. 把返回的简谱粘贴到下方「简谱粘贴」,点击「解析到校对表格」即可校对入库",
        )


class UploadTab(QWidget):
    saved = pyqtSignal()
    live_count_changed = pyqtSignal(int)
    live_stop_requested = pyqtSignal()

    def __init__(
        self,
        db,
        preview_player=None,
        *,
        keymap=None,
        profile=None,
        profiles=None,
    ):
        super().__init__()
        self._db = db
        self._record_keymap = keymap
        self._record_profile = profile
        self._record_profiles = list(profiles or [])
        self._preview_player = (
            preview_player if preview_player is not None else PreviewPlayer(self)
        )
        self._preview_active = False
        self._preview_error = False
        self._last_parse_errors = []
        self._live_recorder = None
        self._live_keyboard_listener = None
        self._live_mouse_listener = None
        self._live_mouse = set()
        self._live_lock = threading.Lock()
        self._build_ui()
        self.live_count_changed.connect(self._on_live_count_changed)
        self.live_stop_requested.connect(self._stop_live_recording)
        self._preview_player.finished.connect(self._on_preview_finished)
        self._preview_player.error_occurred.connect(self._on_preview_error)

    def _card(self, resizable=False):
        if resizable:
            card = BottomResizableCard()
        else:
            card = QFrame()
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card.setObjectName("SectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        return card, layout

    def _fix_card_cursors(self, card):
        """子控件显式光标,避免继承底边拉伸光标。"""
        for w in card.findChildren(QLabel):
            w.setCursor(Qt.CursorShape.ArrowCursor)
        for w in card.findChildren(QTableWidget):
            w.setCursor(Qt.CursorShape.ArrowCursor)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("ContentScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("ContentInner")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(32, 24, 32, 24)
        inner_layout.setSpacing(16)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        # 步骤1:复制识别提示词(交给外部 AI 工具)
        self.prompt_card = PromptCard()
        inner_layout.addWidget(self.prompt_card)

        # 步骤2:粘贴简谱并解析(底部边缘可拖高)
        card2, lay2 = self._card(resizable=True)
        card2.setup(height=220, min_height=160)
        lay2.addLayout(
            _step_row(
                "2",
                "简谱粘贴与解析",
                "外部 AI 工具识别出的简谱直接粘贴到这里 · 支持手动输入与修改,修改后重新「解析到校对表格」",
            )
        )
        self.raw_text = QPlainTextEdit()
        self.raw_text.setPlaceholderText(
            "把外部 AI 工具返回的简谱粘贴到这里,如:1 1 5 5 6 6 5- 4 4 3 3 2 2 1- ..."
        )
        self.raw_text.setCursor(Qt.CursorShape.IBeamCursor)
        lay2.addWidget(self.raw_text, 1)
        btn_row = QHBoxLayout()
        self.parse_btn = QPushButton("解析到校对表格")
        self.parse_btn.setObjectName("BtnPrimary")
        self.parse_btn.clicked.connect(self._parse)
        btn_row.addWidget(self.parse_btn)
        btn_row.addStretch(1)
        lay2.addLayout(btn_row)
        self.parse_status = QLabel()
        self.parse_status.setObjectName("SectionSubtitle")
        self.parse_status.setWordWrap(True)
        lay2.addWidget(self.parse_status)
        self._fix_card_cursors(card2)
        inner_layout.addWidget(card2)

        # 步骤3:校对表格(底部边缘可拖高,总滚动条随高度同步)
        card3, lay3 = self._card(resizable=True)
        # 表格下方还包含编辑、步进录制、实时录制和两行状态提示;
        # 300px 会强行压缩布局,导致这些控件在高 DPI 下互相覆盖。
        card3.setup(height=620, min_height=588)
        lay3.addLayout(
            _step_row("3", "校对表格", "音符列可写和弦,如 high_1,mid_3;时值单位:拍")
        )
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["音符", "时值(拍)", "半音"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._update_preview_button)
        lay3.addWidget(self.table, 1)

        # 编辑按钮组
        edit_row = QHBoxLayout()

        # 音符编辑按钮(插入休止符/删除音符)
        insert_before_btn = QPushButton("← 前插半拍休止")
        insert_before_btn.setObjectName("BtnSecondary")
        insert_before_btn.setToolTip("在选中音符前插入半拍休止符 (0_)")
        insert_before_btn.clicked.connect(
            lambda: self._insert_rest_at_selection(before=True)
        )

        insert_after_btn = QPushButton("后插半拍休止 →")
        insert_after_btn.setObjectName("BtnSecondary")
        insert_after_btn.setToolTip("在选中音符后插入半拍休止符 (0_)")
        insert_after_btn.clicked.connect(
            lambda: self._insert_rest_at_selection(before=False)
        )

        delete_note_btn = QPushButton("删除选中音符")
        delete_note_btn.setObjectName("BtnDanger")
        delete_note_btn.setToolTip("删除选中的音符")
        delete_note_btn.clicked.connect(self._delete_selected_note)

        edit_row.addWidget(insert_before_btn)
        edit_row.addWidget(insert_after_btn)
        edit_row.addWidget(delete_note_btn)
        edit_row.addStretch(1)
        lay3.addLayout(edit_row)

        # 表格操作按钮
        table_row = QHBoxLayout()
        add_btn = QPushButton("添加行")
        add_btn.setObjectName("BtnSecondary")
        add_btn.clicked.connect(self._add_table_row)
        del_btn = QPushButton("删除选中行")
        del_btn.setObjectName("BtnSecondary")
        del_btn.clicked.connect(self._delete_selected_rows)
        clear_btn = QPushButton("清空表格")
        clear_btn.setObjectName("BtnDanger")
        clear_btn.clicked.connect(self._clear_table)
        table_row.addWidget(add_btn)
        table_row.addWidget(del_btn)
        table_row.addWidget(clear_btn)
        self.preview_btn = QPushButton("试听当前乐谱")
        self.preview_btn.setObjectName("BtnPrimary")
        self.preview_btn.setToolTip(
            "使用系统大钢琴音色试听当前校对结果,不会向游戏发送按键"
        )
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._toggle_preview)
        table_row.addWidget(self.preview_btn)
        table_row.addStretch(1)
        lay3.addLayout(table_row)

        record_cfg_row = QHBoxLayout()
        record_title = QLabel("步进录制")
        record_title.setObjectName("FieldLabel")
        self.record_octave_combo = QComboBox()
        self.record_octave_combo.addItem("中音", "mid")
        self.record_octave_combo.addItem("高音", "high")
        self.record_octave_combo.addItem("低音", "low")
        self.record_dur_combo = QComboBox()
        for text, value in (
            ("1/4 拍", 0.25),
            ("1/2 拍", 0.5),
            ("1 拍", 1.0),
            ("2 拍", 2.0),
            ("4 拍", 4.0),
        ):
            self.record_dur_combo.addItem(text, value)
        self.record_semitone_btn = QPushButton("#")
        self.record_semitone_btn.setObjectName("BtnSecondary")
        self.record_semitone_btn.setCheckable(True)
        self.record_semitone_btn.setToolTip("开启后追加升半音标记")
        rest_btn = QPushButton("休止")
        rest_btn.setObjectName("BtnSecondary")
        rest_btn.clicked.connect(self._append_recorded_rest)
        undo_btn = QPushButton("撤销最后")
        undo_btn.setObjectName("BtnSecondary")
        undo_btn.clicked.connect(self._delete_last_table_row)
        record_cfg_row.addWidget(record_title)
        record_cfg_row.addWidget(self.record_octave_combo)
        record_cfg_row.addWidget(self.record_dur_combo)
        record_cfg_row.addWidget(self.record_semitone_btn)
        record_cfg_row.addWidget(rest_btn)
        record_cfg_row.addWidget(undo_btn)
        record_cfg_row.addStretch(1)
        lay3.addLayout(record_cfg_row)

        record_note_row = QHBoxLayout()
        for num in range(1, 8):
            btn = QPushButton(str(num))
            btn.setObjectName("BtnSecondary")
            btn.clicked.connect(lambda _checked=False, n=num: self._append_recorded_note(n))
            record_note_row.addWidget(btn)
        record_note_row.addStretch(1)
        lay3.addLayout(record_note_row)

        live_row = QHBoxLayout()
        live_label = QLabel("实时录制")
        live_label.setObjectName("FieldLabel")
        self.record_profile_combo = QComboBox()
        if self._record_profiles:
            for candidate in self._record_profiles:
                self.record_profile_combo.addItem(
                    f"{candidate.group} · {candidate.name}", candidate
                )
            if self._record_profile is not None:
                for index in range(self.record_profile_combo.count()):
                    candidate = self.record_profile_combo.itemData(index)
                    if getattr(candidate, "id", None) == self._record_profile.id:
                        self.record_profile_combo.setCurrentIndex(index)
                        break
        else:
            self.record_profile_combo.addItem("默认 21 键", None)
        self.record_quantize_combo = QComboBox()
        for text, value in (
            ("量化 1/4 拍", 0.25),
            ("量化 1/2 拍", 0.5),
            ("量化 1 拍", 1.0),
            ("不量化", 0.0),
        ):
            self.record_quantize_combo.addItem(text, value)
        self.live_record_btn = QPushButton("开始实时录制")
        self.live_record_btn.setObjectName("BtnPrimary")
        self.live_record_btn.setToolTip(
            "只监听实际按键时长；停止后转换到上方可编辑表格，不会向系统发送输入"
        )
        self.live_record_btn.clicked.connect(self._start_live_recording)
        self.live_record_btn.setEnabled(
            self._record_keymap is not None or bool(self._record_profiles)
        )
        self.live_stop_btn = QPushButton("停止并写入表格")
        self.live_stop_btn.setObjectName("BtnSecondary")
        self.live_stop_btn.setEnabled(False)
        self.live_stop_btn.clicked.connect(self._stop_live_recording)
        live_row.addWidget(live_label)
        live_row.addWidget(self.record_profile_combo)
        live_row.addWidget(self.record_quantize_combo)
        live_row.addWidget(self.live_record_btn)
        live_row.addWidget(self.live_stop_btn)
        live_row.addStretch(1)
        lay3.addLayout(live_row)

        self.live_record_status = QLabel(
            "选择档位后开始；按 Esc 或“停止并写入表格”结束录制。"
        )
        self.live_record_status.setObjectName("SectionSubtitle")
        self.live_record_status.setWordWrap(True)
        lay3.addWidget(self.live_record_status)

        self.preview_status = QLabel(
            "试听使用系统大钢琴音色，不会操作游戏；建议校对完成后先试听，再保存或开始演奏。"
        )
        self.preview_status.setObjectName("SectionSubtitle")
        self.preview_status.setWordWrap(True)
        lay3.addWidget(self.preview_status)

        self._fix_card_cursors(card3)
        inner_layout.addWidget(card3)

        # 步骤4:保存
        card, lay = self._card()
        lay.addLayout(_step_row("4", "保存到乐谱库"))
        save_row = QHBoxLayout()
        save_row.setSpacing(12)
        name_col = QVBoxLayout()
        name_col.setSpacing(6)
        name_label = QLabel("名称")
        name_label.setObjectName("FieldLabel")
        name_col.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("如: 小星星")
        name_col.addWidget(self.name_edit)
        save_row.addLayout(name_col, 1)
        bpm_col = QVBoxLayout()
        bpm_col.setSpacing(6)
        bpm_label = QLabel("默认 BPM")
        bpm_label.setObjectName("FieldLabel")
        bpm_col.addWidget(bpm_label)
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(30, 300)
        self.bpm_spin.setValue(100)
        bpm_col.addWidget(self.bpm_spin)
        save_row.addLayout(bpm_col)
        save_btn = QPushButton("保存入库")
        save_btn.setObjectName("BtnPrimary")
        save_btn.clicked.connect(self._save)
        save_row.addWidget(save_btn, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(save_row)
        inner_layout.addWidget(card)

    def _parse(self):
        self._stop_preview()
        raw = self.raw_text.toPlainText().strip()
        if not raw:
            AppDialog.show_warning(
                self, "提示", "请先把外部 AI 工具识别出的简谱粘贴到输入框"
            )
            return
        try:
            notes, errors = parse_jianpu(raw, collect=True, strict_ai=True)
        except Exception as e:
            AppDialog.show_error(self, "解析失败", str(e))
            return
        self._last_parse_errors = errors
        fatal_errors = [
            error for error in errors if error.reason == AI_MISSING_SEPARATOR_REASON
        ]
        if fatal_errors:
            preview = "；".join(str(e) for e in fatal_errors[:2])
            more = f"；另有 {len(fatal_errors) - 2} 处" if len(fatal_errors) > 2 else ""
            self.table.setRowCount(0)
            self._update_preview_button()
            message = (
                f"检测到 {len(fatal_errors)} 处无法安全判断的 AI 粘连记号："
                f"{preview}{more}。请修正原文并重新解析；为避免错误变调，本次结果未进入校对表格。"
            )
            self.parse_status.setText(message)
            AppDialog.show_warning(self, "AI 乐谱格式有歧义", message)
            return
        if errors:
            preview = "；".join(str(e) for e in errors[:2])
            more = f"；另有 {len(errors) - 2} 处" if len(errors) > 2 else ""
            self.parse_status.setText(
                f"已解析 {len(notes)} 个音符，但发现 {len(errors)} 处未识别记号：{preview}{more}。"
                "请修正原文后重新解析，或在校对表格中确认结果。"
            )
        else:
            self.parse_status.setText(f"解析成功：{len(notes)} 个音符，未发现未识别记号。")
        if not notes:
            AppDialog.show_warning(self, "提示", "未能从文本中解析出音符,请检查格式")
            return
        self.table.setRowCount(0)
        for n in notes:
            row = self.table.rowCount()
            self.table.insertRow(row)
            notes_str = ",".join(n["notes"]) if n["notes"] else "(休止)"
            self.table.setItem(row, 0, QTableWidgetItem(notes_str))
            self.table.setItem(row, 1, QTableWidgetItem(str(n["dur"])))
            self.table.setItem(
                row, 2, QTableWidgetItem("#" if n.get("semitone") else "")
            )
        self._update_preview_button()

    def _clear_table(self):
        self._stop_preview()
        self.table.setRowCount(0)
        self._update_preview_button()

    def _add_table_row(self):
        self.table.insertRow(self.table.rowCount())
        self._update_preview_button()

    def _update_preview_button(self, *_args):
        if not hasattr(self, "preview_btn"):
            return
        self.preview_btn.setEnabled(self._preview_active or self.table.rowCount() > 0)

    def _toggle_preview(self):
        if self._preview_active:
            self._stop_preview()
            return
        try:
            notes = self._table_to_notes()
        except ValueError as e:
            AppDialog.show_error(self, "试听失败", str(e))
            return
        if not notes:
            AppDialog.show_warning(self, "提示", "表格为空,无法试听")
            return
        self._preview_error = False
        self._preview_active = True
        self.preview_btn.setText("停止试听")
        self.preview_status.setText("大钢琴音色试听中…不会向游戏发送按键")
        try:
            started = self._preview_player.play(
                notes,
                bpm=self.bpm_spin.value(),
                score_name=self.name_edit.text().strip() or "当前校对乐谱",
            )
        except (TypeError, ValueError, RuntimeError) as e:
            self._preview_active = False
            self.preview_btn.setText("试听当前乐谱")
            self._update_preview_button()
            AppDialog.show_error(self, "试听失败", str(e))
            return
        if not started:
            self._preview_active = False
            self.preview_btn.setText("试听当前乐谱")
            self.preview_status.setText("已有试听正在播放,请先停止后再试")
            self._update_preview_button()

    def _stop_preview(self):
        if not self._preview_active:
            return
        self._preview_active = False
        self._preview_player.stop()
        self.preview_btn.setText("试听当前乐谱")
        self.preview_status.setText("试听已停止；不会向游戏发送按键")
        self._update_preview_button()

    def _on_preview_error(self, message):
        if self._preview_active:
            self._preview_error = True
            self.preview_status.setText(f"试听失败: {message}")

    def _on_preview_finished(self, normal):
        if not self._preview_active:
            return
        self._preview_active = False
        self.preview_btn.setText("试听当前乐谱")
        if self._preview_error:
            self.preview_status.setText("试听失败,请检查系统扬声器设置")
        elif normal:
            self.preview_status.setText("试听完成；不会向游戏发送按键")
        else:
            self.preview_status.setText("试听已停止；不会向游戏发送按键")
        self._update_preview_button()

    def _delete_selected_rows(self):
        self._stop_preview()
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._update_preview_button()

    def _current_record_duration(self):
        return float(self.record_dur_combo.currentData() or 1.0)

    def _append_table_item(self, item: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        notes_str = ",".join(item["notes"]) if item["notes"] else "(休止)"
        self.table.setItem(row, 0, QTableWidgetItem(notes_str))
        self.table.setItem(row, 1, QTableWidgetItem(str(item["dur"])))
        self.table.setItem(row, 2, QTableWidgetItem("#" if item.get("semitone") else ""))
        self.table.selectRow(row)
        self._update_preview_button()

    def _append_recorded_note(self, num: int):
        octave = self.record_octave_combo.currentData() or "mid"
        note_id = f"{octave}_{int(num)}"
        try:
            item = StepRecorder().append_note(
                note_id,
                self._current_record_duration(),
                semitone=1 if self.record_semitone_btn.isChecked() else 0,
            )
        except ValueError as e:
            AppDialog.show_error(self, "录制失败", str(e))
            return
        self._append_table_item(item)
        self.parse_status.setText(f"已追加 {note_id},可继续录制、试听或保存。")

    def _append_recorded_rest(self):
        try:
            item = StepRecorder().append_rest(self._current_record_duration())
        except ValueError as e:
            AppDialog.show_error(self, "录制失败", str(e))
            return
        self._append_table_item(item)
        self.parse_status.setText("已追加休止符,可继续录制、试听或保存。")

    def _delete_last_table_row(self):
        self._stop_preview()
        row = self.table.rowCount() - 1
        if row < 0:
            return
        self.table.removeRow(row)
        self._update_preview_button()

    @staticmethod
    def _live_key_name(key):
        char = getattr(key, "char", None)
        if char:
            return char.upper() if len(char) == 1 else str(char).lower()
        name = getattr(key, "name", None)
        return str(name).lower() if name else None

    def _start_live_recording(self, *_args):
        if self._live_recorder is not None:
            return
        profile = self.record_profile_combo.currentData()
        resolver = PhysicalNoteResolver(
            keymap=self._record_keymap,
            profile=profile,
        )
        if not resolver.has_mapping:
            AppDialog.show_warning(self, "无法录制", "当前档位没有可反向识别的音键映射")
            return
        self._stop_preview()
        recorder = PerformanceRecorder(resolver)
        recorder.start()
        self._live_recorder = recorder
        try:
            from pynput import keyboard as pk
            from pynput import mouse as pm

            def on_press(key):
                token = self._live_key_name(key)
                if token == "esc":
                    self.live_stop_requested.emit()
                    return False
                if not token:
                    return
                with self._live_lock:
                    mouse = tuple(self._live_mouse)
                recorder.press(token, mouse)

            def on_release(key):
                token = self._live_key_name(key)
                if token and recorder.release(token):
                    self.live_count_changed.emit(recorder.captured_count)

            def on_click(_x, _y, button, pressed):
                token = str(getattr(button, "name", button)).lower()
                with self._live_lock:
                    if pressed:
                        self._live_mouse.add(token)
                    else:
                        self._live_mouse.discard(token)

            self._live_keyboard_listener = pk.Listener(
                on_press=on_press, on_release=on_release
            )
            self._live_mouse_listener = pm.Listener(on_click=on_click)
            self._live_keyboard_listener.start()
            self._live_mouse_listener.start()
        except Exception as exc:
            recorder.cancel()
            self._live_recorder = None
            self._stop_live_listeners()
            AppDialog.show_error(self, "录制监听启动失败", str(exc))
            return
        self.live_record_btn.setEnabled(False)
        self.live_stop_btn.setEnabled(True)
        self.record_profile_combo.setEnabled(False)
        self.record_quantize_combo.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.live_record_status.setText(
            "实时录制中 · 已录到 0 个音；支持同时按键形成和弦，Esc 可停止。"
        )

    def _on_live_count_changed(self, count):
        if self._live_recorder is not None:
            self.live_record_status.setText(
                f"实时录制中 · 已录到 {int(count)} 个音；停止后将按当前网格量化。"
            )

    def _stop_live_listeners(self):
        for attr in ("_live_keyboard_listener", "_live_mouse_listener"):
            listener = getattr(self, attr, None)
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
            setattr(self, attr, None)
        with self._live_lock:
            self._live_mouse.clear()

    def _stop_live_recording(self, *_args):
        recorder = self._live_recorder
        if recorder is None:
            return
        self._stop_live_listeners()
        self._live_recorder = None
        self.live_record_btn.setEnabled(True)
        self.live_stop_btn.setEnabled(False)
        self.record_profile_combo.setEnabled(True)
        self.record_quantize_combo.setEnabled(True)
        try:
            result = recorder.stop(
                bpm=self.bpm_spin.value(),
                quantize_beats=float(self.record_quantize_combo.currentData() or 0.0),
                title=self.name_edit.text().strip() or "实时录制",
            )
        except (RuntimeError, ValueError) as exc:
            self._update_preview_button()
            self.live_record_status.setText(str(exc))
            AppDialog.show_warning(self, "录制未写入", str(exc))
            return
        for item in result.notes:
            self._append_table_item(item)
        report = f"已录到 {result.captured_count} 个音，转换为 {len(result.notes)} 个可编辑时间元素"
        if result.warnings:
            report += "；" + "；".join(result.warnings[:3])
        self.live_record_status.setText(report)
        self.parse_status.setText(report + "。可继续校对、试听或保存。")
        self._update_preview_button()

    def shutdown(self):
        recorder = self._live_recorder
        self._live_recorder = None
        self._stop_live_listeners()
        if recorder is not None:
            recorder.cancel()

    def _insert_rest_at_selection(self, before: bool):
        """在选中音符的前/后插入半拍休止符。"""
        selected = self.table.selectedIndexes()
        if not selected:
            AppDialog.show_warning(self, "提示", "请先在表格中选中一个音符")
            return

        # 获取选中的行索引(第一个选中的)
        row = selected[0].row()

        # 获取原始简谱文本
        raw = self.raw_text.toPlainText().strip()
        if not raw:
            self._insert_table_rest(row, before)
            return

        try:
            # 在指定位置插入休止符
            new_text = insert_rest(raw, row, before=before, rest_token="0_")

            # 更新简谱文本
            self.raw_text.setPlainText(new_text)

            # 重新解析到表格
            self._parse()

            # 自动选中新插入的休止符所在行
            new_row = row if before else row + 1
            if 0 <= new_row < self.table.rowCount():
                self.table.selectRow(new_row)

        except Exception as e:
            self._insert_table_rest(row, before)

    def _insert_table_rest(self, row: int, before: bool):
        insert_at = row if before else row + 1
        self.table.insertRow(insert_at)
        self.table.setItem(insert_at, 0, QTableWidgetItem("(休止)"))
        self.table.setItem(insert_at, 1, QTableWidgetItem("0.5"))
        self.table.setItem(insert_at, 2, QTableWidgetItem(""))
        self.table.selectRow(insert_at)
        self._update_preview_button()

    def _delete_selected_note(self):
        """删除选中的音符。"""
        selected = self.table.selectedIndexes()
        if not selected:
            AppDialog.show_warning(self, "提示", "请先在表格中选中要删除的音符")
            return

        # 获取选中的行索引
        row = selected[0].row()

        # 获取原始简谱文本
        raw = self.raw_text.toPlainText().strip()
        if not raw:
            self.table.removeRow(row)
            self._update_preview_button()
            return

        try:
            # 删除指定音符
            new_text = delete_event(raw, row)

            # 更新简谱文本
            self.raw_text.setPlainText(new_text)

            # 重新解析到表格
            self._parse()

            # 尝试选中下一行(如果存在)
            if row < self.table.rowCount():
                self.table.selectRow(row)
            elif self.table.rowCount() > 0:
                self.table.selectRow(self.table.rowCount() - 1)

        except Exception as e:
            self.table.removeRow(row)
            self._update_preview_button()

    def _table_to_notes(self):
        notes = []
        for row in range(self.table.rowCount()):
            item_notes = self.table.item(row, 0)
            item_dur = self.table.item(row, 1)
            item_semi = self.table.item(row, 2)
            if item_notes is None or item_dur is None:
                raise ValueError(f"第 {row + 1} 行不完整")
            notes_str = item_notes.text().strip()
            dur_str = item_dur.text().strip()
            if not notes_str or not dur_str:
                raise ValueError(f"第 {row + 1} 行为空")
            try:
                dur = float(dur_str)
            except ValueError:
                raise ValueError(f"第 {row + 1} 行时值不是数字: {dur_str}")
            note_ids = (
                []
                if notes_str == "(休止)"
                else [x.strip() for x in notes_str.split(",") if x.strip()]
            )
            item = {"notes": note_ids, "dur": dur}
            semi_text = item_semi.text().strip() if item_semi is not None else ""
            if semi_text not in ("", "0", "#", "1"):
                raise ValueError(f"第 {row + 1} 行半音只能填写 #、1、0 或留空")
            if semi_text in ("#", "1"):
                item["semitone"] = 1
            notes.append(item)
        # 格式/取值校验统一交给 Schema 校验器(与入库校验同一套规则)
        errors = validate_notes(notes)
        if errors:
            summary = "\n".join(str(e) for e in errors[:6])
            if len(errors) > 6:
                summary += f"\n... 共 {len(errors)} 个错误"
            raise ValueError(summary)
        return notes

    def _save(self):
        name = self.name_edit.text().strip()
        if not name:
            AppDialog.show_warning(self, "提示", "请填写乐谱名称")
            return
        try:
            notes = self._table_to_notes()
        except ValueError as e:
            AppDialog.show_error(self, "保存失败", str(e))
            return
        if not notes:
            AppDialog.show_warning(self, "提示", "表格为空,无法保存")
            return
        try:
            self._db.add_score(
                name=name,
                notes=notes,
                raw_text=self.raw_text.toPlainText(),
                source_file="",
                source_type="manual",
                bpm_default=self.bpm_spin.value(),
            )
        except (ValueError, sqlite3.Error) as exc:
            AppDialog.show_error(self, "保存失败", str(exc))
            return
        AppDialog.show_success(self, "成功", f"《{name}》已保存到乐谱库")
        self.saved.emit()
