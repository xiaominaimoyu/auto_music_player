"""上传识别页:复制提示词 -> 外部 AI 识别 -> 粘贴简谱 -> 解析 -> 校对表格 -> 保存入库。

识别只保留复制粘贴这一条途径:本程序不内置任何在线大模型调用,
用户把内置提示词粘贴到任意外部 AI 工具并附上乐谱图片,把返回的简谱
粘贴回来解析即可,全程无需 API Key。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
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

from core.parser import parse_jianpu
from core.prompt import JIANPU_PROMPT
from core.score_model import validate_notes
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
        lay.addLayout(_step_row(
            "1", "复制识别提示词",
            "把提示词粘贴到任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi),并附上乐谱图片一起发送"))
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
            self, "提示词已复制",
            "已复制到剪贴板。\n\n使用方法:\n"
            "1. 粘贴到任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi)\n"
            "2. 附上乐谱图片一起发送\n"
            "3. 把返回的简谱粘贴到下方「简谱粘贴」,点击「解析到校对表格」即可校对入库",
        )


class UploadTab(QWidget):
    saved = pyqtSignal()

    def __init__(self, db):
        super().__init__()
        self._db = db
        self._build_ui()

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
        lay2.addLayout(_step_row(
            "2", "简谱粘贴与解析",
            "外部 AI 工具识别出的简谱直接粘贴到这里 · 支持手动输入与修改,修改后重新「解析到校对表格」"))
        self.raw_text = QPlainTextEdit()
        self.raw_text.setPlaceholderText(
            "把外部 AI 工具返回的简谱粘贴到这里,如:1 1 5 5 6 6 5- 4 4 3 3 2 2 1- ...")
        self.raw_text.setCursor(Qt.CursorShape.IBeamCursor)
        lay2.addWidget(self.raw_text, 1)
        btn_row = QHBoxLayout()
        self.parse_btn = QPushButton("解析到校对表格")
        self.parse_btn.setObjectName("BtnPrimary")
        self.parse_btn.clicked.connect(self._parse)
        btn_row.addWidget(self.parse_btn)
        btn_row.addStretch(1)
        lay2.addLayout(btn_row)
        self._fix_card_cursors(card2)
        inner_layout.addWidget(card2)

        # 步骤3:校对表格(底部边缘可拖高,总滚动条随高度同步)
        card3, lay3 = self._card(resizable=True)
        card3.setup(height=300, min_height=220)
        lay3.addLayout(_step_row("3", "校对表格", "音符列可写和弦,如 high_1,mid_3;时值单位:拍"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["音符", "时值(拍)", "半音"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay3.addWidget(self.table, 1)
        edit_row = QHBoxLayout()
        add_btn = QPushButton("添加行")
        add_btn.setObjectName("BtnSecondary")
        add_btn.clicked.connect(lambda: self.table.insertRow(self.table.rowCount()))
        del_btn = QPushButton("删除选中行")
        del_btn.setObjectName("BtnSecondary")
        del_btn.clicked.connect(self._delete_selected_rows)
        clear_btn = QPushButton("清空表格")
        clear_btn.setObjectName("BtnDanger")
        clear_btn.clicked.connect(lambda: self.table.setRowCount(0))
        edit_row.addWidget(add_btn)
        edit_row.addWidget(del_btn)
        edit_row.addWidget(clear_btn)
        edit_row.addStretch(1)
        lay3.addLayout(edit_row)
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
        raw = self.raw_text.toPlainText().strip()
        if not raw:
            AppDialog.show_warning(self, "提示", "请先把外部 AI 工具识别出的简谱粘贴到输入框")
            return
        try:
            notes = parse_jianpu(raw)
        except Exception as e:
            AppDialog.show_error(self, "解析失败", str(e))
            return
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
            self.table.setItem(row, 2, QTableWidgetItem("#" if n.get("semitone") else ""))

    def _delete_selected_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

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
            note_ids = [] if notes_str == "(休止)" else [x.strip() for x in notes_str.split(",") if x.strip()]
            item = {"notes": note_ids, "dur": dur}
            semi_text = item_semi.text().strip() if item_semi is not None else ""
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
        self._db.add_score(
            name=name,
            notes=notes,
            raw_text=self.raw_text.toPlainText(),
            source_file="",
            source_type="manual",
            bpm_default=self.bpm_spin.value(),
        )
        AppDialog.show_success(self, "成功", f"《{name}》已保存到乐谱库")
        self.saved.emit()
