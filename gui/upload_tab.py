"""上传识别页(对齐 Web 设计稿):上传 -> 识别 -> 解析 -> 校对表格 -> 保存入库。"""

import os
import shutil
import threading
import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
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
from core.recognizer import JIANPU_PROMPT
from core.score_model import validate_notes
from gui.theme import BRAND, BRAND_SOFT, LINE_2, SURFACE_2, STATE_ERROR
from gui.widgets import AppDialog, BottomResizableCard

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_TEXT_EXTS = {".md", ".txt", ".docx", ".doc"}


def _extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".txt"):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    if ext == ".docx":
        # docx 本质是 zip + XML,用标准库提取正文,免去 python-docx/lxml 重依赖
        import xml.etree.ElementTree as ET
        import zipfile

        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        paragraphs = []
        for p in root.iter(f"{ns}p"):
            texts = [t.text or "" for t in p.iter(f"{ns}t")]
            paragraphs.append("".join(texts))
        return "\n".join(paragraphs)
    if ext == ".doc":
        raise ValueError(".doc 旧格式暂不支持,请另存为 .docx 或 .md")
    raise ValueError(f"不支持的文件类型: {ext}")


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


class DropZone(QFrame):
    clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(150)
        self._build()
        self._update_style(False)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("⤒")
        icon.setStyleSheet(f"font-size: 34px; color: {BRAND};")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)
        self.text = QLabel("选择乐谱文件")
        self.text.setStyleSheet("font-size: 15px; font-weight: 600; color: #EDEDF2;")
        self.text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.text)
        self.hint = QLabel("支持 jpg / png / webp / bmp 图片,md / txt / docx 文档(可拖拽文件到此)")
        self.hint.setObjectName("HintText")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hint)

    def _update_style(self, hover: bool):
        base = f"background: {SURFACE_2}; border: 2px dashed {LINE_2}; border-radius: 12px;"
        if hover:
            base = f"background: {BRAND_SOFT}; border: 2px dashed {BRAND}; border-radius: 12px;"
        self.setStyleSheet(base)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._update_style(True)

    def dragLeaveEvent(self, event):
        self._update_style(False)

    def dropEvent(self, event):
        self._update_style(False)
        urls = event.mimeData().urls()
        if urls:
            self.clicked.emit()
            self.setProperty("droppedPath", urls[0].toLocalFile())


class UploadTab(QWidget):
    saved = pyqtSignal()
    recognized = pyqtSignal(str)
    recognize_failed = pyqtSignal(str)

    def __init__(self, db, recognizer, keymap, upload_dir="data/uploads", advisor_fn=None):
        super().__init__()
        self._db = db
        self._recognizer = recognizer
        self._keymap = keymap
        self._upload_dir = upload_dir
        self._advisor_fn = advisor_fn   # () -> (advisor_cfg, provider) 由主窗口提供
        self._selected_path = None
        self._source_type = ""
        self._build_ui()
        self.recognized.connect(self._on_recognized)
        self.recognize_failed.connect(self._on_recognize_failed)

    def set_recognizer(self, recognizer):
        self._recognizer = recognizer

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

        # 步骤1:上传
        card, lay = self._card()
        lay.addLayout(_step_row("1", "上传乐谱文件"))
        self.drop_zone = DropZone()
        self.drop_zone.clicked.connect(self._pick_file)
        lay.addWidget(self.drop_zone)
        file_row = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setObjectName("FieldLabel")
        file_row.addWidget(self.file_label, 1)
        pick_btn = QPushButton("选择文件")
        pick_btn.setObjectName("BtnSecondary")
        pick_btn.clicked.connect(self._pick_file)
        file_row.addWidget(pick_btn)
        lay.addLayout(file_row)
        inner_layout.addWidget(card)

        # 步骤2:识别与解析(两条途径:内置模型在线识别;或外部 AI 工具识别后粘贴简谱)
        card, lay = self._card()
        lay.addLayout(_step_row("2", "识别与解析", "在线识别需模型设置 · 无模型时点「复制提示词」交给任意外部 AI 工具识别,把简谱粘贴到下方识别结果直接解析"))
        btn_row = QHBoxLayout()
        self.recognize_btn = QPushButton("开始识别")
        self.recognize_btn.setObjectName("BtnPrimary")
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.clicked.connect(self._recognize)
        self.parse_btn = QPushButton("解析到校对表格")
        self.parse_btn.setObjectName("BtnSecondary")
        self.parse_btn.clicked.connect(self._parse)
        btn_row.addWidget(self.recognize_btn)
        btn_row.addWidget(self.parse_btn)
        self.copy_prompt_btn = QPushButton("复制提示词")
        self.copy_prompt_btn.setObjectName("BtnSecondary")
        self.copy_prompt_btn.setToolTip("把简谱识别提示词复制到剪贴板:粘贴到任意外部 AI 工具并附上乐谱图片,即可得到本程序可解析的简谱文本")
        self.copy_prompt_btn.clicked.connect(self._copy_prompt)
        btn_row.addWidget(self.copy_prompt_btn)
        self.advisor_btn = QPushButton("AI 编谱建议")
        self.advisor_btn.setObjectName("BtnSecondary")
        self.advisor_btn.setToolTip("实验性:输入旋律描述或简谱片段,生成符合 21 键的编谱建议(需在模型设置页配置供应商)")
        self.advisor_btn.clicked.connect(self._open_advisor)
        btn_row.addWidget(self.advisor_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        inner_layout.addWidget(card)

        # 步骤3:识别结果(底部边缘可拖高,总滚动条随高度同步)
        card3, lay3 = self._card(resizable=True)
        card3.setup(height=200, min_height=160)
        lay3.addLayout(_step_row("3", "识别结果", "外部 AI 工具识别的简谱可直接粘贴到这里 · 可手动修改,修改后重新\"解析到校对表格\""))
        self.raw_text = QPlainTextEdit()
        self.raw_text.setPlaceholderText("识别结果将显示在这里;也可把外部 AI 工具识别出的简谱直接粘贴到此处,点击\"解析到校对表格\"...")
        self.raw_text.setCursor(Qt.CursorShape.IBeamCursor)
        lay3.addWidget(self.raw_text, 1)
        self._fix_card_cursors(card3)
        inner_layout.addWidget(card3)

        # 步骤4:校对表格(底部边缘可拖高,总滚动条随高度同步)
        card4, lay4 = self._card(resizable=True)
        card4.setup(height=300, min_height=220)
        lay4.addLayout(_step_row("4", "校对表格", "音符列可写和弦,如 high_1,mid_3;时值单位:拍"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["音符", "时值(拍)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay4.addWidget(self.table, 1)
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
        lay4.addLayout(edit_row)
        self._fix_card_cursors(card4)
        inner_layout.addWidget(card4)

        # 步骤5:保存
        card, lay = self._card()
        lay.addLayout(_step_row("5", "保存到乐谱库"))
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

    def _pick_file(self):
        path = self.drop_zone.property("droppedPath")
        self.drop_zone.setProperty("droppedPath", None)
        if not path or not os.path.exists(path):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择乐谱",
                "",
                "乐谱文件 (*.jpg *.jpeg *.png *.webp *.bmp *.md *.txt *.docx)",
            )
        if not path:
            return
        self._selected_path = path
        self.file_label.setText(path)
        self.drop_zone.text.setText(os.path.basename(path))
        ext = os.path.splitext(path)[1].lower()
        self._source_type = "image" if ext in _IMAGE_EXTS else "document"
        self.recognize_btn.setEnabled(True)
        self.raw_text.clear()

    def _recognize(self):
        if not self._selected_path:
            return
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.setText("识别中...")
        path = self._selected_path
        source_type = self._source_type
        t = threading.Thread(target=self._recognize_worker, args=(path, source_type), daemon=True)
        t.start()

    # ---------- AI 编谱建议(实验性) ----------

    def _open_advisor(self):
        if self._advisor_fn is None:
            AppDialog.show_warning(self, "提示", "编谱建议功能未启用")
            return
        advisor_cfg, provider = self._advisor_fn()
        if not (advisor_cfg or {}).get("enabled"):
            AppDialog.show_warning(
                self, "功能未开启",
                "AI 编谱建议为实验性功能,请在 config.yaml 中设置:\nadvisor:\n  enabled: true",
            )
            return
        from gui.advisor_dialog import AdvisorDialog

        AdvisorDialog(self, advisor_cfg, provider, on_accept=self._fill_from_advisor).exec()

    def _fill_from_advisor(self, text: str):
        """确认后的简谱填入识别结果框,走既有解析/校对/保存流程(用户确认后才入库)。"""
        self.raw_text.setPlainText(text)
        AppDialog.show_info(self, "已填入", "编谱建议已填入识别结果,请点击「解析到校对表格」继续校对与保存。")

    def _copy_prompt(self):
        """把规范化简谱提示词复制到剪贴板,供任意外部 AI 工具识别乐谱图片,免 API Key。"""
        QApplication.clipboard().setText(JIANPU_PROMPT)
        AppDialog.show_info(
            self, "提示词已复制",
            "已复制到剪贴板。\n\n使用方法:\n"
            "1. 粘贴到任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi 网页版)\n"
            "2. 附上乐谱图片一起发送\n"
            "3. 把返回的简谱粘贴到「识别结果」,点击「解析到校对表格」即可校对入库",
        )

    def _recognize_worker(self, path, source_type):
        try:
            if source_type == "image":
                text = self._recognizer.recognize_image(path)
            else:
                text = self._recognizer.recognize_document(_extract_text(path))
            self.recognized.emit(text)
        except Exception as e:
            self.recognize_failed.emit(str(e))

    def _on_recognized(self, text: str):
        self.raw_text.setPlainText(text)
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("重新识别")

    def _on_recognize_failed(self, msg: str):
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("开始识别")
        AppDialog.show_error(self, "识别失败", msg)

    def _parse(self):
        raw = self.raw_text.toPlainText().strip()
        if not raw:
            AppDialog.show_warning(self, "提示", "识别结果为空,请先识别或手动输入简谱")
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

    def _delete_selected_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _table_to_notes(self):
        notes = []
        for row in range(self.table.rowCount()):
            item_notes = self.table.item(row, 0)
            item_dur = self.table.item(row, 1)
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
            notes.append({"notes": note_ids, "dur": dur})
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
        source_file = ""
        if self._selected_path:
            os.makedirs(self._upload_dir, exist_ok=True)
            stem, ext = os.path.splitext(os.path.basename(self._selected_path))
            dest = os.path.join(self._upload_dir, f"{stem}_{int(time.time())}{ext}")
            shutil.copy2(self._selected_path, dest)
            source_file = dest
        self._db.add_score(
            name=name,
            notes=notes,
            raw_text=self.raw_text.toPlainText(),
            source_file=source_file,
            source_type=self._source_type or "manual",
            bpm_default=self.bpm_spin.value(),
        )
        AppDialog.show_success(self, "成功", f"《{name}》已保存到乐谱库")
        self.saved.emit()