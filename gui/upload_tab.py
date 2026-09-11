"""上传识别页(对齐 Web 设计稿):上传 -> 识别 -> 解析 -> 校对表格 -> 保存入库。"""

import os
import shutil
import sys
import tempfile
import threading
import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.parser import parse_jianpu
from core.jianpu_editor import delete_event, insert_rest_at_event
from core.lyrics import align_lyrics
from core.recognizer import (
    JIANPU_PROMPT,
    add_line_break_rests,
    build_recognition_text,
    fallback_score_name,
    parse_recognition_response,
)
from core.score_model import validate_notes
from core.score_preview import synthesize_preview
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
    settings_requested = pyqtSignal()

    def __init__(self, db, recognizer, keymap, upload_dir="data/uploads", advisor_fn=None):
        super().__init__()
        self._db = db
        self._recognizer = recognizer
        self._keymap = keymap
        self._upload_dir = upload_dir
        self._advisor_fn = advisor_fn   # () -> (advisor_cfg, provider) 由主窗口提供
        self._selected_path = None
        self._source_type = ""
        self._preview_path = None
        self._preview_ready = False
        self._rhythm_notes = ()
        self._lyrics_lines = ()
        self._lyrics_override = None
        self._recognition_started_at = None
        self._recognition_timer = QTimer(self)
        self._recognition_timer.timeout.connect(self._update_recognition_status)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._on_preview_finished)
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

        heading = QLabel("识别乐谱")
        heading.setObjectName("PageTitle")
        inner_layout.addWidget(heading)
        subheading = QLabel("上传谱面，校对曲名、音符和节奏后试听保存")
        subheading.setObjectName("PageSub")
        inner_layout.addWidget(subheading)

        # 上传与识别集中在一个工具区，避免步骤卡片堆叠。
        card, lay = self._card()
        top = QHBoxLayout()
        top.setSpacing(20)
        self.drop_zone = DropZone()
        self.drop_zone.clicked.connect(self._pick_file)
        self.drop_zone.setMaximumHeight(132)
        top.addWidget(self.drop_zone, 2)

        action_col = QVBoxLayout()
        action_col.setSpacing(10)
        self.file_label = QLabel("未选择文件")
        self.file_label.setObjectName("FieldLabel")
        self.file_label.setWordWrap(True)
        action_col.addWidget(self.file_label)
        pick_btn = QPushButton("选择文件")
        pick_btn.setObjectName("BtnSecondary")
        pick_btn.clicked.connect(self._pick_file)
        action_col.addWidget(pick_btn)
        btn_row = QHBoxLayout()
        self.recognize_btn = QPushButton("开始识别")
        self.recognize_btn.setObjectName("BtnPrimary")
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.clicked.connect(self._recognize)
        btn_row.addWidget(self.recognize_btn)
        self.copy_prompt_btn = QPushButton("复制提示词")
        self.copy_prompt_btn.setObjectName("BtnSecondary")
        self.copy_prompt_btn.setToolTip("把简谱识别提示词复制到剪贴板:粘贴到任意外部 AI 工具并附上乐谱图片,即可得到本程序可解析的简谱文本")
        self.copy_prompt_btn.clicked.connect(self._copy_prompt)
        btn_row.addWidget(self.copy_prompt_btn)
        action_col.addLayout(btn_row)
        self.advisor_btn = QPushButton("AI 编谱建议")
        self.advisor_btn.setObjectName("BtnSecondary")
        self.advisor_btn.setToolTip("实验性:输入旋律描述或简谱片段,生成符合 21 键的编谱建议(需在模型设置页配置供应商)")
        self.advisor_btn.clicked.connect(self._open_advisor)
        action_col.addWidget(self.advisor_btn)
        action_col.addStretch(1)
        top.addLayout(action_col, 1)
        lay.addLayout(top)
        inner_layout.addWidget(card)

        # 模型请求期间显示明确状态、耗时和当前分析阶段。
        self.recognition_panel = QFrame()
        self.recognition_panel.setObjectName("RecognitionPanel")
        recognition_lay = QVBoxLayout(self.recognition_panel)
        recognition_lay.setContentsMargins(16, 14, 16, 14)
        recognition_lay.setSpacing(8)
        recognition_top = QHBoxLayout()
        self.recognition_title = QLabel("正在识别中")
        self.recognition_title.setObjectName("RecognitionTitle")
        recognition_top.addWidget(self.recognition_title)
        self.recognition_elapsed = QLabel("0s")
        self.recognition_elapsed.setObjectName("RecognitionElapsed")
        recognition_top.addStretch(1)
        recognition_top.addWidget(self.recognition_elapsed)
        recognition_lay.addLayout(recognition_top)
        self.recognition_phase = QLabel("正在读取图片与曲名")
        self.recognition_phase.setObjectName("FieldLabel")
        recognition_lay.addWidget(self.recognition_phase)
        progress = QProgressBar()
        progress.setRange(0, 0)
        progress.setTextVisible(False)
        recognition_lay.addWidget(progress)
        self.recognition_panel.hide()
        inner_layout.addWidget(self.recognition_panel)

        # 一体化校对工作区：标题、节奏风险、音符、试听和保存。
        card3, lay3 = self._card(resizable=True)
        card3.setup(height=620, min_height=500)
        result_head = QHBoxLayout()
        result_title = QLabel("校对结果")
        result_title.setObjectName("SectionTitle")
        result_head.addWidget(result_title)
        self.result_summary = QLabel("等待识别或粘贴外部 AI 结果")
        self.result_summary.setObjectName("FieldLabel")
        result_head.addWidget(self.result_summary)
        result_head.addStretch(1)
        self.edit_raw_btn = QPushButton("编辑简谱")
        self.edit_raw_btn.setObjectName("BtnSecondary")
        self.edit_raw_btn.clicked.connect(self._toggle_raw_editor)
        result_head.addWidget(self.edit_raw_btn)
        lay3.addLayout(result_head)

        name_row = QHBoxLayout()
        name_label = QLabel("曲名")
        name_label.setObjectName("FieldLabel")
        name_row.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("AI 自动识别，未识别到时生成临时名称")
        name_row.addWidget(self.name_edit, 1)
        lay3.addLayout(name_row)

        self.rhythm_panel = QFrame()
        self.rhythm_panel.setObjectName("RhythmWarning")
        rhythm_lay = QHBoxLayout(self.rhythm_panel)
        rhythm_lay.setContentsMargins(12, 10, 12, 10)
        self.rhythm_label = QLabel("")
        self.rhythm_label.setWordWrap(True)
        self.rhythm_label.setObjectName("RhythmWarningText")
        rhythm_lay.addWidget(self.rhythm_label, 1)
        self.add_rests_btn = QPushButton("按分行补半拍停顿")
        self.add_rests_btn.setObjectName("BtnWarning")
        self.add_rests_btn.clicked.connect(self._add_line_rests)
        rhythm_lay.addWidget(self.add_rests_btn)
        self.rhythm_panel.hide()
        lay3.addWidget(self.rhythm_panel)

        self.raw_text = QPlainTextEdit()
        self.raw_text.setPlaceholderText("粘贴 TITLE/JIANPU/RHYTHM 或纯简谱；修改后点击“应用简谱修改”")
        self.raw_text.setCursor(Qt.CursorShape.IBeamCursor)
        self.raw_text.textChanged.connect(self._invalidate_parsed_result)
        self.raw_text.setMinimumHeight(112)
        self.raw_text.hide()
        lay3.addWidget(self.raw_text)
        raw_actions = QHBoxLayout()
        self.parse_btn = QPushButton("应用简谱修改")
        self.parse_btn.setObjectName("BtnSecondary")
        self.parse_btn.clicked.connect(self._parse)
        self.parse_btn.hide()
        raw_actions.addWidget(self.parse_btn)
        self.insert_rest_btn = QPushButton("在光标处插入半拍休止")
        self.insert_rest_btn.setObjectName("BtnSecondary")
        self.insert_rest_btn.clicked.connect(self._insert_rest)
        self.insert_rest_btn.hide()
        raw_actions.addWidget(self.insert_rest_btn)
        raw_actions.addStretch(1)
        lay3.addLayout(raw_actions)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["音符", "歌词", "时值(拍)", "半音"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(lambda _item: self._invalidate_preview())
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_note_menu)
        self.table.setToolTip("右键任一音符，可在前后插入半拍休止或删除")
        lay3.addWidget(self.table, 1)
        edit_row = QHBoxLayout()
        add_btn = QPushButton("添加行")
        add_btn.setObjectName("BtnSecondary")
        add_btn.clicked.connect(self._add_empty_row)
        del_btn = QPushButton("删除选中行")
        del_btn.setObjectName("BtnSecondary")
        del_btn.clicked.connect(self._delete_selected_rows)
        clear_btn = QPushButton("清空表格")
        clear_btn.setObjectName("BtnDanger")
        clear_btn.clicked.connect(self._clear_table)
        edit_row.addWidget(add_btn)
        edit_row.addWidget(del_btn)
        edit_row.addWidget(clear_btn)
        edit_row.addStretch(1)
        lay3.addLayout(edit_row)

        speed_head = QHBoxLayout()
        speed_title = QLabel("口琴试听")
        speed_title.setObjectName("SectionTitle")
        speed_head.addWidget(speed_title)
        speed_sub = QLabel("拖动调整速度")
        speed_sub.setObjectName("HintText")
        speed_head.addWidget(speed_sub)
        speed_head.addStretch(1)
        self.bpm_value = QLabel("100 BPM")
        self.bpm_value.setObjectName("BpmValue")
        speed_head.addWidget(self.bpm_value)
        lay3.addLayout(speed_head)

        self.bpm_slider = QSlider(Qt.Orientation.Horizontal)
        self.bpm_slider.setRange(0, 300)
        self.bpm_slider.setValue(100)
        self.bpm_slider.setToolTip("试听与保存速度，范围 0-300；0 BPM 时无法试听")
        self.bpm_slider.valueChanged.connect(self._on_bpm_changed)
        lay3.addWidget(self.bpm_slider)
        scale_row = QHBoxLayout()
        low = QLabel("0")
        low.setObjectName("HintText")
        scale_row.addWidget(low)
        scale_row.addStretch(1)
        self.bpm_hint = QLabel("仅使用口琴近似音色")
        self.bpm_hint.setObjectName("HintText")
        scale_row.addWidget(self.bpm_hint)
        scale_row.addStretch(1)
        high = QLabel("300")
        high.setObjectName("HintText")
        scale_row.addWidget(high)
        lay3.addLayout(scale_row)

        preview_row = QHBoxLayout()
        self.preview_btn = QPushButton("▶ 试听")
        self.preview_btn.setObjectName("BtnSecondary")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._preview)
        preview_row.addWidget(self.preview_btn)
        self.stop_preview_btn = QPushButton("■ 停止试听")
        self.stop_preview_btn.setObjectName("BtnStop")
        self.stop_preview_btn.setEnabled(False)
        self.stop_preview_btn.clicked.connect(self._stop_preview_by_user)
        preview_row.addWidget(self.stop_preview_btn)
        self.preview_status = QLabel("解析校对表格后可试听")
        self.preview_status.setObjectName("HintText")
        preview_row.addWidget(self.preview_status, 1)
        lay3.addLayout(preview_row)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_btn = QPushButton("确认无误，保存入库")
        save_btn.setObjectName("BtnPrimary")
        save_btn.setEnabled(False)
        save_btn.clicked.connect(self._save)
        self.save_btn = save_btn
        save_row.addWidget(save_btn)
        lay3.addLayout(save_row)

        self._fix_card_cursors(card3)
        inner_layout.addWidget(card3)

        inner_layout.addStretch(1)

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
        self.stop_preview()
        self.file_label.setText(path)
        self.drop_zone.text.setText(os.path.basename(path))
        ext = os.path.splitext(path)[1].lower()
        self._source_type = "image" if ext in _IMAGE_EXTS else "document"
        self._rhythm_notes = ()
        self._lyrics_lines = ()
        self._lyrics_override = None
        self.rhythm_panel.hide()
        self.recognize_btn.setEnabled(True)
        self.raw_text.clear()
        self.table.setRowCount(0)
        self._reset_review_state()
        self.name_edit.setText(fallback_score_name())

    def _recognize(self):
        if not self._selected_path:
            return
        if self._recognizer is None:
            choice = AppDialog._popup(
                self, "warning", "尚未配置识别模型",
                "开始识别需要一个已激活的多模态模型。你可以先去模型设置，或复制提示词到任意外部 AI 工具识别后粘贴结果。",
                [("去模型设置", "primary"), ("复制提示词", "secondary"), ("取消", "secondary")],
            )
            if choice == "去模型设置":
                self.settings_requested.emit()
            elif choice == "复制提示词":
                self._copy_prompt()
            return
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.setText("正在识别")
        self._recognition_started_at = time.monotonic()
        self.recognition_elapsed.setText("0s")
        self.recognition_phase.setText("正在读取图片与曲名")
        self.recognition_panel.show()
        self._recognition_timer.start(1000)
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
        self._set_raw_editor_visible(True)
        self._parse()
        AppDialog.show_info(self, "已填入", "编谱建议已填入校对区，请检查音符和节奏后试听。")

    def _copy_prompt(self):
        """把规范化简谱提示词复制到剪贴板,供任意外部 AI 工具识别乐谱图片,免 API Key。"""
        QApplication.clipboard().setText(JIANPU_PROMPT)
        AppDialog.show_info(
            self, "提示词已复制",
            "已复制到剪贴板。\n\n使用方法:\n"
            "1. 粘贴到任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi 网页版)\n"
            "2. 附上乐谱图片一起发送\n"
            "3. 点击「编辑简谱」粘贴 TITLE/JIANPU/RHYTHM，应用修改后试听保存",
        )

    def _update_recognition_status(self):
        if self._recognition_started_at is None:
            return
        elapsed = max(0, int(time.monotonic() - self._recognition_started_at))
        if elapsed < 3:
            phase = "正在读取图片与曲名"
        elif elapsed < 8:
            phase = "正在定位音符与乐句"
        elif elapsed < 15:
            phase = "正在对齐音符与歌词"
        else:
            phase = "正在整理校对结果"
        self.recognition_elapsed.setText(f"{elapsed}s")
        self.recognition_phase.setText(phase)

    def _stop_recognition_status(self):
        self._recognition_timer.stop()
        self._recognition_started_at = None
        self.recognition_panel.hide()

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
        self._stop_recognition_status()
        result = parse_recognition_response(text)
        self.raw_text.setPlainText(result.jianpu_text)
        self._rhythm_notes = result.rhythm_notes
        self._lyrics_lines = result.lyrics_lines
        self._lyrics_override = None
        self.table.setRowCount(0)
        if result.title:
            self.name_edit.setText(result.title)
        elif not self.name_edit.text().strip():
            self.name_edit.setText(fallback_score_name())
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("重新识别")
        self._reset_review_state()
        self._set_raw_editor_visible(False)
        self._parse()

    def _on_recognize_failed(self, msg: str):
        self._stop_recognition_status()
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("开始识别")
        AppDialog.show_error(self, "识别失败", msg)

    def _parse(self):
        raw = self.raw_text.toPlainText().strip()
        if not raw:
            AppDialog.show_warning(self, "提示", "识别结果为空,请先识别或手动输入简谱")
            return
        try:
            result = parse_recognition_response(raw)
            notes = parse_jianpu(result.jianpu_text)
        except Exception as e:
            AppDialog.show_error(self, "解析失败", str(e))
            return
        if not notes:
            AppDialog.show_warning(self, "提示", "未能从文本中解析出音符,请检查格式")
            return
        if result.title:
            self.name_edit.setText(result.title)
        elif not self.name_edit.text().strip():
            self.name_edit.setText(fallback_score_name())
        if result.jianpu_text != raw:
            self.raw_text.blockSignals(True)
            self.raw_text.setPlainText(result.jianpu_text)
            self.raw_text.blockSignals(False)
        if result.rhythm_notes:
            self._rhythm_notes = result.rhythm_notes
        if result.lyrics_lines:
            self._lyrics_lines = result.lyrics_lines
        lyrics = self._lyrics_override
        if lyrics is None or len(lyrics) != len(notes):
            lyrics = align_lyrics(result.jianpu_text, self._lyrics_lines)
        self._lyrics_override = None
        self.table.setRowCount(0)
        for index, n in enumerate(notes):
            row = self.table.rowCount()
            self.table.insertRow(row)
            notes_str = ",".join(n["notes"]) if n["notes"] else "(休止)"
            self.table.setItem(row, 0, QTableWidgetItem(notes_str))
            self.table.setItem(row, 1, QTableWidgetItem(lyrics[index] if index < len(lyrics) else ""))
            self.table.setItem(row, 2, QTableWidgetItem(str(n["dur"])))
            self.table.setItem(row, 3, QTableWidgetItem("#" if n.get("semitone") else ""))
        self._preview_ready = False
        self.save_btn.setEnabled(False)
        self.preview_btn.setEnabled(self.bpm_slider.value() > 0)
        rests = sum(1 for note in notes if not note["notes"])
        beats = sum(note["dur"] for note in notes)
        self.result_summary.setText(f"{len(notes)} 个音符 · {rests} 个休止 · {beats:g} 拍")
        self.preview_status.setText("校对完成，请试听确认")
        self._update_rhythm_panel(notes)

    def _show_note_menu(self, point):
        index = self.table.indexAt(point)
        if not index.isValid():
            return
        from PyQt6.QtWidgets import QMenu

        row = index.row()
        self.table.selectRow(row)
        menu = QMenu(self.table)
        menu.setObjectName("NoteActionMenu")
        note = self.table.item(row, 0).text() if self.table.item(row, 0) else ""
        lyric = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
        beats = self.table.item(row, 2).text() if self.table.item(row, 2) else ""
        summary = f"第 {row + 1} 个 · {note}"
        if lyric:
            summary += f" · {lyric}"
        if beats:
            summary += f" · {beats} 拍"
        title = menu.addAction(summary)
        title.setEnabled(False)
        menu.addSeparator()
        before = menu.addAction("前面插入半拍休止")
        after = menu.addAction("后面插入半拍休止")
        menu.addSeparator()
        delete = menu.addAction("删除这个音符")
        action = menu.exec(self.table.viewport().mapToGlobal(point))
        if action == before:
            self._insert_rest_at_row(row, before=True)
        elif action == after:
            self._insert_rest_at_row(row, before=False)
        elif action == delete:
            self._delete_event_at_row(row)

    def _insert_rest_at_row(self, row, *, before):
        lyrics = [self.table.item(i, 1).text() if self.table.item(i, 1) else ""
                  for i in range(self.table.rowCount())]
        updated = insert_rest_at_event(self.raw_text.toPlainText(), row, before=before)
        if updated == self.raw_text.toPlainText():
            return
        self.raw_text.setPlainText(updated)
        lyrics.insert(row if before else row + 1, "")
        self._lyrics_override = lyrics
        self._parse()
        target = row if before else row + 1
        if 0 <= target < self.table.rowCount():
            self.table.selectRow(target)

    def _delete_event_at_row(self, row):
        lyrics = [self.table.item(i, 1).text() if self.table.item(i, 1) else ""
                  for i in range(self.table.rowCount())]
        updated = delete_event(self.raw_text.toPlainText(), row)
        if updated == self.raw_text.toPlainText():
            return
        self.raw_text.setPlainText(updated)
        if 0 <= row < len(lyrics):
            lyrics.pop(row)
        self._lyrics_override = lyrics
        self._parse()
        if self.table.rowCount():
            self.table.selectRow(min(row, self.table.rowCount() - 1))

    def _on_bpm_changed(self, value):
        self.bpm_value.setText(f"{value} BPM")
        self.bpm_hint.setText("仅使用口琴近似音色" if value > 0 else "0 BPM 无法试听")
        self.bpm_value.setProperty("invalid", value == 0)
        self.bpm_value.style().unpolish(self.bpm_value)
        self.bpm_value.style().polish(self.bpm_value)
        self._invalidate_preview()

    def _toggle_raw_editor(self):
        self._set_raw_editor_visible(not self.raw_text.isVisible())

    def _set_raw_editor_visible(self, visible):
        self.raw_text.setVisible(visible)
        self.parse_btn.setVisible(visible)
        self.insert_rest_btn.setVisible(visible)
        self.edit_raw_btn.setText("收起简谱编辑" if visible else "编辑简谱")
        if visible:
            self.raw_text.setFocus()

    def _insert_rest(self):
        cursor = self.raw_text.textCursor()
        before = self.raw_text.toPlainText()[max(0, cursor.selectionStart() - 1):cursor.selectionStart()]
        after = self.raw_text.toPlainText()[cursor.selectionEnd():cursor.selectionEnd() + 1]
        token = ("" if not before or before.isspace() else " ") + "0_" + (
            "" if not after or after.isspace() else " "
        )
        cursor.insertText(token)
        self.raw_text.setTextCursor(cursor)

    def _add_line_rests(self):
        updated = add_line_break_rests(self.raw_text.toPlainText())
        if updated == self.raw_text.toPlainText().strip():
            return
        lyrics = [self.table.item(i, 1).text() if self.table.item(i, 1) else ""
                  for i in range(self.table.rowCount())]
        original_lines = [line for line in self.raw_text.toPlainText().splitlines() if line.strip()]
        if len(original_lines) > 1:
            event_offset = 0
            for line in original_lines[:-1]:
                event_offset += len(parse_jianpu(line))
                lyrics.insert(event_offset, "")
                event_offset += 1
        self.raw_text.setPlainText(updated)
        self._lyrics_override = lyrics
        self._rhythm_notes = tuple(self._rhythm_notes) + ("已按原谱分行补入半拍休止，请试听确认",)
        self._set_raw_editor_visible(True)
        self._parse()

    def _update_rhythm_panel(self, notes):
        rest_count = sum(1 for note in notes if not note["notes"])
        line_count = sum(1 for line in self.raw_text.toPlainText().splitlines() if line.strip())
        needs_review = bool(notes) and rest_count == 0 and line_count > 1
        warnings = list(self._rhythm_notes)
        if needs_review:
            warnings.append("未识别到休止符；原图若有乐句停顿，请试听校对")
        self.rhythm_label.setText("\n".join(warnings))
        self.add_rests_btn.setVisible(needs_review)
        self.rhythm_panel.setVisible(bool(warnings))

    def _reset_review_state(self):
        self._preview_ready = False
        if hasattr(self, "save_btn"):
            self.save_btn.setEnabled(False)
        if hasattr(self, "preview_btn"):
            self.preview_btn.setEnabled(False)
        if hasattr(self, "preview_status"):
            self.preview_status.setText("解析校对表格后可试听")
        if hasattr(self, "result_summary") and self.table.rowCount() == 0:
            self.result_summary.setText("等待识别或粘贴外部 AI 结果")

    def _invalidate_parsed_result(self):
        """识别原文变化后，表格必须重新解析才能继续试听或保存。"""
        if not hasattr(self, "preview_btn"):
            return
        self._preview_ready = False
        self.preview_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.stop_preview()
        if self.raw_text.toPlainText().strip():
            self.preview_status.setText("识别结果已修改，请重新解析")

    def _invalidate_preview(self):
        """校对内容或 BPM 变化后，之前的试听确认不再代表当前结果。"""
        if not hasattr(self, "save_btn"):
            return
        self._preview_ready = False
        self.save_btn.setEnabled(False)
        self.stop_preview()
        if self.table.rowCount() > 0:
            enabled = self.bpm_slider.value() > 0
            self.preview_btn.setEnabled(enabled)
            self.preview_status.setText("内容已修改，请重新试听" if enabled else "0 BPM 无法试听")

    def _preview(self):
        """合成试听音频并交给系统播放器，不触发键盘演奏线程。"""
        try:
            notes = self._table_to_notes()
            if not notes:
                raise ValueError("表格为空,无法试听")
            if self.bpm_slider.value() == 0:
                raise ValueError("0 BPM 无法试听，请向右拖动速度")
            wav_data, duration = synthesize_preview(notes, self.bpm_slider.value())
            self.stop_preview()
            fd, path = tempfile.mkstemp(prefix="amp_preview_", suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(wav_data)
            self._preview_path = path
            if sys.platform != "win32":
                raise RuntimeError("试听播放目前仅支持 Windows")
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._preview_ready = True
            self.preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.preview_status.setText(f"正在试听 · {duration:.1f} 秒，确认无误后可保存")
            self._preview_timer.start(max(1, round(duration * 1000)))
        except Exception as e:
            self.stop_preview()
            AppDialog.show_error(self, "试听失败", str(e))

    def stop_preview(self):
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, 0)
        except Exception:
            pass
        self.stop_preview_btn.setEnabled(False) if hasattr(self, "stop_preview_btn") else None
        if self._preview_path:
            try:
                os.remove(self._preview_path)
            except OSError:
                pass
            self._preview_path = None

    def _on_preview_finished(self):
        self.stop_preview()
        if self._preview_ready:
            self.preview_status.setText("试听完成，确认无误后可保存")

    def _stop_preview_by_user(self):
        self.stop_preview()
        if self._preview_ready:
            self.preview_status.setText("试听已停止，确认无误后可保存")

    def _delete_selected_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        lyrics = [self.table.item(i, 1).text() if self.table.item(i, 1) else ""
                  for i in range(self.table.rowCount())]
        updated = self.raw_text.toPlainText()
        for row in rows:
            updated = delete_event(updated, row)
            if 0 <= row < len(lyrics):
                lyrics.pop(row)
        self.raw_text.setPlainText(updated)
        self._lyrics_override = lyrics
        self._parse()

    def _add_empty_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("mid_1"))
        self.table.setItem(row, 1, QTableWidgetItem(""))
        self.table.setItem(row, 2, QTableWidgetItem("1.0"))
        self.table.setItem(row, 3, QTableWidgetItem(""))
        self._invalidate_preview()

    def _clear_table(self):
        self.table.setRowCount(0)
        self._reset_review_state()

    def _table_to_notes(self):
        notes = []
        for row in range(self.table.rowCount()):
            item_notes = self.table.item(row, 0)
            item_dur = self.table.item(row, 2)
            item_semi = self.table.item(row, 3)
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
        if not self._preview_ready:
            AppDialog.show_info(self, "请先试听", "先点击「试听」确认当前校对结果，确认无误后再保存。")
            return
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
            raw_text=build_recognition_text(
                name,
                self.raw_text.toPlainText(),
                [self.table.item(row, 1).text() if self.table.item(row, 1) else ""
                 for row in range(self.table.rowCount())],
                self._rhythm_notes,
            ),
            source_file=source_file,
            source_type=self._source_type or "manual",
            bpm_default=self.bpm_slider.value(),
        )
        self.stop_preview()
        self._preview_ready = False
        self.save_btn.setEnabled(False)
        self.preview_status.setText("已保存到乐谱库")
        AppDialog.show_success(self, "成功", f"《{name}》已保存到乐谱库")
        self.saved.emit()
