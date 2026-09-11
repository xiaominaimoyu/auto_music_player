"""乐谱库详情：编辑曲名/简谱/歌词/BPM，并用口琴音色试听。"""

import os
import sys
import tempfile

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.lyrics import align_lyrics
from core.parser import parse_jianpu
from core.recognizer import build_recognition_text, parse_recognition_response
from core.score_model import validate_notes
from core.score_preview import synthesize_preview
from gui.widgets import AppDialog


class ScoreDetailDialog(QDialog):
    def __init__(self, parent, db, score):
        super().__init__(parent)
        self._db = db
        self._score = score
        self._preview_path = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._preview_finished)
        self.saved = False
        self.play_requested = False
        self.setObjectName("ScoreDetailDialog")
        self.setWindowTitle(f"乐谱详情 · {score['name']}")
        self.setMinimumSize(760, 640)
        self.resize(920, 760)
        self._build_ui()
        self._load_score()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        head = QHBoxLayout()
        title = QLabel("乐谱详情")
        title.setObjectName("PageTitle")
        head.addWidget(title)
        self.summary = QLabel("")
        self.summary.setObjectName("PageSub")
        head.addWidget(self.summary)
        head.addStretch(1)
        root.addLayout(head)

        meta = QFrame()
        meta.setObjectName("SectionCard")
        meta.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        meta_lay = QVBoxLayout(meta)
        meta_lay.setContentsMargins(18, 16, 18, 16)
        meta_lay.setSpacing(10)
        name_label = QLabel("曲名")
        name_label.setObjectName("FieldLabel")
        meta_lay.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(80)
        meta_lay.addWidget(self.name_edit)

        editors = QHBoxLayout()
        editors.setSpacing(12)
        score_col = QVBoxLayout()
        score_label = QLabel("简谱")
        score_label.setObjectName("FieldLabel")
        score_col.addWidget(score_label)
        self.jianpu_edit = QPlainTextEdit()
        self.jianpu_edit.setPlaceholderText("该乐谱没有可编辑的原始简谱")
        self.jianpu_edit.setFixedHeight(110)
        score_col.addWidget(self.jianpu_edit)
        editors.addLayout(score_col, 2)
        lyrics_col = QVBoxLayout()
        lyrics_label = QLabel("歌词（一音一字，空格分隔）")
        lyrics_label.setObjectName("FieldLabel")
        lyrics_col.addWidget(lyrics_label)
        self.lyrics_edit = QPlainTextEdit()
        self.lyrics_edit.setFixedHeight(110)
        lyrics_col.addWidget(self.lyrics_edit)
        editors.addLayout(lyrics_col, 1)
        meta_lay.addLayout(editors)
        apply_btn = QPushButton("应用文本修改")
        apply_btn.setObjectName("BtnSecondary")
        apply_btn.clicked.connect(self._apply_score_text)
        meta_lay.addWidget(apply_btn, 0, Qt.AlignmentFlag.AlignRight)
        root.addWidget(meta)

        table_head = QHBoxLayout()
        table_title = QLabel("音符与歌词")
        table_title.setObjectName("SectionTitle")
        table_head.addWidget(table_title)
        table_head.addStretch(1)
        self.table_count = QLabel("")
        self.table_count.setObjectName("HintText")
        table_head.addWidget(self.table_count)
        root.addLayout(table_head)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["音符", "歌词", "时值(拍)", "半音"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        preview = QFrame()
        preview.setObjectName("SectionCard")
        preview_lay = QVBoxLayout(preview)
        preview_lay.setContentsMargins(18, 14, 18, 14)
        preview_lay.setSpacing(8)
        speed_head = QHBoxLayout()
        speed_title = QLabel("口琴试听")
        speed_title.setObjectName("SectionTitle")
        speed_head.addWidget(speed_title)
        speed_head.addStretch(1)
        self.bpm_value = QLabel("100 BPM")
        self.bpm_value.setObjectName("BpmValue")
        speed_head.addWidget(self.bpm_value)
        preview_lay.addLayout(speed_head)
        self.bpm_slider = QSlider(Qt.Orientation.Horizontal)
        self.bpm_slider.setRange(1, 300)
        self.bpm_slider.valueChanged.connect(lambda value: self.bpm_value.setText(f"{value} BPM"))
        self.bpm_slider.valueChanged.connect(lambda _value: self.stop_preview())
        preview_lay.addWidget(self.bpm_slider)
        preview_actions = QHBoxLayout()
        self.preview_btn = QPushButton("▶ 试听")
        self.preview_btn.setObjectName("BtnPrimary")
        self.preview_btn.clicked.connect(self._preview)
        preview_actions.addWidget(self.preview_btn)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setObjectName("BtnStop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_preview)
        preview_actions.addWidget(self.stop_btn)
        self.preview_status = QLabel("可直接试听当前编辑内容")
        self.preview_status.setObjectName("HintText")
        preview_actions.addWidget(self.preview_status, 1)
        preview_lay.addLayout(preview_actions)
        root.addWidget(preview)

        actions = QHBoxLayout()
        cancel = QPushButton("关闭")
        cancel.setObjectName("BtnSecondary")
        cancel.clicked.connect(self.reject)
        play = QPushButton("进入演奏")
        play.setObjectName("BtnSecondary")
        play.clicked.connect(self._go_play)
        save = QPushButton("保存修改")
        save.setObjectName("BtnPrimary")
        save.clicked.connect(self._save)
        actions.addStretch(1)
        actions.addWidget(cancel)
        actions.addWidget(play)
        actions.addWidget(save)
        root.addLayout(actions)

    def _load_score(self):
        result = parse_recognition_response(self._score.get("raw_text", ""))
        self._rhythm_notes = result.rhythm_notes
        self.name_edit.setText(self._score["name"])
        self.jianpu_edit.setPlainText(result.jianpu_text)
        lyrics = align_lyrics(result.jianpu_text, result.lyrics_lines)
        self.lyrics_edit.setPlainText(" ".join(item or "_" for item in lyrics) if result.lyrics_lines else "")
        self.bpm_slider.setValue(int(self._score["bpm_default"]))
        self._apply_score_text(silent=True)

    def _notes_and_lyrics(self):
        text = self.jianpu_edit.toPlainText().strip()
        notes = parse_jianpu(text) if text else list(self._score.get("notes") or [])
        lyrics_source = self.lyrics_edit.toPlainText().strip()
        lyrics = align_lyrics(text, (lyrics_source,)) if text else ["" for _ in notes]
        return notes, lyrics

    def _apply_score_text(self, _checked=False, *, silent=False):
        try:
            notes, lyrics = self._notes_and_lyrics()
        except Exception as exc:
            if not silent:
                AppDialog.show_error(self, "解析失败", str(exc))
            return False
        if not notes:
            if not silent:
                AppDialog.show_warning(self, "简谱为空", "没有解析到可用音符")
            return False
        self.table.setRowCount(0)
        for index, note in enumerate(notes):
            row = self.table.rowCount()
            self.table.insertRow(row)
            label = ",".join(note["notes"]) if note["notes"] else "(休止)"
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(lyrics[index] if index < len(lyrics) else ""))
            self.table.setItem(row, 2, QTableWidgetItem(str(note["dur"])))
            self.table.setItem(row, 3, QTableWidgetItem("#" if note.get("semitone") else ""))
        rests = sum(1 for note in notes if not note["notes"])
        self.summary.setText(f"{len(notes)} 个音符 · {rests} 个休止")
        self.table_count.setText(f"共 {len(notes)} 个")
        self.stop_preview()
        if not silent:
            self.preview_status.setText("文本修改已应用")
        return True

    def _preview(self):
        if not self._apply_score_text(silent=True):
            return
        try:
            notes, _lyrics = self._notes_and_lyrics()
            wav_data, duration = synthesize_preview(notes, self.bpm_slider.value())
            self.stop_preview()
            fd, path = tempfile.mkstemp(prefix="amp_library_preview_", suffix=".wav")
            with os.fdopen(fd, "wb") as stream:
                stream.write(wav_data)
            self._preview_path = path
            if sys.platform != "win32":
                raise RuntimeError("试听播放目前仅支持 Windows")
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.stop_btn.setEnabled(True)
            self.preview_status.setText(f"正在试听 · {duration:.1f} 秒")
            self._preview_timer.start(max(1, round(duration * 1000)))
        except Exception as exc:
            self.stop_preview()
            AppDialog.show_error(self, "试听失败", str(exc))

    def stop_preview(self):
        self._preview_timer.stop()
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(None, 0)
        except Exception:
            pass
        self.stop_btn.setEnabled(False)
        if self._preview_path:
            try:
                os.remove(self._preview_path)
            except OSError:
                pass
            self._preview_path = None

    def _preview_finished(self):
        self.stop_preview()
        self.preview_status.setText("试听完成")

    def _save(self):
        name = self.name_edit.text().strip()
        if not name:
            AppDialog.show_warning(self, "无法保存", "曲名不能为空")
            return
        if not self._apply_score_text(silent=True):
            AppDialog.show_warning(self, "无法保存", "没有可保存的音符")
            return
        notes, lyrics = self._notes_and_lyrics()
        errors = validate_notes(notes, bpm=self.bpm_slider.value())
        if errors:
            AppDialog.show_error(self, "无法保存", "\n".join(str(error) for error in errors[:6]))
            return
        jianpu = self.jianpu_edit.toPlainText().strip()
        raw_text = (
            build_recognition_text(name, jianpu, lyrics, self._rhythm_notes)
            if jianpu else self._score.get("raw_text", "")
        )
        self._db.update_score(
            self._score["id"],
            name,
            notes,
            raw_text=raw_text,
            bpm_default=self.bpm_slider.value(),
        )
        self.saved = True
        self.stop_preview()
        self.accept()

    def _go_play(self):
        self.play_requested = True
        self.stop_preview()
        self.accept()

    def closeEvent(self, event):
        self.stop_preview()
        super().closeEvent(event)
