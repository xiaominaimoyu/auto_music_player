"""MIDI import options shown after parsing and before database writes."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from core.melody import extract_melody, recommend_transpose
from core.source_score import AdaptOptions, recommend_track


class MidiImportDialog(QDialog):
    def __init__(self, song, parent=None):
        super().__init__(parent)
        self._song = song
        self.setWindowTitle("MIDI 导入设置")
        self.setModal(True)
        self.setMinimumWidth(480)

        root = QVBoxLayout(self)
        summary = QLabel(
            f"《{song.title}》 · {len(song.notes)} 个来源音符 · "
            f"{len(song.tracks)} 条有效轨道 · {song.duration_s:.1f} 秒"
        )
        summary.setWordWrap(True)
        summary.setObjectName("MidiImportSummary")
        root.addWidget(summary)

        form = QFormLayout()
        self.track_combo = QComboBox()
        self.track_combo.setObjectName("MidiTrackCombo")
        self.track_combo.addItem("自动推荐旋律轨", "auto")
        self.track_combo.addItem("合并全部轨道", None)
        recommended = recommend_track(song) if song.notes else None
        for track, name in sorted(song.tracks.items()):
            suffix = "（推荐）" if track == recommended else ""
            count = sum(note.track == track for note in song.notes)
            self.track_combo.addItem(f"{name}{suffix} · {count} 音符", track)
        form.addRow("音轨", self.track_combo)

        self.style_combo = QComboBox()
        self.style_combo.setObjectName("MidiStyleCombo")
        self.style_combo.addItem("旋律跟踪（推荐）", "melody")
        self.style_combo.addItem("钢琴旋律适配", "piano")
        self.style_combo.addItem("最高活动音单旋律", "original")
        self.style_combo.addItem("保留严格同时和弦", "preserve")
        form.addRow("处理方式", self.style_combo)

        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("MidiKeyboardProfile")
        self.profile_combo.addItem("21 键（鸣潮 / 原神，无黑键）", "21key")
        self.profile_combo.addItem("三角洲口琴（可升半音）", "delta")
        form.addRow("目标键盘", self.profile_combo)

        self.transpose_spin = QSpinBox()
        self.transpose_spin.setObjectName("MidiTransposeSpin")
        self.transpose_spin.setRange(-24, 24)
        self.transpose_spin.setSuffix(" 半音")
        self.transpose_spin.valueChanged.connect(self._on_transpose_edited)
        form.addRow("移调", self.transpose_spin)

        self.recommendation_label = QLabel("")
        self.recommendation_label.setObjectName("MidiTransposeAdvice")
        self.recommendation_label.setWordWrap(True)
        form.addRow("移调建议", self.recommendation_label)

        self.bpm_spin = QSpinBox()
        self.bpm_spin.setObjectName("MidiBpmSpin")
        self.bpm_spin.setRange(30, 300)
        self.bpm_spin.setValue(max(30, min(300, round(song.bpm_hint))))
        self.bpm_spin.setSuffix(" BPM")
        form.addRow("入库速度", self.bpm_spin)

        self.fold_checkbox = QCheckBox("超出 C3-B5 时按八度折回，并写入降级报告")
        self.fold_checkbox.setObjectName("MidiFoldOctaves")
        self.fold_checkbox.setChecked(True)
        form.addRow("范围处理", self.fold_checkbox)
        root.addLayout(form)

        hint = QLabel(
            "导入只生成标准乐谱，不会向游戏发送任何键盘或鼠标输入。"
            "所有自动简化都会在确认页列出。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintText")
        root.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("解析并继续")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._updating_spin = False
        self._follow_recommendation = True
        self._advice = None
        self._apply_recommendation()
        self.profile_combo.currentIndexChanged.connect(self._apply_recommendation)

    def _melody_notes(self):
        track = recommend_track(self._song) if self._song.notes else None
        notes = [
            note for note in self._song.notes
            if track is None or note.track == track
        ]
        return extract_melody(notes)

    def _apply_recommendation(self):
        profile = self.profile_combo.currentData() or "21key"
        self._advice = recommend_transpose(self._melody_notes(), profile=profile)
        advice = self._advice
        self.recommendation_label.setText(
            f"建议 {advice.semitones:+d} 半音"
            f"（折八度 {advice.fold_count}，黑键 {advice.black_key_count}，"
            f"走向破坏 {advice.contour_breaks}）。可以直接改上面的移调。"
        )
        if self._follow_recommendation:
            self._updating_spin = True
            self.transpose_spin.setValue(advice.semitones)
            self._updating_spin = False

    def _on_transpose_edited(self, _value):
        if getattr(self, "_updating_spin", True):
            return
        self._follow_recommendation = False

    def options(self) -> AdaptOptions:
        return AdaptOptions(
            track=self.track_combo.currentData(),
            style=self.style_combo.currentData(),
            transpose=self.transpose_spin.value(),
            fold_octaves=self.fold_checkbox.isChecked(),
            bpm=self.bpm_spin.value(),
        )
