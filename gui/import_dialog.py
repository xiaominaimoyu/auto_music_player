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
        self.style_combo.addItem("钢琴旋律适配（推荐）", "piano")
        self.style_combo.addItem("最高活动音单旋律", "original")
        self.style_combo.addItem("保留严格同时和弦", "preserve")
        form.addRow("处理方式", self.style_combo)

        self.transpose_spin = QSpinBox()
        self.transpose_spin.setObjectName("MidiTransposeSpin")
        self.transpose_spin.setRange(-24, 24)
        self.transpose_spin.setSuffix(" 半音")
        form.addRow("移调", self.transpose_spin)

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

    def options(self) -> AdaptOptions:
        return AdaptOptions(
            track=self.track_combo.currentData(),
            style=self.style_combo.currentData(),
            transpose=self.transpose_spin.value(),
            fold_octaves=self.fold_checkbox.isChecked(),
            bpm=self.bpm_spin.value(),
        )
