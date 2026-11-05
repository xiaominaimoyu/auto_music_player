"""外观草稿、实时预览与显式保存；取消和关闭还原原设置。"""

from dataclasses import replace
from pathlib import Path
import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from core.appearance import AppearanceSettings, GRADIENTS
from gui.background import BackgroundWidget, encode_background_image, load_background_image
from gui.theme import INK, INK_2, LINE_2, STATE_ERROR, SURFACE, appearance_qss


class AppearanceDialog(QDialog):
    preview_changed = pyqtSignal(object, object)

    def __init__(self, store, settings, image, parent=None):
        super().__init__(parent)
        self.setObjectName("AppearanceDialog")
        self.setWindowTitle("外观设置")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setMinimumWidth(730)
        self._store = store
        self.settings = settings
        self.image = QImage(image)
        self._original_image = QImage(image)
        self._draft = settings
        self._image_changed = False
        self._loading = False
        self._drag_offset = None
        self.setStyleSheet(f"""
            QDialog#AppearanceDialog {{ background: {SURFACE}; border: 1px solid {LINE_2}; color: {INK}; }}
            QLabel#AppearanceHeading {{ color: {INK}; font-size: 18px; font-weight: 600; }}
            QLabel#AppearanceError {{ color: {STATE_ERROR}; font-size: 12px; }}
            QLabel#AppearanceValue {{ color: {INK_2}; min-width: 36px; }}
        """)
        self._build_ui()
        self._fill_controls()

    @staticmethod
    def _label(text, name="FieldLabel"):
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        return label

    def _field(self, layout, text, control):
        row = QWidget()
        column = QVBoxLayout(row)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(5)
        column.addWidget(self._label(text))
        column.addWidget(control)
        layout.addWidget(row)
        return row

    @staticmethod
    def _button(text, callback, primary=False):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.setObjectName("BtnPrimary" if primary else "BtnSecondary")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _slider(self, layout, text):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        value = self._label("", "AppearanceValue")
        row.addWidget(slider, 1)
        row.addWidget(value)
        field = self._field(layout, text, container)
        slider.valueChanged.connect(self._controls_changed)
        return slider, value, field

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        header = QHBoxLayout()
        header.addWidget(self._label("外观设置", "AppearanceHeading"))
        header.addStretch()
        close = self._button("×", self.reject)
        close.setAccessibleName("关闭外观设置")
        close.setFixedWidth(46)
        close.setStyleSheet("padding: 0;")
        header.addWidget(close)
        layout.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(24)
        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(self._label("背景预览", "SectionTitle"))
        self.preview = BackgroundWidget()
        self.preview.setObjectName("BackgroundPreview")
        self.preview.setFixedSize(300, 194)
        preview_layout = QVBoxLayout(self.preview)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.addWidget(self._label("自动演奏器", "SectionTitle"))
        preview_layout.addStretch()
        sample = QFrame()
        sample.setObjectName("SectionCard")
        sample_layout = QVBoxLayout(sample)
        sample_layout.addWidget(self._label("我的乐谱", "SectionTitle"))
        sample_layout.addWidget(self._label("清晰阅读 · 专注演奏", "SectionSubtitle"))
        preview_layout.addWidget(sample)
        left.addWidget(self.preview)
        self.file_label = self._label("尚未选择背景图片", "SectionSubtitle")
        self.file_label.setFixedWidth(300)
        left.addWidget(self.file_label)
        hint = self._label("四个页面共用背景。调整会即时预览，点击保存后下次启动继续使用。", "SectionSubtitle")
        hint.setFixedWidth(300)
        left.addWidget(hint)
        left.addStretch()
        body.addLayout(left)

        right = QVBoxLayout()
        right.setSpacing(12)
        self.mode_combo = QComboBox()
        for text, mode in (("默认背景", "default"), ("纯色", "color"), ("预设渐变", "gradient"), ("本地图片", "image")):
            self.mode_combo.addItem(text, mode)
        self._field(right, "背景类型", self.mode_combo)
        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        self.color_edit = QLineEdit()
        self.color_edit.setMaxLength(7)
        self.color_edit.setPlaceholderText("#203858")
        color_layout.addWidget(self.color_edit, 1)
        color_layout.addWidget(self._button("选色", self._choose_color))
        self.color_row = self._field(right, "背景颜色", color_row)
        self.gradient_combo = QComboBox()
        for key, (name, _, _) in GRADIENTS.items():
            self.gradient_combo.addItem(name, key)
        self.gradient_row = self._field(right, "渐变配色", self.gradient_combo)
        self.image_row = self._field(right, "PNG / JPG · 最大 32 MB、4000 万像素", self._button("选择 / 更换图片", self._choose_image))
        self.fit_combo = QComboBox()
        self.fit_combo.addItem("等比例铺满（居中裁切）", "cover")
        self.fit_combo.addItem("完整显示（保留底色）", "contain")
        self.fit_row = self._field(right, "图片布局", self.fit_combo)
        self.dimming_slider, self.dimming_value, self.dimming_row = self._slider(right, "背景压暗")
        self.opacity_slider, self.opacity_value, self.opacity_row = self._slider(right, "卡片不透明度")
        right.addStretch()
        body.addLayout(right, 1)
        layout.addLayout(body)

        self.error_label = self._label("", "AppearanceError")
        self.error_label.setMinimumHeight(18)
        layout.addWidget(self.error_label)
        buttons = QHBoxLayout()
        self.reset_button = self._button("恢复默认", self._reset)
        buttons.addWidget(self.reset_button)
        buttons.addStretch()
        buttons.addWidget(self._button("取消", self.reject))
        self.save_button = self._button("保存", self._save, primary=True)
        self.save_button.setDefault(True)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        for combo in (self.mode_combo, self.gradient_combo, self.fit_combo):
            combo.currentIndexChanged.connect(self._controls_changed)
        self.color_edit.textChanged.connect(self._controls_changed)

    def _fill_controls(self):
        self._loading = True
        settings = self._draft
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(settings.mode))
        self.color_edit.setText(settings.color)
        self.gradient_combo.setCurrentIndex(self.gradient_combo.findData(settings.gradient))
        self.fit_combo.setCurrentIndex(self.fit_combo.findData(settings.fit))
        self.dimming_slider.setValue(settings.dimming)
        self.opacity_slider.setValue(settings.panel_opacity)
        self.file_label.setText("已保存的背景图片" if settings.image_path else "尚未选择背景图片")
        self._loading = False
        self._controls_changed()

    def _controls_changed(self, *_):
        if self._loading:
            return
        mode = self.mode_combo.currentData()
        valid_color = bool(re.fullmatch(r"#[0-9a-fA-F]{6}", self.color_edit.text()))
        self._draft = replace(
            self._draft, mode=mode,
            color=self.color_edit.text().upper() if valid_color else self._draft.color,
            gradient=self.gradient_combo.currentData(), fit=self.fit_combo.currentData(),
            dimming=self.dimming_slider.value(), panel_opacity=self.opacity_slider.value(),
        )
        self.color_row.setVisible(mode == "color")
        self.gradient_row.setVisible(mode == "gradient")
        self.image_row.setVisible(mode == "image")
        self.fit_row.setVisible(mode == "image")
        self.dimming_row.setEnabled(mode != "default")
        self.opacity_row.setEnabled(mode != "default")
        self.dimming_value.setText(f"{self._draft.dimming}%")
        self.opacity_value.setText(f"{self._draft.panel_opacity}%")
        message = ""
        if mode == "color" and not valid_color:
            message = "请输入 # 加六位十六进制颜色，例如 #203858"
        elif mode == "image" and self.image.isNull():
            message = "请选择一张背景图片"
        self.error_label.setText(message)
        self.save_button.setEnabled(not message)
        self.preview.set_background(self._draft, self.image)
        self.preview.setStyleSheet(appearance_qss(self._draft, "BackgroundPreview"))
        self.preview_changed.emit(self._draft, self.image)

    def _choose_color(self):
        dialog = QColorDialog(QColor(self._draft.color), self)
        dialog.setWindowTitle("选择背景颜色")
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        dialog.setStyleSheet(f"QColorDialog {{ background: {SURFACE}; }} QLabel {{ color: {INK_2}; }}")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.color_edit.setText(dialog.selectedColor().name().upper())

    def import_image(self, path):
        image = load_background_image(path)
        self.image = image
        self._image_changed = True
        self.file_label.setText(f"{Path(path).name}\n{image.width()} × {image.height()} · 已优化预览")
        self.mode_combo.setCurrentIndex(self.mode_combo.findData("image"))
        self._controls_changed()

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择背景图片", "", "背景图片 (*.png *.jpg *.jpeg)")
        if path:
            try:
                self.import_image(path)
            except (OSError, ValueError) as exc:
                self.error_label.setText(str(exc))

    def _reset(self):
        self._draft = AppearanceSettings()
        self.image = QImage()
        self._image_changed = False
        self._fill_controls()

    def _save(self):
        if not self.save_button.isEnabled():
            return
        try:
            png = encode_background_image(self.image) if self._image_changed else None
            saved = self._store.save(self._draft, image_png=png)
        except (OSError, ValueError) as exc:
            self.error_label.setText(f"保存失败：{exc}")
            return
        self.settings = saved
        self.accept()

    def done(self, result):
        if result != QDialog.DialogCode.Accepted:
            self.preview_changed.emit(self.settings, self._original_image)
        super().done(result)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 56:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)
