"""静态背景的限量解码与按窗口尺寸缓存绘制。"""

from pathlib import Path

from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QImage, QImageReader, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

from core.appearance import AppearanceSettings, GRADIENTS
from gui.theme import BG


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_EDGE = 2560


def load_background_image(path):
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("请选择不超过 32 MB 的 PNG 或 JPG 图片")
    reader = QImageReader(str(source))
    reader.setAutoTransform(True)
    if bytes(reader.format()).lower() not in (b"png", b"jpeg", b"jpg"):
        raise ValueError("请选择可读取的 PNG 或 JPG 图片")
    size = reader.size()
    if size.width() <= 0 or size.height() <= 0 or size.width() * size.height() > MAX_IMAGE_PIXELS:
        raise ValueError("图片尺寸无效或超过 4000 万像素，请先缩小图片")
    if max(size.width(), size.height()) > MAX_IMAGE_EDGE:
        reader.setScaledSize(size.scaled(QSize(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    if image.isNull():
        raise ValueError("图片无法解码，请更换一张图片")
    if max(image.width(), image.height()) > MAX_IMAGE_EDGE:
        image = image.scaled(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE,
                             Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return image


def encode_background_image(image):
    data = QByteArray()
    buffer = QBuffer(data)
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly) or not image.save(buffer, "PNG"):
        raise ValueError("无法保存背景图片")
    buffer.close()
    return bytes(data)


class BackgroundWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = AppearanceSettings()
        self.image = QImage()
        self._frame = QPixmap()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def set_background(self, settings, image=None):
        image = image if image is not None else QImage()
        if settings == self.settings and image.cacheKey() == self.image.cacheKey():
            return
        self.settings = settings
        self.image = QImage(image)
        self._frame = QPixmap()
        self.update()

    def _render_frame(self):
        ratio = self.devicePixelRatioF()
        frame = QPixmap(max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio)))
        frame.setDevicePixelRatio(ratio)
        frame.fill(QColor(BG))
        painter = QPainter(frame)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect())
        settings = self.settings
        if settings.mode == "color":
            painter.fillRect(rect, QColor(settings.color))
        elif settings.mode == "gradient":
            _, start, end = GRADIENTS[settings.gradient]
            gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
            gradient.setColorAt(0, QColor(start))
            gradient.setColorAt(1, QColor(end))
            painter.fillRect(rect, gradient)
        elif settings.mode == "image" and not self.image.isNull():
            factors = (rect.width() / self.image.width(), rect.height() / self.image.height())
            scale = min(factors) if settings.fit == "contain" else max(factors)
            width, height = self.image.width() * scale, self.image.height() * scale
            target = QRectF((rect.width() - width) / 2, (rect.height() - height) / 2, width, height)
            painter.drawImage(target, self.image)
        if settings.mode != "default":
            painter.fillRect(rect, QColor(0, 0, 0, round(settings.dimming * 255 / 100)))
        painter.end()
        return frame

    def paintEvent(self, event):
        ratio = self.devicePixelRatioF()
        expected = QSize(max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio)))
        if self._frame.isNull() or self._frame.size() != expected or self._frame.devicePixelRatioF() != ratio:
            self._frame = self._render_frame()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._frame)
