"""外观保存事务、图片解码、预览还原与真实主窗口显示回归。"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QScrollArea

import main
from core.appearance import AppearanceSettings, AppearanceStore
from core.database import ScoreDB
from core.keymap import KeyMap
from core.player import Player
from gui.appearance_dialog import AppearanceDialog
from gui.background import BackgroundWidget, encode_background_image, load_background_image
from gui.main_window import MainWindow
from gui.theme import APP_QSS, BG


_app = None


def ensure_app():
    global _app
    _app = QApplication.instance() or QApplication([])
    return _app


def red_image(width=160, height=90):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#E04030"))
    return image


def pixel_at(widget, point):
    shot = widget.grab()
    ratio = shot.devicePixelRatioF()
    return shot.toImage().pixelColor(round(point.x() * ratio), round(point.y() * ratio))


class AppearanceStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = AppearanceStore(self.root / "data")

    def test_reading_defaults_does_not_create_files(self):
        self.assertEqual(self.store.load(), AppearanceSettings())
        self.assertFalse(self.store.data_dir.exists())

    def test_settings_and_image_survive_moving_entire_data_directory(self):
        png = encode_background_image(red_image())
        saved = self.store.save(replace(AppearanceSettings(), mode="image", fit="contain"), png)
        self.assertEqual(AppearanceStore(self.store.data_dir).load(), saved)
        moved = self.root / "便携数据"
        shutil.move(str(self.store.data_dir), moved)
        relocated = AppearanceStore(moved)
        self.assertEqual(relocated.load(), saved)
        self.assertEqual(load_background_image(relocated.image_file(saved.image_path)).pixelColor(0, 0), QColor("#E04030"))

    def test_failed_config_commit_keeps_previous_image_and_settings(self):
        saved = self.store.save(replace(AppearanceSettings(), mode="image"), encode_background_image(red_image()))
        previous_config = self.store.config_path.read_bytes()
        real_replace = os.replace

        def fail_config(source, target):
            if Path(target) == self.store.config_path:
                raise PermissionError("模拟配置不可写")
            return real_replace(source, target)

        with patch("core.appearance.os.replace", side_effect=fail_config):
            with self.assertRaises(PermissionError):
                self.store.save(saved, encode_background_image(red_image(100, 100)))
        self.assertEqual(self.store.config_path.read_bytes(), previous_config)
        self.assertEqual(self.store.load(), saved)
        self.assertEqual(list((self.store.data_dir / "backgrounds").iterdir()), [self.store.image_file(saved.image_path)])
        self.assertFalse(list(self.store.data_dir.glob(".appearance-*.tmp")))

    def test_reset_removes_only_previously_managed_image(self):
        saved = self.store.save(replace(AppearanceSettings(), mode="image"), encode_background_image(red_image()))
        unrelated = self.store.data_dir / "backgrounds" / "my-photo.png"
        unrelated.write_bytes(b"user file")
        self.store.save(AppearanceSettings())
        self.assertFalse(self.store.image_file(saved.image_path).exists())
        self.assertEqual(unrelated.read_bytes(), b"user file")

    def test_bad_configs_and_missing_images_fall_back_to_default(self):
        self.store.data_dir.mkdir()
        cases = ["not json", "[]", json.dumps({"version": 2, "mode": "default"}),
                 json.dumps({"version": 1, "mode": "image", "image_path": "../private.png"}),
                 json.dumps({"version": 1, "mode": "image", "image_path": "backgrounds/" + "a" * 32 + ".png"})]
        for content in cases:
            with self.subTest(content=content):
                self.store.config_path.write_text(content, encoding="utf-8")
                self.assertEqual(self.store.load(), AppearanceSettings())
                self.assertTrue(self.store.warning)

    def test_percentages_are_clamped_and_invalid_color_uses_default(self):
        raw = asdict(replace(AppearanceSettings(), mode="color"))
        raw.update(dimming=1000, panel_opacity=-5, color="red; background: white")
        settings = AppearanceSettings.from_dict(raw)
        self.assertEqual((settings.dimming, settings.panel_opacity), (100, 0))
        self.assertEqual(settings.color, AppearanceSettings().color)


class AppearanceGuiTests(unittest.TestCase):
    def setUp(self):
        self.app = ensure_app()
        previous_style = self.app.styleSheet()
        self.app.setStyleSheet(APP_QSS)
        self.addCleanup(self.app.setStyleSheet, previous_style)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = AppearanceStore(self.root / "data")

    def dialog(self, settings=None, image=None):
        dialog = AppearanceDialog(self.store, settings or AppearanceSettings(), image or QImage())
        self.addCleanup(dialog.deleteLater)
        dialog.show()
        self.app.processEvents()
        return dialog

    def window(self, data_dir=None):
        cfg = main.load_config()
        cfg["update"]["enabled"] = False
        db = ScoreDB(self.root / "scores.db")
        self.addCleanup(db.close)
        keymap = KeyMap(cfg["keymap"])
        player = Player(keymap, driver=Mock())
        with patch("gui.main_window.pk.GlobalHotKeys"):
            win = MainWindow(cfg, db, keymap, player,
                             data_dir=data_dir or self.store.data_dir,
                             event_log_dir=str(self.root / "logs"), export_dir=str(self.root / "exports"))
        self.addCleanup(win.deleteLater)
        self.addCleanup(win._cleanup_on_quit)
        win.show()
        self.app.processEvents()
        return win

    def test_jpeg_import_is_optimized_saved_and_independent_of_original(self):
        source = self.root / "背景.JPG"
        self.assertTrue(red_image(3000, 1500).save(str(source), "JPEG"))
        dialog = self.dialog()
        dialog.import_image(source)
        self.assertLessEqual(max(dialog.image.width(), dialog.image.height()), 2560)
        QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        source.unlink()
        loaded = self.store.load()
        self.assertEqual(loaded.mode, "image")
        self.assertFalse(load_background_image(self.store.image_file(loaded.image_path)).isNull())

    def test_import_cancel_writes_nothing_and_restores_original_preview(self):
        source = self.root / "picture.png"
        red_image().save(str(source))
        original = replace(AppearanceSettings(), mode="gradient", gradient="forest")
        dialog = self.dialog(original)
        previews = []
        dialog.preview_changed.connect(lambda settings, image: previews.append((settings, image)))
        dialog.import_image(source)
        dialog.dimming_slider.setValue(64)
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertEqual(previews[-1][0], original)
        self.assertTrue(previews[-1][1].isNull())
        self.assertFalse(self.store.data_dir.exists())
        self.assertTrue(source.exists())

    def test_jpeg_orientation_is_applied_before_preview_and_save(self):
        source = self.root / "phone.jpg"
        red_image(160, 90).save(str(source), "JPEG")
        # 手机照片常用 EXIF orientation=6，显示时应旋转 90 度。
        exif = (b"Exif\x00\x00II" + struct.pack("<HIH", 42, 8, 1)
                + struct.pack("<HHIH", 0x0112, 3, 1, 6) + b"\x00\x00" + struct.pack("<I", 0))
        jpeg = source.read_bytes()
        source.write_bytes(jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + jpeg[2:])
        image = load_background_image(source)
        self.assertEqual((image.width(), image.height()), (90, 160))

    def test_bad_import_keeps_current_draft_and_size_limits_are_enforced(self):
        source = self.root / "good.png"
        red_image().save(str(source))
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        dialog = self.dialog()
        dialog.import_image(source)
        key = dialog.image.cacheKey()
        with self.assertRaises(ValueError):
            dialog.import_image(bad)
        self.assertEqual(dialog.image.cacheKey(), key)
        with patch("gui.background.MAX_IMAGE_BYTES", 1):
            with self.assertRaises(ValueError):
                load_background_image(source)
        with patch("gui.background.MAX_IMAGE_PIXELS", 10):
            with self.assertRaises(ValueError):
                load_background_image(source)

    def test_failed_save_keeps_dialog_open_and_can_cancel(self):
        original = self.store.save(replace(AppearanceSettings(), mode="gradient"))
        dialog = self.dialog(original)
        dialog.gradient_combo.setCurrentIndex(dialog.gradient_combo.findData("forest"))
        with patch.object(self.store, "save", side_effect=PermissionError("只读目录")):
            QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
        self.assertTrue(dialog.isVisible())
        self.assertIn("保存失败", dialog.error_label.text())
        self.assertEqual(self.store.load(), original)
        dialog.reject()

    def test_restore_default_is_a_draft_until_saved(self):
        original = self.store.save(replace(AppearanceSettings(), mode="gradient"))
        dialog = self.dialog(original)
        QTest.mouseClick(dialog.reset_button, Qt.MouseButton.LeftButton)
        self.assertEqual(dialog.preview.settings, AppearanceSettings())
        self.assertEqual(self.store.load(), original)
        QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.store.load(), AppearanceSettings())

    def test_background_contains_or_crops_and_darkens_without_distorting(self):
        widget = BackgroundWidget()
        self.addCleanup(widget.deleteLater)
        widget.resize(200, 200)
        settings = replace(AppearanceSettings(), mode="image", fit="contain", dimming=0)
        widget.set_background(settings, red_image(200, 100))
        widget.show()
        self.app.processEvents()
        self.assertEqual(pixel_at(widget, QPoint(10, 10)), QColor(BG))
        self.assertEqual(pixel_at(widget, QPoint(100, 100)), QColor("#E04030"))
        widget.set_background(replace(settings, fit="cover"), red_image(200, 100))
        self.assertEqual(pixel_at(widget, QPoint(10, 10)), QColor("#E04030"))
        widget.set_background(replace(settings, fit="cover", dimming=50), red_image(200, 100))
        self.assertLess(pixel_at(widget, QPoint(100, 100)).red(), 120)

    def test_four_pages_show_same_fixed_background_and_resize(self):
        win = self.window()
        settings = replace(AppearanceSettings(), mode="color", color="#38607C", dimming=0, panel_opacity=50)
        win._apply_appearance(settings, QImage())
        for index in range(4):
            win.nav.setCurrentRow(index)
            self.app.processEvents()
            self.assertEqual(win.stack.currentIndex(), index)
            # 每页外边距透出同一根背景；该点避开页面控件。
            point = win.stack.mapTo(win, QPoint(5, 5))
            self.assertEqual(pixel_at(win, point), QColor("#38607C"))
        win.nav.setCurrentRow(0)
        scroll = win.upload_tab.findChild(QScrollArea)
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        win.resize(1260, 820)
        self.app.processEvents()
        point = win.stack.mapTo(win, QPoint(5, 5))
        self.assertEqual(pixel_at(win, point), QColor("#38607C"))
        win._apply_appearance(AppearanceSettings(), QImage())
        self.assertEqual(win.centralWidget().styleSheet(), "")
        self.assertEqual(pixel_at(win, point), QColor(BG))

    def test_main_window_appearance_entry_saves_and_reloads(self):
        win = self.window()

        def edit_and_save(dialog):
            dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("gradient"))
            dialog.gradient_combo.setCurrentIndex(dialog.gradient_combo.findData("violet"))
            dialog.opacity_slider.setValue(73)
            dialog._save()
            return dialog.result()

        with patch.object(AppearanceDialog, "exec", edit_and_save):
            win.appearance_btn.click()
        self.assertEqual(win._appearance.gradient, "violet")
        self.assertEqual(win._appearance.panel_opacity, 73)
        reopened = self.window()
        self.assertEqual(reopened.centralWidget().settings, win._appearance)

    def test_corrupt_saved_image_falls_back_without_overwriting_user_files(self):
        saved = self.store.save(replace(AppearanceSettings(), mode="image"), b"broken png")
        original = self.store.config_path.read_bytes()
        win = self.window()
        self.assertEqual(win.centralWidget().settings.mode, "default")
        self.assertIn("默认背景", win.status_label.text())
        self.assertEqual(self.store.config_path.read_bytes(), original)
        self.assertTrue(self.store.image_file(saved.image_path).exists())


if __name__ == "__main__":
    unittest.main()
