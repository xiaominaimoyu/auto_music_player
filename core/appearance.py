"""全局外观配置与程序管理的背景副本，不依赖曲谱数据库。"""

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


GRADIENTS = {
    "midnight": ("夜航蓝", "#203858", "#0C1321"),
    "forest": ("松林绿", "#254C42", "#0D1918"),
    "amber": ("暮色金", "#533D2A", "#17131D"),
    "violet": ("星云紫", "#443360", "#141526"),
}
_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")
_ASSET = re.compile(r"backgrounds/[0-9a-f]{32}\.png\Z")


@dataclass(frozen=True)
class AppearanceSettings:
    version: int = 1
    mode: str = "default"
    color: str = "#203858"
    gradient: str = "midnight"
    image_path: str = ""
    fit: str = "cover"
    dimming: int = 35
    panel_opacity: int = 85

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 1:
            raise ValueError("无法识别的外观配置版本")
        defaults = cls()
        path = raw.get("image_path", "")
        if not isinstance(path, str) or (path and not _ASSET.fullmatch(path)):
            raise ValueError("背景图片路径无效")
        mode = raw.get("mode")
        if mode not in ("default", "color", "gradient", "image"):
            raise ValueError("无法识别的背景类型")
        if mode == "image" and not path:
            raise ValueError("缺少背景图片")

        def percent(key):
            value = raw.get(key, getattr(defaults, key))
            return max(0, min(100, value)) if type(value) is int else getattr(defaults, key)

        color = raw.get("color", defaults.color)
        return cls(
            mode=mode,
            color=color.upper() if isinstance(color, str) and _COLOR.fullmatch(color) else defaults.color,
            gradient=raw.get("gradient") if raw.get("gradient") in GRADIENTS else defaults.gradient,
            image_path=path,
            fit=raw.get("fit") if raw.get("fit") in ("cover", "contain") else defaults.fit,
            dimming=percent("dimming"),
            panel_opacity=percent("panel_opacity"),
        )


def _atomic_write(path: Path, content: bytes):
    fd, temporary = tempfile.mkstemp(prefix=".appearance-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AppearanceStore:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.config_path = self.data_dir / "appearance.json"
        self.warning = ""

    def image_file(self, relative):
        if not _ASSET.fullmatch(relative):
            raise ValueError("背景图片路径无效")
        path = self.data_dir / relative
        if not path.resolve().is_relative_to(self.data_dir):
            raise ValueError("背景图片必须保存在程序数据目录内")
        return path

    def load(self):
        self.warning = ""
        try:
            if self.config_path.stat().st_size > 64 * 1024:
                raise ValueError("外观配置过大")
            settings = AppearanceSettings.from_dict(json.loads(self.config_path.read_text(encoding="utf-8")))
            if settings.image_path:
                path = self.image_file(settings.image_path)
                if not path.is_file():
                    raise ValueError("背景图片已丢失")
            return settings
        except FileNotFoundError:
            return AppearanceSettings()
        except (OSError, UnicodeError, ValueError, TypeError):
            self.warning = "外观设置或背景图片无法读取，已使用默认背景"
            return AppearanceSettings()

    def save(self, settings, image_png=None):
        """提交已解码并优化的 PNG；失败时保留原配置及其图片。"""
        previous = self.load()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        created = None
        try:
            if image_png is not None:
                relative = f"backgrounds/{uuid.uuid4().hex}.png"
                created = self.image_file(relative)
                created.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(created, image_png)
                settings = replace(settings, image_path=relative)
            settings = AppearanceSettings.from_dict(asdict(settings))
            if settings.image_path and not self.image_file(settings.image_path).is_file():
                raise ValueError("背景图片已丢失，请重新选择图片")
            content = json.dumps(asdict(settings), ensure_ascii=False, indent=2).encode("utf-8")
            _atomic_write(self.config_path, content)
        except Exception:
            if created is not None:
                created.unlink(missing_ok=True)
            raise
        if previous.image_path and previous.image_path != settings.image_path:
            try:
                self.image_file(previous.image_path).unlink(missing_ok=True)
            except (OSError, ValueError):
                pass  # 配置已提交，旧副本回收失败不影响当前背景。
        return settings
