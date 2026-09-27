"""离线 OMR 组件边界。

基础 EXE 不内置 Java/模型运行时；发布包可以把 Audiveris 和简谱 OMR 作为
签名的按需组件放在 ``data/omr``。本模块只负责检测、隔离进程、超时和输出
白名单，识别结果仍必须经过人工确认后才能写入 SQLite。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class OMRComponentUnavailable(RuntimeError):
    pass


class OMRComponentError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentInfo:
    name: str
    version: str
    root: str
    executable: str
    installed: bool


def component_root(data_dir: str) -> str:
    return os.path.join(os.path.abspath(os.fspath(data_dir)), "omr")


def _candidate_paths(root: str, names: tuple[str, ...]):
    base = Path(root)
    for name in names:
        yield base / name
        yield base / "audiveris" / name
        yield base / "jianpu_omr" / name
        yield base / "bin" / name
        yield base / "audiveris" / "bin" / name


def find_audiveris(data_dir: str, configured: str | None = None) -> ComponentInfo:
    configured_paths = []
    if configured:
        configured_paths.append(Path(configured))
    env = os.environ.get("AUTOMUSIC_AUDIVERIS_HOME")
    if env:
        configured_paths.append(Path(env))
    for program_root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramW6432"),
        os.environ.get("ProgramFiles(x86)"),
    ):
        if program_root:
            configured_paths.append(Path(program_root) / "Audiveris")
    root = component_root(data_dir)
    configured_paths.append(Path(root))
    names = ("Audiveris.bat", "audiveris.bat", "Audiveris.exe", "audiveris.exe")
    for candidate_root in configured_paths:
        if candidate_root.is_file() and candidate_root.name.lower() in {
            item.lower() for item in names
        }:
            return ComponentInfo("audiveris", "", str(candidate_root.parent), str(candidate_root), True)
        for candidate in _candidate_paths(str(candidate_root), names):
            if candidate.is_file():
                return ComponentInfo("audiveris", "", str(candidate_root), str(candidate), True)
    return ComponentInfo("audiveris", "", root, "", False)


def _run(command, *, timeout_s: float, cwd: str | None = None):
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, float(timeout_s)),
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OMRComponentError(f"OMR 组件超过 {timeout_s:g} 秒未完成") from exc
    except OSError as exc:
        raise OMRComponentError(f"无法启动 OMR 组件：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stdout or "").strip()[-2000:]
        raise OMRComponentError(
            f"OMR 组件退出码 {completed.returncode}"
            + (f"：{detail}" if detail else "")
        )
    return completed.stdout or ""


def run_audiveris(
    input_path: str,
    data_dir: str,
    *,
    configured: str | None = None,
    output_dir: str | None = None,
    timeout_s: float = 600.0,
) -> str:
    """运行 Audiveris 批处理并返回唯一的 MusicXML/MXL 输出。"""

    info = find_audiveris(data_dir, configured)
    if not info.installed:
        raise OMRComponentUnavailable(
            "未安装离线 Audiveris 组件；请在设置中安装签名的 OMR 组件"
        )
    input_file = os.path.abspath(os.fspath(input_path))
    if not os.path.isfile(input_file):
        raise OMRComponentError("OMR 输入文件不存在")
    temporary = output_dir is None
    destination = output_dir or tempfile.mkdtemp(prefix="automusic-omr-")
    os.makedirs(destination, exist_ok=True)
    try:
        _run(
            [
                info.executable,
                "-batch",
                "-transcribe",
                "-export",
                "-output",
                destination,
                "--",
                input_file,
            ],
            timeout_s=timeout_s,
            cwd=info.root if os.path.isdir(info.root) else None,
        )
        outputs = sorted(
            [
                path
                for path in Path(destination).rglob("*")
                if path.is_file() and path.suffix.lower() in {".mxl", ".musicxml", ".xml"}
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not outputs:
            raise OMRComponentError("Audiveris 未生成 MusicXML/MXL 输出")
        selected = str(outputs[0])
        if temporary:
            # 临时目录必须由调用方在解析/归档完成后删除；通过属性返回目录位置。
            return selected
        return selected
    except Exception:
        if temporary:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def find_jianpu_component(data_dir: str, configured: str | None = None) -> ComponentInfo:
    root = component_root(data_dir)
    configured_paths = []
    if configured:
        configured_paths.append(Path(configured))
    env = os.environ.get("AUTOMUSIC_JIANPU_OMR_HOME")
    if env:
        configured_paths.append(Path(env))
    configured_paths.append(Path(root))
    names = (
        "jianpu_omr.exe",
        "jianpu-omr.exe",
        "jianpu_omr.cmd",
        "jianpu_omr.bat",
    )
    for candidate_root in configured_paths:
        for candidate in _candidate_paths(str(candidate_root), names):
            if candidate.is_file():
                if candidate.suffix.lower() in {".cmd", ".bat"}:
                    # The bundled wrapper deliberately does not fall back to a
                    # mutable global Node installation; avoid reporting a
                    # source-only template as an installed component.
                    wrapper_root = candidate.parent
                    if not (
                        (wrapper_root / "node.exe").is_file()
                        or (wrapper_root / "runtime" / "node.exe").is_file()
                    ):
                        continue
                return ComponentInfo(
                    "jianpu-omr", "", str(candidate_root), str(candidate), True
                )
    return ComponentInfo("jianpu-omr", "", root, "", False)


def run_jianpu_component(
    input_path: str,
    data_dir: str,
    *,
    mode: str = "printed",
    configured: str | None = None,
    timeout_s: float = 600.0,
) -> str:
    """运行简谱/手写谱组件，返回其 ``auto-music-player-source`` JSON。

    组件协议固定为 ``--input PATH --output PATH --mode printed|handwritten``，
    这样基础 EXE 不需要把某个实验性模型硬编码进主进程。
    """

    info = find_jianpu_component(data_dir, configured=configured)
    if not info.installed:
        raise OMRComponentUnavailable(
            "未安装离线简谱 OMR 组件；印刷简谱和手写谱需要按需安装该组件"
        )
    output_dir = tempfile.mkdtemp(prefix="automusic-jianpu-")
    output = os.path.join(output_dir, "result.json")
    try:
        _run(
            [
                info.executable,
                "--input",
                os.path.abspath(os.fspath(input_path)),
                "--output",
                output,
                "--mode",
                mode,
            ],
            timeout_s=timeout_s,
            cwd=info.root,
        )
        if not os.path.isfile(output):
            raise OMRComponentError("简谱 OMR 组件未生成 result.json")
        return output
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


__all__ = [
    "ComponentInfo",
    "IMAGE_EXTENSIONS",
    "OMRComponentError",
    "OMRComponentUnavailable",
    "component_root",
    "find_audiveris",
    "find_jianpu_component",
    "run_audiveris",
    "run_jianpu_component",
]
