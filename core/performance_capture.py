"""Local, privacy-preserving M0 performance capture.

This module persists only already-mapped performance events.  It deliberately
does not know about GUI objects, operating-system key codes, mouse coordinates,
windows, processes, network services, or training frameworks.

Each capture is an append-only JSONL file while it is being recorded.  The
file lives under ``pending`` until a valid end record is written and flushed;
finalization then moves it atomically to ``completed`` (or ``rejected``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
SCHEMA_NAME = "auto_music_player.performance_capture"
MAX_EVENTS_PER_SESSION = 200_000
MAX_EVENT_PAYLOAD_BYTES = 16 * 1024
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_STATUS_DIRS = ("pending", "completed", "rejected")

# A mapped event contains note-level information only.  Rejecting these names
# recursively prevents accidental persistence of raw pynput/Win32 data even if
# a caller passes a nested dictionary.
_FORBIDDEN_NAME_PARTS = (
    "key",
    "vk",
    "scan",
    "mouse",
    "button",
    "coord",
    "window",
    "hwnd",
    "process",
    "pid",
    "handle",
)
_RESERVED_EVENT_FIELDS = {
    "record",
    "session_id",
    "seq",
    "monotonic_ns",
    "wall_time_ns",
    "elapsed_ns",
}


class CaptureClock(Protocol):
    """Clock protocol used by production code and deterministic tests."""

    def monotonic_ns(self) -> int: ...


class MappedEvent(Protocol):
    """Optional protocol for event objects owned by another module."""

    def to_performance_event(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CaptureResult:
    session_id: str
    status: str
    path: Path
    summary: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    """Serialize strict, deterministic UTF-8 JSON or raise ``ValueError``."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"数据不可序列化为严格 JSON: {exc}") from None


def canonical_notes_hash(notes: Any) -> str:
    """Return the SHA-256 of canonical score notes JSON.

    The function intentionally accepts only JSON-compatible values and does
    not import the application's score model, so it can be used by an offline
    capture/export tool without pulling in GUI dependencies.
    """

    return hashlib.sha256(_json_bytes(notes)).hexdigest()


def _validate_hash(value: str, label: str) -> str:
    value = str(value or "")
    if not _HASH_RE.fullmatch(value):
        raise ValueError(f"{label} 必须是 64 位十六进制 SHA-256")
    return value.lower()


def _validate_name(name: str, *, event_field: bool = False) -> None:
    normalized = str(name).casefold().replace("-", "_")
    if event_field and normalized in _RESERVED_EVENT_FIELDS:
        raise ValueError(f"事件字段保留给写入器: {name}")
    if event_field and normalized in {"x", "y"}:
        raise ValueError(f"事件包含禁止的坐标字段: {name}")
    if event_field and any(part in normalized for part in _FORBIDDEN_NAME_PARTS):
        raise ValueError(f"事件包含禁止的物理输入/环境字段: {name}")


def _validate_payload(value: Any, *, event_field: bool = False) -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if not isinstance(name, str):
                raise ValueError("事件对象的字段名必须是字符串")
            _validate_name(name, event_field=event_field)
            _validate_payload(child, event_field=event_field)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_payload(child, event_field=event_field)
        return
    if isinstance(value, (str, int, bool)) or value is None:
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"事件包含不支持的值类型: {type(value).__name__}")


def _coerce_event(event: Mapping[str, Any] | MappedEvent | Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        payload = dict(event)
    elif hasattr(event, "to_performance_event"):
        payload = dict(event.to_performance_event())
    elif hasattr(event, "to_dict"):
        payload = dict(event.to_dict())
    else:
        raise TypeError("event 必须是映射对象或提供 to_performance_event()/to_dict()")
    if not isinstance(payload.get("type"), str) or not payload["type"].strip():
        raise ValueError("映射事件必须包含非空字符串 type")
    if len(payload["type"]) > 64:
        raise ValueError("事件 type 过长")
    _validate_payload(payload, event_field=True)
    # Force a serialization check before anything reaches the append-only file.
    if len(_json_bytes(payload)) > MAX_EVENT_PAYLOAD_BYTES:
        raise ValueError("映射事件超过单条大小限制")
    return payload


class PerformanceCapture:
    """Thread-safe writer for one local M0 capture session."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        app_version: str,
        profile_id: str,
        mapping_mode: str,
        mapping_hash: str,
        bpm: int,
        quantize: float,
        notes: Any = None,
        source_sha256: str | None = None,
        session_id: str | None = None,
        clock: CaptureClock | None = None,
    ):
        if root is None or str(root).strip() == "":
            raise ValueError("必须显式传入 capture root")
        self.root = Path(root).expanduser().resolve()
        self.session_id = session_id or uuid.uuid4().hex
        try:
            uuid.UUID(self.session_id)
        except (ValueError, AttributeError):
            raise ValueError("session_id 必须是 UUID") from None
        for label, value in (
            ("app_version", app_version),
            ("profile_id", profile_id),
            ("mapping_mode", mapping_mode),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"{label} 必须是 1-128 字符的非空字符串")
        self.app_version = app_version
        self.profile_id = profile_id
        self.mapping_mode = mapping_mode
        self.mapping_hash = _validate_hash(mapping_hash, "mapping_hash")
        if isinstance(bpm, bool) or not isinstance(bpm, (int, float)):
            raise ValueError("bpm 必须是数字")
        if not 30 <= float(bpm) <= 300:
            raise ValueError("bpm 必须在 30-300")
        if isinstance(quantize, bool) or not isinstance(quantize, (int, float)):
            raise ValueError("quantize 必须是数字")
        if not math.isfinite(float(quantize)) or float(quantize) < 0:
            raise ValueError("quantize 必须是有限非负数字")
        if source_sha256 is not None:
            source_sha256 = _validate_hash(source_sha256, "source_sha256")
        self.bpm = int(bpm)
        self.quantize = float(quantize)
        self.notes_sha256 = canonical_notes_hash(notes) if notes is not None else None
        self.notes_count = len(notes) if isinstance(notes, (list, tuple)) else None
        self.source_sha256 = source_sha256
        self._clock = clock
        self._lock = threading.RLock()
        self._fh = None
        self._hasher = hashlib.sha256()
        self._last_monotonic_ns: int | None = None
        self._seq = 0
        self._event_count = 0
        self._mapped_event_count = 0
        self._mapped_press_count = 0
        self._ignored_count = 0
        self._control_count = 0
        self._diagnostic_count = 0
        self._synthetic_close_count = 0
        self._active_presses: dict[int, tuple[str, int, str]] = {}
        self._completed_press_ids: set[int] = set()
        self._result: CaptureResult | None = None
        self._status = "active"
        self._pending_path: Path | None = None
        self._start()

    @classmethod
    def start(cls, root: str | os.PathLike[str], **kwargs) -> "PerformanceCapture":
        """Explicit constructor spelling for call sites and future factories."""

        return cls(root, **kwargs)

    @property
    def closed(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> CaptureResult | None:
        return self._result

    def _monotonic_now(self) -> int:
        clock = self._clock
        mono_fn = getattr(clock, "monotonic_ns", time.monotonic_ns) if clock else time.monotonic_ns
        mono = int(mono_fn())
        if mono < 0:
            raise ValueError("时钟值不能为负数")
        if self._last_monotonic_ns is not None and mono < self._last_monotonic_ns:
            raise ValueError("monotonic_ns 不得回退")
        self._last_monotonic_ns = mono
        return mono

    def _start(self) -> None:
        for directory in _STATUS_DIRS:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        paths = [self.root / directory / f"{self.session_id}.jsonl" for directory in _STATUS_DIRS]
        if any(path.exists() for path in paths):
            raise FileExistsError(f"capture session 已存在: {self.session_id}")
        self._pending_path = paths[0]
        try:
            self._fh = open(self._pending_path, "xb")
            mono = self._monotonic_now()
            start = {
                "record": "start",
                "schema": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "status": "active",
                "app_version": self.app_version,
                "profile_id": self.profile_id,
                "mapping_mode": self.mapping_mode,
                "mapping_hash": self.mapping_hash,
                "bpm": self.bpm,
                "quantize": self.quantize,
                "notes_sha256": self.notes_sha256,
                "notes_count": self.notes_count,
                "source_sha256": self.source_sha256,
                "created_date_utc": datetime.now(timezone.utc).date().isoformat(),
                "privacy": {
                    "raw_os_codes": False,
                    "window_identity": False,
                    "coordinates": False,
                },
            }
            self._last_start_monotonic_ns = mono
            self._write_record(start, include_in_hash=True)
        except Exception:
            if self._fh is not None:
                self._fh.close()
            if self._pending_path is not None:
                try:
                    self._pending_path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _ensure_open(self) -> None:
        if self._result is not None or self._fh is None:
            raise RuntimeError("capture session 已结束")

    def _write_record(self, record: dict[str, Any], *, include_in_hash: bool) -> None:
        encoded = _json_bytes(record) + b"\n"
        self._fh.write(encoded)
        self._fh.flush()
        if include_in_hash:
            self._hasher.update(encoded)

    def _event_stamp(self) -> tuple[int, int]:
        mono = self._monotonic_now()
        start = self._start_monotonic_ns
        return mono, mono - start

    @property
    def _start_monotonic_ns(self) -> int:
        # The value is assigned from the start record before any event can be
        # appended; keeping it separately avoids reading/parsing our own file.
        if self._last_start_monotonic_ns is None:
            raise RuntimeError("capture 尚未初始化")
        return self._last_start_monotonic_ns

    _last_start_monotonic_ns: int | None = None

    def append_event(self, event: Mapping[str, Any] | MappedEvent | Any) -> int:
        """Append one already-mapped event and return its sequence number."""

        with self._lock:
            self._ensure_open()
            if self._event_count >= MAX_EVENTS_PER_SESSION:
                raise ValueError("本次研究样本事件数超过安全上限")
            payload = _coerce_event(event)
            event_type = payload["type"]
            press_update = None
            if event_type in {"note_down", "note_up"}:
                press_id = payload.get("press_id")
                if (
                    isinstance(press_id, bool)
                    or not isinstance(press_id, int)
                    or press_id <= 0
                ):
                    raise ValueError("note_down/note_up 必须包含正整数 press_id")
                note_id = payload.get("note_id")
                if not isinstance(note_id, str) or not note_id.strip():
                    raise ValueError("note_down/note_up 必须包含 note_id")
                signature = (
                    note_id,
                    int(payload.get("semitone", 0) or 0),
                    str(payload.get("modifier", "natural")),
                )
                if event_type == "note_down":
                    if press_id in self._active_presses or press_id in self._completed_press_ids:
                        raise ValueError(f"重复的 press_id: {press_id}")
                    press_update = ("down", press_id, signature)
                else:
                    expected = self._active_presses.get(press_id)
                    if expected is None:
                        raise ValueError(f"note_up 没有对应的 note_down: {press_id}")
                    if expected != signature:
                        raise ValueError(f"press_id {press_id} 的按下/释放音符不一致")
                    press_update = ("up", press_id, signature)
            _mono, elapsed = self._event_stamp()
            self._seq += 1
            self._event_count += 1
            if event_type in {"control", "pause", "resume", "stop"}:
                self._control_count += 1
            elif event_type in {"diagnostic", "ignored", "synthetic_close"}:
                self._diagnostic_count += 1
            else:
                self._mapped_event_count += 1
                if event_type in {"down", "note_down", "note_on"}:
                    self._mapped_press_count += 1
            if bool(payload.get("synthetic_close", False)):
                self._synthetic_close_count += 1
            self._write_record(
                {
                    "record": "event",
                    "session_id": self.session_id,
                    "seq": self._seq,
                    "elapsed_ns": elapsed,
                    "event": payload,
                },
                include_in_hash=True,
            )
            if press_update is not None:
                action, press_id, signature = press_update
                if action == "down":
                    self._active_presses[press_id] = signature
                else:
                    self._active_presses.pop(press_id)
                    self._completed_press_ids.add(press_id)
            return self._seq

    def append_ignored(self, reason: str) -> int:
        """Record an ignored mapped-input attempt without retaining its token."""

        if not isinstance(reason, str) or not reason.strip() or len(reason) > 128:
            raise ValueError("ignored reason 必须是 1-128 字符")
        with self._lock:
            self._ensure_open()
            self._ignored_count += 1
            sequence = self.append_event({"type": "ignored", "reason": reason})
            return sequence

    def append_control(self, action: str) -> int:
        if not isinstance(action, str) or not action.strip() or len(action) > 64:
            raise ValueError("control action 必须是 1-64 字符")
        with self._lock:
            self._ensure_open()
            return self.append_event({"type": "control", "action": action})

    def mark_synthetic_close(self, reason: str = "stop") -> int:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 64:
            raise ValueError("synthetic close reason 必须是 1-64 字符")
        with self._lock:
            self._ensure_open()
            return self.append_event(
                {"type": "synthetic_close", "reason": reason, "synthetic_close": True}
            )

    def _quality(self) -> dict[str, Any]:
        inputs = self._mapped_press_count + self._ignored_count
        ignored_ratio = (self._ignored_count / inputs) if inputs else 0.0
        flags = []
        if ignored_ratio > 0.10:
            flags.append("high_ignored_ratio")
        if self._synthetic_close_count:
            flags.append("held_note_closed_on_stop")
        return {
            "input_count": inputs,
            "mapped_event_count": self._mapped_event_count,
            "mapped_press_count": self._mapped_press_count,
            "paired_press_count": len(self._completed_press_ids),
            "unclosed_press_count": len(self._active_presses),
            "ignored_count": self._ignored_count,
            "ignored_ratio": ignored_ratio,
            "synthetic_close": self._synthetic_close_count > 0,
            "synthetic_close_count": self._synthetic_close_count,
            "flags": flags,
            "training_eligible": not flags,
        }

    def _finish(
        self,
        status: str,
        reason: str,
        *,
        synthetic_close: bool = False,
        canonical_notes: Any = None,
        captured_count: int | None = None,
        duration_s: float | None = None,
        warnings: Sequence[str] = (),
        degradations: Sequence[Mapping[str, Any]] = (),
    ) -> CaptureResult:
        with self._lock:
            if self._result is not None:
                return self._result
            self._ensure_open()
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 128:
                raise ValueError("结束原因必须是 1-128 字符")
            if status == "completed":
                if self._mapped_event_count <= 0:
                    raise ValueError("没有可完成的 mapped event")
                if self._active_presses:
                    raise ValueError("存在没有 note_up 的 note_down")
                if not isinstance(canonical_notes, (list, tuple)) or not canonical_notes:
                    raise ValueError("完整研究样本必须包含规范乐谱")
                if (
                    isinstance(captured_count, bool)
                    or not isinstance(captured_count, int)
                    or captured_count <= 0
                ):
                    raise ValueError("完整研究样本 captured_count 必须大于 0")
                if (
                    isinstance(duration_s, bool)
                    or not isinstance(duration_s, (int, float))
                    or not math.isfinite(float(duration_s))
                    or float(duration_s) <= 0
                ):
                    raise ValueError("完整研究样本 duration_s 必须是有限正数")
                if self._synthetic_close_count and synthetic_close:
                    # Idempotency guard: a caller may pass the flag twice only
                    # before the first finalize, so do not append twice.
                    synthetic_close = False
            if synthetic_close:
                self.mark_synthetic_close(reason)
            _mono, _elapsed = self._event_stamp()
            notes = list(canonical_notes) if canonical_notes is not None else None
            if notes is not None:
                _validate_payload(notes)
                _json_bytes(notes)
            clean_warnings = [str(item) for item in warnings]
            clean_degradations = [dict(item) for item in degradations]
            _validate_payload(clean_warnings)
            _validate_payload(clean_degradations)
            content_sha256 = self._hasher.hexdigest()
            end = {
                "record": "end",
                "session_id": self.session_id,
                "status": status,
                "reason": reason,
                "event_count": self._event_count,
                "last_seq": self._seq,
                "content_sha256": content_sha256,
                "quality": self._quality(),
                "captured_count": int(captured_count or 0),
                "duration_s": float(duration_s or 0.0),
                "canonical_notes": notes,
                "canonical_notes_sha256": (
                    canonical_notes_hash(notes) if notes is not None else None
                ),
                "canonical_notes_count": len(notes) if notes is not None else 0,
                "warnings": clean_warnings,
                "degradations": clean_degradations,
            }
            self._write_record(end, include_in_hash=False)
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
            destination = self.root / ("completed" if status == "completed" else "rejected") / self._pending_path.name
            os.replace(self._pending_path, destination)
            self._status = status
            self._result = CaptureResult(self.session_id, status, destination, end)
            return self._result

    def finalize(
        self,
        *,
        reason: str = "stop",
        synthetic_close: bool = False,
        canonical_notes: Any,
        captured_count: int,
        duration_s: float,
        warnings: Sequence[str] = (),
        degradations: Sequence[Mapping[str, Any]] = (),
    ) -> CaptureResult:
        return self._finish(
            "completed",
            reason,
            synthetic_close=synthetic_close,
            canonical_notes=canonical_notes,
            captured_count=captured_count,
            duration_s=duration_s,
            warnings=warnings,
            degradations=degradations,
        )

    def reject(self, reason: str = "rejected") -> CaptureResult:
        return self._finish("rejected", reason)


__all__ = [
    "CaptureClock",
    "CaptureResult",
    "MappedEvent",
    "PerformanceCapture",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "canonical_notes_hash",
]
