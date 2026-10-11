"""Strict read-only loader for completed local performance captures.

The loader accepts only the version-1 JSONL envelope written by
``core.performance_capture``.  It never repairs, renames, moves, or otherwise
changes a capture: malformed or incomplete data is rejected with
``CaptureValidationError`` so it cannot silently enter a training set.

The important invariants are deliberately checked twice where possible:
the raw pre-end bytes must match the recorded content hash, mapped presses
must have a one-to-one matching release, and the paired press pitch multiset
must equal the expanded canonical score pitch multiset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from core.performance_capture import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    canonical_notes_hash,
    validate_event_payload,
)
from core.score_model import require_valid
from core.score_io import note_id_to_midi


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CaptureValidationError(ValueError):
    """Raised when a capture is not a complete, internally consistent sample."""


@dataclass(frozen=True)
class CapturedNote:
    """One validated press/release pair, timed from the first note-down."""

    onset_s: float
    duration_s: float
    midi_pitch: int
    note_id: str
    semitone: int
    press_id: int
    synthetic_close: bool


@dataclass(frozen=True)
class CaptureSample:
    """A validated completed capture with both source records and derived notes.

    ``piece_group`` is deliberately the canonical-score SHA-256, allowing
    callers to keep different performances of the same canonical score in a
    single split group without using any personal/device metadata.
    """

    session_id: str
    path: Path
    bpm: int
    quantize: float
    mapping_hash: str
    piece_group: str
    training_eligible: bool
    quality: Mapping[str, Any]
    canonical_notes: list[Any]
    notes: tuple[CapturedNote, ...]
    start: Mapping[str, Any]
    end: Mapping[str, Any]


def _fail(message: str) -> None:
    raise CaptureValidationError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} 必须是 JSON 对象")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} 必须是非空字符串")
    return value


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(f"{label} 必须是 64 位十六进制 SHA-256")
    return value.lower()


def _strict_json_object(raw_line: bytes, line_number: int) -> dict[str, Any]:
    """Decode one JSON object while rejecting duplicate keys and non-JSON NaN."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CaptureValidationError(f"第 {line_number} 行含重复字段: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise CaptureValidationError(f"第 {line_number} 行含非严格 JSON 数值: {value}")

    try:
        decoded = raw_line.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=no_duplicates,
            parse_constant=invalid_constant,
        )
    except UnicodeDecodeError as exc:
        _fail(f"第 {line_number} 行不是 UTF-8: {exc}")
    except json.JSONDecodeError as exc:
        _fail(f"第 {line_number} 行不是有效 JSON: {exc.msg}")
    return _require_object(value, f"第 {line_number} 行")


def _read_records(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(f"无法读取 capture 文件: {exc}")
    if not raw:
        _fail("capture 文件为空")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("capture 文件不得含 UTF-8 BOM")
    if not raw.endswith(b"\n"):
        _fail("capture JSONL 最后一行必须以换行结束")
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        _fail("capture JSONL 行分隔无效")
    records = [_strict_json_object(line, index) for index, line in enumerate(lines, 1)]
    return records, lines


def _validate_start(start: dict[str, Any]) -> tuple[str, int, float, str]:
    if start.get("record") != "start":
        _fail("第一条记录必须是 start")
    if start.get("schema") != SCHEMA_NAME:
        _fail("capture schema 不匹配")
    if not _is_int(start.get("schema_version")) or start["schema_version"] != SCHEMA_VERSION:
        _fail(f"不支持的 capture schema_version: {start.get('schema_version')!r}")
    if start.get("status") != "active":
        _fail("start.status 必须为 active")
    session_id = _require_nonempty_string(start.get("session_id"), "start.session_id")
    mapping_hash = _require_hash(start.get("mapping_hash"), "start.mapping_hash")
    bpm = start.get("bpm")
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(float(bpm)):
        _fail("start.bpm 必须是有限数字")
    if int(bpm) != float(bpm) or not 30 <= int(bpm) <= 300:
        _fail("start.bpm 必须是 30-300 的整数")
    quantize = start.get("quantize")
    if isinstance(quantize, bool) or not isinstance(quantize, (int, float)) or not math.isfinite(float(quantize)):
        _fail("start.quantize 必须是有限数字")
    if float(quantize) < 0:
        _fail("start.quantize 不得为负数")
    return session_id, int(bpm), float(quantize), mapping_hash


def _event_signature(event: Mapping[str, Any], line_number: int) -> tuple[int, str, int, str, int, bool]:
    press_id = event.get("press_id")
    if not _is_int(press_id) or press_id <= 0:
        _fail(f"第 {line_number} 行 note event 的 press_id 必须是正整数")
    note_id = _require_nonempty_string(event.get("note_id"), f"第 {line_number} 行 note_id")
    semitone = event.get("semitone", 0)
    if not _is_int(semitone) or semitone not in (0, 1):
        _fail(f"第 {line_number} 行 semitone 必须是 0 或 1")
    modifier = event.get("modifier", "natural")
    if not isinstance(modifier, str):
        _fail(f"第 {line_number} 行 modifier 必须是字符串")
    relative_ns = event.get("relative_ns")
    if not _is_int(relative_ns) or relative_ns < 0:
        _fail(f"第 {line_number} 行 note event 的 relative_ns 必须是非负整数")
    try:
        note_id_to_midi(note_id, semitone)
    except ValueError as exc:
        _fail(f"第 {line_number} 行音符无效: {exc}")
    synthetic_close = event.get("synthetic_close", False)
    if not isinstance(synthetic_close, bool):
        _fail(f"第 {line_number} 行 synthetic_close 必须是布尔值")
    return press_id, note_id, semitone, modifier, relative_ns, synthetic_close


def _expand_canonical_pitches(canonical_notes: list[Any]) -> list[int]:
    """Expand score rows (including chord rows) into their MIDI pitch multiset."""

    pitches: list[int] = []
    for index, item in enumerate(canonical_notes, 1):
        if not isinstance(item, dict):
            _fail(f"canonical_notes 第 {index} 项必须是对象")
        note_ids = item.get("notes")
        if not isinstance(note_ids, list):
            _fail(f"canonical_notes 第 {index} 项 notes 必须是列表")
        semitone = item.get("semitone", 0)
        if not _is_int(semitone) or semitone not in (0, 1):
            _fail(f"canonical_notes 第 {index} 项 semitone 必须是 0 或 1")
        for note_id in note_ids:
            note_id = _require_nonempty_string(note_id, f"canonical_notes 第 {index} 项 note_id")
            try:
                pitches.append(note_id_to_midi(note_id, semitone))
            except ValueError as exc:
                _fail(f"canonical_notes 第 {index} 项音符无效: {exc}")
    return pitches


def load_capture(path: str | Path, require_training_eligible: bool = True) -> CaptureSample:
    """Load one completed v1 capture after validating all training invariants.

    The validation reads the file only.  A capture must have exactly
    ``start, event*, end`` order, continuous event sequence/timestamps, a
    correct raw-byte content hash, closed matching mapped presses, and a final
    canonical score whose expanded pitches/count match the paired presses.
    ``require_training_eligible`` controls only whether an otherwise valid
    sample marked ineligible may be returned; it never relaxes integrity checks.
    """

    capture_path = Path(path)
    records, raw_lines = _read_records(capture_path)
    if len(records) < 2:
        _fail("capture 至少需要 start 和 end 记录")
    if records[0].get("record") != "start" or records[-1].get("record") != "end":
        _fail("capture 记录顺序必须为 start、event*、end")
    if any(record.get("record") != "event" for record in records[1:-1]):
        _fail("capture 记录顺序必须为 start、event*、end")

    start = records[0]
    end = records[-1]
    session_id, bpm, quantize, mapping_hash = _validate_start(start)
    if end.get("status") != "completed":
        _fail("只有 completed capture 可加载")
    if end.get("record") != "end":
        _fail("最后一条记录必须是 end")
    if end.get("session_id") != session_id:
        _fail("end.session_id 与 start 不一致")
    expected_content_hash = _require_hash(end.get("content_sha256"), "end.content_sha256")
    actual_content_hash = hashlib.sha256(b"".join(raw_lines[:-1])).hexdigest()
    if actual_content_hash != expected_content_hash:
        _fail("content_sha256 与原始 pre-end JSONL 内容不匹配")

    active: dict[int, tuple[str, int, str, int, bool, int]] = {}
    completed_press_ids: set[int] = set()
    paired: list[tuple[int, int, str, int, int, bool]] = []
    expected_seq = 1
    previous_elapsed = -1
    previous_relative = -1
    mapped_event_count = 0
    mapped_press_count = 0
    ignored_count = 0
    synthetic_close_count = 0
    for index, record in enumerate(records[1:-1], 2):
        if record.get("session_id") != session_id:
            _fail(f"第 {index} 行 session_id 不一致")
        seq = record.get("seq")
        if not _is_int(seq) or seq != expected_seq:
            _fail(f"第 {index} 行 seq 不连续")
        expected_seq += 1
        elapsed_ns = record.get("elapsed_ns")
        if not _is_int(elapsed_ns) or elapsed_ns < 0 or elapsed_ns < previous_elapsed:
            _fail(f"第 {index} 行 elapsed_ns 必须非负且不回退")
        previous_elapsed = elapsed_ns
        event = _require_object(record.get("event"), f"第 {index} 行 event")
        try:
            validate_event_payload(event)
        except ValueError as exc:
            _fail(f"第 {index} 行 event 未通过隐私校验: {exc}")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            _fail(f"第 {index} 行 event.type 必须是非空字符串")
        synthetic_close_value = event.get("synthetic_close", False)
        if not isinstance(synthetic_close_value, bool):
            _fail(f"第 {index} 行 synthetic_close 必须是布尔值")
        if synthetic_close_value:
            synthetic_close_count += 1
        if event_type in {"control", "pause", "resume", "stop"}:
            pass
        elif event_type in {"diagnostic", "ignored", "synthetic_close"}:
            if event_type == "ignored":
                ignored_count += 1
        else:
            mapped_event_count += 1
            if event_type in {"down", "note_down", "note_on"}:
                mapped_press_count += 1
        relative_value = event.get("relative_ns")
        if relative_value is not None:
            if not _is_int(relative_value) or relative_value < 0 or relative_value < previous_relative:
                _fail(f"第 {index} 行 relative_ns 必须非负且不回退")
            previous_relative = relative_value
        if event_type not in {"note_down", "note_up"}:
            continue
        press_id, note_id, semitone, modifier, relative_ns, synthetic_close = _event_signature(event, index)
        if relative_ns < previous_relative:
            _fail(f"第 {index} 行 relative_ns 必须非负且不回退")
        previous_relative = relative_ns
        if event_type == "note_down":
            if press_id in active or press_id in completed_press_ids:
                _fail(f"第 {index} 行 press_id 重复: {press_id}")
            active[press_id] = (note_id, semitone, modifier, relative_ns, synthetic_close, index)
            continue
        down = active.get(press_id)
        if down is None:
            _fail(f"第 {index} 行 note_up 没有匹配的 note_down")
        down_note_id, down_semitone, down_modifier, down_relative_ns, down_synthetic, _down_line = down
        if (note_id, semitone, modifier) != (down_note_id, down_semitone, down_modifier):
            _fail(f"第 {index} 行 note_up 与 note_down 音符签名不一致")
        if relative_ns < down_relative_ns:
            _fail(f"第 {index} 行 note_up 早于对应 note_down")
        active.pop(press_id)
        completed_press_ids.add(press_id)
        paired.append((down_relative_ns, relative_ns, note_id, semitone, press_id, down_synthetic or synthetic_close))

    if active:
        _fail("存在没有 note_up 的 note_down")
    event_count = end.get("event_count")
    last_seq = end.get("last_seq")
    captured_count = end.get("captured_count")
    if not _is_int(event_count) or not _is_int(last_seq):
        _fail("end.event_count 和 last_seq 必须是整数")
    if event_count != len(records) - 2 or last_seq != len(records) - 2:
        _fail("end.event_count 或 last_seq 与事件记录不一致")
    if not _is_int(captured_count) or captured_count <= 0:
        _fail("end.captured_count 必须是正整数")
    if captured_count != len(paired):
        _fail("end.captured_count 与配对按键数量不一致")
    if not paired:
        _fail("capture 不含完整配对音符")

    quality = _require_object(end.get("quality"), "end.quality")
    inputs = mapped_press_count + ignored_count
    ignored_ratio = (ignored_count / inputs) if inputs else 0.0
    quality_flags = []
    if ignored_ratio > 0.10:
        quality_flags.append("high_ignored_ratio")
    if synthetic_close_count:
        quality_flags.append("held_note_closed_on_stop")
    expected_quality = {
        "input_count": inputs,
        "mapped_event_count": mapped_event_count,
        "mapped_press_count": mapped_press_count,
        "paired_press_count": len(completed_press_ids),
        "unclosed_press_count": len(active),
        "ignored_count": ignored_count,
        "ignored_ratio": ignored_ratio,
        "synthetic_close": synthetic_close_count > 0,
        "synthetic_close_count": synthetic_close_count,
        "flags": quality_flags,
        "training_eligible": not quality_flags,
    }
    serialized_quality = json.dumps(
        quality, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    serialized_expected_quality = json.dumps(
        expected_quality,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if serialized_quality != serialized_expected_quality:
        _fail("end.quality 与事件重算结果不一致")
    training_eligible = expected_quality["training_eligible"]
    if require_training_eligible and not training_eligible:
        _fail("capture 未标记为 training_eligible")

    canonical_notes = end.get("canonical_notes")
    if not isinstance(canonical_notes, list) or not canonical_notes:
        _fail("end.canonical_notes 必须是非空列表")
    try:
        require_valid(canonical_notes, bpm=bpm)
    except (TypeError, ValueError) as exc:
        _fail(f"end.canonical_notes 未通过乐谱校验: {exc}")
    canonical_hash = _require_hash(end.get("canonical_notes_sha256"), "end.canonical_notes_sha256")
    if canonical_notes_hash(canonical_notes) != canonical_hash:
        _fail("canonical_notes_sha256 与 canonical_notes 不匹配")
    if end.get("canonical_notes_count") != len(canonical_notes):
        _fail("canonical_notes_count 与 canonical_notes 不一致")
    canonical_pitches = _expand_canonical_pitches(canonical_notes)
    if not canonical_pitches:
        _fail("end.canonical_notes 不含可训练音符")
    paired_pitches = [note_id_to_midi(note_id, semitone) for _, _, note_id, semitone, _, _ in paired]
    if len(canonical_pitches) != len(paired_pitches) or sorted(canonical_pitches) != sorted(paired_pitches):
        _fail("规范乐谱展开后的音高多重集合与配对原始音符不一致")

    paired.sort(key=lambda item: (item[0], item[2], item[3], item[4], item[1]))
    first_onset_ns = min(down_ns for down_ns, *_ in paired) if paired else 0
    notes = tuple(
        CapturedNote(
            onset_s=(down_ns - first_onset_ns) / 1_000_000_000,
            duration_s=(up_ns - down_ns) / 1_000_000_000,
            midi_pitch=note_id_to_midi(note_id, semitone),
            note_id=note_id,
            semitone=semitone,
            press_id=press_id,
            synthetic_close=synthetic_close,
        )
        for down_ns, up_ns, note_id, semitone, press_id, synthetic_close in paired
    )
    return CaptureSample(
        session_id=session_id,
        path=capture_path,
        bpm=bpm,
        quantize=quantize,
        mapping_hash=mapping_hash,
        piece_group=canonical_hash,
        training_eligible=training_eligible,
        quality=quality,
        canonical_notes=canonical_notes,
        notes=notes,
        start=start,
        end=end,
    )


def iter_completed_captures(root: str | Path) -> Iterator[Path]:
    """Yield direct ``completed/*.jsonl`` children in deterministic path order.

    Enumeration is intentionally shallow because the capture writer's atomic
    finalization target is ``<root>/completed``; pending and rejected files are
    never returned and no path is opened or modified here.
    """

    completed = Path(root) / "completed"
    if not completed.is_dir():
        return
    yield from sorted((path for path in completed.glob("*.jsonl") if path.is_file()), key=lambda path: str(path))


__all__ = [
    "CaptureSample",
    "CapturedNote",
    "CaptureValidationError",
    "iter_completed_captures",
    "load_capture",
]
