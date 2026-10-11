"""Read-only performance capture and conversion to canonical score rows.

The recorder observes key/button state and timestamps only.  It never imports
or calls KeyboardDriver, Player, or EventPlayer.  Rendering reuses the same
SourceSong adapter as MIDI import so gaps, chords, semitones and degradation
reports converge on the existing storage schema.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from core.score_io import note_id_to_midi
from core.source_score import AdaptOptions, ImportDegradation, SourceNote, SourceSong, adapt_source_song


def normalize_key(value) -> str:
    text = str(value or "").strip()
    return text.upper() if len(text) == 1 else text.lower()


def normalize_mouse(value) -> str:
    return str(value or "").strip().lower()


@dataclass(frozen=True)
class ResolvedNote:
    note_id: str
    semitone: int = 0
    modifier: str = "natural"


@dataclass(frozen=True)
class RecordedPress:
    start_s: float
    end_s: float
    note_id: str
    semitone: int
    key: str
    mouse: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawInputEvent:
    """Privacy-bounded mapped input event for local performance research.

    ``relative_ns`` is derived from the recorder's monotonic clock.  Deliberately
    omitted are the physical key token, virtual/scan codes, window details and
    mouse coordinates; only an input that has already resolved to a musical
    note can reach this structure.
    """

    sequence: int
    press_id: int
    action: str
    relative_ns: int
    note_id: str
    semitone: int
    modifier: str = "natural"
    mouse: tuple[str, ...] = ()
    closed_by_stop: bool = False

    def to_performance_event(self) -> dict:
        """Return the privacy-safe payload accepted by PerformanceCapture."""

        return {
            "type": "note_down" if self.action == "down" else "note_up",
            "press_id": int(self.press_id),
            "relative_ns": int(self.relative_ns),
            "note_id": self.note_id,
            "semitone": int(self.semitone),
            "modifier": self.modifier,
            "synthetic_close": bool(self.closed_by_stop),
        }


@dataclass(frozen=True)
class RecordingResult:
    notes: list[dict]
    bpm: int
    captured_count: int
    duration_s: float
    warnings: list[str] = field(default_factory=list)
    degradations: list[ImportDegradation] = field(default_factory=list)
    raw_events: list[RawInputEvent] = field(default_factory=list)
    ignored_count: int = 0
    capture_errors: list[str] = field(default_factory=list)


class PhysicalNoteResolver:
    """Invert a legacy keymap or one event-profile physical layout."""

    def __init__(self, *, keymap=None, profile=None):
        self.keymap = keymap
        self.profile = profile
        self.use_event_path = bool(
            profile is not None and getattr(profile, "legacy_keymap", None) is None
        )
        self._legacy = {}
        if keymap is not None:
            for octave in ("low", "mid", "high"):
                for pitch in range(1, 8):
                    note_id = f"{octave}_{pitch}"
                    key = keymap.key_for(note_id)
                    if key:
                        self._legacy[normalize_key(key)] = note_id
        self._pitch = {}
        self._direct = {}
        self._modifiers = {}
        if profile is not None:
            self._pitch = {
                normalize_key(key): pitch
                for pitch, key in enumerate(profile.pitch_keys, start=1)
            }
            self._direct = {
                normalize_key(key): str(note_id)
                for note_id, key in profile.pitch_direct_overrides.items()
            }
            self._modifiers = {
                name: normalize_mouse(button)
                for name, button in profile.modifier_buttons.items()
                if button
            }
        fingerprint_payload = {
            "mode": "event_profile" if self.use_event_path else "legacy_21_key",
            "legacy": sorted(self._legacy.items()),
            "pitch": sorted(self._pitch.items()),
            "direct": sorted(self._direct.items()),
            "modifiers": sorted(self._modifiers.items()),
        }
        encoded = json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._mapping_hash = hashlib.sha256(encoded).hexdigest()

    def resolve(self, key, mouse: Iterable[str] = ()) -> ResolvedNote | None:
        key = normalize_key(key)
        held = {normalize_mouse(button) for button in mouse if button}
        if not self.use_event_path:
            note_id = self._legacy.get(key)
            return ResolvedNote(note_id, modifier="legacy") if note_id else None

        if not held and key in self._direct:
            return ResolvedNote(self._direct[key], modifier="direct")
        pitch = self._pitch.get(key)
        if pitch is None:
            return None
        active = [name for name, button in self._modifiers.items() if button in held]
        if len(active) > 1:
            return None
        modifier = active[0] if active else "natural"
        if modifier == "lower":
            return ResolvedNote(f"low_{pitch}", modifier="lower")
        if modifier == "higher":
            return ResolvedNote(f"high_{pitch}", modifier="higher")
        if modifier == "semitone":
            return ResolvedNote(f"mid_{pitch}", 1, "semitone")
        return ResolvedNote(f"mid_{pitch}", modifier="natural")

    @property
    def has_mapping(self) -> bool:
        return bool(self._pitch if self.use_event_path else self._legacy)

    @property
    def mapping_mode(self) -> str:
        return "event_profile" if self.use_event_path else "legacy_21_key"

    @property
    def mapping_hash(self) -> str:
        """Hash the physical mapping without exposing its key names on disk."""

        return self._mapping_hash


class PerformanceRecorder:
    """Thread-safe press/release timestamp recorder with deterministic render."""

    def __init__(
        self,
        resolver: PhysicalNoteResolver | Callable[[str, Iterable[str]], ResolvedNote | None],
        *,
        clock: Callable[[], float] = time.perf_counter,
        event_sink: Callable[[RawInputEvent], None] | None = None,
    ):
        self._resolver = resolver.resolve if hasattr(resolver, "resolve") else resolver
        self._clock = clock
        self._lock = threading.RLock()
        self._running = False
        self._origin = 0.0
        self._active = {}
        self._events: list[RecordedPress] = []
        self._raw_events: list[RawInputEvent] = []
        self._next_press_id = 0
        self._ignored = 0
        self._event_sink = event_sink
        self._capture_errors: list[str] = []

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._running

    @property
    def captured_count(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def raw_events(self) -> tuple[RawInputEvent, ...]:
        with self._lock:
            return tuple(self._raw_events)

    @property
    def ignored_count(self) -> int:
        with self._lock:
            return self._ignored

    @property
    def capture_errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._capture_errors)

    def _append_raw_event(
        self,
        *,
        action: str,
        press_id: int,
        relative_ns: int,
        resolved: ResolvedNote,
        mouse: tuple[str, ...],
        closed_by_stop: bool = False,
    ):
        event = RawInputEvent(
            sequence=len(self._raw_events) + 1,
            press_id=int(press_id),
            action=str(action),
            relative_ns=max(0, int(relative_ns)),
            note_id=resolved.note_id,
            semitone=int(resolved.semitone),
            modifier=resolved.modifier,
            mouse=tuple(mouse),
            closed_by_stop=bool(closed_by_stop),
        )
        self._raw_events.append(event)
        # Disk or serialization failures must not kill pynput's listener
        # thread.  Preserve the in-memory capture and surface one concise
        # error to the caller at stop time instead.
        if self._event_sink is not None and not self._capture_errors:
            try:
                self._event_sink(event)
            except Exception as exc:
                self._capture_errors.append(
                    f"本地研究样本写入失败: {type(exc).__name__}: {exc}"
                )

    def start(self):
        with self._lock:
            self._origin = float(self._clock())
            self._active.clear()
            self._events.clear()
            self._raw_events.clear()
            self._next_press_id = 0
            self._ignored = 0
            self._capture_errors.clear()
            self._running = True

    def press(self, key, mouse: Iterable[str] = (), *, at: float | None = None) -> bool:
        with self._lock:
            if not self._running:
                return False
            token = normalize_key(key)
            if not token or token in self._active:
                return False
            held = tuple(sorted({normalize_mouse(button) for button in mouse if button}))
            resolved = self._resolver(token, held)
            if resolved is None:
                self._ignored += 1
                return False
            now = float(self._clock() if at is None else at)
            relative_s = max(0.0, now - self._origin)
            self._next_press_id += 1
            press_id = self._next_press_id
            self._active[token] = (
                relative_s,
                resolved,
                held,
                press_id,
            )
            self._append_raw_event(
                action="down",
                press_id=press_id,
                relative_ns=round(relative_s * 1_000_000_000),
                resolved=resolved,
                mouse=held,
            )
            return True

    def release(
        self,
        key,
        *,
        at: float | None = None,
        closed_by_stop: bool = False,
    ) -> bool:
        with self._lock:
            token = normalize_key(key)
            active = self._active.pop(token, None)
            if active is None:
                return False
            now = float(self._clock() if at is None else at)
            start_s, resolved, held, press_id = active
            end_s = max(start_s + 0.01, now - self._origin)
            self._append_raw_event(
                action="up",
                press_id=press_id,
                relative_ns=round(end_s * 1_000_000_000),
                resolved=resolved,
                mouse=held,
                closed_by_stop=closed_by_stop,
            )
            self._events.append(
                RecordedPress(
                    start_s,
                    end_s,
                    resolved.note_id,
                    int(resolved.semitone),
                    token,
                    held,
                )
            )
            return True

    def cancel(self):
        with self._lock:
            self._running = False
            self._active.clear()
            self._events.clear()
            self._raw_events.clear()
            self._ignored = 0
            self._capture_errors.clear()

    def stop(
        self,
        *,
        bpm: int,
        quantize_beats: float = 0.25,
        title: str = "录制乐谱",
        trim_leading: bool = True,
        at: float | None = None,
    ) -> RecordingResult:
        with self._lock:
            if not self._running:
                raise RuntimeError("录制尚未开始")
            now = float(self._clock() if at is None else at)
            for key in list(self._active):
                self.release(key, at=now, closed_by_stop=True)
            self._running = False
            events = sorted(
                self._events,
                key=lambda event: (event.start_s, event.end_s, event.note_id),
            )
            ignored = self._ignored
            raw_events = list(self._raw_events)
            capture_errors = list(self._capture_errors)
        if not events:
            raise ValueError("没有录到可识别的音键")
        if not 30 <= int(bpm) <= 300:
            raise ValueError("录制 BPM 必须在 30-300")
        if quantize_beats < 0:
            raise ValueError("量化拍值不能为负数")

        beats_per_second = int(bpm) / 60.0
        leading = events[0].start_s if trim_leading else 0.0
        source_notes = []
        for event in events:
            start_beats = max(0.0, (event.start_s - leading) * beats_per_second)
            end_beats = max(start_beats, (event.end_s - leading) * beats_per_second)
            if quantize_beats:
                grid = float(quantize_beats)
                start_beats = round(start_beats / grid) * grid
                end_beats = round(end_beats / grid) * grid
                if end_beats <= start_beats:
                    end_beats = start_beats + grid
            elif end_beats <= start_beats:
                end_beats = start_beats + 0.01 * beats_per_second
            source_notes.append(
                SourceNote(
                    start_beats / beats_per_second,
                    end_beats / beats_per_second,
                    note_id_to_midi(event.note_id) + int(event.semitone),
                    0,
                )
            )
        duration_s = max(note.end_s for note in source_notes)
        adapted = adapt_source_song(
            SourceSong(
                title=title,
                notes=source_notes,
                tracks={0: "实时录制"},
                duration_s=duration_s,
                bpm_hint=int(bpm),
                tempo_change_count=1,
            ),
            AdaptOptions(track=None, style="preserve", bpm=int(bpm)),
        )
        warnings = list(adapted.warnings)
        if ignored:
            warnings.insert(0, f"录制期间忽略了 {ignored} 次未映射或冲突输入。")
        warnings.extend(capture_errors)
        return RecordingResult(
            notes=adapted.notes,
            bpm=adapted.bpm,
            captured_count=len(events),
            duration_s=duration_s,
            warnings=warnings,
            degradations=adapted.degradations,
            raw_events=raw_events,
            ignored_count=ignored,
            capture_errors=capture_errors,
        )
