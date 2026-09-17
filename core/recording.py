"""Read-only performance capture and conversion to canonical score rows.

The recorder observes key/button state and timestamps only.  It never imports
or calls KeyboardDriver, Player, or EventPlayer.  Rendering reuses the same
SourceSong adapter as MIDI import so gaps, chords, semitones and degradation
reports converge on the existing storage schema.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class RecordedPress:
    start_s: float
    end_s: float
    note_id: str
    semitone: int
    key: str
    mouse: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingResult:
    notes: list[dict]
    bpm: int
    captured_count: int
    duration_s: float
    warnings: list[str] = field(default_factory=list)
    degradations: list[ImportDegradation] = field(default_factory=list)


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

    def resolve(self, key, mouse: Iterable[str] = ()) -> ResolvedNote | None:
        key = normalize_key(key)
        held = {normalize_mouse(button) for button in mouse if button}
        if not self.use_event_path:
            note_id = self._legacy.get(key)
            return ResolvedNote(note_id) if note_id else None

        if not held and key in self._direct:
            return ResolvedNote(self._direct[key])
        pitch = self._pitch.get(key)
        if pitch is None:
            return None
        active = [name for name, button in self._modifiers.items() if button in held]
        if len(active) > 1:
            return None
        modifier = active[0] if active else "natural"
        if modifier == "lower":
            return ResolvedNote(f"low_{pitch}")
        if modifier == "higher":
            return ResolvedNote(f"high_{pitch}")
        if modifier == "semitone":
            return ResolvedNote(f"mid_{pitch}", 1)
        return ResolvedNote(f"mid_{pitch}")

    @property
    def has_mapping(self) -> bool:
        return bool(self._pitch if self.use_event_path else self._legacy)


class PerformanceRecorder:
    """Thread-safe press/release timestamp recorder with deterministic render."""

    def __init__(
        self,
        resolver: PhysicalNoteResolver | Callable[[str, Iterable[str]], ResolvedNote | None],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._resolver = resolver.resolve if hasattr(resolver, "resolve") else resolver
        self._clock = clock
        self._lock = threading.RLock()
        self._running = False
        self._origin = 0.0
        self._active = {}
        self._events: list[RecordedPress] = []
        self._ignored = 0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._running

    @property
    def captured_count(self) -> int:
        with self._lock:
            return len(self._events)

    def start(self):
        with self._lock:
            self._origin = float(self._clock())
            self._active.clear()
            self._events.clear()
            self._ignored = 0
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
            self._active[token] = (
                max(0.0, now - self._origin),
                resolved,
                held,
            )
            return True

    def release(self, key, *, at: float | None = None) -> bool:
        with self._lock:
            token = normalize_key(key)
            active = self._active.pop(token, None)
            if active is None:
                return False
            now = float(self._clock() if at is None else at)
            start_s, resolved, held = active
            end_s = max(start_s + 0.01, now - self._origin)
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
            self._ignored = 0

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
                self.release(key, at=now)
            self._running = False
            events = sorted(
                self._events,
                key=lambda event: (event.start_s, event.end_s, event.note_id),
            )
            ignored = self._ignored
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
        return RecordingResult(
            notes=adapted.notes,
            bpm=adapted.bpm,
            captured_count=len(events),
            duration_s=duration_s,
            warnings=warnings,
            degradations=adapted.degradations,
        )
