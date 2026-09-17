"""MIDI Type 0/1 reader that preserves tracks and the global tempo map.

The reader deliberately stops at a neutral :class:`SourceSong`.  Mapping a
source pitch to a game profile is handled by ``core.source_score`` so MIDI
parsing never reaches the keyboard/mouse input boundary.

Parts of the track-name recovery and tempo-map approach are adapted from
``gujingyun/delta-melodica`` (MIT License, copyright gujingyun).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from core.score_model import MAX_BPM, MIN_BPM
from core.source_score import SourceNote, SourceSong

MAX_MIDI_BYTES = 10 * 1024 * 1024
MAX_MIDI_EVENTS = 200_000
MAX_MIDI_NOTES = 30_000
MAX_MIDI_SECONDS = 30 * 60


class MidiSourceError(ValueError):
    """A user-facing MIDI format or resource-limit error."""


@dataclass
class MidiReadResult:
    song: SourceSong
    warnings: list[str] = field(default_factory=list)


def _decode_track_name(name: str) -> str:
    """Recover common UTF-8/GB18030 names read losslessly as Latin-1."""
    try:
        raw = name.encode("latin1")
    except UnicodeEncodeError:
        return name.strip()
    try:
        return raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        pass
    try:
        decoded = raw.decode("gb18030")
    except UnicodeDecodeError:
        return name.strip()
    chinese = sum(
        "\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U000323af"
        for char in decoded
    )
    high_bytes = sum(byte >= 0x80 for byte in raw)
    control_bytes = any(0x80 <= byte <= 0x9F for byte in raw)
    if chinese and (high_bytes > chinese or control_bytes):
        return decoded.strip()
    return name.strip()


def _build_tick_converter(tempos, ticks_per_beat):
    """Return ``tick -> seconds`` using tempo events from every track."""
    change_ticks = [0]
    change_seconds = [0.0]
    rates = [500_000]
    for tick, order, tempo in sorted(tempos, key=lambda item: (item[0], item[1])):
        previous_tick = change_ticks[-1]
        elapsed = (
            change_seconds[-1]
            + (tick - previous_tick) * rates[-1] / ticks_per_beat / 1_000_000
        )
        change_ticks.append(tick)
        change_seconds.append(elapsed)
        rates.append(tempo)

    def seconds(tick):
        index = bisect_right(change_ticks, tick) - 1
        return (
            change_seconds[index]
            + (tick - change_ticks[index])
            * rates[index]
            / ticks_per_beat
            / 1_000_000
        )

    return seconds


def read_midi_source(path: str | Path) -> MidiReadResult:
    """Read a PPQ MIDI 0/1 file into an absolute-time, track-aware source song."""
    source_path = Path(path)
    try:
        size = source_path.stat().st_size
    except OSError as exc:
        raise MidiSourceError(f"无法读取 MIDI 文件: {exc}") from exc
    if size > MAX_MIDI_BYTES:
        raise MidiSourceError("MIDI 文件不能超过 10 MB")

    try:
        import mido
    except ImportError as exc:
        raise MidiSourceError("缺少 MIDI 解析组件 mido，请先安装 requirements.txt") from exc

    try:
        midi = mido.MidiFile(source_path, charset="latin1")
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise MidiSourceError(f"MIDI 文件损坏或格式不受支持: {exc}") from exc

    if midi.type == 2:
        raise MidiSourceError("暂不支持 Type 2（异步轨）MIDI，请另存为 Type 0/1")
    if midi.ticks_per_beat <= 0:
        raise MidiSourceError("暂不支持 SMPTE 时基 MIDI，请转换为标准 PPQ 时基")

    tempos = []
    note_events = []
    tracks = {}
    last_tick = 0
    event_count = 0
    percussion_count = 0
    order = 0

    for track_index, track in enumerate(midi.tracks):
        tick = 0
        track_name = f"音轨 {track_index + 1}"
        for message in track:
            event_count += 1
            order += 1
            if event_count > MAX_MIDI_EVENTS:
                raise MidiSourceError("MIDI 事件超过 200000 个，请先精简或导出旋律轨")
            tick += int(message.time)
            if message.type == "track_name":
                decoded = _decode_track_name(message.name)
                if decoded:
                    track_name = decoded
            elif message.type == "set_tempo":
                if message.tempo <= 0:
                    raise MidiSourceError("MIDI 中包含无效速度事件")
                tempos.append((tick, order, int(message.tempo)))
            elif message.type in ("note_on", "note_off"):
                if message.channel == 9:
                    if message.type == "note_on" and message.velocity > 0:
                        percussion_count += 1
                else:
                    note_events.append((tick, order, track_index, message))
        tracks[track_index] = track_name
        last_tick = max(last_tick, tick)

    seconds = _build_tick_converter(tempos, midi.ticks_per_beat)
    duration_s = seconds(last_tick)
    if duration_s > MAX_MIDI_SECONDS:
        raise MidiSourceError("MIDI 时长超过 30 分钟，请先裁剪")

    active = defaultdict(deque)
    notes = []
    unclosed_count = 0
    for tick, _order, track_index, message in sorted(note_events):
        key = (track_index, message.channel, message.note)
        if message.type == "note_on" and message.velocity > 0:
            active[key].append(tick)
        elif active[key]:
            start_tick = active[key].popleft()
            if tick > start_tick:
                notes.append(
                    SourceNote(
                        seconds(start_tick),
                        seconds(tick),
                        int(message.note),
                        track_index,
                    )
                )

    for (track_index, _channel, pitch), starts in active.items():
        for start_tick in starts:
            closing_tick = max(last_tick, start_tick + midi.ticks_per_beat)
            notes.append(
                SourceNote(
                    seconds(start_tick),
                    seconds(closing_tick),
                    int(pitch),
                    track_index,
                )
            )
            duration_s = max(duration_s, seconds(closing_tick))
            unclosed_count += 1

    if len(notes) > MAX_MIDI_NOTES:
        raise MidiSourceError("MIDI 旋律音符超过 30000 个，请先精简")

    warnings = []
    if percussion_count:
        warnings.append(f"已忽略 {percussion_count} 个打击乐音符")
    if unclosed_count:
        warnings.append(f"{unclosed_count} 个未结束音符已按轨道结尾闭合")

    used_tracks = {note.track for note in notes}
    used_track_names = {index: name for index, name in tracks.items() if index in used_tracks}
    real_tempos = sorted(tempos, key=lambda item: (item[0], item[1]))
    first_tempo = real_tempos[0][2] if real_tempos else 500_000
    raw_bpm = round(60_000_000 / first_tempo)
    bpm_hint = min(MAX_BPM, max(MIN_BPM, raw_bpm))
    if bpm_hint != raw_bpm:
        warnings.append(f"MIDI 速度 {raw_bpm} BPM 超出支持范围，已调整为 {bpm_hint}")
    tempo_change_count = max(1, len({(tick, tempo) for tick, _order, tempo in real_tempos}))

    song = SourceSong(
        title=source_path.stem,
        notes=sorted(notes, key=lambda note: (note.start_s, note.pitch, note.track)),
        tracks=used_track_names,
        duration_s=duration_s,
        bpm_hint=bpm_hint,
        tempo_change_count=tempo_change_count,
    )
    return MidiReadResult(song=song, warnings=warnings)
