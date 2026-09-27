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
    # 不参与游戏投影的完整来源信息，供 sidecar 和后续重新适配使用。
    source_metadata: dict = field(default_factory=dict)


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
    time_signatures = []
    control_changes = []
    other_events = []
    note_events = []
    tracks = {}
    track_end_ticks = {}
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
            elif message.type == "time_signature":
                time_signatures.append(
                    {
                        "tick": tick,
                        "track": track_index,
                        "numerator": int(message.numerator),
                        "denominator": int(message.denominator),
                        "clocks_per_click": int(message.clocks_per_click),
                        "notated_32nd_notes_per_beat": int(
                            message.notated_32nd_notes_per_beat
                        ),
                    }
                )
            elif message.type == "control_change":
                control_changes.append(
                    {
                        "tick": tick,
                        "track": track_index,
                        "channel": int(message.channel),
                        "control": int(message.control),
                        "value": int(message.value),
                    }
                )
            elif message.type in ("note_on", "note_off"):
                if message.channel == 9:
                    if message.type == "note_on" and message.velocity > 0:
                        percussion_count += 1
                note_events.append(
                    (
                        tick,
                        order,
                        track_index,
                        int(message.channel),
                        message.type,
                        int(message.note),
                        int(message.velocity),
                        int(message.channel) == 9,
                    )
                )
            elif message.type in (
                "program_change",
                "pitchwheel",
                "aftertouch",
                "polytouch",
            ):
                payload = {
                    "tick": tick,
                    "track": track_index,
                    "channel": int(message.channel),
                    "type": message.type,
                }
                for key in ("program", "pitch", "value", "note"):
                    if hasattr(message, key):
                        payload[key] = int(getattr(message, key))
                other_events.append(payload)
        tracks[track_index] = track_name
        track_end_ticks[track_index] = tick
        last_tick = max(last_tick, tick)

    seconds = _build_tick_converter(tempos, midi.ticks_per_beat)
    duration_s = seconds(last_tick)
    if duration_s > MAX_MIDI_SECONDS:
        raise MidiSourceError("MIDI 时长超过 30 分钟，请先裁剪")

    active = defaultdict(deque)
    notes = []
    note_records = []
    unclosed_count = 0
    for tick, _order, track_index, channel, message_type, pitch, velocity, percussion in sorted(note_events):
        key = (track_index, channel, pitch)
        if message_type == "note_on" and velocity > 0:
            active[key].append({"start_tick": tick, "velocity": velocity, "percussion": percussion})
        elif active[key]:
            started = active[key].popleft()
            start_tick = int(started["start_tick"])
            if tick > start_tick:
                note_records.append(
                    {
                        "track": track_index,
                        "channel": channel,
                        "pitch": pitch,
                        "velocity": int(started["velocity"]),
                        "start_tick": start_tick,
                        "end_tick": int(tick),
                        "percussion": bool(started["percussion"]),
                    }
                )

    for (track_index, channel, pitch), starts in active.items():
        for start_tick in starts:
            start_value = int(start_tick["start_tick"])
            closing_tick = max(
                int(track_end_ticks.get(track_index, last_tick)),
                start_value + midi.ticks_per_beat,
            )
            note_records.append(
                {
                    "track": track_index,
                    "channel": channel,
                    "pitch": int(pitch),
                    "velocity": int(start_tick["velocity"]),
                    "start_tick": start_value,
                    "end_tick": closing_tick,
                    "percussion": bool(start_tick["percussion"]),
                    "unclosed": True,
                }
            )
            duration_s = max(duration_s, seconds(closing_tick))
            unclosed_count += 1

    # CC64 会把实际发声结束时间推迟到踏板抬起；游戏投影仍使用 key end，
    # sidecar 同时保留两者，后续 MIDI 预览可以按 sounding_end 重建。
    pedal_intervals = defaultdict(list)
    pedal_down = {}
    for event in sorted(control_changes, key=lambda item: (item["tick"], item["track"], item["channel"])):
        if event["control"] != 64:
            continue
        key = (event["track"], event["channel"])
        if event["value"] >= 64:
            pedal_down.setdefault(key, event["tick"])
        elif key in pedal_down:
            pedal_intervals[key].append((pedal_down.pop(key), event["tick"]))
    for key, start_tick in pedal_down.items():
        pedal_intervals[key].append((start_tick, track_end_ticks.get(key[0], last_tick)))

    for record in note_records:
        key_end = int(record["end_tick"])
        sounding_end = key_end
        for pedal_start, pedal_end in pedal_intervals.get(
            (record["track"], record["channel"]), ()
        ):
            if pedal_start <= key_end <= pedal_end:
                sounding_end = max(sounding_end, int(pedal_end))
        record["start_s"] = seconds(int(record["start_tick"]))
        record["key_end_s"] = seconds(key_end)
        record["sounding_end_tick"] = sounding_end
        record["sounding_end_s"] = seconds(sounding_end)
        if not record["percussion"]:
            notes.append(
                SourceNote(
                    record["start_s"],
                    record["key_end_s"],
                    int(record["pitch"]),
                    int(record["track"]),
                )
            )

    if len(notes) > MAX_MIDI_NOTES:
        raise MidiSourceError("MIDI 旋律音符超过 30000 个，请先精简")

    warnings = []
    if percussion_count:
        warnings.append(f"已忽略 {percussion_count} 个打击乐音符")
    if unclosed_count:
        warnings.append(f"{unclosed_count} 个未结束音符已按轨道结尾闭合")
    pedal_count = sum(1 for event in control_changes if event["control"] == 64)
    if pedal_count:
        warnings.append(f"已保留 {pedal_count} 个 CC64 踏板事件到来源 sidecar")
    if time_signatures:
        warnings.append(f"已保留 {len(time_signatures)} 个拍号事件到来源 sidecar")

    used_tracks = {note.track for note in notes}
    used_track_names = {index: name for index, name in tracks.items() if index in used_tracks}
    real_tempos = sorted(tempos, key=lambda item: (item[0], item[1]))
    tick_zero_tempos = [item for item in real_tempos if item[0] == 0]
    first_tempo = (tick_zero_tempos[-1][2] if tick_zero_tempos else 500_000)
    raw_bpm = round(60_000_000 / first_tempo)
    bpm_hint = min(MAX_BPM, max(MIN_BPM, raw_bpm))
    if bpm_hint != raw_bpm:
        warnings.append(f"MIDI 速度 {raw_bpm} BPM 超出支持范围，已调整为 {bpm_hint}")
    tempo_change_count = max(1, len({(tick, tempo) for tick, _order, tempo in real_tempos}))

    source_metadata = {
        "schema_version": 1,
        "format": "midi-source",
        "ticks_per_beat": int(midi.ticks_per_beat),
        "duration_ticks": int(last_tick),
        "duration_s": float(duration_s),
        "tracks": {str(index): name for index, name in tracks.items()},
        "track_end_ticks": {str(index): int(value) for index, value in track_end_ticks.items()},
        "tempo_events": [
            {"tick": int(tick), "order": int(order), "tempo_us": int(tempo)}
            for tick, order, tempo in real_tempos
        ],
        "time_signatures": time_signatures,
        "control_changes": control_changes,
        "other_events": other_events,
        "notes": note_records,
    }

    song = SourceSong(
        title=source_path.stem,
        notes=sorted(notes, key=lambda note: (note.start_s, note.pitch, note.track)),
        tracks=used_track_names,
        duration_s=duration_s,
        bpm_hint=bpm_hint,
        tempo_change_count=tempo_change_count,
    )
    return MidiReadResult(song=song, warnings=warnings, source_metadata=source_metadata)
