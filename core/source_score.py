"""将已解析的来源音符转换为 Auto Music Player 的标准曲谱。

本模块刻意只处理内存中的 ``SourceSong``，不读取 MIDI、不触发键盘或鼠标。
外层解析器负责把不同文件格式转换为来源模型；这里的职责是把来源模型稳定地
降级到数据库和播放器共用的 ``[{"notes": ..., "dur": ...}]`` 格式。

算法中的最高活动音旋律提取、钢琴式短缝连接思路，改编自
gujingyun/delta-melodica（MIT License）；此处按本项目的简谱 Schema 重写。
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field
from typing import Sequence

from core.score_model import MAX_BPM, MAX_DUR_BEATS, MIN_BPM, require_valid


# ``low_1`` 是 C3(48)，``high_7`` 是 B5(83)。黑键以其前一个自然音加
# ``semitone: 1`` 表示，正好匹配现有 Score/JSON/MIDI 导出约定。
MIDI_MIN = 48
MIDI_MAX = 83
MAX_SOURCE_PITCH = 127
MAX_SOURCE_TRACKS = 256
MAX_SOURCE_NOTES = 100_000
MAX_SOURCE_DURATION_S = 24 * 60 * 60
_EPSILON = 1e-9
_DEFAULT_BRIDGE_GAP_S = 0.08
_NATURAL_NUM = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
_SHARP_BASE_NUM = {1: 1, 3: 2, 6: 4, 8: 5, 10: 6}


@dataclass(frozen=True)
class SourceNote:
    """一个已配对的来源音符，时间单位为秒，音高采用 MIDI 编号。"""

    start_s: float
    end_s: float
    pitch: int
    track: int = 0


@dataclass(frozen=True)
class SourceSong:
    """格式解析层交给适配器的最小、无依赖中间表示。"""

    title: str
    notes: Sequence[SourceNote]
    tracks: dict[int, str] | int = field(default_factory=lambda: {0: "主旋律"})
    duration_s: float = 0.0
    bpm_hint: float = 100
    tempo_change_count: int = 1


@dataclass(frozen=True)
class AdaptOptions:
    """适配策略；默认适合将普通 MIDI 旋律变成钢琴可弹的单声部。"""

    track: str | int | None = "auto"
    style: str = "piano"  # preserve | original | piano
    transpose: int = 0
    fold_octaves: bool = True
    bpm: float | None = None


@dataclass(frozen=True)
class ImportDegradation:
    """一次有意的信息降级，便于 UI 和后续导入记录做结构化呈现。"""

    code: str
    count: int
    message: str


@dataclass(frozen=True)
class AdaptedScore:
    """已通过 Schema 校验、可直接入库/送入 Profile→IR→Compiler 的曲谱。"""

    name: str
    bpm: int
    notes: list[dict]
    warnings: list[str]
    degradations: list[ImportDegradation]
    tracks: dict[int, str]
    selected_track: int | None


@dataclass(frozen=True)
class _Segment:
    """时间线上一段不重叠的声音，members 为 ``notes`` 的原始索引。"""

    start_s: float
    end_s: float
    members: tuple[int, ...]


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _require_finite(value, label: str) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数字")
    return float(value)


def _require_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} 必须是整数")
    return value


def _validate_note(note: SourceNote, index: int, *, track_limit: int | None = None) -> None:
    if not isinstance(note, SourceNote):
        raise ValueError(f"来源音符 {index + 1} 不是 SourceNote")
    start_s = _require_finite(note.start_s, f"来源音符 {index + 1} 的开始时间")
    end_s = _require_finite(note.end_s, f"来源音符 {index + 1} 的结束时间")
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"来源音符 {index + 1} 的时间范围无效")
    if end_s > MAX_SOURCE_DURATION_S:
        raise ValueError(f"来源音符 {index + 1} 超出最大时长限制")
    pitch = _require_int(note.pitch, f"来源音符 {index + 1} 的音高")
    if not 0 <= pitch <= MAX_SOURCE_PITCH:
        raise ValueError(f"来源音符 {index + 1} 的 MIDI 音高必须在 0-{MAX_SOURCE_PITCH}")
    track = _require_int(note.track, f"来源音符 {index + 1} 的音轨")
    if track < 0 or (track_limit is not None and track >= track_limit):
        raise ValueError(f"来源音符 {index + 1} 的音轨编号无效")


def validate_source_song(song: SourceSong) -> None:
    """验证解析层输入，尽早拒绝 NaN、负时间、越界轨道和非法 MIDI 音高。"""

    if not isinstance(song, SourceSong):
        raise ValueError("来源曲目必须是 SourceSong")
    if not isinstance(song.title, str):
        raise ValueError("曲名必须是字符串")
    if not isinstance(song.notes, (list, tuple)):
        raise ValueError("来源音符必须是列表或元组")
    if len(song.notes) > MAX_SOURCE_NOTES:
        raise ValueError(f"来源音符数超出上限({MAX_SOURCE_NOTES})")
    track_ids = _track_ids(song)
    duration_s = _require_finite(song.duration_s, "曲目时长")
    if not 0 <= duration_s <= MAX_SOURCE_DURATION_S:
        raise ValueError("曲目时长超出允许范围")
    bpm_hint = _require_finite(song.bpm_hint, "BPM 提示")
    if not MIN_BPM <= bpm_hint <= MAX_BPM:
        raise ValueError(f"BPM 提示必须在 {MIN_BPM}-{MAX_BPM}")
    tempo_changes = _require_int(song.tempo_change_count, "速度变化数")
    if tempo_changes < 1:
        raise ValueError("速度变化数至少为 1")
    for index, note in enumerate(song.notes):
        _validate_note(note, index)
        if note.track not in track_ids:
            raise ValueError(f"来源音符 {index + 1} 的音轨编号无效")
        if note.end_s > duration_s + _EPSILON:
            raise ValueError(f"来源音符 {index + 1} 超出曲目时长")


def _validate_note_sequence(notes: Sequence[SourceNote]) -> list[SourceNote]:
    if not isinstance(notes, (list, tuple)):
        raise ValueError("来源音符必须是列表或元组")
    out = list(notes)
    for index, note in enumerate(out):
        _validate_note(note, index)
    return out


def recommend_track(song: SourceSong) -> int:
    """优先明确旋律轨，再推荐单声部、可演奏密度稳定的轨道。"""

    validate_source_song(song)
    groups = {
        track: [note for note in song.notes if note.track == track]
        for track in _track_ids(song)
    }
    groups = {track: notes for track, notes in groups.items() if notes}
    if not groups:
        raise ValueError("所选文件没有可适配的旋律音符")
    names = _track_names(song)
    positive_words = ("melody", "vocal", "lead", "主旋律", "人声", "旋律")
    negative_words = (
        "drum", "perc", "bass", "chord", "accomp", "鼓", "打击", "贝斯", "伴奏", "和弦"
    )
    largest = max(len(notes) for notes in groups.values())
    candidates = [
        track for track, notes in groups.items()
        if len(notes) >= max(1, largest * 0.1)
        or any(word in names.get(track, "").lower() for word in positive_words)
    ]

    def score(track):
        name = names.get(track, "").lower()
        named = any(word in name for word in positive_words)
        negative = any(word in name for word in negative_words)
        notes = groups[track]
        weights = [min(note.end_s - note.start_s, 1.0) for note in notes]
        weighted_pitch = sum(note.pitch * weight for note, weight in zip(notes, weights)) / sum(weights)
        ordered = sorted(notes, key=lambda item: (item.start_s, item.end_s, item.pitch))
        non_overlapping = 0
        active_end = -1.0
        for note in ordered:
            if note.start_s >= active_end - _EPSILON:
                non_overlapping += 1
            active_end = max(active_end, note.end_s)
        monophonic_ratio = non_overlapping / len(ordered)
        playable_ratio = sum(
            note.end_s - note.start_s >= 0.06 for note in ordered
        ) / len(ordered)
        span = max(note.end_s for note in ordered) - min(note.start_s for note in ordered)
        density = len(ordered) / max(span, 0.001)
        density_score = -abs(density - 2.5)
        return (
            named,
            not negative,
            monophonic_ratio,
            playable_ratio,
            density_score,
            weighted_pitch,
            -track,
        )

    return max(candidates, key=score)


def _track_ids(song: SourceSong) -> tuple[int, ...]:
    if isinstance(song.tracks, int) and not isinstance(song.tracks, bool):
        if not 1 <= song.tracks <= MAX_SOURCE_TRACKS:
            raise ValueError(f"音轨数必须在 1-{MAX_SOURCE_TRACKS}")
        return tuple(range(song.tracks))
    if not isinstance(song.tracks, dict) or not song.tracks:
        raise ValueError("音轨必须是非空编号到名称映射")
    if len(song.tracks) > MAX_SOURCE_TRACKS:
        raise ValueError(f"音轨数不能超过 {MAX_SOURCE_TRACKS}")
    ids = []
    for track, name in song.tracks.items():
        _require_int(track, "音轨编号")
        if not 0 <= track < MAX_SOURCE_TRACKS:
            raise ValueError(f"音轨编号必须在 0-{MAX_SOURCE_TRACKS - 1}")
        if not isinstance(name, str):
            raise ValueError("音轨名称必须是字符串")
        ids.append(track)
    return tuple(sorted(ids))


def _track_names(song: SourceSong) -> dict[int, str]:
    if isinstance(song.tracks, int):
        return {track: f"音轨 {track + 1}" for track in range(song.tracks)}
    return dict(song.tracks)


def _active_intervals(notes: Sequence[SourceNote]):
    """Yield active-note state in O(n log n), including the highest candidate.

    MIDI limits allow tens of thousands of notes.  Re-scanning every note at
    every boundary would be quadratic, so starts/ends update one active set and
    a lazily-pruned priority queue instead.
    """

    changes: dict[float, dict[str, list[int]]] = {}
    for index, note in enumerate(notes):
        changes.setdefault(float(note.start_s), {"start": [], "end": []})["start"].append(index)
        changes.setdefault(float(note.end_s), {"start": [], "end": []})["end"].append(index)
    boundaries = sorted(changes)
    active: set[int] = set()
    groups: dict[tuple[float, float], set[int]] = {}
    highest = []

    for position, left in enumerate(boundaries[:-1]):
        change = changes[left]
        for index in change["end"]:
            active.discard(index)
            note = notes[index]
            group_key = (note.start_s, note.end_s)
            members = groups.get(group_key)
            if members is not None:
                members.discard(index)
                if not members:
                    groups.pop(group_key, None)
        for index in change["start"]:
            active.add(index)
            note = notes[index]
            groups.setdefault((note.start_s, note.end_s), set()).add(index)
            # min-heap equivalent of max((pitch, start_s, -index)).
            heapq.heappush(highest, (-note.pitch, -note.start_s, index))

        while highest and highest[0][2] not in active:
            heapq.heappop(highest)
        right = boundaries[position + 1]
        if right - left <= _EPSILON or not active:
            continue
        yield left, right, active, groups, highest[0][2]


def _selected_highest_segments(notes: Sequence[SourceNote]) -> tuple[list[_Segment], int]:
    """在每个时间切片保留最高活动音；返回被清理的重叠声部数。"""

    source = _validate_note_sequence(notes)
    segments: list[_Segment] = []
    cleanup_count = 0
    for left, right, active, _groups, selected in _active_intervals(source):
        # 同音高时优先较晚开始的音，确保重叠的重复音不会被错误合并。
        cleanup_count += len(active) - 1
        if segments and segments[-1].members == (selected,) and abs(segments[-1].end_s - left) <= _EPSILON:
            previous = segments[-1]
            segments[-1] = _Segment(previous.start_s, right, previous.members)
        else:
            segments.append(_Segment(left, right, (selected,)))
    return segments, cleanup_count


def monophonic_highest_active(notes: Sequence[SourceNote]) -> list[SourceNote]:
    """公开的最高活动音提取器；不合并不同来源的相邻重复音。"""

    source = _validate_note_sequence(notes)
    segments, _ = _selected_highest_segments(source)
    return [
        SourceNote(segment.start_s, segment.end_s, source[segment.members[0]].pitch, source[segment.members[0]].track)
        for segment in segments
    ]


def piano_melody(notes: Sequence[SourceNote]) -> list[SourceNote]:
    """生成钢琴友好的单旋律：全局最高活动音，再连接极短的无声缝隙。"""

    return bridge_short_gaps(monophonic_highest_active(notes))


def _bridge_short_gaps_with_count(
    notes: Sequence[SourceNote], max_gap_s: float = _DEFAULT_BRIDGE_GAP_S
) -> tuple[list[SourceNote], int]:
    source = _validate_note_sequence(notes)
    gap_limit = _require_finite(max_gap_s, "短缝阈值")
    if gap_limit < 0:
        raise ValueError("短缝阈值不能小于 0")
    ordered = sorted(source, key=lambda note: (note.start_s, note.end_s, note.pitch, note.track))
    bridged: list[SourceNote] = []
    count = 0
    for note in ordered:
        if bridged:
            previous = bridged[-1]
            gap = note.start_s - previous.end_s
            if _EPSILON < gap <= gap_limit:
                bridged[-1] = SourceNote(previous.start_s, note.start_s, previous.pitch, previous.track)
                count += 1
        bridged.append(note)
    return bridged, count


def bridge_short_gaps(
    notes: Sequence[SourceNote], max_gap_s: float = _DEFAULT_BRIDGE_GAP_S
) -> list[SourceNote]:
    """将不超过阈值的单声部静音缝隙延长到前一个音，避免钢琴式断续感。"""

    return _bridge_short_gaps_with_count(notes, max_gap_s)[0]


def _preserve_segments(notes: Sequence[SourceNote]) -> tuple[list[_Segment], int]:
    """保留完全同起止的和弦；与其他重叠竞争时降到最高活动音。"""

    source = _validate_note_sequence(notes)
    segments: list[_Segment] = []
    cleanup_count = 0
    for left, right, active, groups, selected in _active_intervals(source):
        if len(groups) == 1:
            members = tuple(sorted(next(iter(groups.values()))))
        else:
            members = (selected,)
            cleanup_count += len(active) - 1
        if segments and segments[-1].members == members and abs(segments[-1].end_s - left) <= _EPSILON:
            previous = segments[-1]
            segments[-1] = _Segment(previous.start_s, right, members)
        else:
            segments.append(_Segment(left, right, members))
    return segments, cleanup_count


def _resolve_options(options: AdaptOptions) -> None:
    if not isinstance(options, AdaptOptions):
        raise ValueError("适配选项必须是 AdaptOptions")
    if options.track != "auto" and options.track is not None:
        _require_int(options.track, "选定音轨")
    if options.style not in {"preserve", "original", "piano"}:
        raise ValueError("style 必须为 preserve、original 或 piano")
    transpose = _require_int(options.transpose, "移调")
    if not -24 <= transpose <= 24:
        raise ValueError("移调必须在 -24 到 24 个半音之间")
    if not isinstance(options.fold_octaves, bool):
        raise ValueError("fold_octaves 必须是布尔值")
    if options.bpm is not None:
        bpm = _require_finite(options.bpm, "指定 BPM")
        if not MIN_BPM <= bpm <= MAX_BPM:
            raise ValueError(f"指定 BPM 必须在 {MIN_BPM}-{MAX_BPM}")


def _selected_source_notes(song: SourceSong, options: AdaptOptions) -> tuple[list[SourceNote], int | None]:
    if options.track == "auto":
        selected_track: int | None = recommend_track(song)
    elif options.track is None:
        selected_track = None
    else:
        selected_track = options.track
        if selected_track not in _track_ids(song):
            raise ValueError(f"不存在音轨 {selected_track}")
    selected = [note for note in song.notes if selected_track is None or note.track == selected_track]
    if not selected:
        raise ValueError("所选音轨没有可适配的旋律音符")
    return selected, selected_track


def _output_bpm(song: SourceSong, options: AdaptOptions) -> int:
    bpm = song.bpm_hint if options.bpm is None else options.bpm
    # ``require_valid`` 接受数值；这里归一为稳定的 UI/数据库整数 BPM。
    return int(round(float(bpm)))


def _map_pitch(pitch: int, *, transpose: int, fold_octaves: bool) -> tuple[str, int, bool] | None:
    mapped = pitch + transpose
    folded = False
    if not MIDI_MIN <= mapped <= MIDI_MAX:
        if not fold_octaves:
            return None
        while mapped < MIDI_MIN:
            mapped += 12
        while mapped > MIDI_MAX:
            mapped -= 12
        folded = True
    octave = ("low", "mid", "high")[(mapped - MIDI_MIN) // 12]
    pc = mapped % 12
    if pc in _NATURAL_NUM:
        return f"{octave}_{_NATURAL_NUM[pc]}", 0, folded
    return f"{octave}_{_SHARP_BASE_NUM[pc]}", 1, folded


def _append_storage_item(output: list[dict], notes: list[str], dur_beats: float, semitone: int = 0) -> None:
    """按模型的 16 拍上限切分，不删除任何时间。"""

    remaining = float(dur_beats)
    while remaining > _EPSILON:
        piece = min(remaining, MAX_DUR_BEATS)
        item = {"notes": list(notes), "dur": piece}
        if notes and semitone:
            item["semitone"] = 1
        output.append(item)
        remaining -= piece


def _add_degradation(
    degradations: list[ImportDegradation], warnings: list[str], code: str, count: int, message: str
) -> None:
    if count <= 0:
        return
    degradation = ImportDegradation(code=code, count=count, message=message)
    degradations.append(degradation)
    warnings.append(message)


def adapt_source_song(song: SourceSong, options: AdaptOptions = AdaptOptions()) -> AdaptedScore:
    """把来源曲目适配为可直接入库的规范简谱。

    ``original`` 和 ``piano`` 都先全局单声部化；``piano`` 额外桥接 80ms 内的
    无声短缝。``preserve`` 会保留完全同起止、半音状态一致的和弦，其余无法表达的
    重叠也会显式记录为降级。整个结果不含任何播放或设备副作用。
    """

    validate_source_song(song)
    _resolve_options(options)
    bpm = _output_bpm(song, options)
    # 上游和 options 都已经保证范围；保留这一断言可防止未来改动绕过 Schema 限制。
    if not MIN_BPM <= bpm <= MAX_BPM:
        raise ValueError(f"输出 BPM 必须在 {MIN_BPM}-{MAX_BPM}")

    selected, selected_track = _selected_source_notes(song, options)
    source = list(selected)
    if options.style == "preserve":
        segments, polyphony_count = _preserve_segments(source)
        bridge_count = 0
    else:
        segments, polyphony_count = _selected_highest_segments(source)
        if options.style == "piano":
            mono_notes = [
                SourceNote(segment.start_s, segment.end_s, source[segment.members[0]].pitch, source[segment.members[0]].track)
                for segment in segments
            ]
            bridged, bridge_count = _bridge_short_gaps_with_count(mono_notes)
            # Each monophonic segment corresponds to one source index.  Rebind the bridged
            # intervals by position so mapping/degradation accounting remains deterministic.
            segments = [
                _Segment(note.start_s, note.end_s, segments[index].members)
                for index, note in enumerate(bridged)
            ]
        else:
            bridge_count = 0

    warnings: list[str] = []
    degradations: list[ImportDegradation] = []
    _add_degradation(
        degradations, warnings, "polyphony_cleanup", polyphony_count,
        f"已将 {polyphony_count} 个重叠声部片段简化为单旋律。",
    )
    _add_degradation(
        degradations, warnings, "short_gap_bridged", bridge_count,
        f"已连接 {bridge_count} 个不超过 {_DEFAULT_BRIDGE_GAP_S:g} 秒的演奏短缝。",
    )

    rendered: list[tuple[float, float, list[str], int]] = []
    folded_sources: set[int] = set()
    unsupported_sources: set[int] = set()
    mixed_chords = 0
    for segment in segments:
        mapped_members = []
        for index in segment.members:
            mapped = _map_pitch(
                source[index].pitch, transpose=options.transpose, fold_octaves=options.fold_octaves
            )
            if mapped is None:
                unsupported_sources.add(index)
                continue
            note_id, semitone, folded = mapped
            if folded:
                folded_sources.add(index)
            mapped_members.append((source[index].pitch + options.transpose, note_id, semitone))
        if not mapped_members:
            rendered.append((segment.start_s, segment.end_s, [], 0))
            continue
        # Schema 的 semitone 位于整条 chord，而不是单音：混合时不能无声写错，
        # 因此按要求只保留最高音并报告这一降级。
        semitone_states = {member[2] for member in mapped_members}
        if len(mapped_members) > 1 and len(semitone_states) > 1:
            highest = max(mapped_members, key=lambda member: member[0])
            rendered.append((segment.start_s, segment.end_s, [highest[1]], highest[2]))
            mixed_chords += 1
            continue
        mapped_members.sort(key=lambda member: member[0])
        note_ids = []
        for _, note_id, _ in mapped_members:
            if note_id not in note_ids:
                note_ids.append(note_id)
        rendered.append((segment.start_s, segment.end_s, note_ids, mapped_members[0][2]))

    _add_degradation(
        degradations, warnings, "octave_folded", len(folded_sources),
        f"已将 {len(folded_sources)} 个超出 C3-B5 范围的音按八度折回。",
    )
    _add_degradation(
        degradations, warnings, "unsupported_note", len(unsupported_sources),
        f"有 {len(unsupported_sources)} 个超出 C3-B5 范围的音已以等时值休止保留。",
    )
    _add_degradation(
        degradations, warnings, "mixed_semitone_chord", mixed_chords,
        f"有 {mixed_chords} 个混合自然音/半音和弦已保留最高音。",
    )
    if song.tempo_change_count > 1:
        _add_degradation(
            degradations, warnings, "tempo_flattened", song.tempo_change_count - 1,
            f"来源含 {song.tempo_change_count} 段速度变化，已按 {bpm} BPM 固定换算。",
        )

    storage: list[dict] = []
    split_count = 0
    cursor_s = 0.0
    beats_per_second = bpm / 60.0
    for start_s, end_s, note_ids, semitone in rendered:
        if start_s > cursor_s + _EPSILON:
            rest_beats = (start_s - cursor_s) * beats_per_second
            split_count += max(0, math.ceil(rest_beats / MAX_DUR_BEATS) - 1)
            _append_storage_item(storage, [], rest_beats)
        # ``rendered`` 来自切片，不会重叠；此处仍防御性截断以保证绝对时间单调。
        event_start = max(start_s, cursor_s)
        if end_s > event_start + _EPSILON:
            note_beats = (end_s - event_start) * beats_per_second
            split_count += max(0, math.ceil(note_beats / MAX_DUR_BEATS) - 1)
            _append_storage_item(storage, note_ids, note_beats, semitone)
            cursor_s = end_s
    if song.duration_s > cursor_s + _EPSILON:
        rest_beats = (song.duration_s - cursor_s) * beats_per_second
        split_count += max(0, math.ceil(rest_beats / MAX_DUR_BEATS) - 1)
        _append_storage_item(storage, [], rest_beats)

    _add_degradation(
        degradations, warnings, "duration_split", split_count,
        f"有 {split_count} 个超长时值片段已按 16 拍上限切分，时间轴保持不变。",
    )

    require_valid(storage, bpm=bpm)
    return AdaptedScore(
        name=song.title.strip() or "未命名乐曲",
        bpm=bpm,
        notes=storage,
        warnings=warnings,
        degradations=degradations,
        tracks=_track_names(song),
        selected_track=selected_track,
    )


__all__ = [
    "AdaptOptions",
    "AdaptedScore",
    "ImportDegradation",
    "SourceNote",
    "SourceSong",
    "adapt_source_song",
    "bridge_short_gaps",
    "monophonic_highest_active",
    "piano_melody",
    "recommend_track",
    "validate_source_song",
]
