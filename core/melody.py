"""MIDI 旋律提取：只在新起音上跟踪旋律，并给出全局移调建议。

和 ``monophonic_highest_active`` 的差别：

- 起音落在约 25–40ms 内的音并成一个和弦，只留一个旋律音，避免滚奏被切碎。
- 已经按住的伴奏不会填进旋律的空隙，长音在更高的音结束后也不会被重新触发。
- 移调在 -24..+24 里搜索，尽量少折八度、少黑键（21 键没有黑键）、少破坏走向。
- 八度折回按乐句整体移动，而不是每个音各自折。
"""

from __future__ import annotations

from dataclasses import dataclass

from core.source_score import MIDI_MAX, MIDI_MIN, SourceNote

_BLACK_PCS = {1, 3, 6, 8, 10}
_DEFAULT_WINDOW_S = 0.032
_PHRASE_GAP_S = 0.45


@dataclass(frozen=True)
class TransposeAdvice:
    """一次移调建议。调用方可以原样采用，也可以让用户改掉。"""

    semitones: int
    fold_count: int
    black_key_count: int
    contour_breaks: int
    score: float
    profile: str = "21key"


def cluster_onsets(notes, window_s: float = _DEFAULT_WINDOW_S) -> list[list[SourceNote]]:
    """把落在同一窗口内的起音收成和弦。窗口锚点是簇里最早的起音，避免音阶被粘成一串。"""

    window = float(window_s)
    if window < 0:
        raise ValueError("起音聚类窗口不能为负")
    ordered = sorted(notes, key=lambda note: (note.start_s, note.pitch, note.track))
    clusters: list[list[SourceNote]] = []
    for note in ordered:
        if clusters and note.start_s - clusters[-1][0].start_s <= window + 1e-9:
            clusters[-1].append(note)
        else:
            clusters.append([note])
    return clusters


def _pick(cluster: list[SourceNote], previous: int | None, *, cost: bool) -> SourceNote:
    if not cost or previous is None:
        return max(cluster, key=lambda note: (note.pitch, -note.start_s, -note.track))
    return min(
        cluster,
        key=lambda note: (abs(note.pitch - previous), -note.pitch, note.start_s, note.track),
    )


def extract_melody(notes, *, window_s: float = _DEFAULT_WINDOW_S, cost: bool = False) -> list[SourceNote]:
    """新起音旋律线。返回的音符互不重叠，时长被下一处起音截断。"""

    chosen: list[SourceNote] = []
    previous = None
    for cluster in cluster_onsets(notes, window_s):
        pick = _pick(cluster, previous, cost=cost)
        start = cluster[0].start_s
        end = pick.end_s
        if end <= start:
            continue
        if chosen and chosen[-1].end_s > start + 1e-9:
            prev = chosen[-1]
            if start <= prev.start_s + 1e-6:
                chosen.pop()
            else:
                chosen[-1] = SourceNote(prev.start_s, start, prev.pitch, prev.track)
        chosen.append(SourceNote(start, max(end, start + 1e-4), pick.pitch, pick.track))
        previous = pick.pitch
    return chosen


def quantize_notes(notes, *, bpm: float, grid_beats: float = 0.25, strength: float = 0.0):
    """把起音往网格上拉。strength=0 不动，1 完全吸附。时值长度保持不变。"""

    strength = float(strength)
    if strength <= 0 or not notes:
        return list(notes)
    if not 0 <= strength <= 1:
        raise ValueError("量化强度必须在 0 到 1 之间")
    grid = float(grid_beats) * 60.0 / float(bpm)
    if grid <= 0:
        raise ValueError("量化网格必须为正")
    quantized = []
    for note in notes:
        snapped = round(note.start_s / grid) * grid
        start = note.start_s + strength * (snapped - note.start_s)
        if start < 0:
            start = 0.0
        end = note.end_s + (start - note.start_s)
        if end <= start:
            end = start + 1e-4
        quantized.append(SourceNote(start, end, note.pitch, note.track))
    return quantized


def _split_phrases(notes, gap_s: float = _PHRASE_GAP_S) -> list[list[SourceNote]]:
    phrases: list[list[SourceNote]] = []
    for note in notes:
        if not phrases or note.start_s - phrases[-1][-1].end_s > gap_s:
            phrases.append([note])
        else:
            phrases[-1].append(note)
    return phrases


def phrase_fold(notes, *, gap_s: float = _PHRASE_GAP_S) -> tuple[list[SourceNote], int]:
    """每个乐句共用一个八度位移，句内音程保持不变。返回 (音符, 被移动的乐句数)。"""

    folded: list[SourceNote] = []
    moved = 0
    for phrase in _split_phrases(notes, gap_s):
        best_shift = 0
        best_key = None
        for octaves in range(-4, 5):
            shift = octaves * 12
            outside = sum(
                not MIDI_MIN <= note.pitch + shift <= MIDI_MAX for note in phrase
            )
            key = (outside, abs(shift), shift)
            if best_key is None or key < best_key:
                best_key = key
                best_shift = shift
        if best_shift:
            moved += 1
        for note in phrase:
            folded.append(SourceNote(
                note.start_s, note.end_s, note.pitch + best_shift, note.track,
            ))
    return folded, moved


def _contour_breaks(original, shifted) -> int:
    breaks = 0
    for left, right, new_left, new_right in zip(original, original[1:], shifted, shifted[1:]):
        old_step = right - left
        new_step = new_right - new_left
        if old_step == 0 or new_step == 0:
            continue
        if (old_step > 0) != (new_step > 0):
            breaks += 1
    return breaks


def _individual_fold(pitch: int) -> tuple[int, bool]:
    mapped = pitch
    changed = False
    while mapped < MIDI_MIN:
        mapped += 12
        changed = True
    while mapped > MIDI_MAX:
        mapped -= 12
        changed = True
    return mapped, changed


def _place_for_shift(pitches, shift: int, *, profile: str) -> tuple[int, int, int, float]:
    """按「每个音各自折八度」估计代价。真正导入时会改用乐句整体折回。"""

    final = []
    folds = 0
    for pitch in pitches:
        mapped, changed = _individual_fold(pitch + shift)
        folds += int(changed)
        final.append(mapped)
    blacks = sum((pitch % 12) in _BLACK_PCS for pitch in final)
    breaks = _contour_breaks(pitches, final)
    # 21 键没有黑键，会把黑键弹成相邻白键；三角洲可以用中键升半音。
    black_weight = 4.0 if profile == "21key" else 0.0
    score = folds * 5.0 + blacks * black_weight + breaks * 6.0 + abs(shift) * 0.02
    return folds, blacks, breaks, score


def recommend_transpose(notes, *, profile: str = "21key") -> TransposeAdvice:
    """在 -24..+24 里选一个让折八度、黑键和走向破坏都尽量少的移调。"""

    if profile not in {"21key", "delta"}:
        raise ValueError("键盘档案必须是 21key 或 delta")
    pitches = [note.pitch for note in notes]
    if not pitches:
        return TransposeAdvice(0, 0, 0, 0, 0.0, profile)
    best = None
    for shift in range(-24, 25):
        folds, blacks, breaks, score = _place_for_shift(pitches, shift, profile=profile)
        advice = TransposeAdvice(shift, folds, blacks, breaks, score, profile)
        if best is None or (advice.score, abs(advice.semitones), advice.semitones) < (
            best.score, abs(best.semitones), best.semitones,
        ):
            best = advice
    return best
