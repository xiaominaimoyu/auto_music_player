"""把结构化简谱合成为口琴近似音色的 PCM WAV，不经过游戏输入通道。"""

import io
import math
import struct
import wave

from core.score_model import require_valid

SAMPLE_RATE = 16000
MAX_PREVIEW_SECONDS = 300.0
_MAJOR_OFFSETS = (0, 2, 4, 5, 7, 9, 11)
_OCTAVE_SHIFT = {"low": -12, "mid": 0, "high": 12}


def _frequency(note_id: str, semitone: int = 0) -> float:
    octave, number = note_id.split("_", 1)
    midi = 60 + _OCTAVE_SHIFT[octave] + _MAJOR_OFFSETS[int(number) - 1] + int(semitone)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _harmonica_voice(freq: float, frame: int, frames: int, sample_rate: int) -> float:
    time_s = frame / sample_rate
    angle = 2.0 * math.pi * freq * time_s
    attack = min(1.0, frame / max(1, sample_rate * 0.018))
    tail = min(1.0, max(0.0, (frames - frame) / (sample_rate * 0.035)))
    wave_value = math.sin(angle) + 0.31 * math.sin(3 * angle) + 0.13 * math.sin(5 * angle)
    return wave_value / 1.44 * attack * tail


def synthesize_preview(notes: list, bpm: int, *, sample_rate: int = SAMPLE_RATE) -> tuple[bytes, float]:
    """返回 ``(wav_bytes, duration_seconds)``，支持休止与简单和弦。"""
    require_valid(notes, bpm=bpm)
    seconds_per_beat = 60.0 / float(bpm)
    duration = sum(float(item["dur"]) * seconds_per_beat for item in notes)
    if duration > MAX_PREVIEW_SECONDS:
        raise ValueError(f"试听时长不能超过 {MAX_PREVIEW_SECONDS / 60:g} 分钟")

    pcm = bytearray()
    amplitude = 9800
    for item in notes:
        frame_count = max(1, round(float(item["dur"]) * seconds_per_beat * sample_rate))
        semitone = int(item.get("semitone", 0))
        freqs = [_frequency(note_id, semitone) for note_id in (item.get("notes") or [])]
        for frame in range(frame_count):
            sample = sum(_harmonica_voice(freq, frame, frame_count, sample_rate) for freq in freqs)
            if freqs:
                sample /= len(freqs)
            pcm.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * amplitude)))

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue(), duration
