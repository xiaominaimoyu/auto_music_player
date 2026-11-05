"""使用 Windows General MIDI 试听原谱或游戏编排，仅向音源发送消息。"""

import ctypes
import io
import math
import struct
import sys
import threading
import time
import wave
from ctypes import wintypes

from PyQt6.QtCore import QObject, pyqtSignal

from core.score_io import note_id_to_midi
from core.timing import wait_until


PREVIEW_GAP_MS = 0
MIDI_ACOUSTIC_GRAND_PIANO = 0
PREVIEW_INSTRUMENTS = {
    "大钢琴": MIDI_ACOUSTIC_GRAND_PIANO,
    "竖琴": 46,
    "口琴": 22,
    "筝类近似音色（Koto）": 107,
}
MIDI_PREVIEW_VELOCITY = 92
_SAMPLE_RATE = 22050
_MIDI_MAPPER = 0xFFFFFFFF


def note_midi_number(note_id: str, semitone: int = 0) -> int:
    """把存储格式的音符 ID 转为 MIDI 音高编号。"""
    try:
        semitone = int(semitone or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"半音标记必须是 0 或 1: {semitone!r}") from exc
    if semitone not in (0, 1):
        raise ValueError(f"半音标记必须是 0 或 1: {semitone!r}")
    try:
        return note_id_to_midi(str(note_id).strip()) + semitone
    except ValueError as exc:
        raise ValueError(f"无法试听未知音符: {note_id!r}") from exc


def note_frequency(note_id: str, semitone: int = 0) -> float:
    """把存储格式的音符 ID 转为平均律频率。

    存储中的中音 1 以 C4 为基准;低/高音分别相差一个八度。
    """
    midi = note_midi_number(note_id, semitone)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def note_frequencies(note: dict) -> list[float]:
    """返回一个谱面事件需要同时发出的频率;空列表表示休止符。"""
    return [440.0 * 2.0 ** ((midi - 69) / 12.0) for midi in note_midi_numbers(note)]


def note_midi_numbers(note: dict) -> list[int]:
    """返回一个谱面事件需要同时发出的 MIDI 音高;空列表表示休止符。"""
    note_ids = note.get("notes") or []
    offsets = note.get("semitones")
    if offsets is None:
        offsets = [note.get("semitone", 0)] * len(note_ids)
    elif not isinstance(offsets, list) or len(offsets) != len(note_ids):
        raise ValueError("和弦的 semitones 必须与 notes 等长")
    return [note_midi_number(note_id, offset) for note_id, offset in zip(note_ids, offsets)]


def preview_duration_ms(note: dict, bpm: float) -> int:
    """按谱面时值和 BPM 计算试听事件时长。"""
    return max(1, int(round(float(note["dur"]) * 60000.0 / max(1.0, float(bpm)))))


def render_wav(frequencies: list[float], duration_ms: int) -> bytes:
    """生成单声道 16-bit PCM WAV；保留给兼容调用，试听播放不再使用。"""
    sample_count = max(1, int(round(_SAMPLE_RATE * max(1, duration_ms) / 1000.0)))
    channels = max(1, len(frequencies))
    frames = bytearray(sample_count * 2)
    attack = max(1, int(_SAMPLE_RATE * 0.005))
    release = max(1, int(_SAMPLE_RATE * 0.008))

    for index in range(sample_count):
        envelope = min(1.0, index / attack, (sample_count - index) / release)
        if frequencies:
            sample = sum(
                math.sin(2.0 * math.pi * frequency * index / _SAMPLE_RATE)
                for frequency in frequencies
            ) / channels
            sample *= 0.32 * envelope
        else:
            sample = 0.0
        sample = max(-1.0, min(1.0, sample))
        struct.pack_into("<h", frames, index * 2, int(sample * 32767))

    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(frames)
    return stream.getvalue()


def _midi_message(status: int, data1: int = 0, data2: int = 0) -> int:
    """把 MIDI 短消息的三个字节打包为 WinMM DWORD。"""
    return (status & 0xFF) | ((data1 & 0x7F) << 8) | ((data2 & 0x7F) << 16)


class _WindowsMidiOutput:
    """WinMM MIDI 输出封装，默认使用 General MIDI 标准大钢琴。"""

    def __init__(self, program=MIDI_ACOUSTIC_GRAND_PIANO):
        self._lock = threading.RLock()
        self._closed = True
        self._dll = ctypes.WinDLL("winmm")
        self._configure_api()
        self._handle = wintypes.HANDLE()
        result = self._midi_out_open(
            ctypes.byref(self._handle), _MIDI_MAPPER, 0, 0, 0
        )
        if result != 0:
            raise RuntimeError(f"无法打开 Windows MIDI 音源（错误码 {result}）")
        self._closed = False
        try:
            self._send(_midi_message(0xC0, int(program)))
        except Exception:
            self.close()
            raise

    def _configure_api(self):
        self._midi_out_open = self._dll.midiOutOpen
        self._midi_out_open.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        self._midi_out_open.restype = wintypes.UINT

        self._midi_out_short_msg = self._dll.midiOutShortMsg
        self._midi_out_short_msg.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._midi_out_short_msg.restype = wintypes.UINT

        self._midi_out_reset = self._dll.midiOutReset
        self._midi_out_reset.argtypes = [wintypes.HANDLE]
        self._midi_out_reset.restype = wintypes.UINT

        self._midi_out_close = self._dll.midiOutClose
        self._midi_out_close.argtypes = [wintypes.HANDLE]
        self._midi_out_close.restype = wintypes.UINT

    def _send(self, message: int):
        with self._lock:
            if self._closed:
                raise RuntimeError("Windows MIDI 音源已经关闭")
            result = self._midi_out_short_msg(self._handle, message)
            if result != 0:
                raise RuntimeError(f"Windows MIDI 音符发送失败（错误码 {result}）")

    def note_on(self, midi_note: int, velocity=MIDI_PREVIEW_VELOCITY):
        self._send(_midi_message(0x90, midi_note, velocity))

    def note_off(self, midi_note: int):
        self._send(_midi_message(0x80, midi_note, 0))

    def reset(self):
        """立即停止当前通道上的发声，供用户点击停止时调用。"""
        with self._lock:
            if not self._closed:
                self._midi_out_reset(self._handle)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._midi_out_reset(self._handle)
            self._midi_out_close(self._handle)
            self._closed = True


class PreviewPlayer(QObject):
    """在后台线程播放试听音频,并向 GUI 报告进度。"""

    started = pyqtSignal(int)  # 总事件数
    progress = pyqtSignal(int, int)  # 已完成事件数, 总事件数
    finished = pyqtSignal(bool)  # 自然结束 / 用户停止
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._midi_output = None

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, notes, bpm=100, gap_ms=PREVIEW_GAP_MS, score_name="", *,
             program=MIDI_ACOUSTIC_GRAND_PIANO) -> bool:
        """试听原谱。默认按谱面起音，显式 gap_ms 保留旧接口的额外间隔。"""
        beat_ms = 60000.0 / max(1.0, float(bpm))
        gap_ms = max(0.0, float(gap_ms))
        timeline = []
        start_ms = 0.0
        count = 0
        for count, note in enumerate(notes, start=1):
            end_ms = start_ms + max(1.0, float(note["dur"]) * beat_ms)
            pitches = note_midi_numbers(note)
            timeline.extend((start_ms, pitch, True, 0) for pitch in pitches)
            timeline.extend((end_ms, pitch, False, 0) for pitch in pitches)
            timeline.append((end_ms, None, False, count))
            start_ms = end_ms + gap_ms
        if not count:
            raise ValueError("没有可试听的音符")
        return self._start(timeline, count, program)

    def play_events(self, events, *, program=MIDI_ACOUSTIC_GRAND_PIANO, duration_ms=0) -> bool:
        """播放 (绝对毫秒, MIDI 音高, 是否起音) 序列，duration_ms 包含尾部休止。"""
        duration_ms = float(duration_ms)
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("试听时长必须为非负有限数")
        ordered = []
        for at_ms, pitch, is_down in events:
            at_ms = float(at_ms)
            if not math.isfinite(at_ms) or at_ms < 0:
                raise ValueError("试听事件时间必须为非负有限数")
            if not isinstance(pitch, int) or not 0 <= pitch <= 127:
                raise ValueError("MIDI 音高必须是 0 至 127 的整数")
            ordered.append((at_ms, pitch, is_down))
        ordered.sort(key=lambda event: event[0])
        timeline = [(at, pitch, down, index) for index, (at, pitch, down)
                    in enumerate(ordered, start=1)]
        if not timeline or timeline[-1][0] < duration_ms:
            timeline.append((duration_ms, None, False, 0))
        return self._start(timeline, len(ordered), program)

    def _start(self, timeline, count, program):
        if not isinstance(program, int) or not 0 <= program <= 127:
            raise ValueError("MIDI 音色必须是 0 至 127 的整数")
        with self._lock:
            if self.is_playing:
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(timeline, count, program),
                daemon=True,
            )
            self._thread.start()
        self.started.emit(count)
        return True

    def stop(self):
        """停止试听并立即关闭仍在发声的 MIDI 音符。"""
        self._stop_event.set()
        with self._lock:
            midi_output = self._midi_output
        if midi_output is not None:
            midi_output.reset()

    def shutdown(self, join_timeout=1.0):
        self.stop()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(join_timeout)

    def _create_midi_output(self, program):
        return _WindowsMidiOutput(program)

    def _run(self, timeline, count, program):
        normal = False
        midi_output = None
        try:
            if sys.platform != "win32":
                raise RuntimeError("试听目前仅支持 Windows 系统 MIDI 音源")
            if self._stop_event.is_set():
                return
            midi_output = self._create_midi_output(program)
            with self._lock:
                self._midi_output = midi_output
            origin = time.perf_counter()
            for at_ms, pitch, is_down, progress in timeline:
                if wait_until(self._stop_event, origin + at_ms / 1000.0):
                    break
                if pitch is not None:
                    if is_down:
                        midi_output.note_on(pitch)
                    else:
                        midi_output.note_off(pitch)
                if progress:
                    self.progress.emit(progress, count)
            normal = not self._stop_event.is_set()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if midi_output is not None:
                midi_output.reset()
                midi_output.close()
            with self._lock:
                if self._midi_output is midi_output:
                    self._midi_output = None
            self.finished.emit(normal)
