"""本地试听器测试:钢琴 MIDI 映射、时值与兼容 WAV,不触发真实扬声器。"""

import io
import sys
import wave

import pytest
from PyQt6.QtCore import Qt

from core.preview_player import (
    MIDI_ACOUSTIC_GRAND_PIANO,
    PREVIEW_INSTRUMENTS,
    _midi_message,
    note_frequencies,
    note_frequency,
    note_midi_number,
    note_midi_numbers,
    preview_duration_ms,
    render_wav,
    PreviewPlayer,
)


class FakeMidiOutput:
    def __init__(self):
        self.events = []

    def note_on(self, midi_note):
        self.events.append(("on", midi_note))

    def note_off(self, midi_note):
        self.events.append(("off", midi_note))

    def reset(self):
        self.events.append(("reset",))

    def close(self):
        self.events.append(("close",))


def test_note_frequency_uses_c4_as_middle_one():
    assert note_frequency("mid_1") == pytest.approx(261.6256, rel=1e-4)
    assert note_frequency("high_1") == pytest.approx(523.2511, rel=1e-4)
    assert note_frequency("low_1") == pytest.approx(130.8128, rel=1e-4)


def test_semitone_and_chord_frequencies():
    assert note_frequency("mid_1", 1) == pytest.approx(277.1826, rel=1e-4)
    frequencies = note_frequencies(
        {"notes": ["mid_1", "mid_3", "mid_5"], "dur": 1.0}
    )
    assert len(frequencies) == 3
    assert frequencies[0] < frequencies[1] < frequencies[2]


def test_numbered_notation_uses_natural_major_scale_midi_notes():
    assert [note_midi_number(f"mid_{number}") for number in range(1, 8)] == [
        60, 62, 64, 65, 67, 69, 71
    ]
    assert note_midi_number("mid_1", 1) == 61
    assert note_midi_numbers(
        {"notes": ["mid_1", "mid_3", "mid_5"], "dur": 1.0}
    ) == [60, 64, 67]
    assert note_midi_numbers({"notes": ["mid_1", "mid_3"], "semitones": [1, 0]}) == [61, 64]
    with pytest.raises(ValueError, match="等长"):
        note_midi_numbers({"notes": ["mid_1"], "semitones": []})


def test_general_midi_program_is_acoustic_grand_piano():
    assert MIDI_ACOUSTIC_GRAND_PIANO == 0
    assert _midi_message(0xC0, MIDI_ACOUSTIC_GRAND_PIANO) == 0xC0
    assert _midi_message(0x90, 60, 92) == 0x5C3C90


def test_rest_has_no_frequency():
    assert note_frequencies({"notes": [], "dur": 1.0}) == []


def test_invalid_note_is_rejected():
    with pytest.raises(ValueError, match="未知音符"):
        note_frequency("mid_8")


def test_preview_duration_uses_bpm_and_never_returns_zero():
    assert preview_duration_ms({"dur": 1.0}, 120) == 500
    assert preview_duration_ms({"dur": 0.0}, 120) == 1


def test_render_wav_returns_playable_pcm_wave():
    audio = render_wav([note_frequency("mid_1")], 40)
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() > 0


def test_preview_player_sends_midi_note_events_without_real_audio(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    player = PreviewPlayer()
    midi = FakeMidiOutput()
    monkeypatch.setattr(player, "_create_midi_output", lambda program: midi)
    finished = []
    player.finished.connect(finished.append, Qt.ConnectionType.DirectConnection)

    assert player.play(
        [{"notes": ["mid_1", "mid_3", "mid_5"], "dur": 0.0}],
        120,
        0.0,
        "",
    )
    player._thread.join(1)

    assert finished == [True]
    assert midi.events == [
        ("on", 60),
        ("on", 64),
        ("on", 67),
        ("off", 60),
        ("off", 64),
        ("off", 67),
        ("reset",),
        ("close",),
    ]


def test_preview_player_emits_finished_for_stop_without_audio(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    player = PreviewPlayer()
    player._create_midi_output = lambda program: pytest.fail("已停止时不应打开 MIDI 音源")
    player._stop_event.set()
    finished = []
    player.finished.connect(finished.append)

    player._run([(0.0, 60, True, 1)], 1, 0)

    assert finished == [False]


def test_stop_resets_active_midi_notes():
    player = PreviewPlayer()
    midi = FakeMidiOutput()
    player._midi_output = midi

    player.stop()

    assert player._stop_event.is_set()
    assert midi.events == [("reset",)]


@pytest.mark.parametrize("requested_gap", [None, 30.0])
def test_original_score_preserves_onsets_and_explicit_gap(monkeypatch, requested_gap):
    player = PreviewPlayer()
    timeline = []
    def capture(schedule, count, program):
        timeline.extend(schedule)
        return True
    monkeypatch.setattr(player, "_start", capture)
    notes = [{"notes": ["mid_1"], "dur": 0.5}, {"notes": [], "dur": 0.25},
             {"notes": ["mid_2"], "dur": 0.25}]
    options = {} if requested_gap is None else {"gap_ms": requested_gap}
    assert player.play(notes, bpm=120, **options)
    starts = [at for at, pitch, down, _ in timeline if pitch is not None and down]
    assert starts == [0.0, 375.0 + 2 * (requested_gap or 0)]


@pytest.mark.parametrize("program", PREVIEW_INSTRUMENTS.values())
def test_absolute_midi_timeline_uses_selected_program_and_trailing_rest(monkeypatch, program):
    monkeypatch.setattr(sys, "platform", "win32")
    player = PreviewPlayer()
    midi = FakeMidiOutput()
    opened = []
    def open_midi(selected):
        opened.append(selected)
        return midi
    monkeypatch.setattr(player, "_create_midi_output", open_midi)
    deadlines = []
    def record_deadline(stop_event, deadline):
        deadlines.append(deadline)
        return False
    monkeypatch.setattr("core.preview_player.wait_until", record_deadline)

    assert player.play_events([(20, 60, False), (0, 60, True), (0, 64, True),
                               (20, 64, False)], program=program, duration_ms=60)
    player._thread.join(1)
    assert opened == [program]
    assert midi.events == [("on", 60), ("on", 64), ("off", 60), ("off", 64),
                           ("reset",), ("close",)]
    assert [time - deadlines[0] for time in deadlines] == pytest.approx([0, 0, .020, .020, .060])
