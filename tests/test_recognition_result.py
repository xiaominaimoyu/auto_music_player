"""识别结果协议、曲名兜底与本地试听合成测试。"""

import io
import unittest
import wave

from core.recognizer import (
    JIANPU_PROMPT,
    OpenAIStyleRecognizer,
    add_line_break_rests,
    fallback_score_name,
    build_recognition_text,
    get_recognizer_from_provider,
    parse_recognition_response,
)
from core.jianpu_editor import delete_event, insert_rest_at_event
from core.lyrics import align_lyrics
from core.score_preview import synthesize_preview


class TestRecognitionResult(unittest.TestCase):
    def test_prompt_requests_title_and_jianpu(self):
        self.assertIn("TITLE:", JIANPU_PROMPT)
        self.assertIn("JIANPU:", JIANPU_PROMPT)
        self.assertIn("UNKNOWN", JIANPU_PROMPT)
        self.assertIn("RHYTHM:", JIANPU_PROMPT)
        self.assertIn("LYRICS:", JIANPU_PROMPT)
        self.assertIn("禁止仅根据横向字距", JIANPU_PROMPT)
        self.assertIn("换行本身不要转换成休止", JIANPU_PROMPT)

    def test_structured_multiline_result(self):
        result = parse_recognition_response(
            "TITLE:\n小星星\nJIANPU:\n1 1 5 5\n6 6 5-"
        )
        self.assertEqual(result.title, "小星星")
        self.assertEqual(result.jianpu_text, "1 1 5 5\n6 6 5-")

    def test_inline_chinese_labels(self):
        result = parse_recognition_response("曲名：茉莉花\n简谱：3 3 5 6 1' 1'")
        self.assertEqual(result.title, "茉莉花")
        self.assertEqual(result.jianpu_text, "3 3 5 6 1' 1'")

    def test_unknown_title_becomes_empty(self):
        result = parse_recognition_response("TITLE: UNKNOWN\nJIANPU: 1 2 3")
        self.assertEqual(result.title, "")
        self.assertEqual(result.jianpu_text, "1 2 3")

    def test_structured_result_with_empty_jianpu_stays_empty(self):
        result = parse_recognition_response("TITLE: 导入曲\nJIANPU:\nLYRICS: UNKNOWN\nRHYTHM: OK")
        self.assertEqual(result.title, "导入曲")
        self.assertEqual(result.jianpu_text, "")

    def test_rhythm_note_is_separate_from_score(self):
        result = parse_recognition_response(
            "TITLE: 琵琶曲\nJIANPU:\n6 5 3\n3 2 1\nRHYTHM:\n原谱未标时值，停顿按分行推断"
        )
        self.assertEqual(result.jianpu_text, "6 5 3\n3 2 1")
        self.assertEqual(result.rhythm_notes, ("原谱未标时值，停顿按分行推断",))

    def test_lyrics_are_separate_and_unknown_is_empty(self):
        result = parse_recognition_response(
            "TITLE: 琵琶曲\nJIANPU: 6 5 3\nLYRICS: 人 间 琴\nRHYTHM: OK"
        )
        self.assertEqual(result.lyrics_lines, ("人 间 琴",))
        self.assertEqual(result.jianpu_text, "6 5 3")
        unknown = parse_recognition_response("TITLE: 小星星\nJIANPU: 1 1 5\nLYRICS: UNKNOWN")
        self.assertEqual(unknown.lyrics_lines, ())

    def test_add_line_break_rests(self):
        self.assertEqual(add_line_break_rests("6 5 3\n3 2 1"), "6 5 3 0_\n3 2 1")
        self.assertEqual(add_line_break_rests("1 2 3"), "1 2 3")

    def test_insert_rest_before_or_after_specific_event(self):
        source = "6 5 3 5 3\n6 5 6"
        self.assertEqual(
            insert_rest_at_event(source, 5, before=True),
            "6 5 3 5 3\n0_ 6 5 6",
        )
        self.assertEqual(
            insert_rest_at_event(source, 4, before=False),
            "6 5 3 5 3 0_\n6 5 6",
        )

    def test_delete_specific_event(self):
        self.assertEqual(delete_event("6 5 3\n2 1", 1), "6 3\n2 1")
        self.assertEqual(delete_event("6 5 3\n2 1", 2), "6 5\n2 1")

    def test_lyrics_align_by_line_without_shifting_mismatch(self):
        self.assertEqual(
            align_lyrics("6 5 0_ 3 5 3\n6 5 3", ("人 间 琴 悠 扬", "数 量 不 对")),
            ["人", "间", "", "琴", "悠", "扬", "", "", ""],
        )

    def test_saved_global_lyrics_restore_across_score_lines(self):
        self.assertEqual(
            align_lyrics("6 5 0_\n3 2 1", ("人 间 _ 琴 悠 扬",)),
            ["人", "间", "", "琴", "悠", "扬"],
        )

    def test_build_recognition_text_round_trip(self):
        raw = build_recognition_text("琵琶语", "6 5 0_\n3 2 1", ["人", "间", "", "琴", "悠", "扬"])
        result = parse_recognition_response(raw)
        self.assertEqual(result.title, "琵琶语")
        self.assertEqual(align_lyrics(result.jianpu_text, result.lyrics_lines), ["人", "间", "", "琴", "悠", "扬"])

    def test_legacy_plain_jianpu_compatible(self):
        result = parse_recognition_response("1 2 3_ 4_ 5-")
        self.assertEqual(result.title, "")
        self.assertEqual(result.jianpu_text, "1 2 3_ 4_ 5-")

    def test_fallback_name_is_stable_with_token(self):
        self.assertEqual(fallback_score_name("a1b2"), "未命名乐谱-A1B2")


class TestProviderSelection(unittest.TestCase):
    def test_missing_or_incomplete_provider_has_no_fake_recognizer(self):
        self.assertIsNone(get_recognizer_from_provider(None))
        self.assertIsNone(get_recognizer_from_provider({"api_key": "key"}))

    def test_complete_provider_creates_real_recognizer(self):
        recognizer = get_recognizer_from_provider({
            "base_url": "https://example.test/v1",
            "api_key": "key",
            "model": "vision-model",
        })
        self.assertIsInstance(recognizer, OpenAIStyleRecognizer)


class TestScorePreview(unittest.TestCase):
    def test_generates_valid_wav_with_expected_duration(self):
        data, duration = synthesize_preview([
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": [], "dur": 0.5},
            {"notes": ["mid_3", "mid_5"], "dur": 0.5},
        ], bpm=120, sample_rate=8000)
        self.assertAlmostEqual(duration, 1.0)
        with wave.open(io.BytesIO(data), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 8000)
            self.assertEqual(wav.getnframes(), 8000)

    def test_rejects_invalid_note(self):
        with self.assertRaises(ValueError):
            synthesize_preview([{"notes": ["bad_1"], "dur": 1.0}], bpm=100)

    def test_supports_slow_nonzero_bpm(self):
        data, duration = synthesize_preview([{"notes": ["mid_1"], "dur": 0.01}], bpm=1, sample_rate=100)
        self.assertTrue(data)
        self.assertAlmostEqual(duration, 0.6)

    def test_semitone_changes_preview_pitch(self):
        natural, natural_duration = synthesize_preview(
            [{"notes": ["mid_1"], "dur": 0.1}], bpm=120
        )
        sharp, sharp_duration = synthesize_preview(
            [{"notes": ["mid_1"], "dur": 0.1, "semitone": 1}], bpm=120
        )
        self.assertEqual(natural_duration, sharp_duration)
        self.assertNotEqual(natural, sharp)


if __name__ == "__main__":
    unittest.main()
