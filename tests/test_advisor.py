"""AI 编谱建议的输出拆分、容错和无网络调用测试。"""

import unittest

from core.advisor import ADVISOR_PROMPT, AdvisorResult, generate_advice, parse_advice_response

WELL_FORMED = """JIANPU:
1 2 3_ 3_ 5- 0 [1' 3' 5']-
TIPS:
节奏先慢后快
结尾落在高音上
"""


class TestParseAdvice(unittest.TestCase):
    def test_well_formed(self):
        result = parse_advice_response(WELL_FORMED)
        self.assertTrue(result.ok)
        self.assertEqual(result.jianpu_text, "1 2 3_ 3_ 5- 0 [1' 3' 5']-")
        self.assertEqual(result.tips, ["节奏先慢后快", "结尾落在高音上"])
        self.assertEqual(result.warnings, [])

    def test_missing_format_fallback(self):
        result = parse_advice_response("一段旋律:\n1 2 3 5, 6 5' 3 1")
        self.assertTrue(result.ok)
        self.assertTrue(any("未按" in warning for warning in result.warnings))

    def test_invalid_tokens_kept_with_warning(self):
        result = parse_advice_response("JIANPU:\n1 8 @ 2\nTIPS:\n建议")
        self.assertEqual(result.jianpu_text, "1 8 @ 2")
        self.assertTrue(result.ok)
        self.assertTrue(any("无法识别" in warning for warning in result.warnings))

    def test_empty_output(self):
        result = parse_advice_response("抱歉,我无法完成该请求。")
        self.assertFalse(result.ok)
        self.assertTrue(any("未能" in warning for warning in result.warnings))


class TestGenerateAdvice(unittest.TestCase):
    def test_uses_injected_chat_and_prompt(self):
        captured = {}

        def fake_chat(api_base, api_key, model, prompt, timeout):
            captured.update(base=api_base, key=api_key, model=model, prompt=prompt)
            return WELL_FORMED

        result = generate_advice(
            "一段欢快旋律",
            "https://api.example.com/v1",
            "sk-test",
            "text-model",
            chat_fn=fake_chat,
        )
        self.assertTrue(result.ok)
        self.assertEqual(captured["model"], "text-model")
        self.assertIn("一段欢快旋律", captured["prompt"])
        self.assertNotIn("{desc}", captured["prompt"])

    def test_empty_desc_rejected_before_network(self):
        with self.assertRaises(ValueError):
            generate_advice("   ", "b", "k", "m", chat_fn=lambda *_args: WELL_FORMED)

    def test_default_result_not_ok(self):
        self.assertFalse(AdvisorResult().ok)
        self.assertIn("{desc}", ADVISOR_PROMPT)


if __name__ == "__main__":
    unittest.main()
