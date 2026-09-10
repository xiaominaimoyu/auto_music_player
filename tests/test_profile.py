"""core.profile 单元测试(档位分区、回落、校验、与编译器对接)。"""

import os
import tempfile
import unittest

from core.compiler import ChordPolicy, Modifier, ModifierPolicy, compile_score
from core.ir import IRNote
from core.profile import (
    DEFAULT_PROFILE_ID,
    GameProfile,
    ensure_profiles,
    grouped,
    legacy_profile,
    load_profiles,
    resolve_profile,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(REPO_ROOT, "profiles")

LEGACY_KEYMAP = {
    "high": list("QWERTYU"),
    "mid": list("ASDFGHJ"),
    "low": list("ZXCVBNM"),
}


class TestLoadRealProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_profiles(PROFILES_DIR, fallback_keymap=LEGACY_KEYMAP)
        cls.by_id = {p.id: p for p in cls.profiles}

    def test_both_profiles_present(self):
        self.assertIn("default", self.by_id)
        self.assertIn("delta_force_harmonica", self.by_id)

    def test_separate_groups(self):
        """需求:鸣潮/原神 与 三角洲行动 必须在不同分区。"""
        self.assertEqual(self.by_id["default"].group, "默认")
        self.assertEqual(self.by_id["delta_force_harmonica"].group, "三角洲行动")
        self.assertEqual(sorted(grouped(self.profiles)), ["三角洲行动", "默认"])

    def test_default_profile_is_21_direct_keys(self):
        p = self.by_id["default"]
        self.assertEqual(len(p.pitch_direct_overrides), 21)
        self.assertEqual(p.modifier_buttons, {}, "默认档位不应使用鼠标修饰键")
        self.assertEqual(p.legacy_keymap["high"], list("QWERTYU"))
        self.assertEqual(p.pitch_direct_overrides["high_1"], "Q")
        self.assertEqual(p.pitch_direct_overrides["low_7"], "M")

    def test_delta_profile_shape(self):
        p = self.by_id["delta_force_harmonica"]
        self.assertEqual(p.pitch_keys, tuple("ZXCVBNM"))
        self.assertEqual(p.modifier_buttons,
                         {"lower": "left", "semitone": "middle", "higher": "right"})
        self.assertIsNone(p.legacy_keymap, "三角洲档位没有 21 键旧模型")
        self.assertIs(p.modifier_policy, ModifierPolicy.OCTAVE_FIRST)
        self.assertIs(p.chord_policy, ChordPolicy.CHORD_FIRST)

    def test_delta_only_enables_high_1_override(self):
        """D3:仅逗号启用;. 与 / 证据不足,MVP 不放开。"""
        p = self.by_id["delta_force_harmonica"]
        self.assertEqual(p.pitch_direct_overrides, {"high_1": ","})


class TestResolveProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_profiles(PROFILES_DIR, fallback_keymap=LEGACY_KEYMAP)

    def test_by_id(self):
        self.assertEqual(resolve_profile(self.profiles, "delta_force_harmonica").id,
                         "delta_force_harmonica")

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(resolve_profile(self.profiles, "nope").id, DEFAULT_PROFILE_ID)

    def test_none_falls_back_to_default(self):
        self.assertEqual(resolve_profile(self.profiles, None).id, DEFAULT_PROFILE_ID)


class TestFallbackAndValidation(unittest.TestCase):
    def test_missing_dir_synthesizes_default(self):
        """profiles/ 缺失时必须回落到 config.yaml 的 keymap,老用户配置不失效(R8)。"""
        with tempfile.TemporaryDirectory() as tmp:
            profiles = load_profiles(os.path.join(tmp, "nope"),
                                     fallback_keymap=LEGACY_KEYMAP)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].id, DEFAULT_PROFILE_ID)
        self.assertEqual(profiles[0].pitch_direct_overrides["mid_5"], "G")

    def test_empty_dir_synthesizes_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = load_profiles(tmp, fallback_keymap=LEGACY_KEYMAP)
        self.assertEqual([p.id for p in profiles], [DEFAULT_PROFILE_ID])

    def test_dir_without_default_gets_one_inserted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "only.yaml"), "w", encoding="utf-8") as f:
                f.write("id: only\nname: Only\n")
            profiles = load_profiles(tmp, fallback_keymap=LEGACY_KEYMAP)
        self.assertEqual([p.id for p in profiles], [DEFAULT_PROFILE_ID, "only"])

    def test_duplicate_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fn in ("a.yaml", "b.yaml"):
                with open(os.path.join(tmp, fn), "w", encoding="utf-8") as f:
                    f.write("id: dup\nname: Dup\n")
            with self.assertRaises(ValueError):
                load_profiles(tmp)

    def test_missing_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "x.yaml"), "w", encoding="utf-8") as f:
                f.write("name: NoId\n")
            with self.assertRaises(ValueError):
                load_profiles(tmp)

    def test_unknown_policy_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "x.yaml"), "w", encoding="utf-8") as f:
                f.write("id: x\nname: X\nmodifier_policy: bogus\n")
            with self.assertRaises(ValueError):
                load_profiles(tmp)

    def test_invalid_override_id_raises(self):
        with self.assertRaises(ValueError):
            GameProfile(id="x", name="X", pitch_direct_overrides={"top_1": "Q"})

    def test_unknown_modifier_name_raises(self):
        with self.assertRaises(ValueError):
            GameProfile(id="x", name="X", modifier_buttons={"side": "left"})

    def test_bad_pitch_keys_length_raises(self):
        with self.assertRaises(ValueError):
            GameProfile(id="x", name="X", pitch_keys=("Z", "X"))


class TestBuildCompileParams(unittest.TestCase):
    def _params(self, profile):
        return profile.build_compile_params(
            bpm=100, settle_ms=30.0, release_settle_ms=20.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=0.0)

    def test_delta_maps_modifier_names_to_enum(self):
        p = load_profiles(PROFILES_DIR)[1]
        params = self._params(p)
        self.assertEqual(params.modifier_buttons[Modifier.LOWER], "left")
        self.assertEqual(params.modifier_buttons[Modifier.SEMITONE], "middle")
        self.assertEqual(params.modifier_buttons[Modifier.HIGHER], "right")
        self.assertEqual(params.pitch_direct_overrides, {"high_1": ","})

    def test_default_has_no_modifier_mapping(self):
        params = self._params(legacy_profile(LEGACY_KEYMAP))
        self.assertEqual(params.pitch_direct_overrides["high_1"], "Q")


class TestEndToEndWithProfiles(unittest.TestCase):
    """两个档位对同一份 IR 必须产出不同但各自正确的输入。"""

    def _params(self, profile):
        return profile.build_compile_params(
            bpm=100, settle_ms=30.0, release_settle_ms=20.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=0.0)

    def test_default_profile_uses_21_keys_no_mouse(self):
        profile = legacy_profile(LEGACY_KEYMAP)
        r = compile_score([IRNote(1, 1, 0, 0.01), IRNote(3, 0, 0, 0.01)],
                          self._params(profile))
        self.assertEqual([(e.device, e.key) for e in r.events],
                         [("kb", "Q"), ("kb", "Q"), ("kb", "D"), ("kb", "D")])
        self.assertEqual(r.degradations, [])

    def test_delta_profile_uses_modifier(self):
        profile = [p for p in load_profiles(PROFILES_DIR)
                   if p.id == "delta_force_harmonica"][0]
        r = compile_score([IRNote(3, 1, 0, 0.01)], self._params(profile))
        self.assertEqual([(e.device, e.key, e.action) for e in r.events], [
            ("mouse", "right", "down"),
            ("kb", "C", "down"),
            ("kb", "C", "up"),
            ("mouse", "right", "up"),
        ])

    def test_delta_high_1_uses_comma_override(self):
        profile = [p for p in load_profiles(PROFILES_DIR)
                   if p.id == "delta_force_harmonica"][0]
        r = compile_score([IRNote(1, 1, 0, 0.01)], self._params(profile))
        self.assertEqual([e.device for e in r.events], ["kb", "kb"])
        self.assertEqual(r.events[0].key, ",")


class TestEnsureProfiles(unittest.TestCase):
    def test_dev_mode_returns_path_without_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ensure_profiles(tmp), os.path.join(tmp, "profiles"))


if __name__ == "__main__":
    unittest.main()
