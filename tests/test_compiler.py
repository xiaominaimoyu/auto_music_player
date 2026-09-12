"""core.compiler 单元测试(含 D2 冲突策略全覆盖)。"""

import unittest

from core.compiler import (
    ChordPolicy,
    CompileParams,
    Degradation,
    Modifier,
    ModifierPolicy,
    compile_score,
    resolve_chord,
    resolve_modifier,
)
from core.ir import IRChord, IRNote, IRRest

FAST = CompileParams(bpm=100, settle_ms=30.0, release_settle_ms=20.0,
                     hold_ratio=1.0, max_hold_ms=None, gap_ms=0.0)


def sig(events):
    return [(e.device, e.key, e.action, round(e.t_ms, 3)) for e in events]


class TestResolveModifierNoConflict(unittest.TestCase):
    def test_natural(self):
        mod, deg = resolve_modifier(0, 0)
        self.assertIs(mod, Modifier.NATURAL)
        self.assertIsNone(deg)

    def test_lower(self):
        mod, deg = resolve_modifier(-1, 0)
        self.assertIs(mod, Modifier.LOWER)
        self.assertIsNone(deg)

    def test_higher(self):
        mod, deg = resolve_modifier(1, 0)
        self.assertIs(mod, Modifier.HIGHER)
        self.assertIsNone(deg)

    def test_semitone_alone_is_not_a_conflict(self):
        """自然音 + 半音是合法组合,不算冲突。"""
        mod, deg = resolve_modifier(0, 1)
        self.assertIs(mod, Modifier.SEMITONE)
        self.assertIsNone(deg)


class TestResolveModifierConflict(unittest.TestCase):
    """D2:两种冲突 × 三种策略 = 6 组,全覆盖。"""

    def test_octave_first_keeps_octave(self):
        mod, deg = resolve_modifier(-1, 1, ModifierPolicy.OCTAVE_FIRST)
        self.assertIs(mod, Modifier.LOWER)
        self.assertEqual((deg.requested, deg.actual, deg.reason),
                         ("low#", "low", "octave_first"))

        mod, deg = resolve_modifier(1, 1, ModifierPolicy.OCTAVE_FIRST)
        self.assertIs(mod, Modifier.HIGHER)
        self.assertEqual((deg.requested, deg.actual), ("high#", "high"))

    def test_keep_accidental_keeps_semitone(self):
        for octave in (-1, 1):
            mod, deg = resolve_modifier(octave, 1, ModifierPolicy.KEEP_ACCIDENTAL)
            self.assertIs(mod, Modifier.SEMITONE)
            self.assertEqual(deg.actual, "semitone")

    def test_reject_note_drops(self):
        for octave in (-1, 1):
            mod, deg = resolve_modifier(octave, 1, ModifierPolicy.REJECT_NOTE)
            self.assertIsNone(mod)
            self.assertEqual(deg.actual, "skipped")

    def test_default_policy_is_octave_first(self):
        mod, _ = resolve_modifier(1, 1)
        self.assertIs(mod, Modifier.HIGHER)


class TestResolveChord(unittest.TestCase):
    def _chord(self):
        return IRChord((IRNote(1, 0, 0), IRNote(3, 0, 0), IRNote(5, 0, 0)), 2.0)

    def test_first(self):
        notes, deg = resolve_chord(self._chord(), ChordPolicy.CHORD_FIRST)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].note_id, "mid_1")
        self.assertEqual(deg.reason, "first")

    def test_reject(self):
        notes, deg = resolve_chord(self._chord(), ChordPolicy.CHORD_REJECT)
        self.assertEqual(notes, [])
        self.assertEqual(deg.actual, "skipped")

    def test_arpeggiate_splits_duration(self):
        notes, deg = resolve_chord(self._chord(), ChordPolicy.CHORD_ARPEGGIATE)
        self.assertEqual(len(notes), 3)
        self.assertEqual([n.pitch for n in notes], [1, 3, 5])
        self.assertAlmostEqual(notes[0].dur, 2.0 / 3)
        self.assertEqual(deg.actual, "arpeggio×3")


class TestCompileEventOrder(unittest.TestCase):
    def test_natural_note_has_no_mouse_events(self):
        r = compile_score([IRNote(1, 0, 0, 0.01)], FAST)
        self.assertEqual(sig(r.events), [("kb", "Z", "down", 0.0),
                                         ("kb", "Z", "up", 6.0)])

    def test_high_note_order(self):
        """顺序硬约束:mouse down → settle → key down → key up → mouse up。"""
        r = compile_score([IRNote(1, 1, 0, 0.01)], FAST)
        self.assertEqual(sig(r.events), [
            ("mouse", "right", "down", 0.0),
            ("kb", "Z", "down", 30.0),
            ("kb", "Z", "up", 36.0),
            ("mouse", "right", "up", 56.0),
        ])

    def test_low_note_uses_left(self):
        r = compile_score([IRNote(5, -1, 0, 0.01)], FAST)
        self.assertEqual(sig(r.events), [
            ("mouse", "left", "down", 0.0),
            ("kb", "B", "down", 30.0),
            ("kb", "B", "up", 36.0),
            ("mouse", "left", "up", 56.0),
        ])

    def test_semitone_uses_middle(self):
        r = compile_score([IRNote(3, 0, 1, 0.01)], FAST)
        self.assertEqual([e.key for e in r.events if e.device == "mouse"],
                         ["middle", "middle"])

    def test_pitch_key_mapping(self):
        for pitch, key in enumerate("ZXCVBNM", start=1):
            r = compile_score([IRNote(pitch, 0, 0, 0.01)], FAST)
            self.assertEqual(r.events[0].key, key)


class TestCompileTiming(unittest.TestCase):
    def test_humanize_jitter_does_not_accumulate_between_beats(self):
        """每拍 +10ms 只偏移自身，不能变成 +10/+20/+30… 的长曲漂移。"""
        params = CompileParams(
            bpm=100,
            settle_ms=0.0,
            release_settle_ms=0.0,
            hold_ratio=0.5,
            max_hold_ms=None,
            gap_ms=0.0,
        )
        result = compile_score(
            [IRNote(1, 0, 0, 1.0) for _ in range(5)],
            params,
            timings=[(10.0, 0.5)] * 5,
        )
        downs = [e.t_ms for e in result.events if e.device == "kb" and e.action == "down"]
        self.assertEqual(downs, [10.0, 610.0, 1210.0, 1810.0, 2410.0])

    def test_source_markers_end_on_last_event_of_logical_note(self):
        result = compile_score(
            [IRNote(1, 1, 0, 0.01), IRRest(1.0), IRNote(2, 0, 0, 0.01)],
            FAST,
            source_index_offset=4,
        )
        ends = [(e.source_index, e.key) for e in result.events if e.source_end]
        self.assertEqual(ends, [(4, "Z"), (6, "X")])

    def test_same_modifier_held_across_notes(self):
        """相邻同修饰态不得 release→press 于同一时刻,否则游戏可能漏掉重新按下。"""
        notes = [IRNote(1, 1, 0, 0.01), IRNote(2, 1, 0, 0.01)]
        r = compile_score(notes, FAST)
        self.assertEqual(sig(r.events), [
            ("mouse", "right", "down", 0.0),
            ("kb", "Z", "down", 30.0),
            ("kb", "Z", "up", 36.0),
            ("kb", "X", "down", 36.0),
            ("kb", "X", "up", 42.0),
            ("mouse", "right", "up", 62.0),   # 只在末尾释放一次
        ])

    def test_modifier_switch_releases_then_presses(self):
        """切换修饰键:先松旧的,留足 release_settle 再按新的。"""
        notes = [IRNote(1, 1, 0, 0.01), IRNote(2, -1, 0, 0.01)]
        r = compile_score(notes, FAST)
        self.assertEqual(sig(r.events)[:5], [
            ("mouse", "right", "down", 0.0),
            ("kb", "Z", "down", 30.0),
            ("kb", "Z", "up", 36.0),
            ("mouse", "right", "up", 56.0),
            ("mouse", "left", "down", 56.0),
        ])

    def test_rest_releases_held_modifier(self):
        """休止前必须松开修饰键,避免长时间按住(右键在 FPS 中有独立语义)。"""
        r = compile_score([IRNote(1, 1, 0, 0.01), IRRest(1.0)], FAST)
        ups = [e for e in r.events if e.device == "mouse" and e.action == "up"]
        self.assertEqual(len(ups), 1)
        self.assertAlmostEqual(ups[0].t_ms, 56.0)

    def test_repeated_same_note_respects_gap(self):
        """同音重复:必须留出 gap,否则 up/down 同刻会漏触发(调研「不要多点」)。"""
        p = CompileParams(bpm=100, settle_ms=0.0, release_settle_ms=0.0,
                          gap_ms=20.0, max_hold_ms=None)
        r = compile_score([IRNote(1, 0, 0, 0.01), IRNote(1, 0, 0, 0.01)], p)
        downs = [e.t_ms for e in r.events if e.action == "down"]
        self.assertGreaterEqual(downs[1] - downs[0], 20.0)

    def test_hold_ratio(self):
        p = CompileParams(bpm=100, hold_ratio=0.5, max_hold_ms=None, settle_ms=0.0,
                          release_settle_ms=0.0, gap_ms=0.0)
        r = compile_score([IRNote(1, 0, 0, 1.0)], p)
        self.assertAlmostEqual(r.events[1].t_ms, 300.0)

    def test_max_hold_truncates_and_records(self):
        p = CompileParams(bpm=100, hold_ratio=1.0, max_hold_ms=100.0,
                          settle_ms=0.0, release_settle_ms=0.0, gap_ms=0.0)
        r = compile_score([IRNote(1, 0, 0, 1.0)], p)
        self.assertAlmostEqual(r.events[1].t_ms, 100.0)
        self.assertTrue(any(d.reason == "MAX_HOLD" for d in r.degradations))

    def test_rest_advances_time(self):
        r = compile_score([IRRest(1.0), IRNote(1, 0, 0, 0.01)], FAST)
        self.assertAlmostEqual(r.events[0].t_ms, 600.0)

    def test_duration_ms(self):
        r = compile_score([IRNote(1, 1, 0, 0.01)], FAST)
        self.assertAlmostEqual(r.duration_ms, 56.0)


class TestCompileDegradation(unittest.TestCase):
    def test_chord_degradation_index(self):
        r = compile_score(
            [IRNote(1, 0, 0, 0.5),
             IRChord((IRNote(1, 0, 0), IRNote(3, 0, 0)), 1.0)],
            FAST)
        self.assertEqual(len(r.degradations), 1)
        self.assertEqual(r.degradations[0].index, 1)
        self.assertEqual(r.degradations[0].reason, "first")

    def test_rejected_modifier_counts_as_skipped(self):
        p = CompileParams(modifier_policy=ModifierPolicy.REJECT_NOTE)
        r = compile_score([IRNote(1, 1, 1, 0.5)], p)
        self.assertEqual(r.skipped, 1)
        self.assertEqual(r.note_count, 0)
        self.assertEqual(r.events, [])

    def test_rejected_modifier_after_held_modifier_keeps_timeline(self):
        """被拒绝音降级为休止时仍携带源索引，不能破坏前一修饰音的收尾。"""
        p = CompileParams(modifier_policy=ModifierPolicy.REJECT_NOTE)
        r = compile_score([
            IRNote(1, -1, 0, 0.25),
            IRNote(2, 1, 1, 0.25),
        ], p)
        self.assertEqual(r.skipped, 1)
        self.assertEqual(r.note_count, 1)
        self.assertEqual({e.source_index for e in r.events}, {0})
        self.assertEqual(r.events[-1].device, "mouse")
        self.assertEqual(r.events[-1].action, "up")

    def test_direct_override_skips_modifier(self):
        """D3:命中直达键后不再按修饰键。"""
        p = CompileParams(pitch_direct_overrides={"high_1": ","},
                          settle_ms=30.0, release_settle_ms=20.0, max_hold_ms=None)
        r = compile_score([IRNote(1, 1, 0, 0.01)], p)
        self.assertEqual([e.device for e in r.events], ["kb", "kb"])
        self.assertEqual(r.events[0].key, ",")

    def test_direct_override_semitone_guard(self):
        """带 semitone 的音不走直达键,必须走修饰键路径。"""
        p = CompileParams(pitch_direct_overrides={"high_1": ","},
                          settle_ms=30.0, release_settle_ms=20.0, max_hold_ms=None)
        r = compile_score([IRNote(1, 1, 1, 0.01)], p)
        # 不应命中直达键 ",",应走修饰键路径(默认 OCTAVE_FIRST → right)
        kb_keys = [e.key for e in r.events if e.device == "kb"]
        self.assertNotIn(",", kb_keys)
        self.assertIn("mouse", [e.device for e in r.events])
        # OCTAVE_FIRST 策略下 high_1 + semitone → HIGHER (right),半音被降级
        mouse_keys = [e.key for e in r.events if e.device == "mouse"]
        self.assertIn("right", mouse_keys)
        self.assertTrue(any(d.reason == "octave_first" for d in r.degradations))

    def test_direct_override_no_semitone_still_hits(self):
        """semitone=0 的音仍命中直达键,行为不变。"""
        p = CompileParams(pitch_direct_overrides={"high_1": ","},
                          settle_ms=30.0, release_settle_ms=20.0, max_hold_ms=None)
        r = compile_score([IRNote(1, 1, 0, 0.01)], p)
        self.assertEqual([e.device for e in r.events], ["kb", "kb"])
        self.assertEqual(r.events[0].key, ",")


class TestParamsValidation(unittest.TestCase):
    def test_bpm_bounds(self):
        with self.assertRaises(ValueError):
            CompileParams(bpm=0)
        with self.assertRaises(ValueError):
            CompileParams(bpm=301)

    def test_pitch_keys_length(self):
        with self.assertRaises(ValueError):
            CompileParams(pitch_keys=("Z", "X"))

    def test_negative_timing(self):
        for kw in ("settle_ms", "release_settle_ms", "gap_ms"):
            with self.assertRaises(ValueError):
                CompileParams(**{kw: -1})

    def test_hold_ratio_bounds(self):
        with self.assertRaises(ValueError):
            CompileParams(hold_ratio=0)
        with self.assertRaises(ValueError):
            CompileParams(hold_ratio=1.5)

    def test_max_hold_none_allowed(self):
        self.assertIsNone(CompileParams(max_hold_ms=None).max_hold_ms)


class TestDegradationStr(unittest.TestCase):
    def test_readable(self):
        self.assertEqual(str(Degradation(2, "low#", "low", "octave_first")),
                         "元素 3: low# → low(octave_first)")


if __name__ == "__main__":
    unittest.main()
