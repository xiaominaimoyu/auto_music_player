"""真人化节奏塑形单元测试:plan_timings 结构规则 + 编译器接入。"""

import random
import unittest

from core.compiler import CompileParams, compile_score
from core.humanize import HumanizeParams, plan_timings
from core.ir import IRNote, IRRest

# 结构性测试用零 jitter/breath/leap,让塑形规则的贡献可被精确观测
STILL = HumanizeParams(jitter_ms=0.0, breath_ms=0.0, leap_ms=0.0)


def el(notes, dur=1.0):
    return {"notes": notes, "dur": dur}


class TestPlanTimingsStructure(unittest.TestCase):
    def test_length_and_rest_placeholder(self):
        """输出与输入等长;休止为 (0.0, None) 占位。"""
        els = [el(["mid_1"]), {"notes": [], "dur": 1.0}, el(["mid_2"])]
        ts = plan_timings(els, STILL, random.Random(1))
        self.assertEqual(len(ts), 3)
        self.assertEqual(ts[1], (0.0, None))
        self.assertIsInstance(ts[0][0], float)
        self.assertIsInstance(ts[0][1], float)

    def test_deterministic_with_seeded_rng(self):
        els = [el(["mid_1", "mid_3"]), el(["high_2"], 2.0), el(["mid_7"], 0.5)]
        a = plan_timings(els, HumanizeParams(), random.Random(42))
        b = plan_timings(els, HumanizeParams(), random.Random(42))
        self.assertEqual(a, b)

    def test_hold_tiers_by_duration(self):
        """短/中/长音符的 hold 落在各自档位区间内。"""
        p = HumanizeParams(jitter_ms=0.0)
        els = [el(["mid_1"], 0.25), el(["mid_2"], 1.0), el(["mid_3"], 2.0)]
        ts = plan_timings(els, p, random.Random(7))
        lo, hi = p.short_hold
        self.assertGreaterEqual(ts[0][1], lo)
        self.assertLessEqual(ts[0][1], hi)
        lo, hi = p.mid_hold
        self.assertGreaterEqual(ts[1][1], lo)
        self.assertLessEqual(ts[1][1], hi)
        lo, hi = p.long_hold
        self.assertGreaterEqual(ts[2][1], lo)
        self.assertLessEqual(ts[2][1], hi)

    def test_breath_after_rest(self):
        """休止后的下一音应附加呼吸停顿(jitter=0 时 offset 必为正)。"""
        p = HumanizeParams(jitter_ms=0.0, breath_ms=20.0, leap_ms=0.0)
        els = [el(["mid_1"]), {"notes": [], "dur": 1.0}, el(["mid_2"])]
        ts = plan_timings(els, p, random.Random(3))
        self.assertEqual(ts[0][0], 0.0)   # 首音无前置元素
        self.assertGreater(ts[2][0], 0.0)  # 呼吸 × uniform(0.6,1.0) ≥ 12ms
        self.assertLess(ts[2][0], 20.0)

    def test_breath_after_long_note(self):
        """长音(≥2 拍)结尾后的下一音按乐句开头处理,带呼吸。"""
        p = HumanizeParams(jitter_ms=0.0, breath_ms=20.0, leap_ms=0.0)
        els = [el(["mid_1"], 2.0), el(["mid_2"], 1.0)]
        ts = plan_timings(els, p, random.Random(3))
        self.assertGreater(ts[1][0], 0.0)

    def test_no_breath_after_short_note(self):
        """普通短音符之间不引入呼吸。"""
        p = HumanizeParams(jitter_ms=0.0, breath_ms=20.0, leap_ms=0.0)
        els = [el(["mid_1"], 1.0), el(["mid_2"], 1.0)]
        ts = plan_timings(els, p, random.Random(3))
        self.assertEqual(ts[1][0], 0.0)

    def test_leap_adds_offset(self):
        """音高大跳进(≥阈值)附加换指时间。"""
        p = HumanizeParams(jitter_ms=0.0, breath_ms=0.0, leap_ms=5.0)
        els = [el(["mid_1"]), el(["high_1"])]   # 差值 7 ≥ 4
        ts = plan_timings(els, p, random.Random(3))
        self.assertGreater(ts[1][0], 0.0)
        # 小步进(mid_1 → mid_2)不附加
        els = [el(["mid_1"]), el(["mid_2"])]
        ts = plan_timings(els, p, random.Random(3))
        self.assertEqual(ts[1][0], 0.0)

    def test_jitter_zero_means_exact(self):
        """jitter=0 时普通连续音符 offset 恒为 0(零均值高斯的退化)。"""
        els = [el(["mid_1"]), el(["mid_2"]), el(["mid_3"])]
        ts = plan_timings(els, STILL, random.Random(5))
        self.assertEqual([t[0] for t in ts], [0.0, 0.0, 0.0])


class TestHumanizeParams(unittest.TestCase):
    def test_reject_negative_amplitudes(self):
        with self.assertRaises(ValueError):
            HumanizeParams(jitter_ms=-1.0)
        with self.assertRaises(ValueError):
            HumanizeParams(min_gap_ms=-1.0)

    def test_reject_bad_hold_tier(self):
        with self.assertRaises(ValueError):
            HumanizeParams(short_hold=(0.9, 0.5))
        with self.assertRaises(ValueError):
            HumanizeParams(mid_hold=(0.0, 0.5))


class TestCompilerWithTimings(unittest.TestCase):
    """真人化 timings 接入 compile_score 的行为验证。"""

    P = CompileParams(bpm=100, settle_ms=30.0, release_settle_ms=20.0,
                      hold_ratio=1.0, max_hold_ms=None, gap_ms=0.0)

    def _events(self, els, timings):
        return compile_score(els, self.P, timings=timings)

    def test_per_note_hold_ratio_applied(self):
        """timings 的逐音符 hold_ratio 覆盖全局 hold_ratio。"""
        els = [IRNote(1, 0, 0, 1.0), IRNote(2, 0, 0, 1.0)]  # 600ms 每音
        timings = [(0.0, 0.5), (0.0, 0.5)]
        ev = self._events(els, timings)
        downs = [e for e in ev.events if e.action == "down" and e.device == "kb"]
        ups = [e for e in ev.events if e.action == "up" and e.device == "kb"]
        for d, u in zip(downs, ups):
            self.assertAlmostEqual(u.t_ms - d.t_ms, 300.0, places=3)

    def test_positive_offset_delays_attack(self):
        """正偏移把起音推迟(呼吸/跳进);零偏移时链式钳制仅多 +1ms 安全边。"""
        els = [IRNote(1, 0, 0, 1.0), IRNote(2, 0, 0, 1.0)]  # 600ms/音
        base = self._events(els, [(0.0, 1.0), (0.0, 1.0)])
        moved = self._events(els, [(0.0, 1.0), (30.0, 1.0)])
        b = [e for e in base.events if e.action == "down" and e.device == "kb"]
        m = [e for e in moved.events if e.action == "down" and e.device == "kb"]
        self.assertAlmostEqual(b[1].t_ms, 601.0, places=3)   # 链式钳制:上一音释放+1ms
        self.assertAlmostEqual(m[1].t_ms, 630.0, places=3)   # 偏移 30ms,大于钳制下限

    def test_negative_offset_clamped_by_prev_release(self):
        """负偏移不得让下一音早于上一音完全释放(链式防叠键)。"""
        els = [IRNote(1, 0, 0, 1.0), IRNote(2, 0, 0, 1.0)]  # 600ms/音,hold 100%
        timings = [(0.0, 1.0), (-200.0, 1.0)]
        ev = self._events(els, timings)
        downs = [e for e in ev.events if e.action == "down" and e.device == "kb"]
        ups = [e for e in ev.events if e.action == "up" and e.device == "kb"]
        self.assertGreaterEqual(downs[1].t_ms, ups[0].t_ms)

    def test_none_timings_keeps_mechanical(self):
        """timings=None 完全保持历史机械时序(与旧行为 bit 级一致)。"""
        els = [IRNote(1, 0, 0, 1.0), IRRest(1.0), IRNote(3, 0, 0, 2.0)]
        self.assertEqual(
            self._events(els, None),
            compile_score(els, self.P),
        )

    def test_rest_timings_ignored(self):
        """休止元素的 (0.0, None) 占位不参与发声;等效于 None 供休止使用。"""
        els = [IRNote(1, 0, 0, 1.0), IRRest(1.0), IRNote(3, 0, 0, 1.0)]
        timings = [(0.0, 0.5), (0.0, None), (0.0, 0.5)]
        ev = self._events(els, timings)
        downs = [e for e in ev.events if e.action == "down" and e.device == "kb"]
        self.assertEqual(len(downs), 2)
