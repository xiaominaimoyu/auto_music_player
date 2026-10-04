package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.Random
import kotlin.math.abs

/**
 * 真人化塑形单测。
 *
 * 锁两件事:默认参数必须与桌面版 core/humanize.py 一致(历史上安卓端长期停留在
 * 桌面 v1.1 之前的旧值,导致手机听起来比桌面更平更紧);以及 seed 可重放、
 * 抖动带乐句内相关性。
 */
class HumanizeTest {

    private fun melody(size: Int, key: String = "mid_1") =
        (0 until size).map { NoteEvent(listOf(key), 1.0) }

    /** 相邻偏移的滞后 1 自相关:白噪声≈0,AR(1) 相关≈φ。 */
    private fun lag1Autocorr(xs: List<Double>): Double {
        val n = xs.size - 1
        val a = xs.dropLast(1)
        val b = xs.drop(1)
        val ma = a.average()
        val mb = b.average()
        var cov = 0.0
        var va = 0.0
        var vb = 0.0
        for (i in 0 until n) {
            val da = a[i] - ma
            val db = b[i] - mb
            cov += da * db
            va += da * da
            vb += db * db
        }
        return cov / (kotlin.math.sqrt(va * vb))
    }

    @Test
    fun `默认参数与桌面版 humanize 对齐`() {
        val p = HumanizeParams()
        assertEquals(12.0, p.jitterMs, 1e-9)
        assertEquals(25.0, p.breathMs, 1e-9)
        assertEquals(5.0, p.leapMs, 1e-9)
        assertEquals(1.5, p.minGapMs, 1e-9)
        assertEquals(0.65, p.jitterCorrelation, 1e-9)
        assertEquals(4, p.leapThreshold)
        assertEquals(0.58 to 0.70, p.shortHold)
        assertEquals(0.70 to 0.85, p.midHold)
        assertEquals(0.86 to 0.98, p.longHold)
    }

    @Test
    fun `同 seed 重放同一份塑形计划`() {
        val notes = melody(40)
        val a = Humanize.buildPlan(notes, HumanizeParams(), 20261004L)
        val b = Humanize.buildPlan(notes, HumanizeParams(), 20261004L)
        assertEquals(20261004L, a.seed)
        assertEquals(a.timings, b.timings)
    }

    @Test
    fun `抖动在乐句内相关而关闭相关时接近白噪声`() {
        val notes = melody(600)
        val correlated = Humanize.planTimings(
            notes, HumanizeParams(jitterCorrelation = 0.85), Random(7L),
        ).map { it.offsetMs }
        val independent = Humanize.planTimings(
            notes, HumanizeParams(jitterCorrelation = 0.0), Random(7L),
        ).map { it.offsetMs }

        assertTrue("相关抖动应产生连续性", lag1Autocorr(correlated) > 0.5)
        assertTrue("correlation=0 应接近独立", abs(lag1Autocorr(independent)) < 0.2)
    }

    @Test
    fun `休止不发声并作为乐句边界追加呼吸`() {
        val notes = listOf(
            NoteEvent(listOf("mid_1"), 1.0),
            NoteEvent(emptyList(), 2.0),
            NoteEvent(listOf("mid_3"), 1.0),
        )
        // 关掉抖动与跳进,断言只考察"休止后呼吸"这一条规则
        val p = HumanizeParams(jitterMs = 0.0, leapMs = 0.0)
        (1..50).forEach { seed ->
            val t = Humanize.buildPlan(notes, p, seed.toLong()).timings
            assertEquals(3, t.size)
            assertEquals(0.0, t[1].offsetMs, 1e-9)
            assertNull(t[1].holdRatio)
            assertTrue("休止后的音符应带正呼吸停顿", t[2].offsetMs > 0.0)
        }
    }

    @Test
    fun `休止重置相关抖动状态`() {
        val notes = listOf(
            NoteEvent(listOf("mid_1"), 1.0),
            NoteEvent(listOf("mid_1"), 1.0),
            NoteEvent(emptyList(), 1.0),
            NoteEvent(listOf("mid_1"), 1.0),
        )
        val p = HumanizeParams(jitterCorrelation = 0.95, breathMs = 0.0)
        // 休止把相关状态归零:该音偏移只能来自 (1-φ)·σ 这一小步,而不是链上累积量
        val maxAfterRest = (1..200L).maxOf { abs(Humanize.buildPlan(notes, p, it).timings[3].offsetMs) }
        assertTrue("休止后偏移应被重置,实测 $maxAfterRest ms", maxAfterRest < 5.0)
    }

    @Test
    fun `动态按住分档落在声明区间内`() {
        val notes = listOf(
            NoteEvent(listOf("mid_1"), 0.25),
            NoteEvent(listOf("mid_1"), 1.0),
            NoteEvent(listOf("mid_1"), 4.0),
        )
        val p = HumanizeParams(jitterMs = 0.0, breathMs = 0.0, leapMs = 0.0)
        (1..50).forEach { seed ->
            val t = Humanize.buildPlan(notes, p, seed.toLong()).timings
            assertTrue(t[0].holdRatio!! in 0.58..0.70)
            assertTrue(t[1].holdRatio!! in 0.70..0.85)
            assertTrue(t[2].holdRatio!! in 0.86..0.98)
        }
    }

    @Test
    fun `非法参数直接拒绝`() {
        runCatching { HumanizeParams(jitterMs = -1.0) }.onSuccess { fail("负抖动应被拒绝") }
        runCatching { HumanizeParams(jitterCorrelation = 1.0) }.onSuccess { fail("相关系数=1 应被拒绝") }
        runCatching { HumanizeParams(minGapMs = -0.5) }.onSuccess { fail("负 minGap 应被拒绝") }
        runCatching { HumanizeParams(longHold = 0.9 to 1.4) }.onSuccess { fail("hold 上限越界应被拒绝") }
    }
}
