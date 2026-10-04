package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 手势折叠规划器单测。
 *
 * 这里锁死的是"手机端不会把上一个音掐断"的前提条件:批内 stroke 数、时间跨度、
 * 同坐标不重叠,以及慢速段落仍然逐音派发(保持与桌面版可对比的行为)。
 */
class GestureBatcherTest {

    private fun note(index: Int, start: Long, hold: Long, vararg keys: String) = ScheduledNote(
        index = index,
        keys = keys.toList(),
        startUptimeMs = start,
        holdMs = hold,
        // 每个 key 一个确定且互不相同的坐标,便于构造"同坐标重叠"场景
        coords = keys.map { k -> k.length.toFloat() to k.sumOf { c -> c.code }.toFloat() },
    )

    @Test
    fun `空序列不产生批次`() {
        assertEquals(0, GestureBatcher.batch(emptyList()).size)
    }

    @Test
    fun `快速段落折叠成少数手势`() {
        val notes = (0 until 12).map { note(it, it * 120L, 100L, "mid_${it % 7 + 1}") }
        val batches = GestureBatcher.batch(notes)

        assertTrue(12 > batches.size)
        assertEquals(12, batches.sumOf { it.strokes.size })
        batches.forEach {
            assertTrue("stroke 不能超上限", it.strokes.size <= GestureBatcher.MAX_STROKES_PER_GESTURE)
            assertTrue("单手势跨度不能超上限", it.lastReleaseOffsetMs <= GestureBatcher.DEFAULT_MAX_SPAN_MS)
        }
        // 批次必须连续覆盖整段,进度显示不会漏音或重播
        assertEquals(0, batches.first().startIndex)
        assertEquals(11, batches.last().endIndex)
        batches.zip(batches.drop(1)).forEach { (prev, next) ->
            assertEquals(prev.endIndex + 1, next.startIndex)
        }
    }

    @Test
    fun `慢速长音仍逐音派发`() {
        val notes = (0 until 3).map { note(it, it * 2000L, 1900L, "mid_${it + 1}") }
        val batches = GestureBatcher.batch(notes)

        assertEquals(3, batches.size)
        batches.forEach { assertEquals(1, it.strokes.size) }
    }

    @Test
    fun `和弦按指数占用 stroke 额度`() {
        val notes = (0 until 6).map { note(it, it * 100L, 80L, "mid_1", "mid_3", "mid_5") }
        val batches = GestureBatcher.batch(notes)

        batches.forEach {
            assertTrue("和弦也要受 10 stroke 限制", it.strokes.size <= 10)
            assertTrue("每条手势至少一个音", it.strokes.isNotEmpty())
        }
        assertEquals(18, batches.sumOf { it.strokes.size })
    }

    @Test
    fun `同一根手指时间重叠必须拆批`() {
        val overlapping = listOf(
            note(0, 0L, 300L, "mid_4"),
            note(1, 200L, 300L, "mid_4"),  // 同坐标且重叠
        )
        assertEquals(2, GestureBatcher.batch(overlapping).size)

        val distinctFingers = listOf(
            note(0, 0L, 300L, "mid_4"),
            note(1, 200L, 300L, "mid_5"),  // 不同坐标,重叠无妨(等同和弦式并发)
        )
        assertEquals(1, GestureBatcher.batch(distinctFingers).size)
    }

    @Test
    fun `批次锚点与松手时刻`() {
        val notes = listOf(note(3, 5000L, 120L, "mid_1"), note(4, 5150L, 100L, "mid_2"))
        val batch = GestureBatcher.batch(notes).single()

        assertEquals(5000L, batch.anchorUptimeMs)
        assertEquals(250L, batch.lastReleaseOffsetMs)
        assertEquals(5250L, batch.lastReleaseUptimeMs)
        assertEquals(listOf(0L, 150L), batch.strokes.map { it.startMs })
    }

    @Test
    fun `超长单音被夹到系统手势上限`() {
        val notes = listOf(note(0, 0L, 90_000L, "mid_1"))
        val batch = GestureBatcher.batch(notes).single()

        assertTrue(batch.strokes.single().holdMs <= GestureBatcher.MAX_GESTURE_MS)
    }
}
