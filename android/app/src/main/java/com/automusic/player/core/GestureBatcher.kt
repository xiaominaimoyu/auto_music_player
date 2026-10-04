package com.automusic.player.core

import com.automusic.player.input.TimedTouch

/**
 * 已排期的发声元素:绝对起音时刻用 uptimeMillis 计,坐标已是像素。
 * 休止元素不产生 stroke,因此不进入这里。
 */
data class ScheduledNote(
    val index: Int,
    val keys: List<String>,
    val startUptimeMs: Long,
    val holdMs: Long,
    val coords: List<Pair<Float, Float>>,
) {
    val lastReleaseUptimeMs: Long
        get() = startUptimeMs + holdMs
}

/**
 * 一条无障碍手势 = 若干带相对时刻的 stroke。
 *
 * 同一批次内的 stroke 由系统在同一手势时间轴上派发,因此批内不存在
 * "后一次 dispatchGesture 取消前一次未结束手势"的语义;批次之间才是边界。
 */
data class GestureBatch(
    val anchorUptimeMs: Long,
    val strokes: List<TimedTouch>,
    /** 批内第一个与最后一个发声元素的谱面序号(用于进度显示)。 */
    val startIndex: Int,
    val endIndex: Int,
    /** 相对 anchor 的最后一次松手时刻,用于估算停止后的残留按住时长。 */
    val lastReleaseOffsetMs: Long,
) {
    val lastReleaseUptimeMs: Long
        get() = anchorUptimeMs + lastReleaseOffsetMs
}

/**
 * 把逐音符排期折叠成尽量少的无障碍手势。
 *
 * 折叠只在"时间余量本来就紧"的快速段落起作用:慢速长音仍然各自成批,
 * 行为与逐音派发一致。约束来自系统:单手势 stroke 数、单手势总时长上限,
 * 以及同一根手指(同坐标)在一条手势内不能有时间重叠。
 */
object GestureBatcher {

    const val MAX_STROKES_PER_GESTURE = 10
    const val MAX_GESTURE_MS = 60_000L

    /** 单批时间跨度上限:决定"按停止后手指还会按住多久"的最坏值。 */
    const val DEFAULT_MAX_SPAN_MS = 1200L

    fun batch(
        notes: List<ScheduledNote>,
        maxStrokes: Int = MAX_STROKES_PER_GESTURE,
        maxSpanMs: Long = DEFAULT_MAX_SPAN_MS,
    ): List<GestureBatch> {
        if (notes.isEmpty()) return emptyList()

        val batches = ArrayList<GestureBatch>()
        val current = ArrayList<ScheduledNote>()
        var anchor = notes.first().startUptimeMs

        fun flush() {
            if (current.isEmpty()) return
            batches += toBatch(current, anchor)
            current.clear()
        }

        for (note in notes) {
            if (current.isEmpty()) {
                anchor = note.startUptimeMs
                current += note
                continue
            }
            val used = current.sumOf { it.coords.size }
            val relStart = note.startUptimeMs - anchor
            val relEnd = relStart + effectiveHoldMs(note, relStart)
            val fits = used + note.coords.size <= maxStrokes &&
                relEnd <= maxSpanMs &&
                relEnd <= MAX_GESTURE_MS &&
                !overlapsSameFinger(current, note)
            if (fits) {
                current += note
            } else {
                flush()
                anchor = note.startUptimeMs
                current += note
            }
        }
        flush()
        return batches
    }

    /** 同坐标的两次按下若时间重叠,系统会当成一根手指走两段路径,必须避免。 */
    private fun overlapsSameFinger(
        pending: List<ScheduledNote>,
        note: ScheduledNote,
    ): Boolean {
        for (other in pending) {
            val sharedFinger = note.coords.any { other.coords.contains(it) }
            if (!sharedFinger) continue
            if (note.startUptimeMs < other.startUptimeMs + other.holdMs &&
                other.startUptimeMs < note.startUptimeMs + note.holdMs
            ) return true
        }
        return false
    }

    private fun toBatch(notes: List<ScheduledNote>, anchor: Long): GestureBatch {
        val strokes = ArrayList<TimedTouch>()
        var lastRelease = 0L
        for (note in notes) {
            val start = (note.startUptimeMs - anchor).coerceAtLeast(0L)
            val hold = effectiveHoldMs(note, start)
            for ((x, y) in note.coords) strokes += TimedTouch(x, y, start, hold)
            lastRelease = maxOf(lastRelease, start + hold)
        }
        return GestureBatch(
            anchorUptimeMs = anchor,
            strokes = strokes,
            startIndex = notes.first().index,
            endIndex = notes.last().index,
            lastReleaseOffsetMs = lastRelease,
        )
    }

    /** 系统对单条手势有总时长硬上限,越界的按住时长就地夹住。 */
    private fun effectiveHoldMs(note: ScheduledNote, relStartMs: Long): Long =
        note.holdMs.coerceAtLeast(1L).coerceAtMost(MAX_GESTURE_MS - relStartMs)
}
