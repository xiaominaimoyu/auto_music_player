package com.automusic.player.input.delta

import com.automusic.player.core.delta.TouchAction
import com.automusic.player.core.delta.TouchActionType
import com.automusic.player.input.TimedTouch
import com.automusic.player.input.TouchInjector
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * 三角洲触摸执行器:将编译产出的 TouchAction 序列转换为原子手势并派发。
 *
 * 设计说明:Android dispatchGesture 的 stroke 在 duration 结束后自动抬起,
 * 无单独"抬起"API。因此将 DOWN+UP 对合并为一个 dispatchChord 调用
 * (按下-按住-抬起一体)。同一逻辑音的修饰钮与音格会放到一个
 * GestureDescription 中，保证二者重叠，不会被下一次 dispatchGesture 取消。
 *
 * 中断时取消协程即可停止派发新手势;已派发的手势会自动完成(stroke 到期抬起)。
 */
object DeltaTouchExecutor {

    private const val TAG = "DeltaTouchExecutor"
    private const val MAX_STROKE_MS = 60000L

    /** 一个"按下-按住-抬起"操作。 */
    data class HoldAction(
        val tMs: Double,
        val x: Float,
        val y: Float,
        val holdMs: Long,
        val key: String,
        val sourceIndex: Int?,
    )

    data class GestureStroke(
        val x: Float,
        val y: Float,
        val startMs: Long,
        val holdMs: Long,
        val key: String,
    )

    data class AtomicGesture(
        val tMs: Double,
        val sourceIndex: Int?,
        val strokes: List<GestureStroke>,
    )

    /**
     * 将 TouchAction 序列配对为"按住操作"序列。
     * 每个 DOWN 与后续同 key 的 UP 配对,holdMs = UP.tMs - DOWN.tMs。
     */
    fun toHoldActions(actions: List<TouchAction>): List<HoldAction> {
        val pending = mutableMapOf<String, TouchAction>()
        val result = mutableListOf<HoldAction>()
        for (action in actions.sortedWith(compareBy<TouchAction> { it.tMs }.thenBy { it.action.ordinal })) {
            when (action.action) {
                TouchActionType.DOWN -> pending[action.key] = action
                TouchActionType.UP -> {
                    val down = pending.remove(action.key)
                    if (down != null) {
                        val holdMs = (action.tMs - down.tMs).toLong().coerceIn(1, MAX_STROKE_MS)
                        result.add(HoldAction(
                            down.tMs,
                            down.x,
                            down.y,
                            holdMs,
                            down.key,
                            down.sourceIndex ?: action.sourceIndex,
                        ))
                    }
                }
            }
        }
        return result.sortedBy { it.tMs }
    }

    /** 将同一逻辑音的修饰钮和音格合并为一条原子手势。 */
    fun toAtomicGestures(actions: List<TouchAction>): List<AtomicGesture> {
        val groups = linkedMapOf<String, MutableList<HoldAction>>()
        var anonymous = 0
        for (hold in toHoldActions(actions)) {
            val key = hold.sourceIndex?.let { "source:$it" } ?: "anonymous:${anonymous++}"
            groups.getOrPut(key) { mutableListOf() }.add(hold)
        }
        return groups.values.map { holds ->
            val start = holds.minOf { it.tMs }
            AtomicGesture(
                tMs = start,
                sourceIndex = holds.first().sourceIndex,
                strokes = holds.map { hold ->
                    GestureStroke(
                        hold.x,
                        hold.y,
                        (hold.tMs - start).toLong().coerceAtLeast(0L),
                        hold.holdMs,
                        hold.key,
                    )
                }.sortedBy { it.startMs },
            )
        }.sortedBy { it.tMs }
    }

    /** 挂起至原子手势完成；下一音只会在上一手势结束后再派发。 */
    suspend fun dispatchAtomic(gesture: AtomicGesture) {
        suspendCancellableCoroutine<Unit> { continuation ->
            try {
                val accepted = TouchInjector.dispatchTimeline(
                    gesture.strokes.map { TimedTouch(it.x, it.y, it.startMs, it.holdMs) }
                ) { completed ->
                    if (continuation.isActive) {
                        if (completed) continuation.resume(Unit)
                        else continuation.resumeWithException(
                            IllegalStateException("无障碍手势被系统取消")
                        )
                    }
                }
                if (!accepted && continuation.isActive) {
                    continuation.resumeWithException(IllegalStateException("dispatchGesture 被系统拒绝"))
                }
            } catch (e: Throwable) {
                if (continuation.isActive) continuation.resumeWithException(e)
            }
        }
    }

    /** 派发一个"按下-按住-抬起"操作。 */
    fun dispatchHold(x: Float, y: Float, holdMs: Long) {
        TouchInjector.dispatchChord(listOf(x to y), holdMs)
    }

    /** 单点轻触(标定/测试用)。 */
    fun tap(x: Float, y: Float) {
        TouchInjector.tap(x, y)
    }

    /**
     * 尽最大努力释放全部按住态。
     *
     * 已派发的原子手势会在当前音结束时自动抬起；中断路径停止后续派发，
     * 当前音最多持续其受限的 holdMs。
     */
    fun releaseAll() {
        android.util.Log.i(TAG, "releaseAll: 已派发手势将自动完成,协程已停止派发新手势")
    }
}
