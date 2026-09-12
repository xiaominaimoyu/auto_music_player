package com.automusic.player.input.delta

import com.automusic.player.core.delta.TouchAction
import com.automusic.player.core.delta.TouchActionType
import com.automusic.player.input.TouchInjector

/**
 * 三角洲触摸执行器:将编译产出的 TouchAction 序列转换为"按住操作"并派发。
 *
 * 设计说明:Android dispatchGesture 的 stroke 在 duration 结束后自动抬起,
 * 无单独"抬起"API。因此将 DOWN+UP 对合并为一个 dispatchChord 调用
 * (按下-按住-抬起一体),复用现有注入通道,不改动 AmpAccessibilityService。
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
    )

    /**
     * 将 TouchAction 序列配对为"按住操作"序列。
     * 每个 DOWN 与后续同 key 的 UP 配对,holdMs = UP.tMs - DOWN.tMs。
     */
    fun toHoldActions(actions: List<TouchAction>): List<HoldAction> {
        val pending = mutableMapOf<String, TouchAction>()
        val result = mutableListOf<HoldAction>()
        for (action in actions) {
            when (action.action) {
                TouchActionType.DOWN -> pending[action.key] = action
                TouchActionType.UP -> {
                    val down = pending.remove(action.key)
                    if (down != null) {
                        val holdMs = (action.tMs - down.tMs).toLong().coerceIn(1, MAX_STROKE_MS)
                        result.add(HoldAction(down.tMs, down.x, down.y, holdMs, down.key))
                    }
                }
            }
        }
        return result.sortedBy { it.tMs }
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
     * 已派发的 dispatchGesture 手势会自动完成(stroke duration 到期抬起);
     * 此方法用于中断路径记录日志,无法主动取消已派发的手势。
     */
    fun releaseAll() {
        android.util.Log.i(TAG, "releaseAll: 已派发手势将自动完成,协程已停止派发新手势")
    }
}