package com.automusic.player.core.delta

/**
 * 触摸动作类型。
 *
 * - DOWN 按下(手指落下)
 * - UP   抬起(手指离开)
 */
enum class TouchActionType {
    DOWN,
    UP,
}

/**
 * 带时刻与键名的触摸事件(编译器产出,执行器消费)。
 *
 * @param tMs    绝对时刻(毫秒,从演奏起点计)
 * @param x      屏幕 X 像素坐标
 * @param y      屏幕 Y 像素坐标
 * @param action DOWN 或 UP
 * @param key    键名(用于释放追踪,如 "note_3"、"mod_lower"、"submit")
 */
data class TouchAction(
    val tMs: Double,
    val x: Float,
    val y: Float,
    val action: TouchActionType,
    val key: String,
)