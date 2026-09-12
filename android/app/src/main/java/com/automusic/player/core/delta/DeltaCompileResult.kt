package com.automusic.player.core.delta

/**
 * 三角洲谱面编译结果。
 *
 * @param events      触摸事件序列(按时刻递增)
 * @param degradations 降级清单
 * @param noteCount   音符总数(含休止)
 * @param durationMs  总时长(毫秒)
 */
data class DeltaCompileResult(
    val events: List<TouchAction>,
    val degradations: List<Degradation>,
    val noteCount: Int,
    val durationMs: Double,
)