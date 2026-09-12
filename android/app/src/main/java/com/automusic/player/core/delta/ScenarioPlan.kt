package com.automusic.player.core.delta

/**
 * 场景演奏计划:编译产出的事件序列经场景适配后的最终执行计划。
 *
 * @param events      触摸事件序列(可能含场景追加的提交动作)
 * @param interruptMode 中断语义(PAUSE 可续 / ABORT 不可续)
 * @param abortHint   中断提示文案(空=无提示)
 * @param scenarioId  场景标识
 * @param scenarioName 场景显示名
 */
data class ScenarioPlan(
    val events: List<TouchAction>,
    val interruptMode: InterruptMode,
    val abortHint: String,
    val scenarioId: String,
    val scenarioName: String,
)