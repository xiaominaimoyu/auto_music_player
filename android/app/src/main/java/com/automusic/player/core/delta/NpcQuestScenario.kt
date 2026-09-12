package com.automusic.player.core.delta

/**
 * S2 NPC 听旋律复刻任务:NPC 示范后自动装备口琴,中断不可续播,自动提交。
 *
 * - interruptMode = ABORT(不可续播,需重新听 NPC 示范)
 * - humanize = false(识别友好,宁可慢不可叠键)
 * - 间隔:settleMs=60, gapMs=120(保守值,降低叠键风险)
 * - 演奏完毕自动点击提交按钮
 */
class NpcQuestScenario : Scenario() {
    override val id: String = "npc_quest"
    override val name: String = "NPC 任务"
    override val interruptMode: InterruptMode = InterruptMode.ABORT
    override val abortHint: String = "NPC 任务中断后不可续播,请重新听 NPC 示范"
    override val humanize: Boolean = false

    override fun intervals(base: DeltaCompileParams): DeltaCompileParams {
        return base.copy(settleMs = 60L, gapMs = 120L)
    }

    override fun plan(events: List<TouchAction>, submitCoord: Pair<Float, Float>?, lastTMs: Double): ScenarioPlan {
        val finalEvents = if (submitCoord != null) {
            val submitGap = 200.0
            val submitT = lastTMs + submitGap
            events + listOf(
                TouchAction(submitT, submitCoord.first, submitCoord.second, TouchActionType.DOWN, DeltaKeyPoint.KEY_SUBMIT),
                TouchAction(submitT + 50.0, submitCoord.first, submitCoord.second, TouchActionType.UP, DeltaKeyPoint.KEY_SUBMIT),
            )
        } else {
            events
        }
        return ScenarioPlan(
            events = finalEvents,
            interruptMode = interruptMode,
            abortHint = abortHint,
            scenarioId = id,
            scenarioName = name,
        )
    }
}