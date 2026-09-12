package com.automusic.player.core.delta

/**
 * S1 自由演奏场景:口琴由用户手动从背包启用,中断可续播,无提交。
 *
 * - interruptMode = PAUSE(可续播)
 * - humanize = true(真人化演奏)
 * - 间隔:settleMs=30, gapMs=20(音乐时值友好)
 */
class FreePlayScenario : Scenario() {
    override val id: String = "free_play"
    override val name: String = "自由演奏"
    override val interruptMode: InterruptMode = InterruptMode.PAUSE
    override val abortHint: String = ""
    override val humanize: Boolean = true

    override fun intervals(base: DeltaCompileParams): DeltaCompileParams = base

    override fun plan(events: List<TouchAction>, submitCoord: Pair<Float, Float>?, lastTMs: Double): ScenarioPlan {
        return ScenarioPlan(
            events = events,
            interruptMode = interruptMode,
            abortHint = abortHint,
            scenarioId = id,
            scenarioName = name,
        )
    }
}