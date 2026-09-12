package com.automusic.player.core.delta

/**
 * 场景适配抽象基类:收敛 FreePlay 与 NpcQuest 场景差异。
 *
 * 音符层不 fork:两场景共用同一编译器事件流,差异收敛在会话层:
 * - 中断语义(pause 可续 / abort 不可续)
 * - 提交行为(NpcQuest 追加提交动作)
 * - 间隔策略(识别友好保守值)
 * - 真人化开关
 */
abstract class Scenario {
    abstract val id: String
    abstract val name: String
    abstract val interruptMode: InterruptMode
    abstract val abortHint: String
    abstract val humanize: Boolean

    /** 返回本场景的编译参数间隔策略。 */
    abstract fun intervals(base: DeltaCompileParams): DeltaCompileParams

    /**
     * 将编译产出的事件序列适配为场景演奏计划。
     *
     * @param events      编译产出的触摸事件序列
     * @param submitCoord 提交按钮像素坐标(NpcQuest 用,可缺省)
     * @param lastTMs     最后一个事件的时刻(用于追加提交动作)
     */
    abstract fun plan(events: List<TouchAction>, submitCoord: Pair<Float, Float>?, lastTMs: Double): ScenarioPlan
}