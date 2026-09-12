package com.automusic.player.core.delta

import android.content.Context
import com.automusic.player.PlaybackService
import com.automusic.player.core.NoteEvent
import com.automusic.player.core.Timing
import com.automusic.player.input.AmpAccessibilityService

/**
 * 三角洲演奏会话管理器:统一编排编译→场景适配→执行→中断处理的生命周期。
 *
 * 四条中断路径均通过此入口:
 * - 用户停止 → endSession()
 * - 系统失焦 → onActivityPause()
 * - 异常退出 → 协程捕获 → abort()
 * - 无障碍断开 → onAccessibilityDisconnected()
 */
class SessionManager(
    private val player: DeltaPlayerEngine,
    private val context: Context,
) {
    private var active: Boolean = false
    private var currentPlan: ScenarioPlan? = null

    /**
     * 启动演奏会话。
     *
     * @param notes    简谱音符列表
     * @param params   编译参数(场景仅施加安全下限)
     * @param layout   三角洲布局
     * @param scenario 场景适配器
     * @param screenW  屏幕宽度
     * @param screenH  屏幕高度
     * @param timings  真人化偏移(可选)
     * @return 编译结果(含降级清单),失败返回 null
     */
    fun startSession(
        notes: List<NoteEvent>,
        params: DeltaCompileParams,
        layout: DeltaLayout,
        scenario: Scenario,
        screenW: Int,
        screenH: Int,
        timings: List<Timing>? = null,
    ): DeltaCompileResult? {
        if (!AmpAccessibilityService.ready) return null

        val sceneParams = scenario.intervals(params)
        val result = DeltaCompiler.compile(notes, sceneParams, layout, screenW, screenH, timings)
        val lastTMs = result.events.maxOfOrNull { it.tMs } ?: 0.0
        val submitCoord = layout.submitPixels(screenW, screenH)
        val plan = scenario.plan(result.events, submitCoord, lastTMs)
        currentPlan = plan

        PlaybackService.start(context)
        active = true
        player.play(plan)
        return result
    }

    /** 从暂停态续播(FreePlay 场景)。失败时返回 false，供 UI 提示重新开始。 */
    fun resumeSession(): Boolean {
        if (!AmpAccessibilityService.ready) return false
        val plan = currentPlan ?: return false
        val st = player.state.value as? DeltaPlayerEngine.State.Paused ?: return false
        val startIndex = st.done
        PlaybackService.start(context)
        active = true
        player.play(plan, startIndex)
        return true
    }

    /** 用户主动结束会话。 */
    fun endSession() {
        if (!active) return
        player.stop()
        PlaybackService.stop(context)
        active = false
    }

    /** 系统失焦(Activity onPause)。 */
    fun onActivityPause() {
        if (!active) return
        player.stop()
        PlaybackService.stop(context)
        active = false
    }

    /** 无障碍服务断开。 */
    fun onAccessibilityDisconnected() {
        if (!active) return
        player.abort("无障碍服务断开,请重新开启触摸注入服务")
        PlaybackService.stop(context)
        active = false
    }

    /** 异常退出。 */
    fun onError(reason: String) {
        if (!active) return
        player.abort(reason)
        PlaybackService.stop(context)
        active = false
    }

    /** 重置会话到未开始态。 */
    fun resetSession() {
        player.reset()
        PlaybackService.stop(context)
        active = false
        currentPlan = null
    }

    val isActive: Boolean get() = active
}
