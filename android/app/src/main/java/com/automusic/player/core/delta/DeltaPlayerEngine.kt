package com.automusic.player.core.delta

import android.os.SystemClock
import com.automusic.player.input.delta.DeltaTouchExecutor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * 三角洲演奏引擎:协程驱动的事件序列消费器,支持 pause/abort 双中断语义。
 *
 * 与 PlayerEngine 的区别:
 * - State 增加 Aborted(reason)(NpcQuest 不可续播)
 * - 消费编译产出的原子 Gesture 序列(修饰态与音格同一次派发)
 * - stop() 按 plan.interruptMode 决定 pause(保留进度)或 abort(清空进度)
 *
 * 不改动 PlayerEngine 现有代码,独立实现。
 */
class DeltaPlayerEngine(private val scope: kotlinx.coroutines.CoroutineScope) {

    sealed interface State {
        data object Idle : State
        data class Playing(val done: Int, val total: Int) : State
        data class Paused(val done: Int, val total: Int) : State
        data class Aborted(val reason: String) : State
        data class Finished(val complete: Boolean, val error: String?) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private var job: Job? = null
    private var currentPlan: ScenarioPlan? = null

    @Volatile
    private var lastDone = 0

    @Volatile
    private var lastTotal = 0

    val isPlaying: Boolean
        get() = job?.isActive == true

    /**
     * 开始演奏(或从指定进度续播)。
     *
     * @param plan       场景演奏计划
     * @param startIndex 起始原子手势索引(暂停续播用)
     */
    fun play(plan: ScenarioPlan, startIndex: Int = 0) {
        if (isPlaying) return
        currentPlan = plan
        job = scope.launch {
            val gestures = DeltaTouchExecutor.toAtomicGestures(plan.events)
            val total = gestures.size
            lastTotal = total
            var complete = true
            var errorMsg: String? = null
            try {
                val safeStart = startIndex.coerceIn(0, total)
                // 续播时以第一条未完成手势为新的零点，避免等待整段旧时间轴。
                val baseMs = gestures.getOrNull(safeStart)?.tMs?.toLong() ?: 0L
                val start = SystemClock.uptimeMillis() - baseMs
                for (idx in safeStart until total) {
                    val gesture = gestures[idx]
                    val target = start + gesture.tMs.toLong()
                    delayUntil(target)
                    DeltaTouchExecutor.dispatchAtomic(gesture)
                    lastDone = idx + 1
                    _state.value = State.Playing(lastDone, total)
                }
                lastDone = 0
                lastTotal = 0
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                lastDone = 0
                lastTotal = 0
                complete = false
                errorMsg = e.message ?: e.toString()
            } finally {
                if (kotlin.coroutines.coroutineContext.isActive) {
                    _state.value = State.Finished(complete, errorMsg)
                }
            }
        }
    }

    /**
     * 停止:按计划的中断模式决定 pause(可续)或 abort(不可续)。
     */
    fun stop() {
        val plan = currentPlan
        job?.cancel()
        job = null
        when (plan?.interruptMode) {
            InterruptMode.ABORT -> {
                DeltaTouchExecutor.releaseAll()
                lastDone = 0
                lastTotal = 0
                _state.value = State.Aborted(plan.abortHint.ifBlank { "演奏已中止" })
            }
            InterruptMode.PAUSE -> {
                DeltaTouchExecutor.releaseAll()
                _state.value = if (_state.value is State.Playing) {
                    State.Paused(lastDone, lastTotal)
                } else {
                    State.Idle
                }
            }
            null -> {
                _state.value = State.Idle
            }
        }
    }

    /** 重置:清空进度回到未开始态。 */
    fun reset() {
        job?.cancel()
        job = null
        lastDone = 0
        lastTotal = 0
        currentPlan = null
        _state.value = State.Idle
    }

    /** 强制中止(异常/失焦/无障碍断开等)。 */
    fun abort(reason: String) {
        job?.cancel()
        job = null
        DeltaTouchExecutor.releaseAll()
        lastDone = 0
        lastTotal = 0
        _state.value = State.Aborted(reason)
    }

    private suspend fun delayUntil(target: Long) {
        while (kotlin.coroutines.coroutineContext.isActive) {
            val now = SystemClock.uptimeMillis()
            if (now >= target) return
            delay(target - now)
        }
    }
}
