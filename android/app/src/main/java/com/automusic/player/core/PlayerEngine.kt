package com.automusic.player.core

import android.os.SystemClock
import android.util.Log
import com.automusic.player.input.TouchInjector
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * 演奏引擎:按音符序列的节奏派发触摸和弦(协程 + 绝对时钟调度,抗节奏漂移)。
 *
 * 三态控制:
 * - 开始/继续:play(startIndex) 从指定进度开始
 * - 停止:stop() 暂停演奏 -> State.Paused(done, total),进度保留
 * - 重置:reset() 进度归零 -> State.Idle,下次 play 从头开始
 *
 * 时序:每音符周期严格等于 时值 + gap;按住时长 = 时值 × hold_ratio。
 *
 * 派发方式:先把全部音符排成绝对时刻表,再由 [GestureBatcher] 折叠成尽量少的
 * 无障碍手势。原因是系统的 dispatchGesture 有"后一次派发取消前一次未结束手势"
 * 的语义,逐音派发时两个音之间只剩 `时值×(1-hold)+gap` 的余量,一次调度抖动就会
 * 把上一个音掐断。折叠后批内由同一条手势的时间轴保证精度,批间等待自然完成,
 * 只在边界承担一次派发延迟(记入日志 late_ms)。
 *
 * 残留按住:已派发的单条手势无法取消(Android 无公开取消 API),所以"停止"只能
 * 保证不再派发新事件,当前这条手势内最后一个松手时刻仍会到点执行;Paused 状态
 * 用 residualMs 把它显式告诉界面,不再假装立即静音。
 *
 * 真人化(humanize 非 null):起音叠加微偏移、长短音动态按住、乐句呼吸,
 * 每次演奏一枚会话级 seed,暂停续播沿用同一份计划;链式防叠键保证实际起音
 * 不早于上一音完全释放。
 */
class PlayerEngine(private val scope: kotlinx.coroutines.CoroutineScope) {

    companion object {
        private const val TAG = "AmpPlay"
    }

    sealed interface State {
        data object Idle : State
        data class Playing(val done: Int, val total: Int) : State
        data class Paused(
            val done: Int,
            val total: Int,
            val residualMs: Long = 0L,
            /** 非空表示是"前台目标丢失"触发的自动暂停,而不是用户按的停止。 */
            val guardNote: String? = null,
        ) : State
        data class Finished(
            val complete: Boolean,
            val error: String?,
            val residualMs: Long = 0L,
        ) : State
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private var job: Job? = null

    @Volatile
    private var lastDone = 0

    @Volatile
    private var lastTotal = 0

    /** 当前在飞手势的最后一次松手时刻(uptime),用于估算停止后的残留按住。 */
    @Volatile
    private var inFlightReleaseMs = 0L

    /** 本次演奏的会话级 seed;暂停续播沿用,重置时丢弃。 */
    @Volatile
    private var planSeed: Long? = null

    val isPlaying: Boolean
        get() = job?.isActive == true

    fun play(
        notes: List<NoteEvent>,
        bpm: Int,
        holdRatio: Double,
        gapMs: Long,
        layout: Map<String, Pair<Float, Float>>,
        screenW: Int,
        screenH: Int,
        startIndex: Int = 0,
        humanize: HumanizeParams? = null,
        /** 允许接收点击的前台包名(目标游戏 + 本应用);为空则不做派发前核对。 */
        allowedForeground: Set<String> = emptySet(),
        /** 读取当前前台包名的回调(由无障碍服务提供)。 */
        foregroundPackage: () -> String? = { null },
    ) {
        if (isPlaying) return
        job = scope.launch {
            run(
                notes, bpm, holdRatio, gapMs, layout, screenW, screenH, startIndex, humanize,
                allowedForeground, foregroundPackage,
            )
        }
    }

    /** 停止(暂停):保留当前进度,之后可 play(startIndex=done) 继续或 reset() 归零。 */
    fun stop() {
        val st = _state.value
        job?.cancel()
        job = null
        val residual = residualHoldMs()
        _state.value = if (st is State.Playing) {
            State.Paused(lastDone, lastTotal, residual)
        } else {
            State.Idle
        }
        Log.i(TAG, "stop paused_at=$lastDone/$lastTotal residual_ms=$residual")
    }

    /** 重置:清空进度回到未开始态(仅暂停态下调用有意义)。 */
    fun reset() {
        job?.cancel()
        job = null
        lastDone = 0
        lastTotal = 0
        inFlightReleaseMs = 0L
        planSeed = null
        _state.value = State.Idle
    }

    /** 距离当前在飞手势自然松手还剩多少毫秒;无在飞事件为 0。 */
    fun residualHoldMs(): Long =
        (inFlightReleaseMs - SystemClock.uptimeMillis()).coerceAtLeast(0L)

    private suspend fun run(
        notes: List<NoteEvent>,
        bpm: Int,
        holdRatio: Double,
        gapMs: Long,
        layout: Map<String, Pair<Float, Float>>,
        screenW: Int,
        screenH: Int,
        startIndex: Int,
        humanize: HumanizeParams?,
        allowedForeground: Set<String>,
        foregroundPackage: () -> String?,
    ) {
        val beatMs = 60000.0 / bpm.coerceAtLeast(1)
        val total = notes.size
        lastTotal = total
        var complete = true
        var errorMsg: String? = null
        var guardNote: String? = null
        var terminalHandled = false

        // 真人化塑形:续播沿用同一枚 seed -> 重放同一份计划(与桌面版一致)
        val timings: List<Timing>? = humanize?.let { params ->
            val carried = planSeed
            val seed = if (startIndex > 0 && carried != null) carried
            else Humanize.makeSeed().also { planSeed = it }
            Humanize.buildPlan(notes, params, seed).timings
        }
        val minGapMs = humanize?.minGapMs ?: 0.0
        if (humanize == null) planSeed = null

        try {
            val planned = schedule(
                notes, beatMs, holdRatio, gapMs, layout, screenW, screenH, startIndex, timings, minGapMs,
            )
            val scheduled = planned.notes
            val batches = GestureBatcher.batch(scheduled)
            val semitoneIgnored = notes.count { it.semitone && it.notes.isNotEmpty() }
            Log.i(
                TAG,
                "plan seed=$planSeed bpm=$bpm elements=$total sounding=${scheduled.size} " +
                    "batches=${batches.size} humanize=${humanize != null} min_gap_ms=$minGapMs " +
                    "allowed_fg=${allowedForeground.joinToString(",")}",
            )
            if (semitoneIgnored > 0) {
                Log.w(
                    TAG,
                    "degraded: $semitoneIgnored 个升半音音符按自然音演奏(21 键档位无半音层," +
                        "与桌面版 default profile 一致)",
                )
            }

            var cancelled = 0
            var maxLateMs = 0L
            for (batch in batches) {
                delayUntil(batch.anchorUptimeMs)
                // 派发前核对前台:注入是按绝对坐标落点的,前台换了就等于往别的应用上点
                val fg = foregroundPackage()
                if (allowedForeground.isNotEmpty() && fg != null && fg !in allowedForeground) {
                    guardNote = "前台已切到 $fg(允许 ${allowedForeground.joinToString()})," +
                        "已停止派发新点击"
                    Log.w(TAG, "guard: $guardNote")
                    break
                }
                val late = SystemClock.uptimeMillis() - batch.anchorUptimeMs
                maxLateMs = maxOf(maxLateMs, late)
                inFlightReleaseMs = batch.lastReleaseUptimeMs
                val completed = TouchInjector.dispatchTimelineAwait(batch.strokes)
                if (completed) {
                    Log.i(
                        TAG,
                        "batch idx=${batch.startIndex}..${batch.endIndex} " +
                            "strokes=${batch.strokes.size} late_ms=$late",
                    )
                } else {
                    cancelled += 1
                    Log.w(
                        TAG,
                        "gesture CANCELLED idx=${batch.startIndex}..${batch.endIndex} " +
                            "late_ms=$late (音被掐断/松手提前)",
                    )
                }
                lastDone = batch.endIndex + 1
                _state.value = State.Playing(lastDone, total)
            }
            if (guardNote != null) {
                terminalHandled = true
                _state.value = State.Paused(lastDone, total, residualHoldMs(), guardNote)
                Log.i(TAG, "guard paused_at=$lastDone/$total")
            } else {
                // 等完尾部休止与最后一个音的自然释放,再宣布演奏完成
                delayUntil(planned.endUptimeMs)
                Log.i(
                    TAG,
                    "finish notes=$lastDone/$total cancelled_gestures=$cancelled max_late_ms=$maxLateMs",
                )
                lastDone = 0
                lastTotal = 0
                inFlightReleaseMs = 0L
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            lastDone = 0
            lastTotal = 0
            complete = false
            errorMsg = e.message ?: e.toString()
            Log.e(TAG, "play aborted: $errorMsg", e)
        } finally {
            // 取消(暂停/重置)与护栏暂停的状态各自负责,这里勿覆盖
            if (!terminalHandled && kotlin.coroutines.coroutineContext.isActive) {
                _state.value = State.Finished(complete, errorMsg, residualHoldMs())
            }
        }
    }

    /**
     * 谱面 -> 绝对时刻表(只含发声元素,休止只推进时间轴)。
     *
     * 全部时刻由 `nextStart` 的纯算术推出,不依赖"实际睡到什么时候",因此预先
     * 排完整首不会引入累计漂移;真实派发误差只会出现在批边界,由日志 late_ms 观测。
     */
    private fun schedule(
        notes: List<NoteEvent>,
        beatMs: Double,
        holdRatio: Double,
        gapMs: Long,
        layout: Map<String, Pair<Float, Float>>,
        screenW: Int,
        screenH: Int,
        startIndex: Int,
        timings: List<Timing>?,
        minGapMs: Double,
    ): Schedule {
        val result = ArrayList<ScheduledNote>()
        var nextStart = SystemClock.uptimeMillis()
        var prevRelease: Long? = null   // 上一发声音符完全释放时刻(链式防叠键下限)
        for (idx in startIndex.coerceIn(0, notes.size) until notes.size) {
            val note = notes[idx]
            val durMs = note.dur * beatMs
            val timing = timings?.get(idx)
            val offMs = timing?.offsetMs ?: 0.0
            val ratio = timing?.holdRatio ?: holdRatio
            // 目标起音 = 理想时刻 + 真人化偏移;链式保护:不早于上一音完全释放
            var target = nextStart + offMs.toLong()
            if (prevRelease != null && note.notes.isNotEmpty()) {
                val lower = prevRelease + Math.ceil(minGapMs).toLong()
                if (target < lower) target = lower
            }
            val coords = KeyPointMap.coordsFor(note.notes, layout, screenW, screenH)
            if (coords.isNotEmpty()) {
                val holdMs = (durMs * ratio).toLong().coerceAtLeast(1L)
                result += ScheduledNote(idx, note.notes, target, holdMs, coords)
                prevRelease = target + holdMs
                nextStart += (durMs + gapMs).toLong()
            } else {
                // 休止符:只占时值,不加 gap(与桌面版一致)
                nextStart += durMs.toLong()
            }
        }
        // 收尾时刻 = 最后一个元素的时值边界与最后一次松手的较晚者(尾部休止也算)
        val endUptime = maxOf(nextStart, prevRelease ?: nextStart)
        return Schedule(result, endUptime)
    }

    private suspend fun delayUntil(target: Long) {
        while (kotlin.coroutines.coroutineContext.isActive) {
            val now = SystemClock.uptimeMillis()
            if (now >= target) return
            delay(target - now)
        }
    }
}

/** 排期结果:发声元素时刻表 + 整段演奏的收尾时刻。 */
private data class Schedule(val notes: List<ScheduledNote>, val endUptimeMs: Long)
