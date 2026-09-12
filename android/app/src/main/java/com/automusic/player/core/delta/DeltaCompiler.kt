package com.automusic.player.core.delta

import com.automusic.player.core.NoteEvent
import com.automusic.player.core.Timing

/**
 * 三角洲口琴谱面编译器:将简谱音符序列编译为带时刻的触摸事件序列。
 *
 * 纯函数无副作用,相同输入产出相同输出。移植自桌面端 core/compiler.py 的核心逻辑,
 * 适配安卓触摸注入模型(状态钮按住态 + 音格点按)。
 *
 * 流程:扁平化(resolve_chord) → 修饰态解析(resolve_modifier) → 时序编译(compile_timeline)
 */
object DeltaCompiler {

    /** 解析后的单音(内部 IR)。 */
    private data class ResolvedNote(
        val index: Int,
        val noteNum: Int?,
        val modifier: Modifier,
        val dur: Double,
        val degradations: List<Degradation>,
    )

    /**
     * 解析 note_id 的八度请求。
     * "low" → LOWER, "mid" → NATURAL, "high" → HIGHER
     */
    private fun octaveToModifier(octave: String): Modifier = when (octave) {
        "low" -> Modifier.LOWER
        "high" -> Modifier.HIGHER
        else -> Modifier.NATURAL
    }

    /**
     * 从 note_id 解析音名(1-7)。
     * "mid_4" → 4, "high_1" → 1, "low_3" → 3
     */
    private fun parseNoteId(noteId: String): Pair<String, Int>? {
        val parts = noteId.split("_")
        if (parts.size != 2) return null
        val num = parts[1].toIntOrNull() ?: return null
        if (num !in 1..7) return null
        return parts[0] to num
    }

    /**
     * 修饰态解析:根据 note_id 的八度 + semitone 字段,按策略解析为单一修饰态。
     *
     * @return (修饰态, 降级记录或 null)
     */
    fun resolveModifier(
        octaveMod: Modifier,
        hasSemitone: Boolean,
        policy: ModifierPolicy,
        index: Int,
    ): Pair<Modifier, Degradation?> {
        val octaveRequested = octaveMod != Modifier.NATURAL
        if (!octaveRequested && !hasSemitone) return Modifier.NATURAL to null
        if (octaveRequested && !hasSemitone) return octaveMod to null
        if (!octaveRequested && hasSemitone) return Modifier.SEMITONE to null

        // 八度 + 半音同时请求,按策略解析
        val requested = "${octaveMod.name}+SEMITONE"
        return when (policy) {
            ModifierPolicy.OCTAVE_FIRST -> octaveMod to Degradation(
                index, requested, octaveMod.name, "OCTAVE_FIRST:保留八度,丢弃半音",
            )
            ModifierPolicy.REJECT_NOTE -> Modifier.NATURAL to Degradation(
                index, requested, "NATURAL", "REJECT_NOTE:冲突拒绝,回自然音",
            )
            ModifierPolicy.KEEP_ACCIDENTAL -> Modifier.SEMITONE to Degradation(
                index, requested, "SEMITONE", "KEEP_ACCIDENTAL:保留半音,丢弃八度",
            )
        }
    }

    /**
     * 和弦降级:三角洲为单旋律乐器,不支持和弦。
     *
     * @return (保留的单音 note_id 或 null=休止, 降级记录或 null)
     */
    fun resolveChord(
        notes: List<String>,
        policy: ChordPolicy,
        index: Int,
    ): Pair<String?, Degradation?> {
        if (notes.size <= 1) return notes.firstOrNull() to null
        val requested = notes.joinToString(",")
        return when (policy) {
            ChordPolicy.CHORD_FIRST -> notes.first() to Degradation(
                index, requested, notes.first(), "CHORD_FIRST:取首音",
            )
            ChordPolicy.CHORD_REJECT -> null to Degradation(
                index, requested, "rest", "CHORD_REJECT:和弦拒绝,降级为休止",
            )
            ChordPolicy.CHORD_ARPEGGIATE -> notes.first() to Degradation(
                index, requested, notes.first(), "CHORD_ARPEGGIATE:拆分未实现,取首音",
            )
        }
    }

    /**
     * 修饰态时序编译:按时序状态机编排触摸事件。
     *
     * 规则:
     * - 修饰态变化时:释放旧状态钮(up) → releaseSettleMs → 按下新状态钮(down) → settleMs
     * - 相邻同态保持按住(不释放不重按)
     * - 音格:down → 按住 holdMs → up
     * - 相邻音符间完全释放后再按下(禁止多点)
     */
    private fun compileTimeline(
        items: List<ResolvedNote>,
        params: DeltaCompileParams,
        layout: DeltaLayout,
        screenW: Int,
        screenH: Int,
    ): List<TouchAction> {
        val events = mutableListOf<TouchAction>()
        var t = 0.0
        var prevMod = Modifier.NATURAL
        var modDown = false

        for (note in items) {
            if (note.noteNum == null) {
                t += note.dur * params.beatMs
                continue
            }

            // 修饰态切换
            if (note.modifier != prevMod) {
                if (modDown) {
                    val modCoord = layout.modifierPixels(prevMod, screenW, screenH)
                    if (modCoord != null) {
                        events.add(TouchAction(t, modCoord.first, modCoord.second, TouchActionType.UP, modifierKey(prevMod)))
                    }
                    t += params.releaseSettleMs
                    modDown = false
                }
                if (note.modifier != Modifier.NATURAL) {
                    val modCoord = layout.modifierPixels(note.modifier, screenW, screenH)
                    if (modCoord != null) {
                        events.add(TouchAction(t, modCoord.first, modCoord.second, TouchActionType.DOWN, modifierKey(note.modifier)))
                    }
                    t += params.settleMs
                    modDown = true
                }
            }

            // 音格按下 → 按住 → 抬起
            val durMs = note.dur * params.beatMs
            val holdMs = minOf(durMs * params.holdRatio, params.maxHoldMs.toDouble())
            val noteCoord = layout.notePixels(note.noteNum, screenW, screenH)
            if (noteCoord != null) {
                events.add(TouchAction(t, noteCoord.first, noteCoord.second, TouchActionType.DOWN, noteKey(note.noteNum)))
                t += holdMs
                events.add(TouchAction(t, noteCoord.first, noteCoord.second, TouchActionType.UP, noteKey(note.noteNum)))
            } else {
                t += holdMs
            }

            // 间隙
            val remaining = durMs - holdMs
            t += maxOf(remaining, params.gapMs.toDouble())

            prevMod = note.modifier
        }

        // 最终释放状态钮
        if (modDown) {
            val modCoord = layout.modifierPixels(prevMod, screenW, screenH)
            if (modCoord != null) {
                events.add(TouchAction(t, modCoord.first, modCoord.second, TouchActionType.UP, modifierKey(prevMod)))
            }
        }

        return events
    }

    private fun modifierKey(mod: Modifier): String = when (mod) {
        Modifier.LOWER -> DeltaKeyPoint.KEY_MOD_LOWER
        Modifier.SEMITONE -> DeltaKeyPoint.KEY_MOD_SEMITONE
        Modifier.HIGHER -> DeltaKeyPoint.KEY_MOD_HIGHER
        Modifier.NATURAL -> "natural"
    }

    private fun noteKey(num: Int): String = "${DeltaKeyPoint.NOTE_PREFIX}$num"

    /**
     * 编译主入口:将简谱音符序列编译为触摸事件序列。
     *
     * @param notes    简谱音符列表
     * @param params   编译参数
     * @param layout   三角洲布局(音格+状态钮坐标)
     * @param screenW  屏幕宽度(像素)
     * @param screenH  屏幕高度(像素)
     * @param timings  真人化偏移(可选,每音符对应一条)
     * @return 编译结果(事件序列 + 降级清单 + 统计)
     */
    fun compile(
        notes: List<NoteEvent>,
        params: DeltaCompileParams,
        layout: DeltaLayout,
        screenW: Int,
        screenH: Int,
        timings: List<Timing>? = null,
    ): DeltaCompileResult {
        require(notes.isNotEmpty()) { "音符列表不能为空" }
        require(screenW > 0 && screenH > 0) { "屏幕尺寸必须为正" }

        val allDegradations = mutableListOf<Degradation>()
        val resolved = mutableListOf<ResolvedNote>()

        for ((idx, note) in notes.withIndex()) {
            // 和弦降级
            val (chordNoteId, chordDeg) = resolveChord(note.notes, params.chordPolicy, idx)
            if (chordDeg != null) allDegradations.add(chordDeg)

            if (chordNoteId == null) {
                resolved.add(ResolvedNote(idx, null, Modifier.NATURAL, note.dur, emptyList()))
                continue
            }

            val parsed = parseNoteId(chordNoteId)
            if (parsed == null) {
                allDegradations.add(Degradation(idx, chordNoteId, "rest", "无法解析 note_id:$chordNoteId"))
                resolved.add(ResolvedNote(idx, null, Modifier.NATURAL, note.dur, emptyList()))
                continue
            }

            val (octave, noteNum) = parsed
            val octaveMod = octaveToModifier(octave)

            // 修饰态解析
            val (modifier, modDeg) = resolveModifier(octaveMod, note.semitone, params.modifierPolicy, idx)
            if (modDeg != null) allDegradations.add(modDeg)

            resolved.add(ResolvedNote(idx, noteNum, modifier, note.dur, emptyList()))
        }

        // 时序编译
        var events = compileTimeline(resolved, params, layout, screenW, screenH)

        // 真人化偏移叠加
        if (timings != null && timings.isNotEmpty()) {
            events = applyTimings(events, timings)
        }

        val durationMs = events.maxOfOrNull { it.tMs } ?: 0.0
        return DeltaCompileResult(events, allDegradations, notes.size, durationMs)
    }

    /** 将真人化偏移叠加到事件时刻。 */
    private fun applyTimings(events: List<TouchAction>, timings: List<Timing>): List<TouchAction> {
        return events.mapIndexed { i, e ->
            val offset = timings.getOrNull(i)?.offsetMs ?: 0.0
            e.copy(tMs = maxOf(0.0, e.tMs + offset))
        }
    }
}