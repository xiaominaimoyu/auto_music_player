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
     * Android 无障碍手势不能跨多次 dispatchGesture 保持手指状态，因此每个
     * 逻辑音都编成一个可原子派发的窗口：修饰钮 down → settle → 音格
     * down/up → 修饰钮 up。执行器会把同一 sourceIndex 的多条动作放入同一个
     * GestureDescription，避免新音符取消仍按住的修饰钮。
     */
    private fun compileTimeline(
        items: List<ResolvedNote>,
        params: DeltaCompileParams,
        layout: DeltaLayout,
        screenW: Int,
        screenH: Int,
        timings: List<Timing>?,
    ): List<TouchAction> {
        val events = mutableListOf<TouchAction>()
        // 未加真人化偏移的逻辑时间轴。每音 jitter 只能影响本音，不能累计到后续拍点。
        var cursor = 0.0

        for (note in items) {
            val durMs = note.dur * params.beatMs
            if (note.noteNum == null) {
                cursor += durMs + params.gapMs.toDouble()
                continue
            }

            val timing = timings?.getOrNull(note.index)
            val offsetMs = timing?.offsetMs ?: 0.0
            val holdRatio = timing?.holdRatio ?: params.holdRatio
            val nominalStart = cursor
            val start = maxOf(0.0, nominalStart + offsetMs)
            val noteCoord = layout.notePixels(note.noteNum, screenW, screenH)
            if (noteCoord == null) {
                cursor = maxOf(
                    cursor + durMs + params.gapMs.toDouble(),
                    start + durMs + params.gapMs.toDouble(),
                )
                continue
            }

            val modCoord = if (note.modifier == Modifier.NATURAL) null
                else layout.modifierPixels(note.modifier, screenW, screenH)
            val settle = if (modCoord == null) 0.0 else params.settleMs.toDouble()
            val releaseSettle = if (modCoord == null) 0.0 else params.releaseSettleMs.toDouble()
            val holdMs = minOf(durMs * holdRatio, params.maxHoldMs.toDouble())
            val keyDown = start + settle
            val keyUp = keyDown + holdMs

            if (modCoord != null) {
                events.add(TouchAction(
                    start, modCoord.first, modCoord.second, TouchActionType.DOWN,
                    modifierKey(note.modifier), note.index,
                ))
            }
            events.add(TouchAction(
                keyDown, noteCoord.first, noteCoord.second, TouchActionType.DOWN,
                noteKey(note.noteNum), note.index,
            ))
            events.add(TouchAction(
                keyUp, noteCoord.first, noteCoord.second, TouchActionType.UP,
                noteKey(note.noteNum), note.index,
            ))
            val releaseEnd = keyUp + releaseSettle
            if (modCoord != null) {
                events.add(TouchAction(
                    releaseEnd, modCoord.first, modCoord.second, TouchActionType.UP,
                    modifierKey(note.modifier), note.index,
                ))
            }

            val nominalReleaseEnd = nominalStart + settle + holdMs + releaseSettle
            cursor = maxOf(
                nominalStart + durMs + params.gapMs.toDouble(),
                nominalReleaseEnd + params.gapMs.toDouble(),
                releaseEnd + params.gapMs.toDouble(),
            )
        }
        return events.sortedWith(compareBy<TouchAction> { it.tMs }.thenBy { it.action.ordinal })
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
        val events = compileTimeline(resolved, params, layout, screenW, screenH, timings)

        val durationMs = events.maxOfOrNull { it.tMs } ?: 0.0
        return DeltaCompileResult(events, allDegradations, notes.size, durationMs)
    }

}
