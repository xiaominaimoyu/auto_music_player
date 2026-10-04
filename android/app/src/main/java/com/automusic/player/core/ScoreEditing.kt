package com.automusic.player.core

/**
 * 校对表的可撤销编辑状态。
 *
 * 与桌面版上传页对齐:音符/时值/半音三列可改,可前插后插休止、删除行、追加行、
 * 清空,以及"撤销最后"。每次改动前把整表快照压栈,简单但足够可靠。
 */
data class EditState(val notes: List<NoteEvent>, val undoStack: List<List<NoteEvent>> = emptyList()) {
    val canUndo: Boolean get() = undoStack.isNotEmpty()

    fun undo(): EditState = if (undoStack.isEmpty()) this
    else EditState(undoStack.last(), undoStack.dropLast(1))
}

object ScoreEditing {

    const val MAX_UNDO = 50
    const val HALF_BEAT_REST = 0.5

    private val NOTE_ID_RE = Regex("^(high|mid|low)_([1-7])$")

    fun load(notes: List<NoteEvent>): EditState = EditState(notes, emptyList())

    /**
     * 音符列文本 -> 键位列表。支持 `high_1,mid_3`(逗号/顿号/空格分隔),
     * 空串表示休止。含非法键位时返回 null,由界面提示而不是静默丢弃。
     */
    fun parseCells(text: String): List<String>? {
        val cleaned = text.trim().replace('，', ',').replace('、', ',').replace('+', ',')
        if (cleaned.isEmpty()) return emptyList()
        val ids = cleaned.split(',').map { it.trim() }.filter { it.isNotEmpty() }
        if (ids.isEmpty()) return emptyList()
        if (ids.any { !NOTE_ID_RE.matches(it) }) return null
        return ids.distByOrder()
    }

    /** 时值列文本 -> 拍数。 */
    fun parseDur(text: String): Double? =
        text.trim().toDoubleOrNull()?.takeIf { it.isFinite() && it > 0.0 }

    fun setCells(state: EditState, index: Int, text: String): Pair<EditState, String?> {
        val ids = parseCells(text)
            ?: return state to "音符列非法:\"$text\"(应为 high/mid/low_1~7,和弦用逗号分隔,留空=休止)"
        if (index !in state.notes.indices) return state to "没有选中行"
        val rows = state.notes.toMutableList()
        rows[index] = rows[index].copy(notes = ids)
        return push(state).copy(notes = rows) to null
    }

    fun setDur(state: EditState, index: Int, text: String): Pair<EditState, String?> {
        val dur = parseDur(text)
            ?: return state to "时值必须是大于 0 的数字(拍)"
        if (dur > ScoreValidation.MAX_DUR_BEATS) {
            return state to "时值超出上限(最大 ${ScoreValidation.MAX_DUR_BEATS.toInt()} 拍)"
        }
        if (index !in state.notes.indices) return state to "没有选中行"
        val rows = state.notes.toMutableList()
        rows[index] = rows[index].copy(dur = dur)
        return push(state).copy(notes = rows) to null
    }

    fun toggleSemitone(state: EditState, index: Int): EditState {
        if (index !in state.notes.indices) return state
        val rows = state.notes.toMutableList()
        rows[index] = rows[index].copy(semitone = !rows[index].semitone)
        return push(state).copy(notes = rows)
    }

    /** 在选中行之前/之后插入一个半拍休止(与桌面版"前插半拍休止/后插半拍休止"一致)。 */
    fun insertRest(state: EditState, index: Int, before: Boolean, dur: Double = HALF_BEAT_REST): EditState {
        val at = when {
            state.notes.isEmpty() -> 0
            before -> index.coerceIn(0, state.notes.size)
            else -> (index + 1).coerceIn(0, state.notes.size)
        }
        val rows = state.notes.toMutableList()
        rows.add(at, NoteEvent(emptyList(), dur))
        return push(state).copy(notes = rows)
    }

    fun deleteAt(state: EditState, index: Int): EditState {
        if (index !in state.notes.indices) return state
        val rows = state.notes.toMutableList()
        rows.removeAt(index)
        return push(state).copy(notes = rows)
    }

    /** 追加一行中音 1,方便从空表开始手敲。 */
    fun appendRow(state: EditState): EditState =
        push(state).copy(notes = state.notes + NoteEvent(listOf("mid_1"), 1.0))

    fun clear(state: EditState): EditState =
        if (state.notes.isEmpty()) state else push(state).copy(notes = emptyList())

    private fun push(state: EditState): EditState =
        state.copy(undoStack = (state.undoStack + listOf(state.notes)).takeLast(MAX_UNDO))

    private fun List<String>.distByOrder(): List<String> {
        val seen = LinkedHashSet<String>()
        forEach { seen.add(it) }
        return seen.toList()
    }
}
