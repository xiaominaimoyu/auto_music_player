package com.automusic.player.core.delta

import com.automusic.player.core.NoteEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 三角洲档琶音展开。桌面版 core/compiler.py 早就把和弦拆成时值均分的单音序列,
 * 安卓侧此前是"取首音"的占位,两端行为不一致。
 */
class DeltaArpeggiateTest {

    private fun layout() = DeltaKeyPoint.buildLayout(DeltaKeyPoint.defaultLayout())

    private fun noteDowns(events: List<TouchAction>) =
        events.filter { it.action == TouchActionType.DOWN && it.key.startsWith("note_") }

    @Test
    fun `和弦按拍序均分展开为单音序列`() {
        val notes = listOf(NoteEvent(listOf("mid_1", "mid_3", "mid_5"), 3.0))
        val params = DeltaCompileParams(bpm = 120, chordPolicy = ChordPolicy.CHORD_ARPEGGIATE)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)

        val downs = noteDowns(result.events)
        assertEquals("三个音应各自落指一次", 3, downs.size)
        assertEquals(listOf("note_1", "note_3", "note_5"), downs.map { it.key })
        // 每子音 1 拍 = 500ms,再加 20ms 间隙
        assertEquals(listOf(0.0, 520.0, 1040.0), downs.map { it.tMs })
        assertTrue(result.degradations.any { it.actual.startsWith("arpeggio") })
    }

    @Test
    fun `单音不受琶音策略影响`() {
        val notes = listOf(NoteEvent(listOf("mid_1"), 1.0))
        val params = DeltaCompileParams(bpm = 120, chordPolicy = ChordPolicy.CHORD_ARPEGGIATE)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)

        assertTrue(result.degradations.isEmpty())
        assertEquals(1, noteDowns(result.events).size)
    }

    @Test
    fun `子音共享同一 sourceIndex 以便装进同一条原子手势`() {
        val notes = listOf(NoteEvent(listOf("mid_1", "mid_5"), 2.0))
        val params = DeltaCompileParams(bpm = 120, chordPolicy = ChordPolicy.CHORD_ARPEGGIATE)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)

        assertEquals(setOf(0), result.events.mapNotNull { it.sourceIndex }.toSet())
    }

    @Test
    fun `首音策略仍然只取一个音`() {
        val notes = listOf(NoteEvent(listOf("mid_1", "mid_5"), 1.0))
        val result = DeltaCompiler.compile(
            notes, DeltaCompileParams(bpm = 120), layout(), 1080, 2400,
        )
        assertEquals(1, noteDowns(result.events).size)
        assertEquals(1, result.degradations.size)
    }
}
