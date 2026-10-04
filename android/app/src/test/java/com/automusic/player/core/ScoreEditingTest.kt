package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** 校对表编辑操作单测:桌面版上传页的插删/改格/撤销语义在安卓端的等价实现。 */
class ScoreEditingTest {

    private fun rows(vararg keys: String) = keys.map { NoteEvent(listOf(it), 1.0) }

    @Test
    fun `单元格解析支持和弦与休止并拒绝非法键位`() {
        assertEquals(listOf("high_1", "mid_3"), ScoreEditing.parseCells("high_1,mid_3"))
        assertEquals(listOf("mid_1", "mid_3"), ScoreEditing.parseCells(" mid_1、mid_3 "))
        assertEquals(listOf("mid_1", "mid_3"), ScoreEditing.parseCells("mid_1+mid_3"))
        assertEquals(emptyList<String>(), ScoreEditing.parseCells("   "))
        assertNull(ScoreEditing.parseCells("mid_8"))
        assertNull(ScoreEditing.parseCells("sub_1"))
    }

    @Test
    fun `改音符与改时值失败时不改动状态`() {
        val s = ScoreEditing.load(rows("mid_1", "mid_2"))
        val (after, err) = ScoreEditing.setCells(s, 0, "mid_5,mid_7")
        assertNull(err)
        assertEquals(listOf("mid_5", "mid_7"), after.notes[0].notes)
        assertEquals(1.0, after.notes[0].dur, 1e-9)

        val (same, bad) = ScoreEditing.setCells(s, 0, "mid_9")
        assertNotNull(bad)
        assertEquals(s, same)

        val (durOk, durErr) = ScoreEditing.setDur(s, 1, "0.5")
        assertNull(durErr)
        assertEquals(0.5, durOk.notes[1].dur, 1e-9)
        val (_, durBad) = ScoreEditing.setDur(s, 1, "0")
        assertNotNull(durBad)
        val (_, durTooLong) = ScoreEditing.setDur(s, 1, "17")
        assertNotNull(durTooLong)
    }

    @Test
    fun `前插与后插半拍休止位置正确`() {
        val s = ScoreEditing.load(rows("mid_1", "mid_2"))
        val before = ScoreEditing.insertRest(s, 1, before = true)
        assertEquals(3, before.notes.size)
        assertTrue(before.notes[1].notes.isEmpty())
        assertEquals(0.5, before.notes[1].dur, 1e-9)
        assertEquals("mid_2", before.notes[2].notes.single())

        val after = ScoreEditing.insertRest(s, 0, before = false)
        assertTrue(after.notes[1].notes.isEmpty())
        assertEquals("mid_1", after.notes[0].notes.single())
    }

    @Test
    fun `删除追加清空都能撤销`() {
        var s = ScoreEditing.load(rows("mid_1", "mid_2", "mid_3"))
        s = ScoreEditing.deleteAt(s, 1)
        assertEquals(listOf("mid_1", "mid_3"), s.notes.map { it.notes.single() })
        assertTrue(s.canUndo)
        s = s.undo()
        assertEquals(3, s.notes.size)
        assertFalse(s.canUndo)

        s = ScoreEditing.appendRow(s)
        assertEquals(4, s.notes.size)
        s = ScoreEditing.clear(s)
        assertTrue(s.notes.isEmpty())
        assertEquals(4, s.undo().notes.size)
    }

    @Test
    fun `撤销栈只保留最近 50 步`() {
        var s = ScoreEditing.load(rows("mid_1"))
        repeat(ScoreEditing.MAX_UNDO + 20) { s = ScoreEditing.toggleSemitone(s, 0) }
        assertEquals(ScoreEditing.MAX_UNDO, s.undoStack.size)
    }

    @Test
    fun `升号切换只影响选中行`() {
        val s = ScoreEditing.load(rows("mid_1", "mid_2"))
        val t = ScoreEditing.toggleSemitone(s, 1)
        assertFalse(t.notes[0].semitone)
        assertTrue(t.notes[1].semitone)
    }
}
