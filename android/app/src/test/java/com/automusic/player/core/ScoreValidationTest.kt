package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScoreValidationTest {

    private fun note(vararg keys: String, dur: Double = 1.0, semi: Boolean = false) =
        NoteEvent(keys.toList(), dur, semi)

    @Test
    fun `合法谱面没有错误`() {
        val v = ScoreValidation.validate(
            listOf(note("mid_1"), note("high_5"), note(dur = 1.0)), 100,
        )
        assertTrue(v.errors.joinToString(), v.errors.isEmpty())
    }

    @Test
    fun `空谱与全休止被拒绝`() {
        assertFalse(ScoreValidation.validate(emptyList(), 100).ok)
        assertFalse(ScoreValidation.validate(listOf(note(dur = 1.0)), 100).ok)
    }

    @Test
    fun `时值越界逐项报错`() {
        val bad = listOf(
            note("mid_1", dur = 0.0),
            note("mid_1", dur = -1.0),
            note("mid_1", dur = 17.0),
            note("mid_1", dur = Double.NaN),
        )
        val v = ScoreValidation.validate(bad, 100)
        assertEquals(4, v.errors.size)
        assertEquals(listOf(0, 1, 2, 3), v.errors.map { it.elementIndex })
    }

    @Test
    fun `非法键位被拒绝`() {
        val v = ScoreValidation.validate(listOf(note("sub_1", "mid_8")), 100)
        assertEquals(2, v.errors.size)
        assertTrue(v.errors.all { "high/mid/low" in it.message })
    }

    @Test
    fun `BPM 超出允许范围被拒绝`() {
        val notes = listOf(note("mid_1"))
        assertFalse(ScoreValidation.validate(notes, 29).ok)
        assertTrue(ScoreValidation.validate(notes, 30).ok)
        assertTrue(ScoreValidation.validate(notes, 300).ok)
        assertFalse(ScoreValidation.validate(notes, 301).ok)
    }

    @Test
    fun `档位无半音层时升号只给警告不给错误`() {
        val notes = listOf(note("mid_1", semi = true), note("mid_2"))
        val v = ScoreValidation.validate(notes, 100, semitoneSupported = false)
        assertTrue(v.ok)
        assertTrue(v.warnings.any { "降为自然音" in it.message || "自然音" in it.message })
        assertTrue(ScoreValidation.validate(notes, 100, semitoneSupported = true).warnings.none { "自然音" in it.message })
    }
}
