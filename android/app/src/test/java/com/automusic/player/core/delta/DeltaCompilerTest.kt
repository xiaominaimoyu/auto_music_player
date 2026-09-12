package com.automusic.player.core.delta

import com.automusic.player.core.NoteEvent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DeltaCompilerTest {

    private fun layout(): DeltaLayout = DeltaKeyPoint.buildLayout(DeltaKeyPoint.defaultLayout())

    @Test
    fun resolveModifier_natural_noDegradation() {
        val (mod, deg) = DeltaCompiler.resolveModifier(Modifier.NATURAL, false, ModifierPolicy.OCTAVE_FIRST, 0)
        assertEquals(Modifier.NATURAL, mod)
        assertNull(deg)
    }

    @Test
    fun resolveModifier_lowerOnly() {
        val (mod, deg) = DeltaCompiler.resolveModifier(Modifier.LOWER, false, ModifierPolicy.OCTAVE_FIRST, 0)
        assertEquals(Modifier.LOWER, mod)
        assertNull(deg)
    }

    @Test
    fun resolveModifier_semitoneOnly() {
        val (mod, deg) = DeltaCompiler.resolveModifier(Modifier.NATURAL, true, ModifierPolicy.OCTAVE_FIRST, 0)
        assertEquals(Modifier.SEMITONE, mod)
        assertNull(deg)
    }

    @Test
    fun resolveModifier_octaveFirst_keepsOctave() {
        val (mod, deg) = DeltaCompiler.resolveModifier(Modifier.LOWER, true, ModifierPolicy.OCTAVE_FIRST, 0)
        assertEquals(Modifier.LOWER, mod)
        assertNotNull(deg)
        assertTrue(deg!!.reason.contains("OCTAVE_FIRST"))
    }

    @Test
    fun resolveModifier_rejectNote_returnsNatural() {
        val (mod, deg) = DeltaCompiler.resolveModifier(Modifier.HIGHER, true, ModifierPolicy.REJECT_NOTE, 0)
        assertEquals(Modifier.NATURAL, mod)
        assertNotNull(deg)
    }

    @Test
    fun resolveModifier_keepAccidental_keepsSemitone() {
        val (mod, deg) = DeltaCompiler.resolveModifier(Modifier.LOWER, true, ModifierPolicy.KEEP_ACCIDENTAL, 0)
        assertEquals(Modifier.SEMITONE, mod)
        assertNotNull(deg)
    }

    @Test
    fun resolveChord_singleNote_noDegradation() {
        val (note, deg) = DeltaCompiler.resolveChord(listOf("mid_3"), ChordPolicy.CHORD_FIRST, 0)
        assertEquals("mid_3", note)
        assertNull(deg)
    }

    @Test
    fun resolveChord_chordFirst_takesFirst() {
        val (note, deg) = DeltaCompiler.resolveChord(listOf("high_1", "high_3", "high_5"), ChordPolicy.CHORD_FIRST, 0)
        assertEquals("high_1", note)
        assertNotNull(deg)
    }

    @Test
    fun resolveChord_reject_returnsNull() {
        val (note, deg) = DeltaCompiler.resolveChord(listOf("mid_1", "mid_3"), ChordPolicy.CHORD_REJECT, 0)
        assertNull(note)
        assertNotNull(deg)
    }

    @Test
    fun compile_simpleNote_producesDownUp() {
        val notes = listOf(NoteEvent(listOf("mid_4"), 1.0))
        val params = DeltaCompileParams(bpm = 120)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        assertEquals(1, result.noteCount)
        assertTrue(result.events.isNotEmpty())
        assertTrue(result.degradations.isEmpty())
    }

    @Test
    fun compile_lowerThenNatural_producesModifierSequence() {
        val notes = listOf(
            NoteEvent(listOf("low_3"), 1.0),
            NoteEvent(listOf("mid_1"), 1.0),
        )
        val params = DeltaCompileParams(bpm = 120)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        val downs = result.events.filter { it.action == TouchActionType.DOWN }
        assertTrue(downs.any { it.key == "mod_lower" })
        assertTrue(downs.any { it.key == "note_3" })
        assertTrue(downs.any { it.key == "note_1" })
    }

    @Test
    fun compile_chord_producesDegradation() {
        val notes = listOf(NoteEvent(listOf("mid_1", "mid_3", "mid_5"), 1.0))
        val params = DeltaCompileParams(bpm = 120)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        assertTrue(result.degradations.isNotEmpty())
    }

    @Test
    fun compile_emptyNotes_throws() {
        try {
            DeltaCompiler.compile(emptyList(), DeltaCompileParams(), layout(), 1080, 2400)
            assert(false) { "应抛出 IllegalArgumentException" }
        } catch (e: IllegalArgumentException) {
        }
    }

    @Test
    fun compile_sameInput_sameOutput() {
        val notes = listOf(NoteEvent(listOf("mid_4"), 1.0), NoteEvent(listOf("high_1"), 0.5))
        val params = DeltaCompileParams(bpm = 100)
        val r1 = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        val r2 = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        assertEquals(r1.events.size, r2.events.size)
        assertEquals(r1.durationMs, r2.durationMs, 0.01)
    }
}