package com.automusic.player.core.delta

import com.automusic.player.core.NoteEvent
import com.automusic.player.input.delta.DeltaTouchExecutor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class IntegrationTest {

    private fun layout(): DeltaLayout = DeltaKeyPoint.buildLayout(DeltaKeyPoint.defaultLayout())

    @Test
    fun fullPipeline_compileToHoldActions_producesCorrectSequence() {
        val notes = listOf(
            NoteEvent(listOf("low_3"), 1.0),
            NoteEvent(listOf("mid_1"), 1.0),
        )
        val params = DeltaCompileParams(bpm = 120)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)

        val scenario = FreePlayScenario()
        val plan = scenario.plan(result.events, null, result.durationMs)

        val holds = DeltaTouchExecutor.toHoldActions(plan.events)
        assertTrue(holds.isNotEmpty())

        val keys = holds.map { it.key }.toSet()
        assertTrue("note_3" in keys)
        assertTrue("note_1" in keys)
        assertTrue("mod_lower" in keys)
    }

    @Test
    fun fullPipeline_npcQuest_appendsSubmitHold() {
        val notes = listOf(NoteEvent(listOf("mid_4"), 1.0))
        val params = DeltaCompileParams(bpm = 120)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)

        val submitCoord = layout().submitPixels(1080, 2400)
        val scenario = NpcQuestScenario()
        val plan = scenario.plan(result.events, submitCoord, result.durationMs)

        val holds = DeltaTouchExecutor.toHoldActions(plan.events)
        assertTrue(holds.any { it.key == DeltaKeyPoint.KEY_SUBMIT })
    }

    @Test
    fun fullPipeline_chordDegradation_propagatesToResult() {
        val notes = listOf(
            NoteEvent(listOf("mid_1", "mid_3", "mid_5"), 1.0),
            NoteEvent(listOf("mid_2"), 0.5),
        )
        val params = DeltaCompileParams(bpm = 100)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        assertTrue(result.degradations.isNotEmpty())
        assertEquals(2, result.noteCount)
    }

    @Test
    fun fullPipeline_holdActions_timeMonotonic() {
        val notes = listOf(
            NoteEvent(listOf("mid_1"), 1.0),
            NoteEvent(listOf("mid_2"), 1.0),
            NoteEvent(listOf("mid_3"), 1.0),
        )
        val params = DeltaCompileParams(bpm = 120)
        val result = DeltaCompiler.compile(notes, params, layout(), 1080, 2400)
        val holds = DeltaTouchExecutor.toHoldActions(result.events)
        for (i in 1 until holds.size) {
            assertTrue("time not monotonic at $i", holds[i].tMs >= holds[i - 1].tMs)
        }
    }
}