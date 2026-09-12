package com.automusic.player.core.delta

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ScenarioTest {

    @Test
    fun freePlay_interruptMode_isPause() {
        val s = FreePlayScenario()
        assertEquals(InterruptMode.PAUSE, s.interruptMode)
    }

    @Test
    fun freePlay_humanize_isTrue() {
        val s = FreePlayScenario()
        assertTrue(s.humanize)
    }

    @Test
    fun freePlay_intervals_unchanged() {
        val s = FreePlayScenario()
        val base = DeltaCompileParams(settleMs = 30, gapMs = 20)
        val result = s.intervals(base)
        assertEquals(30L, result.settleMs)
        assertEquals(20L, result.gapMs)
    }

    @Test
    fun freePlay_plan_noSubmitAction() {
        val s = FreePlayScenario()
        val events = listOf(
            TouchAction(0.0, 100f, 200f, TouchActionType.DOWN, "note_1"),
            TouchAction(500.0, 100f, 200f, TouchActionType.UP, "note_1"),
        )
        val plan = s.plan(events, submitCoord = 500f to 600f, lastTMs = 500.0)
        assertEquals(events.size, plan.events.size)
        assertEquals("", plan.abortHint)
    }

    @Test
    fun npcQuest_interruptMode_isAbort() {
        val s = NpcQuestScenario()
        assertEquals(InterruptMode.ABORT, s.interruptMode)
    }

    @Test
    fun npcQuest_humanize_isFalse() {
        val s = NpcQuestScenario()
        assertFalse(s.humanize)
    }

    @Test
    fun npcQuest_intervals_conservativeValues() {
        val s = NpcQuestScenario()
        val base = DeltaCompileParams(settleMs = 30, gapMs = 20)
        val result = s.intervals(base)
        assertEquals(60L, result.settleMs)
        assertEquals(120L, result.gapMs)
    }

    @Test
    fun npcQuest_intervals_preserveSlowerUserCalibration() {
        val s = NpcQuestScenario()
        val result = s.intervals(DeltaCompileParams(settleMs = 123L, gapMs = 177L))
        assertEquals(123L, result.settleMs)
        assertEquals(177L, result.gapMs)
    }

    @Test
    fun npcQuest_plan_appendsSubmitAction() {
        val s = NpcQuestScenario()
        val events = listOf(
            TouchAction(0.0, 100f, 200f, TouchActionType.DOWN, "note_1"),
            TouchAction(500.0, 100f, 200f, TouchActionType.UP, "note_1"),
        )
        val plan = s.plan(events, submitCoord = 500f to 600f, lastTMs = 500.0)
        assertTrue(plan.events.size > events.size)
        val submitActions = plan.events.filter { it.key == DeltaKeyPoint.KEY_SUBMIT }
        assertTrue(submitActions.isNotEmpty())
    }

    @Test
    fun npcQuest_plan_noSubmitCoord_noAppend() {
        val s = NpcQuestScenario()
        val events = listOf(
            TouchAction(0.0, 100f, 200f, TouchActionType.DOWN, "note_1"),
            TouchAction(500.0, 100f, 200f, TouchActionType.UP, "note_1"),
        )
        val plan = s.plan(events, submitCoord = null, lastTMs = 500.0)
        assertEquals(events.size, plan.events.size)
    }

    @Test
    fun npcQuest_abortHint_notEmpty() {
        val s = NpcQuestScenario()
        assertNotNull(s.abortHint)
        assertTrue(s.abortHint.isNotEmpty())
    }
}
