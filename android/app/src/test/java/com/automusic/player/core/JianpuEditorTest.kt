package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Test

class JianpuEditorTest {
    @Test
    fun insertsHalfBeatRestBeforeOrAfterSelectedEvent() {
        val source = "6 5 3 5 3\n6 5 6"
        assertEquals(
            "6 5 3 5 3\n0_ 6 5 6",
            JianpuEditor.insertRest(source, eventIndex = 5, before = true),
        )
        assertEquals(
            "6 5 3 5 3 0_\n6 5 6",
            JianpuEditor.insertRest(source, eventIndex = 4, before = false),
        )
    }

    @Test
    fun deletesSelectedEventWithoutJoiningNeighbors() {
        assertEquals("6 3\n2 1", JianpuEditor.deleteEvent("6 5 3\n2 1", 1))
        assertEquals("6 5\n2 1", JianpuEditor.deleteEvent("6 5 3\n2 1", 2))
    }

    @Test
    fun parserPreservesChordAndSingleNoteOrder() {
        val notes = JianpuParser.parse("1 [3 5] 2")
        assertEquals(listOf("mid_1"), notes[0].notes)
        assertEquals(listOf("mid_3", "mid_5"), notes[1].notes)
        assertEquals(listOf("mid_2"), notes[2].notes)
    }

    @Test
    fun lyricsAlignPerLineAndLeaveMismatchedLineBlank() {
        assertEquals(
            listOf("人", "间", "", "琴", "悠", "扬", "", "", ""),
            JianpuEditor.alignLyrics(
                "6 5 0_ 3 5 3\n6 5 3",
                listOf("人 间 琴 悠 扬", "数 量 不 对"),
            ),
        )
    }

    @Test
    fun savedGlobalLyricsRestoreAcrossScoreLines() {
        assertEquals(
            listOf("人", "间", "", "琴", "悠", "扬"),
            JianpuEditor.alignLyrics("6 5 0_\n3 2 1", listOf("人 间 _ 琴 悠 扬")),
        )
    }
}
