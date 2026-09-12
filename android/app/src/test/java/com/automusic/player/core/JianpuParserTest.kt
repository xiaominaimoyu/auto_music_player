package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class JianpuParserTest {

    @Test
    fun prefixSharp_matchesCanonicalSuffixSharp() {
        assertEquals(
            JianpuParser.parse("1# 2'# 3,#"),
            JianpuParser.parse("#1 #2' #3,"),
        )
    }

    @Test
    fun communityRegions_preservePitchAndOrder() {
        val notes = JianpuParser.parse("1 【2 3】 4 （5 6） (7) #1")
        assertEquals(
            listOf("mid_1", "high_2", "high_3", "mid_4", "low_5", "low_6", "low_7", "mid_1"),
            notes.map { it.notes.single() },
        )
        assertTrue(notes.last().semitone)
    }

    @Test
    fun mixedChordAndSingles_keepTextualOrder() {
        val notes = JianpuParser.parse("1 [2 3] 4")
        assertEquals(listOf("mid_1", "mid_2", "mid_4"), notes.map { it.notes.first() })
    }

    @Test
    fun noteCodec_writesDesktopCompatibleNumberAndReadsLegacyBoolean() {
        val encoded = NoteCodec.encode(listOf(NoteEvent(listOf("mid_1"), 1.0, true)))
        assertTrue(encoded.contains("\"semitone\":1"))
        assertTrue(NoteCodec.decode("[{\"notes\":[\"mid_1\"],\"dur\":1,\"semitone\":true}]").first().semitone)
        assertTrue(NoteCodec.decode("[{\"notes\":[\"mid_1\"],\"dur\":1,\"semitone\":1}]").first().semitone)
        assertFalse(NoteCodec.decode("[{\"notes\":[\"mid_1\"],\"dur\":1}]").first().semitone)
    }
}
