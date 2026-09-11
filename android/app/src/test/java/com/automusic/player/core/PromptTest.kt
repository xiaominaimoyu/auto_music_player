package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Test

class PromptTest {
    @Test
    fun structuredResultWithEmptyJianpuStaysEmpty() {
        val result = Prompt.parseRecognitionResult(
            "TITLE: 导入曲\nJIANPU:\nLYRICS: UNKNOWN\nRHYTHM: OK"
        )
        assertEquals("导入曲", result.title)
        assertEquals("", result.jianpu)
    }
}
