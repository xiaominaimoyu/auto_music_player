package com.automusic.player.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScoreJsonTest {

    @Test
    fun `识别 json 与简谱文本`() {
        assertTrue(ScoreJson.looksLikeJson("  {\"notes\":[]}"))
        assertTrue(ScoreJson.looksLikeJson("[{\"notes\":[]}]"))
        assertFalse(ScoreJson.looksLikeJson("1 2 3 4 5"))
    }

    @Test
    fun `解析桌面导出的完整格式`() {
        val text = """
            {"format":"auto-music-player-score","version":1,"name":"小星星","bpm":88,
             "notes":[{"notes":["mid_1"],"dur":1},{"notes":["mid_1"],"dur":1},
                      {"notes":[],"dur":1},{"notes":["mid_3","mid_5"],"dur":2,"semitone":1}]}
        """.trimIndent()
        val p = ScoreJson.parse(text)
        assertEquals("小星星", p.name)
        assertEquals(88, p.bpm)
        assertEquals(4, p.notes.size)
        assertEquals(listOf("mid_1"), p.notes[0].notes)
        assertTrue(p.notes[2].notes.isEmpty())
        assertEquals(2.0, p.notes[3].dur, 1e-9)
        assertTrue(p.notes[3].semitone)
    }

    @Test
    fun `接受裸数组并回落到默认 bpm`() {
        val p = ScoreJson.parse("""[{"notes":["mid_5"],"dur":0.5}]""")
        assertEquals(100, p.bpm)
        assertEquals("", p.name)
        assertEquals(0.5, p.notes.single().dur, 1e-9)
    }

    @Test
    fun `半音字段兼容 0 1 true 与字符串`() {
        fun semi(literal: String) =
            ScoreJson.parse("""[{"notes":["mid_1"],"dur":1,"semitone":$literal}]""").notes.single().semitone
        assertTrue(semi("1"))
        assertFalse(semi("0"))
        assertTrue(semi("true"))
        assertTrue(semi("\"1\""))
        assertFalse(semi("\"0\""))
    }

    @Test
    fun `不受支持的格式被拒绝`() {
        val e = runCatching { ScoreJson.parse("""{"format":"other","notes":[]}""") }
            .exceptionOrNull()
        assertTrue(e is ScoreJsonException)
        assertTrue(e!!.message!!.contains("不支持的乐谱格式"))
    }

    @Test
    fun `缺字段与类型错误带元素序号`() {
        val missingDur = runCatching { ScoreJson.parse("""[{"notes":["mid_1"],"dur":1},{"notes":["mid_2"]}]""") }
            .exceptionOrNull()
        assertTrue(missingDur is ScoreJsonException)
        assertTrue(missingDur!!.message!!.contains("第 2 个元素"))

        val missingNotes = runCatching { ScoreJson.parse("""[{"dur":1}]""") }.exceptionOrNull()
        assertTrue(missingNotes!!.message!!.contains("缺少 notes"))
    }

    @Test
    fun `非法 json 报解析失败而不是崩溃`() {
        val e = runCatching { ScoreJson.parse("{not json") }.exceptionOrNull()
        assertTrue(e is ScoreJsonException)
        assertTrue(e!!.message!!.contains("不是有效的 JSON"))
    }

    @Test
    fun `编码后再解析结果一致`() {
        val notes = listOf(
            NoteEvent(listOf("mid_1"), 1.0),
            NoteEvent(emptyList(), 2.0),
            NoteEvent(listOf("high_5", "mid_3"), 0.5, true),
        )
        val json = """{"name":"t","bpm":120,"notes":${NoteCodec.encode(notes)}}"""
        val p = ScoreJson.parse(json)
        assertEquals(notes, p.notes)
        assertEquals(120, p.bpm)
    }
}
