package com.automusic.player.core

/**
 * 简谱解析器:规范化简谱文本 -> 结构化音符序列。
 *
 * 输出协议(与大模型 prompt 一致,移植自桌面版 core/parser.py):
 * - 音高:1-7 中音;数字+' 高音;数字+, 低音;0 休止
 * - 时值:无后缀=四分音符(1拍);_ = 八分(0.5拍), __ = 十六分(0.25拍);
 *   - = 二分(2拍), -- = 全音符(4拍);附点用 . 或 · 跟在时值符号后,时值 ×1.5
 * - 和弦:[音1 音2 ...]时值后缀,如 [1' 3' 5']- ;和弦内部只写音高
 * - 小节线 | ‖ 、调号行(1=C)、歌词行自动忽略
 */
object JianpuParser {

    private val CHORD_RE = Regex("""[\[(]([^\])]+)[\])]([_\-.·]*)""")
    private val NOTE_RE = Regex("""[0-7](?:'|,|\.|·|_|-)*""")
    private val TUNE_LINE_RE = Regex("""^\s*1\s*=\s*[A-Ga-g]""")
    private val CHINESE_RE = Regex("""[\u4e00-\u9fff]""")
    private const val PITCH_HIGH = "high"
    private const val PITCH_MID = "mid"
    private const val PITCH_LOW = "low"

    fun parse(text: String): List<NoteEvent> {
        val result = mutableListOf<NoteEvent>()
        for (raw in text.lines()) {
            val line = raw.trim()
            if (line.isEmpty()) continue
            if (TUNE_LINE_RE.containsMatchIn(line) || isLyricLine(line)) continue
            val cleaned = line.replace("|", " ").replace("‖", " ")

            val tokens = mutableListOf<Pair<Int, NoteEvent>>()
            val chordMatches = CHORD_RE.findAll(cleaned).toList()
            val singleScan = CHORD_RE.replace(cleaned, " ")
            for (m in NOTE_RE.findAll(singleScan)) {
                buildSingle(m.value)?.let { tokens.add(m.range.first to it) }
            }
            for (cm in chordMatches) {
                val note = buildChord(cm.groupValues[1], cm.groupValues[2])
                if (note.notes.isNotEmpty()) tokens.add(cm.range.first to note)
            }
            result.addAll(tokens.sortedBy { it.first }.map { it.second })
        }
        return result
    }

    private fun isLyricLine(line: String): Boolean {
        val chinese = CHINESE_RE.findAll(line).count()
        val digits = Regex("[0-7]").findAll(line).count()
        return chinese > digits
    }

    /** 把数字后的修饰符串拆成(音高, 时值部分)。 */
    private fun splitPitchDur(suffix: String): Pair<String, String> {
        var pitch = PITCH_MID
        var rest = suffix
        when {
            rest.startsWith("'") -> {
                pitch = PITCH_HIGH
                rest = rest.substring(1)
            }
            rest.startsWith(",") -> {
                pitch = PITCH_LOW
                rest = rest.substring(1)
            }
            rest.startsWith("·") -> {
                // · 是明确的附点符号,始终作为附点
            }
            rest.startsWith(".") -> {
                // 容错:单独的 . 且后面没有时值符号时,视为低音
                if (rest.length == 1 || rest[1] !in "_-.·") {
                    pitch = PITCH_LOW
                    rest = rest.substring(1)
                }
            }
        }
        return pitch to rest
    }

    private fun parseDur(rest: String): Double {
        var dur = 1.0
        var dotted = false
        for (ch in rest) {
            when (ch) {
                '_' -> dur *= 0.5
                '-' -> dur *= 2.0
                '.', '·' -> dotted = true
            }
        }
        if (dotted) dur *= 1.5
        return dur
    }

    private fun noteId(num: Int, pitch: String): String = "${pitch}_$num"

    private fun buildSingle(token: String): NoteEvent? {
        val num = token[0] - '0'
        val (pitch, rest) = splitPitchDur(token.substring(1))
        val dur = parseDur(rest)
        if (num == 0) return NoteEvent(emptyList(), dur)
        if (num !in 1..7) return null
        return NoteEvent(listOf(noteId(num, pitch)), dur)
    }

    private fun buildChord(inner: String, suffix: String): NoteEvent {
        val dur = parseDur(suffix)
        val ids = mutableListOf<String>()
        for (part in inner.trim().split(Regex("[,\\s]+"))) {
            val m = NOTE_RE.matchEntire(part) ?: NOTE_RE.find(part) ?: continue
            val token = m.value
            val num = token[0] - '0'
            if (num == 0 || num !in 1..7) continue
            val (pitch, _) = splitPitchDur(token.substring(1))
            ids.add(noteId(num, pitch))
        }
        return NoteEvent(ids, dur)
    }
}
