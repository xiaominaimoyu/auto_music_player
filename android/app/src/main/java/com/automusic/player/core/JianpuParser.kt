package com.automusic.player.core

/**
 * 简谱解析器:规范化简谱文本 -> 结构化音符序列。
 *
 * 输出协议(与大模型 prompt 一致,移植自桌面版 core/parser.py):
 * - 音高:1-7 中音;数字+' 高音;数字+, 低音;0 休止；升半音可写 1# 或 #1
 * - 时值:无后缀=四分音符(1拍);_ = 八分(0.5拍), __ = 十六分(0.25拍);
 *   - = 二分(2拍), -- = 全音符(4拍);附点用 . 或 · 跟在时值符号后,时值 ×1.5
 * - 和弦:[音1 音2 ...]时值后缀,如 [1' 3' 5']- ;和弦内部只写音高
 * - 小节线 | ‖ 、调号行(1=C)、歌词行自动忽略
 */
object JianpuParser {

    private val CHORD_RE = Regex("""[\[(]([^\])]+)[\])]([_\-.·]*)""")
    private val NOTE_RE = Regex("""#?[0-7](?:'|,|\.|·|_|-|#)*""")
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
            val cleaned = normalizeDeltaCommunityNotation(
                line.replace("|", " ").replace("‖", " ")
            )
            val consumed = BooleanArray(cleaned.length)
            val tokens = mutableListOf<Pair<Int, NoteEvent>>()

            for (cm in CHORD_RE.findAll(cleaned)) {
                for (i in cm.range) consumed[i] = true
                val note = buildChord(cm.groupValues[1], cm.groupValues[2])
                if (note.notes.isNotEmpty()) tokens.add(cm.range.first to note)
            }
            for (m in NOTE_RE.findAll(cleaned)) {
                if ((m.range.first..m.range.last).any { consumed[it] }) continue
                buildSingle(m.value)?.let { tokens.add(m.range.first to it) }
            }
            tokens.sortBy { it.first }
            result.addAll(tokens.map { it.second })
        }
        return result
    }

    private fun isLyricLine(line: String): Boolean {
        val chinese = CHINESE_RE.findAll(line).count()
        val digits = Regex("[0-7]").findAll(line).count()
        return chinese > digits
    }

    /** 把数字后的修饰符串拆成(音高, 时值部分, 半音标记)。 */
    private fun splitPitchDur(suffix: String): Triple<String, String, Boolean> {
        var pitch = PITCH_MID
        var rest = suffix
        var semitone = false
        var hasOctave = false
        if (rest.startsWith("#")) {
            semitone = true
            rest = rest.substring(1)
        }
        if (rest.startsWith("'")) {
            pitch = PITCH_HIGH
            rest = rest.substring(1)
            hasOctave = true
        } else if (rest.startsWith(",")) {
            pitch = PITCH_LOW
            rest = rest.substring(1)
            hasOctave = true
        }
        if (rest.contains("#")) {
            semitone = true
            rest = rest.replace("#", "")
        }
        if (rest == ".") {
            if (!hasOctave) pitch = PITCH_LOW
            rest = ""
        }
        return Triple(pitch, rest, semitone)
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

    private fun canonicalNoteToken(token: String): String =
        if (token.startsWith("#")) token.substring(1) + "#" else token

    private fun withOctave(token: String, marker: Char): String {
        val prefix = if (token.startsWith("#")) "#" else ""
        val body = token.removePrefix(prefix)
        if (body.length > 1 && (body[1] == '\'' || body[1] == ',')) return token
        return prefix + body[0] + marker + body.substring(1)
    }

    /** 与桌面端一致：兼容三角洲社区谱的高低音分区和 #1 写法。 */
    private fun normalizeDeltaCommunityNotation(line: String): String {
        fun normalizeRegion(inner: String, marker: Char): String =
            NOTE_RE.replace(inner) { withOctave(it.value, marker) }

        var out = Regex("【([^】]+)】").replace(line) {
            normalizeRegion(it.groupValues[1], '\'')
        }
        out = Regex("（([^）]+)）").replace(out) {
            normalizeRegion(it.groupValues[1], ',')
        }
        return Regex("""\((#?[0-7](?:'|,|\.|·|_|-|#)*)\)""").replace(out) {
            withOctave(it.groupValues[1], ',')
        }
    }

    private fun buildSingle(token: String): NoteEvent? {
        val canonical = canonicalNoteToken(token)
        val num = canonical[0] - '0'
        val (pitch, rest, semitone) = splitPitchDur(canonical.substring(1))
        val dur = parseDur(rest)
        if (num == 0) return NoteEvent(emptyList(), dur)
        if (num !in 1..7) return null
        return NoteEvent(listOf(noteId(num, pitch)), dur, semitone)
    }

    private fun buildChord(inner: String, suffix: String): NoteEvent {
        val dur = parseDur(suffix)
        val ids = mutableListOf<String>()
        var semitone = false
        for (part in inner.trim().split(Regex("[,\\s]+"))) {
            val m = NOTE_RE.matchEntire(part) ?: NOTE_RE.find(part) ?: continue
            val token = canonicalNoteToken(m.value)
            val num = token[0] - '0'
            if (num == 0 || num !in 1..7) continue
            val (pitch, _, sharp) = splitPitchDur(token.substring(1))
            ids.add(noteId(num, pitch))
            semitone = semitone || sharp
        }
        return NoteEvent(ids, dur, semitone)
    }
}
