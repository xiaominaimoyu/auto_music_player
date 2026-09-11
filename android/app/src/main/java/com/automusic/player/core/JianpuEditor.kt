package com.automusic.player.core

/** 对规范化简谱执行位置稳定的编辑操作。 */
object JianpuEditor {
    private val chordRegex = Regex("""[\[(]([^\])]+)[\])]([_\-.·]*)""")
    private val noteRegex = Regex("""[0-7](?:'|,|\.|·|_|-)*""")
    private val tuneLineRegex = Regex("""^\s*1\s*=\s*[A-Ga-g]""")
    private val chineseRegex = Regex("""[\u4e00-\u9fff]""")

    /** 在第 [eventIndex] 个解析音符前/后插入休止，并保留原有换行。 */
    fun insertRest(text: String, eventIndex: Int, before: Boolean, restToken: String = "0_"): String {
        val ranges = eventRanges(text)
        if (eventIndex !in ranges.indices) return text
        val range = ranges[eventIndex]
        val position = if (before) range.first else range.last + 1
        val leftNeedsSpace = position > 0 && !text[position - 1].isWhitespace()
        val rightNeedsSpace = position < text.length && !text[position].isWhitespace()
        val insertion = buildString {
            if (leftNeedsSpace) append(' ')
            append(restToken)
            if (rightNeedsSpace) append(' ')
        }
        return text.substring(0, position) + insertion + text.substring(position)
    }

    /** 删除指定音符事件，并清理相邻的水平空白。 */
    fun deleteEvent(text: String, eventIndex: Int): String {
        val ranges = eventRanges(text)
        if (eventIndex !in ranges.indices) return text
        var start = ranges[eventIndex].first
        var end = ranges[eventIndex].last + 1
        val originalEnd = end
        while (end < text.length && text[end] in " \t") end += 1
        if (end == originalEnd) {
            while (start > 0 && text[start - 1] in " \t") start -= 1
        }
        return text.substring(0, start) + text.substring(end)
    }

    /** 逐谱行严格对齐歌词；数量不匹配的行全部留空，避免歌词串位。 */
    fun alignLyrics(jianpu: String, lyricsLines: List<String>): List<String> {
        val scoreLines = jianpu.lines().filter(String::isNotBlank)
        val eventsByLine = scoreLines.map(JianpuParser::parse)
        if (lyricsLines.size == 1 && scoreLines.size > 1) {
            val tokens = lyricTokens(lyricsLines[0])
            val allEvents = eventsByLine.flatten()
            val soundingCount = allEvents.count { it.notes.isNotEmpty() }
            if (tokens.size == allEvents.size) {
                return tokens.map { if (it == "_") "" else it }
            }
            if (tokens.size == soundingCount) {
                var tokenIndex = 0
                return allEvents.map { event ->
                    if (event.notes.isEmpty()) "" else tokens[tokenIndex++].takeUnless { it == "_" }.orEmpty()
                }
            }
        }
        val aligned = mutableListOf<String>()
        for ((lineIndex, events) in eventsByLine.withIndex()) {
            val tokens = lyricTokens(lyricsLines.getOrNull(lineIndex).orEmpty())
            val soundingCount = events.count { it.notes.isNotEmpty() }
            when {
                tokens.size == events.size -> aligned.addAll(tokens.map { if (it == "_") "" else it })
                tokens.size == soundingCount -> {
                    var tokenIndex = 0
                    events.forEach { event ->
                        aligned.add(if (event.notes.isEmpty()) "" else tokens[tokenIndex++].takeUnless { it == "_" }.orEmpty())
                    }
                }
                else -> repeat(events.size) { aligned.add("") }
            }
        }
        return aligned
    }

    private fun lyricTokens(value: String): List<String> {
        val tokens = value.trim().split(Regex("\\s+")).filter(String::isNotBlank)
        return if (tokens.size == 1 && tokens[0].length > 1) tokens[0].map(Char::toString) else tokens
    }

    private fun eventRanges(text: String): List<IntRange> {
        val result = mutableListOf<IntRange>()
        var lineStart = 0
        for (lineWithBreak in text.splitToSequence(Regex("(?<=\\n)"))) {
            val line = lineWithBreak.removeSuffix("\n").removeSuffix("\r")
            val trimmed = line.trim()
            if (trimmed.isNotEmpty() && !tuneLineRegex.containsMatchIn(trimmed) && !isLyricLine(trimmed)) {
                val occupied = BooleanArray(line.length)
                val local = mutableListOf<IntRange>()
                for (match in chordRegex.findAll(line)) {
                    val range = match.range
                    for (index in range) occupied[index] = true
                    local.add(range)
                }
                for (match in noteRegex.findAll(line)) {
                    if (match.range.none { occupied[it] }) local.add(match.range)
                }
                local.sortedBy { it.first }.forEach { range ->
                    result.add((lineStart + range.first)..(lineStart + range.last))
                }
            }
            lineStart += lineWithBreak.length
        }
        return result
    }

    private fun isLyricLine(line: String): Boolean {
        val chinese = chineseRegex.findAll(line).count()
        val digits = Regex("[0-7]").findAll(line).count()
        return chinese > digits
    }
}
