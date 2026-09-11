package com.automusic.player.core

/** 大模型输出协议，与桌面端 recognizer.py 保持一致。 */
object Prompt {
    const val JIANPU_PROMPT: String = """你是一位简谱识别专家。请识别输入乐谱的曲名、音符、时值与歌词，并严格按指定格式输出。

曲名规则:
1. 优先读取乐谱图片或文档中明确出现的标题、歌名或曲名
2. 没有明确曲名时输出 UNKNOWN，禁止猜测或根据旋律臆造

简谱规则:
1. 音高:中音用数字 1-7;高音在数字后加英文单引号 ' (如 5');低音在数字后加英文逗号 , (如 5,);休止符用 0
2. 时值:不加符号=四分音符;数字后跟一个下划线 _ = 八分音符,两个 __ = 十六分音符;跟一个减号 - = 二分音符,两个 -- = 全音符;附点用 · 加在时值符号后(如 5_· 是附点八分音符)
3. 和弦:多个同时演奏的音用方括号括起、空格分隔,时值符号加在右括号后(如 [1' 3' 5']- 是二分和弦);和弦内的音只写音高不写时值
4. 每个音符(或和弦)之间用空格分隔
5. 只有原谱明确标出的休止符才输出为 0;禁止仅根据横向字距、歌词语义、换气或乐句感觉添加休止
6. 保留原谱的乐句分行:JIANPU 中每个原谱行单独输出一行;换行本身不要转换成休止，后续由用户按原谱分行补入并试听校对
7. 小节线、调号(如 1=C)、拍号、歌词、装饰音符号、力度记号不写入 JIANPU
8. 如果原图未标注或看不清时值/休止,必须在 RHYTHM 中说明;保持谱面原有分行，不要自行推断休止
9. 不要输出其他解释文字,不要用代码块包裹

歌词规则:
1. 如果谱面有歌词,LYRICS 必须与 JIANPU 逐行对应,每个音符事件写一个歌词字并用空格分隔
2. 一字多音时后续音符用 _ 占位;休止符、过门或没有歌词的音符也用 _ 占位
3. 如果谱面没有歌词,输出 UNKNOWN;不要根据曲名补写歌词

输出格式(严格保留以下四个标签):
TITLE:
<图片或文档中的曲名;未识别到则写 UNKNOWN>
JIANPU:
<仅包含规范化简谱;保留原谱分行>
LYRICS:
<与 JIANPU 逐行、逐音符对应的歌词;无歌词写 UNKNOWN>
RHYTHM:
<一句话说明时值与停顿是明确识别还是推断;没有疑问写 OK>

示例输出:
TITLE:
小星星
JIANPU:
1 1 5 5 6 6 5-
4 4 3 3 2 2 1-
LYRICS:
一 闪 一 闪 亮 晶 晶
满 天 都 是 小 星 星
RHYTHM:
时值来自原谱标记"""

    data class RecognitionResult(
        val title: String,
        val jianpu: String,
        val lyricsLines: List<String> = emptyList(),
        val rhythmNotes: List<String> = emptyList(),
        val raw: String = "",
    )

    fun parseRecognitionResult(raw: String): RecognitionResult {
        val text = raw.trim()
        var title = ""
        val jianpu = mutableListOf<String>()
        val lyrics = mutableListOf<String>()
        val rhythmNotes = mutableListOf<String>()
        var section = ""
        var sawProtocol = false
        val titlePattern = Regex("^(?:TITLE|曲名|歌名|标题)\\s*[:：]\\s*(.*)$", RegexOption.IGNORE_CASE)
        val jianpuPattern = Regex("^(?:JIANPU|简谱)\\s*[:：]\\s*(.*)$", RegexOption.IGNORE_CASE)
        val lyricsPattern = Regex("^(?:LYRICS|歌词)\\s*[:：]\\s*(.*)$", RegexOption.IGNORE_CASE)
        val rhythmPattern = Regex("^(?:RHYTHM|节奏说明|节奏)\\s*[:：]\\s*(.*)$", RegexOption.IGNORE_CASE)
        val unknown = setOf(
            "unknown", "n/a", "na", "none", "null", "未知", "未识别", "未识别到",
            "无", "未命名", "无法识别", "无法确定",
        )

        for (rawLine in text.lines()) {
            val line = rawLine.trim()
            if (line.isEmpty() || line.startsWith("```")) continue
            val titleMatch = titlePattern.matchEntire(line)
            if (titleMatch != null) {
                sawProtocol = true
                section = "title"
                title = cleanTitle(titleMatch.groupValues[1], unknown)
                continue
            }
            val jianpuMatch = jianpuPattern.matchEntire(line)
            if (jianpuMatch != null) {
                sawProtocol = true
                section = "jianpu"
                jianpuMatch.groupValues[1].trim().takeIf(String::isNotEmpty)?.let(jianpu::add)
                continue
            }
            val lyricsMatch = lyricsPattern.matchEntire(line)
            if (lyricsMatch != null) {
                sawProtocol = true
                section = "lyrics"
                lyricsMatch.groupValues[1].trim()
                    .takeIf { it.isNotEmpty() && it.lowercase() !in unknown }
                    ?.let(lyrics::add)
                continue
            }
            val rhythmMatch = rhythmPattern.matchEntire(line)
            if (rhythmMatch != null) {
                sawProtocol = true
                section = "rhythm"
                rhythmMatch.groupValues[1].trim()
                    .takeIf { it.isNotEmpty() && !it.equals("OK", ignoreCase = true) }
                    ?.let(rhythmNotes::add)
                continue
            }
            when (section) {
                "title" -> {
                    if (title.isEmpty()) title = cleanTitle(line, unknown)
                    section = ""
                }
                "jianpu" -> jianpu.add(line)
                "lyrics" -> if (line.lowercase() !in unknown) lyrics.add(line)
                "rhythm" -> if (!line.equals("OK", ignoreCase = true)) rhythmNotes.add(line)
            }
        }
        return RecognitionResult(
            title = title,
            jianpu = if (sawProtocol) jianpu.joinToString("\n") else text,
            lyricsLines = lyrics,
            rhythmNotes = rhythmNotes,
            raw = text,
        )
    }

    fun addLineBreakRests(jianpu: String, restToken: String = "0_"): String {
        val lines = jianpu.lines().map(String::trim).filter(String::isNotEmpty)
        if (lines.size < 2) return jianpu.trim()
        return lines.joinToString(" $restToken\n")
    }

    fun fallbackScoreName(token: String = java.util.UUID.randomUUID().toString()): String =
        "未命名乐谱-${token.replace("-", "").take(4).uppercase()}"

    private fun cleanTitle(value: String, unknown: Set<String>): String {
        val title = value.trim().trim('`', '#', '*', ' ', '\"', '\'', '《', '》').take(80)
        return if (title.lowercase() in unknown) "" else title
    }

    /** 内置样例(空库时端到端跑通流程用)。 */
    const val SAMPLE_JIANPU: String =
        "1 1 5, 5, 6 6 5'- 4 4 3 3 2 2 1- 0 0 [1' 3' 5']- 1 2 3_ 3_ 5_· 5_"
}
