package com.automusic.player.core

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/** JSON 乐谱格式非法或不受支持。 */
class ScoreJsonException(message: String) : IllegalArgumentException(message)

/**
 * 解析桌面版「导出 JSON」得到的曲谱,让电脑导好的谱能直接进手机。
 *
 * 桌面导出格式(core/score_io.py export_json):
 * `{"format":"auto-music-player-score","version":1,"name":...,"bpm":...,"notes":[{notes,dur,semitone}]}`
 * 同时接受裸数组(即桌面 scores.notes_json 那一列的原样内容)。
 */
object ScoreJson {

    const val FORMAT = "auto-music-player-score"
    const val SUPPORTED_VERSION = 1
    const val MAX_CHARS = 10 * 1024 * 1024

    data class Parsed(
        val name: String,
        val bpm: Int,
        val notes: List<NoteEvent>,
        val warnings: List<String>,
    )

    /** 输入框里的内容看起来像 JSON 时才走本解析器。 */
    fun looksLikeJson(text: String): Boolean {
        val t = text.trimStart()
        return t.startsWith("{") || t.startsWith("[")
    }

    fun parse(text: String): Parsed {
        if (text.length > MAX_CHARS) throw ScoreJsonException("JSON 乐谱不能超过 10 MB")
        val trimmed = text.trim()
        val notesArray: JSONArray
        var name = ""
        var bpm = 100
        val warnings = ArrayList<String>()

        try {
            if (trimmed.startsWith("[")) {
                notesArray = JSONArray(trimmed)
            } else {
                val root = JSONObject(trimmed)
                val format = root.optString("format")
                if (format.isNotEmpty() && format != FORMAT) {
                    throw ScoreJsonException("不支持的乐谱格式:$format(应为 $FORMAT)")
                }
                val version = root.optInt("version", SUPPORTED_VERSION)
                if (version > SUPPORTED_VERSION) {
                    warnings += "文件版本 $version 高于本应用支持的 $SUPPORTED_VERSION,可能忽略了部分字段"
                }
                name = root.optString("name")
                bpm = root.optInt("bpm", 100)
                notesArray = root.optJSONArray("notes")
                    ?: throw ScoreJsonException("JSON 里缺少 notes 音符数组")
            }
        } catch (e: JSONException) {
            throw ScoreJsonException("不是有效的 JSON:${e.message}")
        }

        val notes = ArrayList<NoteEvent>(notesArray.length())
        for (i in 0 until notesArray.length()) {
            val obj = notesArray.optJSONObject(i)
                ?: throw ScoreJsonException("第 ${i + 1} 个元素不是对象")
            val ids = ArrayList<String>()
            val raw = obj.optJSONArray("notes")
            if (raw == null) {
                // 空 notes 在桌面协议里表示休止,但字段本身必须存在
                throw ScoreJsonException("第 ${i + 1} 个元素缺少 notes 字段(休止应写作空数组)")
            }
            for (j in 0 until raw.length()) ids += raw.optString(j)
            if (!obj.has("dur")) throw ScoreJsonException("第 ${i + 1} 个元素缺少时值 dur")
            val dur = obj.optDouble("dur", Double.NaN)
            if (dur.isNaN()) throw ScoreJsonException("第 ${i + 1} 个元素时值不是数字")
            notes += NoteEvent(ids, dur, semitoneOf(obj.opt("semitone")))
        }
        return Parsed(name, bpm, notes, warnings)
    }

    /** 桌面写 0/1,历史安卓包写 true/false,两种都要能吃。 */
    private fun semitoneOf(value: Any?): Boolean = when (value) {
        is Boolean -> value
        is Number -> value.toInt() == 1
        is String -> value == "1" || value.equals("true", ignoreCase = true)
        else -> false
    }
}
