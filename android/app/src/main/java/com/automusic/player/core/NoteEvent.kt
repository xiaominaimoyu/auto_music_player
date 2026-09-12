package com.automusic.player.core

import org.json.JSONArray

/** 音符事件:notes 为 note_id 列表(和弦),dur 单位为拍;休止符 notes 为空。 */
data class NoteEvent(
    val notes: List<String>,
    val dur: Double,
    val semitone: Boolean = false,
)

/** notes <-> JSON 数组(与原桌面版 notes_json 协议一致)。 */
object NoteCodec {

    fun encode(notes: List<NoteEvent>): String {
        val arr = JSONArray()
        for (n in notes) {
            val obj = org.json.JSONObject()
            obj.put("notes", JSONArray(n.notes))
            obj.put("dur", n.dur)
            if (n.semitone) obj.put("semitone", true)
            arr.put(obj)
        }
        return arr.toString()
    }

    fun decode(json: String): List<NoteEvent> {
        val result = mutableListOf<NoteEvent>()
        val arr = JSONArray(json)
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            val ids = mutableListOf<String>()
            val nArr = obj.optJSONArray("notes") ?: JSONArray()
            for (j in 0 until nArr.length()) ids.add(nArr.getString(j))
            result.add(NoteEvent(ids, obj.optDouble("dur", 1.0), obj.optBoolean("semitone", false)))
        }
        return result
    }
}
