package com.automusic.player.calib

import android.content.Context
import com.automusic.player.core.KeyPointMap
import com.automusic.player.core.delta.DeltaKeyPoint
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** 一套琴键布局:note_id -> 归一化坐标。 */
data class KeyLayout(
    val id: String,
    val name: String,
    val points: Map<String, Pair<Float, Float>>,
    val gameType: String = KeyPointMap.GAME_WUTHERING,
)

data class LayoutState(
    val layouts: List<KeyLayout> = emptyList(),
    val activeId: String = "",
) {
    val active: KeyLayout?
        get() = layouts.firstOrNull { it.id == activeId } ?: layouts.firstOrNull()
}

/** 布局导入结果。 */
data class LayoutImportOutcome(
    val layout: KeyLayout,
    val renamed: Boolean,        // id 与现有布局冲突时自动改名
    val warnings: List<String>,
)

/** 布局 JSON 非法或结构不符。 */
class LayoutFormatException(message: String) : IllegalArgumentException(message)

/**
 * 标定布局存储:filesDir/layouts.json。
 * 首次启动自动生成鸣潮/原神两套默认预设(真机截图估算值)。
 * 支持把布局导出为可分享的 JSON 文本,并从 JSON 导入(校验后存为独立布局)。
 */
class LayoutStore(context: Context) {

    private val file = File(context.filesDir, "layouts.json")

    private val _state = MutableStateFlow(load())
    val state: StateFlow<LayoutState> = _state.asStateFlow()

    fun get(id: String): KeyLayout? = _state.value.layouts.firstOrNull { it.id == id }

    /** 导出指定布局为可分享 JSON;布局不存在返回 null。 */
    fun exportJson(id: String): String? {
        val layout = get(id) ?: return null
        return exportLayoutJson(layout)
    }

    /** 解析、校验并保存导入的布局(设为激活)。 */
    fun importJson(json: String): LayoutImportOutcome {
        val outcome = parseLayoutJson(json, _state.value.layouts)
        save(outcome.layout, makeActive = true)
        return outcome
    }

    fun save(layout: KeyLayout, makeActive: Boolean = true) {
        val current = _state.value
        val list = current.layouts.filterNot { it.id == layout.id } + layout
        val activeId = when {
            makeActive -> layout.id
            current.activeId.isNotEmpty() -> current.activeId
            else -> list.first().id
        }
        persist(LayoutState(list, activeId))
    }

    fun delete(id: String) {
        val current = _state.value
        val list = current.layouts.filterNot { it.id == id }
        val activeId = if (current.activeId == id) list.firstOrNull()?.id ?: "" else current.activeId
        persist(LayoutState(list, activeId))
    }

    fun setActive(id: String) {
        val current = _state.value
        if (current.layouts.any { it.id == id }) persist(current.copy(activeId = id))
    }

    private fun persist(state: LayoutState) {
        _state.value = state
        runCatching { write(state) }
    }

    private fun load(): LayoutState {
        return try {
            if (file.exists()) {
                decode(file.readText())
            } else {
                // 首次启动:写入鸣潮/原神/三角洲三套默认预设
                val initial = listOf(
                    default("wuthering"),
                    default("genshin"),
                    default(DeltaKeyPoint.GAME_DELTA),
                )
                val state = LayoutState(initial, initial.first().id)
                runCatching { write(state) }
                state
            }
        } catch (e: Exception) {
            LayoutState()
        }
    }

    private fun default(game: String): KeyLayout {
        val id = game
        val name = when (game) {
            DeltaKeyPoint.GAME_DELTA -> "三角洲"
            else -> KeyPointMap.GAME_NAMES[game] ?: game
        }
        val points = when (game) {
            DeltaKeyPoint.GAME_DELTA -> DeltaKeyPoint.defaultLayout()
            else -> KeyPointMap.defaultLayout(game)
        }
        return KeyLayout(id, name, points, game)
    }

    private fun write(state: LayoutState) {
        val root = JSONObject()
        root.put("active", state.activeId)
        val arr = JSONArray()
        for (l in state.layouts) {
            arr.put(encodeLayout(l))
        }
        root.put("layouts", arr)
        file.writeText(root.toString())
    }

    private fun decode(json: String): LayoutState {
        val root = JSONObject(json)
        val arr = root.optJSONArray("layouts") ?: JSONArray()
        val layouts = mutableListOf<KeyLayout>()
        for (i in 0 until arr.length()) {
            layouts.add(decodeLayout(arr.getJSONObject(i)))
        }
        return LayoutState(layouts, root.optString("active"))
    }

    companion object {

        const val FORMAT_TYPE = "amp-key-layout"
        const val FORMAT_VERSION = 1

        /** 单布局 -> 可分享 JSON(带类型标记,便于导入时识别文件种类)。 */
        fun exportLayoutJson(layout: KeyLayout): String {
            val root = JSONObject()
            root.put("type", FORMAT_TYPE)
            root.put("version", FORMAT_VERSION)
            root.put("id", layout.id)
            root.put("name", layout.name)
            root.put("gameType", layout.gameType)
            root.put("points", encodeLayout(layout).getJSONObject("points"))
            return root.toString(2)
        }

        /**
         * 解析并校验导入的布局(纯函数,无 Context,便于 JVM 单测):
         * - type 标记可选但存在时必须匹配;id 必填;points 键须为合法 note_id、坐标 0..1
         * - id 与现有布局冲突时自动改名加后缀,避免覆盖鸣潮/原神预设或既有布局
         * - 非法键位忽略并计入警告;坐标全部无效视为格式错误
         */
        fun parseLayoutJson(json: String, existing: List<KeyLayout>): LayoutImportOutcome {
            val root = try {
                JSONObject(json)
            } catch (e: Exception) {
                throw LayoutFormatException("不是有效的布局 JSON:${e.message}")
            }
            val type = root.optString("type")
            if (type.isNotEmpty() && type != FORMAT_TYPE) {
                throw LayoutFormatException("不支持的布局类型:$type")
            }
            val id = root.optString("id").trim()
            if (id.isEmpty()) throw LayoutFormatException("缺少布局 ID")
            val gameType = root.optString("gameType", KeyPointMap.GAME_WUTHERING)
            val ptsObj = root.optJSONObject("points")
                ?: throw LayoutFormatException("缺少 points 琴键坐标数据")
            val validNotes = when (gameType) {
                DeltaKeyPoint.GAME_DELTA -> DeltaKeyPoint.KEY_POINTS.toSet()
                else -> KeyPointMap.ALL_NOTES.toSet()
            }
            val expectedCount = validNotes.size
            val points = LinkedHashMap<String, Pair<Float, Float>>()
            val unknown = mutableListOf<String>()
            for (key in ptsObj.keys()) {
                val arr = ptsObj.optJSONArray(key) ?: continue
                if (key !in validNotes || arr.length() < 2) {
                    unknown.add(key)
                    continue
                }
                val x = arr.optDouble(0, Double.NaN).toFloat()
                val y = arr.optDouble(1, Double.NaN).toFloat()
                if (!x.isFinite() || !y.isFinite() || x < 0f || x > 1f || y < 0f || y > 1f) {
                    throw LayoutFormatException("坐标越界:$key ($x, $y),应为 0..1 归一化坐标")
                }
                points[key] = x to y
            }
            if (points.isEmpty()) throw LayoutFormatException("布局不含任何有效琴键坐标")
            var name = root.optString("name").ifBlank { id }
            var finalId = id
            val existingIds = existing.map { it.id }.toSet()
            var renamed = false
            if (finalId in existingIds) {
                finalId = "${id}_import"
                var n = 1
                while (finalId in existingIds) {
                    finalId = "${id}_import${n}"
                    n++
                }
                name = "$name (导入)"
                renamed = true
            }
            val warnings = buildList {
                if (unknown.isNotEmpty()) {
                    add("已忽略 ${unknown.size} 个无效键位:${unknown.take(3).joinToString()}")
                }
                if (points.size < expectedCount) {
                    add("布局仅含 ${points.size}/$expectedCount 个琴键坐标,缺失键位演奏时不会触发")
                }
            }
            return LayoutImportOutcome(KeyLayout(finalId, name, points, gameType), renamed, warnings)
        }

        private fun encodeLayout(l: KeyLayout): JSONObject {
            val obj = JSONObject()
            obj.put("id", l.id)
            obj.put("name", l.name)
            obj.put("gameType", l.gameType)
            val pts = JSONObject()
            for ((noteId, p) in l.points) {
                pts.put(noteId, JSONArray().put(p.first.toDouble()).put(p.second.toDouble()))
            }
            obj.put("points", pts)
            return obj
        }

        private fun decodeLayout(o: JSONObject): KeyLayout {
            val pts = mutableMapOf<String, Pair<Float, Float>>()
            val p = o.optJSONObject("points") ?: JSONObject()
            for (key in p.keys()) {
                val pair = p.getJSONArray(key)
                pts[key] = pair.getDouble(0).toFloat() to pair.getDouble(1).toFloat()
            }
            return KeyLayout(o.optString("id"), o.optString("name"), pts, o.optString("gameType", KeyPointMap.GAME_WUTHERING))
        }
    }
}
