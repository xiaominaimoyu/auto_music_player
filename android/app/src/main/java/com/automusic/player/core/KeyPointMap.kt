package com.automusic.player.core

/**
 * 音符 -> 屏幕坐标映射(归一化 0..1,适配任意分辨率)。
 *
 * 默认预设取自真机截图(华为 nova 11,2412x1084 横屏)估算值,
 * 用户可通过标定页重新标定并保存覆盖。
 */
object KeyPointMap {

    const val GAME_WUTHERING = "wuthering"
    const val GAME_GENSHIN = "genshin"
    const val GAME_GENSHIN_WOYANISHA = "genshin_woyanisha"

    val GAME_NAMES = linkedMapOf(
        GAME_WUTHERING to "鸣潮",
        GAME_GENSHIN to "原神",
        GAME_GENSHIN_WOYANISHA to "沃雅妮莎",
    )

    /** 全部 21 个 note_id,标定顺序:高音 1-7、中音 1-7、低音 1-7。 */
    val ALL_NOTES: List<String> = buildList {
        for (octave in listOf("high", "mid", "low")) {
            for (i in 1..7) add("${octave}_$i")
        }
    }

    /** 沃雅妮莎 14 个 note_id,标定顺序:高音 1-7、中音 1-7。 */
    val WOYANISHA_NOTES: List<String> = buildList {
        for (octave in listOf("high", "mid")) {
            for (i in 1..7) add("${octave}_$i")
        }
    }

    fun notesFor(gameType: String): List<String> = when (gameType) {
        GAME_GENSHIN_WOYANISHA -> WOYANISHA_NOTES
        else -> ALL_NOTES
    }

    /** note_id -> 归一化坐标 (x, y)。 */
    fun defaultLayout(game: String): Map<String, Pair<Float, Float>> {
        if (game == GAME_GENSHIN_WOYANISHA) return emptyMap()

        val cols: FloatArray
        val rows: FloatArray
        when (game) {
            GAME_GENSHIN -> {
                cols = floatArrayOf(0.240f, 0.325f, 0.408f, 0.500f, 0.591f, 0.676f, 0.760f)
                rows = floatArrayOf(0.530f, 0.702f, 0.857f)
            }
            else -> { // GAME_WUTHERING 默认
                cols = floatArrayOf(0.269f, 0.347f, 0.424f, 0.500f, 0.577f, 0.654f, 0.731f)
                rows = floatArrayOf(0.572f, 0.719f, 0.868f)
            }
        }
        return buildMap {
            val octaves = listOf("high", "mid", "low")
            octaves.forEachIndexed { r, octave ->
                for (i in 1..7) {
                    put("${octave}_$i", cols[i - 1] to rows[r])
                }
            }
        }
    }

    /** 归一化坐标 -> 当前屏幕像素坐标。 */
    fun toPixels(point: Pair<Float, Float>, w: Int, h: Int): Pair<Float, Float> =
        (point.first * w) to (point.second * h)

    /** 音符列表 -> 像素坐标列表(跳过未映射项)。 */
    fun coordsFor(
        noteIds: List<String>,
        layout: Map<String, Pair<Float, Float>>,
        w: Int,
        h: Int,
    ): List<Pair<Float, Float>> =
        noteIds.mapNotNull { layout[it] }.map { toPixels(it, w, h) }

    /**
     * 四角快速标定:根据 high_1 / high_7 / low_1 / low_7 双线性插值推算其余 17 点。
     * 任一角度缺失时返回 null。
     */
    fun deriveGrid(
        high1: Pair<Float, Float>,
        high7: Pair<Float, Float>,
        low1: Pair<Float, Float>,
        low7: Pair<Float, Float>,
    ): Map<String, Pair<Float, Float>>? {
        if (high1.first >= high7.first || low1.first >= low7.first) return null
        return buildMap {
            val octaves = listOf("high", "mid", "low")
            octaves.forEachIndexed { r, octave ->
                val rowT = r / 2f
                for (i in 1..7) {
                    val colT = (i - 1) / 6f
                    val topX = high1.first + (high7.first - high1.first) * colT
                    val topY = high1.second + (high7.second - high1.second) * colT
                    val botX = low1.first + (low7.first - low1.first) * colT
                    val botY = low1.second + (low7.second - low1.second) * colT
                    val x = topX + (botX - topX) * rowT
                    val y = topY + (botY - topY) * rowT
                    put("${octave}_$i", x to y)
                }
            }
        }
    }

    /** 沃雅妮莎两排四角标定:每排独立在左右端点之间等距插值。 */
    fun deriveGrid14(
        high1: Pair<Float, Float>,
        high7: Pair<Float, Float>,
        mid1: Pair<Float, Float>,
        mid7: Pair<Float, Float>,
    ): Map<String, Pair<Float, Float>>? {
        if (high1.first >= high7.first || mid1.first >= mid7.first) return null
        return buildMap {
            val rows = listOf(
                Triple("high", high1, high7),
                Triple("mid", mid1, mid7),
            )
            for ((octave, start, end) in rows) {
                for (i in 1..7) {
                    val t = (i - 1) / 6f
                    val x = start.first + (end.first - start.first) * t
                    val y = start.second + (end.second - start.second) * t
                    put("${octave}_$i", x to y)
                }
            }
        }
    }
}
