package com.automusic.player.core.delta

/**
 * 三角洲口琴手游键点定义:7 音格 + 3 状态钮 + 1 提交按钮,共 11 个标定点。
 *
 * 键名空间与鸣潮/原神的 21 个 note_id(high_1..low_7)完全无交集,
 * 保证两套布局在 LayoutStore 中互不干扰。
 */
object DeltaKeyPoint {

    const val GAME_DELTA = "delta_mobile"

    /** 11 个标定键名(按标定顺序)。 */
    val KEY_POINTS: List<String> = listOf(
        "note_1", "note_2", "note_3", "note_4", "note_5", "note_6", "note_7",
        "mod_lower", "mod_semitone", "mod_higher",
        "submit",
    )

    /** 音格键名前缀。 */
    const val NOTE_PREFIX = "note_"

    /** 状态钮键名。 */
    const val KEY_MOD_LOWER = "mod_lower"
    const val KEY_MOD_SEMITONE = "mod_semitone"
    const val KEY_MOD_HIGHER = "mod_higher"

    /** 提交按钮键名。 */
    const val KEY_SUBMIT = "submit"

    /**
     * 默认布局(基于三角洲手游界面估算):
     * - 7 音格横排居中,占屏幕中下部
     * - 3 状态钮在音格下方并排
     * - 提交按钮在右下角
     */
    fun defaultLayout(): Map<String, Pair<Float, Float>> {
        val noteCols = floatArrayOf(0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f)
        val noteRow = 0.62f
        val modCols = floatArrayOf(0.30f, 0.45f, 0.60f)
        val modRow = 0.78f
        return linkedMapOf(
            "note_1" to (noteCols[0] to noteRow),
            "note_2" to (noteCols[1] to noteRow),
            "note_3" to (noteCols[2] to noteRow),
            "note_4" to (noteCols[3] to noteRow),
            "note_5" to (noteCols[4] to noteRow),
            "note_6" to (noteCols[5] to noteRow),
            "note_7" to (noteCols[6] to noteRow),
            KEY_MOD_LOWER to (modCols[0] to modRow),
            KEY_MOD_SEMITONE to (modCols[1] to modRow),
            KEY_MOD_HIGHER to (modCols[2] to modRow),
            KEY_SUBMIT to (0.88f to 0.88f),
        )
    }

    /** 归一化坐标 -> 屏幕像素坐标。 */
    fun toPixels(point: Pair<Float, Float>, w: Int, h: Int): Pair<Float, Float> =
        (point.first * w) to (point.second * h)

    /**
     * 将键名映射为音名(1-7);非音格键返回 null。
     */
    fun noteIndex(key: String): Int? {
        if (!key.startsWith(NOTE_PREFIX)) return null
        return key.removePrefix(NOTE_PREFIX).toIntOrNull()
    }

    /** 将键名映射为修饰态;非状态钮键返回 null。 */
    fun modifierOf(key: String): Modifier? = when (key) {
        KEY_MOD_LOWER -> Modifier.LOWER
        KEY_MOD_SEMITONE -> Modifier.SEMITONE
        KEY_MOD_HIGHER -> Modifier.HIGHER
        else -> null
    }

    /**
     * 从扁平键值对坐标 Map 构建 DeltaLayout。
     */
    fun buildLayout(points: Map<String, Pair<Float, Float>>): DeltaLayout {
        val noteCoords = mutableMapOf<Int, Pair<Float, Float>>()
        val modCoords = mutableMapOf<Modifier, Pair<Float, Float>>()
        var submit: Pair<Float, Float>? = null
        for ((key, coord) in points) {
            noteIndex(key)?.let { noteCoords[it] = coord }
            modifierOf(key)?.let { modCoords[it] = coord }
            if (key == KEY_SUBMIT) submit = coord
        }
        return DeltaLayout(noteCoords, modCoords, submit)
    }
}