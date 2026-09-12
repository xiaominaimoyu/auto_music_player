package com.automusic.player.core.delta

/**
 * 三角洲口琴手游布局:7 个数字音格 + 3 个状态钮 + 提交按钮(可缺省)。
 *
 * @param noteCoords    音格归一化坐标,键为音名 1-7
 * @param modifierCoords 状态钮归一化坐标(LOWER/SEMITONE/HIGHER)
 * @param submitCoord   提交按钮归一化坐标(NpcQuest 场景用,可缺省)
 */
data class DeltaLayout(
    val noteCoords: Map<Int, Pair<Float, Float>>,
    val modifierCoords: Map<Modifier, Pair<Float, Float>>,
    val submitCoord: Pair<Float, Float>?,
) {
    /** 获取音格坐标(像素)。 */
    fun notePixels(note: Int, w: Int, h: Int): Pair<Float, Float>? {
        val p = noteCoords[note] ?: return null
        return (p.first * w) to (p.second * h)
    }

    /** 获取状态钮坐标(像素)。 */
    fun modifierPixels(mod: Modifier, w: Int, h: Int): Pair<Float, Float>? {
        if (mod == Modifier.NATURAL) return null
        val p = modifierCoords[mod] ?: return null
        return (p.first * w) to (p.second * h)
    }

    /** 获取提交按钮坐标(像素)。 */
    fun submitPixels(w: Int, h: Int): Pair<Float, Float>? {
        val p = submitCoord ?: return null
        return (p.first * w) to (p.second * h)
    }
}