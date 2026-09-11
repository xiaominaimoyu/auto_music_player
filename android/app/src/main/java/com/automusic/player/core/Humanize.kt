package com.automusic.player.core

import java.util.Random

/**
 * 真人化节奏塑形(与桌面版 core/humanize.py 规则一致):
 *
 * 1. 起音微偏移(jitter):每个音符起始时刻叠加零均值正态偏移——
 *    真人不可能每个音都精准踩在拍点上;
 * 2. 动态按住(articulation):短音按得轻快(较低 hold),长音按得饱满(较高 hold),
 *    替代全局固定 holdRatio,让颗粒感随音符密度自然变化;
 * 3. 乐句呼吸(breath):休止之后、长音结尾之后的下一音稍作停顿再进入,
 *    模拟换气与重新起手;音高大跳进附加少量换指时间。
 *
 * 防叠键:本模块只产出"相对理想起音时刻的偏移",不直接移动时间轴;
 * PlayerEngine 负责链式下限保护——本音符实际起音不早于上一音符完全释放。
 */
data class HumanizeParams(
    /** 起音正态微偏移 σ(毫秒)。 */
    val jitterMs: Double = 8.0,
    /** 乐句边界后的呼吸停顿基准(毫秒,实际 ×0.6~1.0 随机)。 */
    val breathMs: Double = 18.0,
    /** 短音符(≤0.5 拍)hold 区间:轻快断奏。 */
    val shortHold: Pair<Double, Double> = 0.62 to 0.72,
    /** 中等音符 hold 区间。 */
    val midHold: Pair<Double, Double> = 0.72 to 0.82,
    /** 长音符(≥2 拍)hold 区间:绵延饱满。 */
    val longHold: Pair<Double, Double> = 0.88 to 0.96,
    /** 大跳进附加换指时间(毫秒,实际 ×0.5~1.0 随机)。 */
    val leapMs: Double = 3.0,
    /** 音高差 ≥ 该值视为大跳进。 */
    val leapThreshold: Int = 4,
    /** 塑形后保证的最小不叠键间隙(毫秒)。 */
    val minGapMs: Double = 2.0,
)

/** 单元素塑形结果:offsetMs 为相对理想起音的偏移;holdRatio 为 null 表示休止占位。 */
data class Timing(val offsetMs: Double, val holdRatio: Double?)

object Humanize {
    /** ≥ 该拍数的音符视为长音,其后的音符按乐句开头处理(换气)。 */
    const val PHRASE_LONG_DUR = 2.0

    private val OCTAVE_VAL = mapOf("low" to 0, "mid" to 7, "high" to 14)

    /**
     * 音符序列 → 与元素等长的 [Timing] 列表。
     * 休止元素为 (0.0, null)——不发声,由调用方跳过;
     * 但休止参与"乐句边界"判定:休止之后的音符带呼吸停顿。
     */
    fun planTimings(
        notes: List<NoteEvent>,
        params: HumanizeParams = HumanizeParams(),
        rng: Random = Random(),
    ): List<Timing> {
        val timings = ArrayList<Timing>(notes.size)
        var prevPitch: Int? = null
        var prevDur: Double? = null
        var prevWasRest = false

        for (el in notes) {
            val dur = el.dur
            if (el.notes.isEmpty()) {
                // 休止:占位,不发声;重置跳进链(下一音的呼吸已覆盖换指)
                timings.add(Timing(0.0, null))
                prevPitch = null
                prevDur = dur
                prevWasRest = true
                continue
            }

            var offset = rng.nextGaussian() * params.jitterMs
            // 乐句边界:前一元素是休止(换气),或长音结尾(重新起手)
            if (prevWasRest || (prevDur != null && prevDur >= PHRASE_LONG_DUR)) {
                offset += params.breathMs * (0.6 + 0.4 * rng.nextDouble())
            }
            // 大跳进:换指需要一点额外时间
            val pitch = pitchValue(el.notes.first())
            if (pitch != null && prevPitch != null &&
                kotlin.math.abs(pitch - prevPitch) >= params.leapThreshold
            ) {
                offset += params.leapMs * (0.5 + 0.5 * rng.nextDouble())
            }

            // 动态按住:按音符长度分档,同档内随机
            val (lo, hi) = when {
                dur <= 0.5 -> params.shortHold
                dur < PHRASE_LONG_DUR -> params.midHold
                else -> params.longHold
            }
            val holdRatio = lo + (hi - lo) * rng.nextDouble()

            timings.add(Timing(offset, holdRatio))
            if (pitch != null) prevPitch = pitch
            prevDur = dur
            prevWasRest = false
        }
        return timings
    }

    /** note_id → 音高量值(low=0..6, mid=7..13, high=14..20,跨八度连续);无法解析返回 null。 */
    private fun pitchValue(noteId: String): Int? {
        val name = noteId.substringBefore("_")
        val num = noteId.substringAfter("_", "")
        val n = num.toIntOrNull() ?: return null
        if (n !in 1..7) return null
        return (OCTAVE_VAL[name] ?: return null) + n
    }
}
