package com.automusic.player.core.delta

/**
 * 三角洲谱面编译参数。
 *
 * @param bpm            每分钟节拍数(1-300)
 * @param settleMs       修饰键按下后稳定间隔(默认 30ms)
 * @param releaseSettleMs 修饰键释放后稳定间隔(默认 20ms)
 * @param gapMs          相邻音符间最小间隙(默认 20ms)
 * @param maxHoldMs      单音按住上限(默认 3000ms,防卡死)
 * @param holdRatio      按住时长占时值比例(默认 1.0)
 * @param modifierPolicy 修饰态冲突策略(默认 OCTAVE_FIRST)
 * @param chordPolicy    和弦降级策略(默认 CHORD_FIRST)
 */
data class DeltaCompileParams(
    val bpm: Int = 120,
    val settleMs: Long = 30L,
    val releaseSettleMs: Long = 20L,
    val gapMs: Long = 20L,
    val maxHoldMs: Long = 3000L,
    val holdRatio: Double = 1.0,
    val modifierPolicy: ModifierPolicy = ModifierPolicy.OCTAVE_FIRST,
    val chordPolicy: ChordPolicy = ChordPolicy.CHORD_FIRST,
) {
    init {
        require(bpm in 1..300) { "bpm 必须在 1-300 范围内,当前:$bpm" }
    }

    /** 一拍时长(毫秒)。 */
    val beatMs: Double get() = 60000.0 / bpm
}