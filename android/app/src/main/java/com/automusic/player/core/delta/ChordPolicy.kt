package com.automusic.player.core.delta

/**
 * 和弦(多音同响)降级策略。三角洲口琴为单旋律乐器,不支持和弦。
 *
 * - CHORD_FIRST      取首音,其余降级
 * - CHORD_REJECT     返回休止并降级
 * - CHORD_ARPEGGIATE 拆为时值均分的单音序列(与桌面版 compiler 一致)
 */
enum class ChordPolicy {
    CHORD_FIRST,
    CHORD_REJECT,
    CHORD_ARPEGGIATE,
}