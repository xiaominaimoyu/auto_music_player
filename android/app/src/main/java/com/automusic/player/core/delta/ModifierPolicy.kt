package com.automusic.player.core.delta

/**
 * 八度与半音冲突时的修饰态解析策略。
 *
 * - OCTAVE_FIRST    保留八度、丢半音并告警(默认)
 * - REJECT_NOTE     返回 NATURAL 并降级该音
 * - KEEP_ACCIDENTAL 保留半音、丢八度并告警
 */
enum class ModifierPolicy {
    OCTAVE_FIRST,
    REJECT_NOTE,
    KEEP_ACCIDENTAL,
}