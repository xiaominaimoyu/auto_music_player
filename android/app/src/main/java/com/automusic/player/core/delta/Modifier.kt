package com.automusic.player.core.delta

/**
 * 三角洲口琴修饰态:四态互斥枚举(对应鼠标左/中/右键按住态)。
 *
 * - NATURAL  自然音(不按修饰键)
 * - LOWER    降调/低八度(按住左键)
 * - SEMITONE 升半音(按住中键/滚轮)
 * - HIGHER   升调/高八度(按住右键)
 */
enum class Modifier {
    NATURAL,
    LOWER,
    SEMITONE,
    HIGHER,
}