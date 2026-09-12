package com.automusic.player.core.delta

/**
 * 降级清单条目:记录某音符因冲突或限制未能按请求执行的降级信息。
 *
 * @param index    音符序号(0-based)
 * @param requested 请求的修饰态/和弦描述
 * @param actual   实际执行的修饰态/和弦描述
 * @param reason   降级原因
 */
data class Degradation(
    val index: Int,
    val requested: String,
    val actual: String,
    val reason: String,
)