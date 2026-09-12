package com.automusic.player.core.delta

/**
 * 演奏中断语义。
 *
 * - PAUSE  可续播(FreePlay 自由演奏):释放全部触摸,保留进度
 * - ABORT  不可续播(NpcQuest NPC 任务):释放全部触摸,清空进度
 */
enum class InterruptMode {
    PAUSE,
    ABORT,
}