package com.automusic.player.input

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.view.accessibility.AccessibilityEvent

/**
 * 无障碍手势注入服务(备用注入通道)。
 *
 * dispatchGesture 是公开 API,不受华为对 injectInputEvent 的调用方进程校验限制;
 * 多 stroke 同一手势 = 多指和弦,stroke duration = 按住时长,结束自动抬起。
 */
class AmpAccessibilityService : AccessibilityService() {

    companion object {
        @Volatile
        var instance: AmpAccessibilityService? = null
            private set

        val ready: Boolean
            get() = instance != null
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {}

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    /** 一次派发和弦手势:所有手指同时按下,按住 holdMs 后自动抬起。 */
    fun chord(coords: List<Pair<Float, Float>>, holdMs: Long) {
        require(coords.isNotEmpty()) { "和弦坐标为空" }
        val accepted = timeline(
            coords.map { (x, y) -> TimedTouch(x, y, 0L, holdMs) },
            onComplete = null,
        )
        if (!accepted) error("dispatchGesture 被系统拒绝")
    }

    /**
     * 一次派发带相对时间的多 stroke 手势。所有需要同时生效的触点必须位于
     * 同一个 GestureDescription 中，避免系统取消前一条未结束的手势。
     */
    fun timeline(strokes: List<TimedTouch>, onComplete: ((Boolean) -> Unit)?): Boolean {
        require(strokes.isNotEmpty()) { "手势 stroke 不能为空" }
        require(strokes.size <= 10) { "单次无障碍手势最多支持 10 个 stroke" }
        val builder = GestureDescription.Builder()
        for (stroke in strokes) {
            val start = stroke.startMs.coerceAtLeast(0L)
            val duration = stroke.holdMs.coerceIn(1L, 60000L)
            require(start + duration <= 60000L) { "单次无障碍手势不能超过 60 秒" }
            val path = Path().apply { moveTo(stroke.x, stroke.y) }
            builder.addStroke(GestureDescription.StrokeDescription(path, start, duration))
        }
        val callback = if (onComplete == null) null else object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription) {
                onComplete(true)
            }

            override fun onCancelled(gestureDescription: GestureDescription) {
                onComplete(false)
            }
        }
        return dispatchGesture(builder.build(), callback, null)
    }
}
