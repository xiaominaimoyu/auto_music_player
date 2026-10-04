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

        /**
         * 覆盖层类包名:它们浮在游戏之上(华为小窗由 hwdockbar 承载、游戏助手侧边栏、
         * 系统栏),窗口状态变化会上报它们,但点下去仍然落在底下的游戏,不算"丢失目标"。
         *
         * 刻意不含桌面与负一屏(com.huawei.android.launcher / com.huawei.intelligent):
         * 那两类是真替换前台,点到上面的绝对坐标就是误触。
         */
        val TRANSPARENT_PACKAGES = setOf(
            "com.android.systemui",
            "com.huawei.hwdockbar",
            "com.huawei.gameassistant",
        )
    }

    /**
     * 最近一次窗口状态变化上报的前台包名。
     *
     * 触摸注入是按屏幕绝对坐标落点的,前台是谁就点谁:下拉通知栏、切到别的应用、
     * 游戏弹出暂停界面,都会让同一串坐标点到完全不同的东西上。派发前核对这个值,
     * 是桌面版"目标窗口失焦自动暂停"在安卓端的等价物。
     */
    @Volatile
    var foregroundPackage: String? = null
        private set

    /**
     * 最近一次"真正的全屏应用"包名:排除本应用与覆盖层。
     *
     * 本应用常被当成悬浮小窗盖在游戏上用,此时窗口状态变化上报的是小窗自己
     * (或承载小窗的 hwdockbar),拿它当演奏目标会把游戏排除在外。
     */
    @Volatile
    var lastExternalPackage: String? = null
        private set

    /** 该包名是否"透明"——浮层或本应用自身,不算丢失演奏目标。 */
    fun isTransparent(pkg: String?): Boolean =
        pkg == null || pkg == packageName || pkg in TRANSPARENT_PACKAGES

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event?.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) return
        val pkg = event.packageName?.toString()?.takeIf { it.isNotEmpty() } ?: return
        foregroundPackage = pkg
        if (!isTransparent(pkg)) lastExternalPackage = pkg
    }

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
