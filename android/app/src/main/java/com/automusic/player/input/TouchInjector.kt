package com.automusic.player.input

/** 单个 Stroke 的相对开始时刻与按住时长，用于同一原子无障碍手势。 */
data class TimedTouch(
    val x: Float,
    val y: Float,
    val startMs: Long,
    val holdMs: Long,
)

/**
 * 触摸注入器:无障碍 dispatchGesture 通道(完全自主运行,零外部依赖)。
 *
 * - 公开 API,免 root、免 adb、无需任何外部激活工具
 * - 多 stroke 同一手势 = 多指和弦;stroke duration = 按住时长,结束自动抬起
 * - 手势由系统无障碍手势控制器派发到前台窗口,按屏幕绝对坐标路由
 *
 * 说明:曾实现 Shizuku + InputManager.injectInputEvent 路径,实测部分厂商
 * (华为等)对非 adb 会话进程的注入有对抗校验(返回 false),且该路径需要
 * 外部激活;本通道在同等场景下工作稳定,故成为唯一通道。
 */
object TouchInjector {

    private const val TAG = "TouchInjector"
    private const val TAP_HOLD_MS = 40L

    /** 无障碍注入服务是否已开启并绑定。 */
    val accessibilityReady: Boolean
        get() = AmpAccessibilityService.ready

    /**
     * 派发一个完整和弦:全部手指同时按下,按住 holdMs 后自动抬起。
     * 同步返回,手势由系统异步执行;结束时保证无残留按键。
     */
    fun dispatchChord(coords: List<Pair<Float, Float>>, holdMs: Long) {
        require(coords.isNotEmpty()) { "和弦坐标为空" }
        val svc = AmpAccessibilityService.instance
            ?: error(
                "无障碍注入服务未开启:" +
                    "请到「演奏」页点「去开启」,在系统无障碍设置中启用本服务的触摸注入"
            )
        svc.chord(coords, holdMs)
        android.util.Log.i(TAG, "gesture dispatched pointers=${coords.size} hold=${holdMs}ms")
    }

    /**
     * 派发同一条 GestureDescription 中的多段 stroke。
     *
     * 修饰钮和音格必须通过此方法同批派发，不能拆成多次 dispatchGesture；
     * 后者会取消先前仍在执行的手势。
     */
    fun dispatchTimeline(strokes: List<TimedTouch>, onComplete: (Boolean) -> Unit): Boolean {
        require(strokes.isNotEmpty()) { "手势 stroke 不能为空" }
        val svc = AmpAccessibilityService.instance
            ?: error(
                "无障碍注入服务未开启:" +
                    "请到「演奏」页点「去开启」,在系统无障碍设置中启用本服务的触摸注入"
            )
        val accepted = svc.timeline(strokes, onComplete)
        if (accepted) {
            android.util.Log.i(TAG, "timeline dispatched strokes=${strokes.size}")
        }
        return accepted
    }

    /** 单点轻触(标定/测试用)。 */
    fun tap(x: Float, y: Float) {
        dispatchChord(listOf(x to y), TAP_HOLD_MS)
    }
}
