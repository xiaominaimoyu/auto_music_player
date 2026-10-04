package com.automusic.player

import android.app.Application
import android.content.Context
import androidx.room.Room
import com.automusic.player.calib.LayoutStore
import com.automusic.player.core.PlayerEngine
import com.automusic.player.core.db.ScoreDb
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import java.io.File

/** 全局依赖容器:App 进程单例,页面直接引用。 */
class AppContainer(context: Context) {
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val db: ScoreDb = openScoreDb(context.applicationContext)

    val layouts = LayoutStore(context)
    val player = PlayerEngine(appScope)

    /** 播放页 -> 库页联动的当前选中乐谱。 */
    var selectedScoreId: Long = -1L

    /**
     * 用户显式锁定的演奏目标包名;null = 自动取"最近一个非覆盖层前台"。
     * 绝对坐标注入必须知道自己在往谁身上点,自动检测在悬浮小窗下不可靠。
     */
    var targetPackage: String? = null
}

private const val DB_NAME = "scores.db"

/**
 * 打开曲谱库。
 *
 * 刻意不使用 fallbackToDestructiveMigration():那条路径在 schema 不匹配时会
 * 静默删掉用户手机上存过的全部乐谱。这里改为"先把库文件连同 -wal/-shm 挪成带
 * 时间戳的旁路副本,再新建空库",数据可找回,失败也看得见。
 */
private fun openScoreDb(context: Context): ScoreDb {
    fun build() = Room.databaseBuilder(context, ScoreDb::class.java, DB_NAME).build()
    return try {
        build().also { it.openHelper.writableDatabase.close() }
    } catch (e: Exception) {
        val stamp = System.currentTimeMillis()
        val primary = context.getDatabasePath(DB_NAME)
        val dir = primary.parentFile
        runCatching {
            for (suffix in listOf("", "-wal", "-shm")) {
                val stale = File(dir, DB_NAME + suffix)
                if (stale.exists()) {
                    stale.renameTo(File(dir, "${stale.name}.incompatible-$stamp"))
                }
            }
        }
        android.util.Log.e(
            "AmpApp",
            "曲谱库无法打开(旧版 schema 或文件损坏),已保留副本 scores.db.incompatible-$stamp 并新建空库: ${e.message}",
        )
        build()
    }
}

object AppHolder {
    @Volatile
    private var container: AppContainer? = null

    fun get(context: Context): AppContainer =
        container ?: synchronized(this) {
            container ?: AppContainer(context.applicationContext).also { container = it }
        }
}

class AmpApp : Application() {

    override fun onCreate() {
        super.onCreate()
        AppHolder.get(this)
    }
}
