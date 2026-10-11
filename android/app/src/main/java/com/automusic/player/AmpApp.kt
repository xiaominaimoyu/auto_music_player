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

/** 全局依赖容器:App 进程单例,页面直接引用。 */
class AppContainer(context: Context) {
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val db: ScoreDb = Room.databaseBuilder(context, ScoreDb::class.java, "scores.db")
        .fallbackToDestructiveMigration()
        .build()

    val layouts = LayoutStore(context)
    val player = PlayerEngine(appScope)

    /** 播放页 -> 库页联动的当前选中乐谱。 */
    var selectedScoreId: Long = -1L
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
