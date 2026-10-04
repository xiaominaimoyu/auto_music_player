package com.automusic.player

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder

/** 演奏期间的前台保活服务:仅提升进程优先级,不做实际演奏工作。 */
class PlaybackService : Service() {

    companion object {
        private const val CHANNEL_ID = "playback"
        private const val NOTIFICATION_ID = 1
        private const val ACTION_STOP = "com.automusic.player.action.STOP"

        fun start(context: Context) {
            context.startForegroundService(Intent(context, PlaybackService::class.java))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, PlaybackService::class.java))
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "演奏中", NotificationManager.IMPORTANCE_LOW)
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // 通知栏「停止演奏」:下拉通知栏不必离开游戏,是当前唯一的即时停止入口。
        if (intent?.action == ACTION_STOP) {
            AppHolder.get(applicationContext).player.stop()
            stopSelf()
            return START_NOT_STICKY
        }
        val stopIntent = Intent(this, PlaybackService::class.java).setAction(ACTION_STOP)
        val stopPending = PendingIntent.getService(
            this, 0, stopIntent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val notification: Notification =
            androidx.core.app.NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_media_play)
                .setContentTitle("正在演奏")
                .setContentText("自动演奏进行中")
                .setOngoing(true)
                .addAction(0, "停止演奏", stopPending)
                .build()
        startForeground(NOTIFICATION_ID, notification)
        return START_NOT_STICKY
    }
}
