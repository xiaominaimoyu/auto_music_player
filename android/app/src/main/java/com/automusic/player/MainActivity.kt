package com.automusic.player

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.automusic.player.ui.AmpNavHost
import com.automusic.player.ui.theme.AmpTheme

class MainActivity : ComponentActivity() {

    /** 拒绝也不阻塞演奏,只是失去前台服务通知上的「停止演奏」入口。 */
    private val requestNotificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Android 13+ 未授予通知权限时,前台服务通知被系统抑制,演奏期间就没有
        // 不切回应用也能停止的入口(手机端没有桌面版 F8 的等价物)。
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            requestNotificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        enableEdgeToEdge()
        setContent {
            AmpTheme {
                AmpNavHost(container = AppHolder.get(this))
            }
        }
    }
}
