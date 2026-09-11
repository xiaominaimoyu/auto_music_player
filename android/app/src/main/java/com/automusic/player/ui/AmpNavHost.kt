package com.automusic.player.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.LibraryMusic
import androidx.compose.material.icons.outlined.PlayCircle
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.TouchApp
import androidx.compose.material.icons.outlined.UploadFile
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.automusic.player.AppContainer
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Ink2

enum class Tab(val title: String) {
    UPLOAD("识别"),
    LIBRARY("乐谱库"),
    PLAY("演奏"),
    CALIB("标定"),
    SETTINGS("设置"),
}

@Composable
fun AmpNavHost(container: AppContainer) {
    var tab by rememberSaveable { mutableStateOf(Tab.UPLOAD) }

    Scaffold(
        bottomBar = {
            NavigationBar(containerColor = com.automusic.player.ui.theme.Surface1) {
                val icons = mapOf(
                    Tab.UPLOAD to Icons.Outlined.UploadFile,
                    Tab.LIBRARY to Icons.Outlined.LibraryMusic,
                    Tab.PLAY to Icons.Outlined.PlayCircle,
                    Tab.CALIB to Icons.Outlined.TouchApp,
                    Tab.SETTINGS to Icons.Outlined.Settings,
                )
                for (t in Tab.entries) {
                    val selected = tab == t
                    NavigationBarItem(
                        selected = selected,
                        onClick = { tab = t },
                        icon = { Icon(icons.getValue(t), contentDescription = t.title) },
                        label = { Text(t.title) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Brand,
                            selectedTextColor = Brand,
                            unselectedIconColor = Ink2,
                            unselectedTextColor = Ink2,
                        ),
                    )
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding)) {
            when (tab) {
                Tab.UPLOAD -> UploadScreen(container, onOpenSettings = { tab = Tab.SETTINGS })
                Tab.LIBRARY -> LibraryScreen(container, onGoPlay = { tab = Tab.PLAY })
                Tab.PLAY -> PlayScreen(container)
                Tab.CALIB -> CalibScreen(container)
                Tab.SETTINGS -> SettingsScreen(container)
            }
        }
    }
}
