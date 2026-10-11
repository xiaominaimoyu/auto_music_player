package com.automusic.player

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.automusic.player.ui.AmpNavHost
import com.automusic.player.ui.theme.AmpTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AmpTheme {
                AmpNavHost(container = AppHolder.get(this))
            }
        }
    }
}
