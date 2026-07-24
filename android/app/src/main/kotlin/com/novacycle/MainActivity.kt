package com.novacycle

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.novacycle.ui.navigation.NovaCycleNavHost
import com.novacycle.ui.theme.NovaCycleTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * Single-activity architecture. All navigation is handled by Jetpack Navigation Compose.
 * Hilt injects dependencies throughout the Compose tree via hiltViewModel().
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Edge-to-edge display for modern Android look
        enableEdgeToEdge()
        setContent {
            NovaCycleTheme {
                NovaCycleNavHost()
            }
        }
    }
}
