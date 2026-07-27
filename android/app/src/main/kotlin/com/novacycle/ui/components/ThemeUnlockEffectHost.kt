package com.novacycle.ui.components

import android.media.AudioAttributes
import android.media.SoundPool
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import com.novacycle.R
import com.novacycle.viewmodel.ThemeViewModel

/**
 * Invisible host that reacts to theme-unlock events with a SoundPool "ping" and
 * haptic feedback, no matter which screen the user is on when the milestone is
 * crossed. The visual shimmer lives in the Settings theme picker.
 */
@Composable
fun ThemeUnlockEffectHost(themeViewModel: ThemeViewModel) {
    val context = LocalContext.current
    val haptics = LocalHapticFeedback.current

    val soundPool = remember {
        SoundPool.Builder()
            .setMaxStreams(1)
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .build()
    }
    val pingId = remember { soundPool.load(context, R.raw.theme_unlock_ping, 1) }

    DisposableEffect(Unit) {
        onDispose { soundPool.release() }
    }

    LaunchedEffect(themeViewModel) {
        themeViewModel.unlockEvents.collect {
            soundPool.play(pingId, 1f, 1f, 1, 0, 1f)
            haptics.performHapticFeedback(HapticFeedbackType.LongPress)
        }
    }
}
