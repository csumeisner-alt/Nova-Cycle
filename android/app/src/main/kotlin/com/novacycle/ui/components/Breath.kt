package com.novacycle.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf

/**
 * Shared "breath" of the whole UI. Tapping the logo calls [breathe]; anything
 * that wants to feel alive (ambient background, gauge glows, card rims, the
 * logo itself) reads [pulse] — 0 at rest, swelling to 1 at the peak of a
 * breath — and scales its glow/brightness with it.
 *
 * A single instance is provided at the activity root via [LocalBreathState],
 * so one tap ripples through every subscribed layer at once.
 */
class BreathState {
    /** 0 = at rest, 1 = peak of a breath. Animate-only; read from draw code. */
    val pulse = Animatable(0f)

    /** One full breath: quick swell, slow release. Safe to call repeatedly. */
    suspend fun breathe() {
        pulse.animateTo(1f, tween(durationMillis = 450, easing = FastOutSlowInEasing))
        pulse.animateTo(0f, tween(durationMillis = 1400, easing = FastOutSlowInEasing))
    }
}

val LocalBreathState = staticCompositionLocalOf { BreathState() }

@Composable
fun rememberBreathState(): BreathState = remember { BreathState() }
