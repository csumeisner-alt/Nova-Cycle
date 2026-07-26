package com.novacycle.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.novacycle.R

/**
 * The tappable NovaCycle emblem shown in the dashboard header.
 *
 * Each tap plays the brand animation (Compose equivalent of the spec's
 * star_glow + ring_fade XML anims):
 *  1. star-glow pulse — the emblem scales up 1.0→1.18 and back with a bright flash
 *  2. ring fade — an expanding gold ring fades out around the emblem
 * and reports the tap to [onTap] (which feeds the 20,000-tap achievement).
 */
@Composable
fun NovaLogo(
    onTap: () -> Unit,
    modifier: Modifier = Modifier,
    size: Dp = 44.dp
) {
    val scale = remember { Animatable(1f) }
    val ringScale = remember { Animatable(1f) }
    val ringAlpha = remember { Animatable(0f) }
    var tapPulse by remember { mutableIntStateOf(0) }

    // Restartable animation: every tap bumps tapPulse, cancelling & replaying.
    LaunchedEffect(tapPulse) {
        if (tapPulse == 0) return@LaunchedEffect
        scale.snapTo(1f)
        scale.animateTo(1.18f, tween(90))
        scale.animateTo(1f, tween(160))
    }
    LaunchedEffect(tapPulse) {
        if (tapPulse == 0) return@LaunchedEffect
        ringScale.snapTo(1f)
        ringAlpha.snapTo(0.8f)
        ringScale.animateTo(1.9f, tween(350))
        ringAlpha.animateTo(0f, tween(350))
    }

    Box(modifier = modifier.size(size), contentAlignment = Alignment.Center) {
        // Expanding, fading gold ring
        Box(
            Modifier
                .size(size)
                .graphicsLayer {
                    scaleX = ringScale.value
                    scaleY = ringScale.value
                    alpha = ringAlpha.value
                }
                .border(2.dp, MaterialTheme.colorScheme.primary, CircleShape)
        )
        Image(
            painter = painterResource(R.drawable.nova_logo),
            contentDescription = "NovaCycle logo — tap for the 20,000-tap achievement",
            modifier = Modifier
                .size(size)
                .scale(scale.value)
                .clip(CircleShape)
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null
                ) {
                    tapPulse++
                    onTap()
                }
        )
    }
}
