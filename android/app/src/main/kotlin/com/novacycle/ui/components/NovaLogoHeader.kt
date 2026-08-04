package com.novacycle.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.draw.clip
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.novacycle.R
import com.novacycle.ui.theme.LocalNovaTheme
import com.novacycle.ui.theme.spec
import kotlinx.coroutines.launch

/**
 * Top-center brand header for the main dashboard: the gold ring logo with a
 * living glow halo, plus the wordmark underneath.
 *
 * Tapping the logo:
 *  1. Spins it (Y-flip) with a shimmer sweep and a glow bloom, and
 *  2. Triggers the shared [LocalBreathState] breath — the ambient background,
 *     gauge glows and card rims all swell in response, so the whole page
 *     visibly reacts to the touch.
 *
 * Taps naturally count toward theme unlocks via the activity-level global
 * tap counter (this composable never consumes the down event before it).
 */
@Composable
fun NovaLogoHeader(modifier: Modifier = Modifier) {
    val spec = LocalNovaTheme.current.spec()
    val breath = LocalBreathState.current
    val scope = rememberCoroutineScope()

    val spin = remember { Animatable(0f) }      // 0..360 Y-rotation
    val shimmer = remember { Animatable(-1f) }  // -1..2 sweep position
    val burst = remember { Animatable(0f) }     // tap bloom, 0..1
    val orbit = remember { Animatable(0f) }     // orbit ring rotation

    Column(
        modifier = modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(116.dp)
                // Glow halo behind the ring — breathes with the shared pulse.
                .drawBehind {
                    val pulse = breath.pulse.value
                    val bloom = burst.value
                    val radius = size.minDimension * (0.55f + 0.12f * pulse + 0.12f * bloom)
                    drawCircle(
                        brush = Brush.radialGradient(
                            listOf(
                                spec.glowBright.copy(alpha = 0.28f + 0.42f * pulse + 0.28f * bloom),
                                spec.glow.copy(alpha = 0.10f + 0.25f * pulse + 0.16f * bloom),
                                Color.Transparent
                            ),
                            center = center, radius = radius
                        ),
                        radius = radius, center = center
                    )
                    // A fine orbit makes the tap read as a branded interaction,
                    // not just a rotating bitmap.
                    drawArc(
                        color = spec.glowBright.copy(alpha = 0.72f + 0.2f * bloom),
                        startAngle = -52f + orbit.value,
                        sweepAngle = 112f,
                        useCenter = false,
                        style = Stroke(width = 1.5.dp.toPx(), cap = StrokeCap.Round)
                    )
                    drawCircle(
                        color = spec.rim.copy(alpha = 0.55f),
                        radius = size.minDimension * 0.39f,
                        style = Stroke(width = 1.dp.toPx())
                    )
                }
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication = null // the breath IS the feedback
                ) {
                    scope.launch {
                        launch { breath.breathe() }
                        launch {
                            burst.snapTo(0f)
                            burst.animateTo(1f, tween(240, easing = FastOutSlowInEasing))
                            burst.animateTo(0f, tween(800, easing = FastOutSlowInEasing))
                        }
                        launch {
                            orbit.snapTo(0f)
                            orbit.animateTo(360f, tween(900, easing = FastOutSlowInEasing))
                        }
                        launch {
                            shimmer.snapTo(-1f)
                            shimmer.animateTo(2f, tween(700, easing = FastOutSlowInEasing))
                        }
                        spin.snapTo(0f)
                        spin.animateTo(360f, tween(900, easing = FastOutSlowInEasing))
                    }
                }
        ) {
            Image(
                painter = painterResource(R.drawable.nova_logo),
                contentDescription = "NovaCycle logo — tap to make the app breathe",
                modifier = Modifier
                    // The source artwork has a dark square background. Clipping
                    // it to a medallion makes that background part of the mark
                    // instead of a visible bitmap rectangle.
                    .size(88.dp)
                    .clip(CircleShape)
                    .graphicsLayer {
                        rotationY = spin.value
                        val scale = 1f + 0.06f * burst.value
                        scaleX = scale
                        scaleY = scale
                        cameraDistance = 12f * density
                    }
            )
            // Diagonal shimmer sweep across the ring while spinning
            val sweep = shimmer.value
            if (sweep > -1f && sweep < 2f) {
                Canvas(Modifier.size(88.dp)) {
                    drawRect(
                        brush = Brush.linearGradient(
                            listOf(Color.Transparent, spec.glowBright.copy(alpha = 0.55f), Color.Transparent),
                            start = Offset(size.width * sweep - size.width * 0.4f, 0f),
                            end = Offset(size.width * sweep + size.width * 0.4f, size.height)
                        )
                    )
                }
            }
        }
        androidx.compose.material3.Text(
            text = "NOVACYCLE",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.5.sp,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(top = 0.dp)
        )
        androidx.compose.material3.Text(
            text = "— ${LocalNovaTheme.current.displayName.uppercase()} —",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.7f)
        )
    }
}
