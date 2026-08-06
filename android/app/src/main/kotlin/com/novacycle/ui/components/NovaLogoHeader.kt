package com.novacycle.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.novacycle.ui.theme.LocalNovaTheme
import com.novacycle.ui.theme.NovaTheme
import com.novacycle.ui.theme.spec
import kotlinx.coroutines.delay
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
    val theme = LocalNovaTheme.current
    val spec = theme.spec()
    val breath = LocalBreathState.current
    val scope = rememberCoroutineScope()

    val spin = remember { Animatable(0f) }      // 0..360 Y-rotation (or Z for Heritage)
    val scaleAnim = remember { Animatable(1f) } // pulse scale for Datastream
    val shimmer = remember { Animatable(-1f) }  // -1..2 sweep position
    val burst = remember { Animatable(0f) }     // tap bloom, 0..1
    val orbit = remember { Animatable(0f) }     // orbit ring rotation

    Column(
        modifier = modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        val baseAlpha = when (theme) {
            NovaTheme.AURORA_FLUX -> 0.55f
            NovaTheme.DARK_LUXE -> 0.35f
            NovaTheme.MINT_LUXE -> 0.20f
            NovaTheme.CRIMSON_PULSE -> 0.15f
        }

        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(116.dp)
                // Glow halo behind the ring — breathes with the shared pulse.
                .drawBehind {
                    val pulse = breath.pulse.value
                    val bloom = burst.value
                    val radius = size.minDimension * (baseAlpha + 0.12f * pulse + 0.12f * bloom)
                    drawCircle(
                        brush = Brush.radialGradient(
                            listOf(
                                spec.glowBright.copy(alpha = baseAlpha + 0.42f * pulse + 0.28f * bloom),
                                spec.glow.copy(alpha = (baseAlpha * 0.4f) + 0.25f * pulse + 0.16f * bloom),
                                Color.Transparent
                            ),
                            center = center, radius = radius
                        ),
                        radius = radius, center = center
                    )
                    // A fine orbit makes the tap read as a branded interaction
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
                        
                        if (theme == NovaTheme.DARK_LUXE || theme == NovaTheme.MINT_LUXE) {
                            launch {
                                burst.snapTo(0f)
                                burst.animateTo(1f, tween(200, easing = FastOutSlowInEasing))
                            }
                        }

                        when (theme) {
                            NovaTheme.DARK_LUXE -> {
                                launch {
                                    spin.snapTo(0f)
                                    spin.animateTo(360f, spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessMediumLow))
                                }
                                launch {
                                    orbit.snapTo(0f)
                                    orbit.animateTo(720f, tween(1400, easing = FastOutSlowInEasing))
                                    orbit.animateTo(360f, tween(600, easing = FastOutSlowInEasing))
                                }
                                launch {
                                    shimmer.snapTo(-1f)
                                    shimmer.animateTo(2f, tween(800, easing = FastOutSlowInEasing))
                                }
                            }
                            NovaTheme.MINT_LUXE -> {
                                launch {
                                    scaleAnim.snapTo(1f)
                                    scaleAnim.animateTo(1.12f, tween(300))
                                    scaleAnim.animateTo(1f, tween(300))
                                }
                                launch {
                                    orbit.snapTo(0f)
                                    orbit.animateTo(180f, tween(400))
                                    delay(100)
                                    orbit.animateTo(360f, tween(400))
                                }
                                launch {
                                    repeat(3) {
                                        shimmer.snapTo(-1f)
                                        shimmer.animateTo(2f, tween(300))
                                    }
                                }
                            }
                            NovaTheme.AURORA_FLUX -> {
                                launch {
                                    spin.snapTo(0f)
                                    spin.animateTo(180f, tween(800))
                                    delay(200)
                                    spin.animateTo(360f, tween(800))
                                }
                                launch {
                                    burst.snapTo(0f)
                                    burst.animateTo(1f, tween(200))
                                    delay(600)
                                    burst.animateTo(0f, tween(400))
                                }
                                launch {
                                    orbit.snapTo(0f)
                                    orbit.animateTo(360f, tween(1800, easing = LinearEasing))
                                }
                                launch {
                                    shimmer.snapTo(-1f)
                                    shimmer.animateTo(2f, tween(1200))
                                }
                            }
                            NovaTheme.CRIMSON_PULSE -> {
                                launch {
                                    spin.snapTo(0f)
                                    spin.animateTo(15f, tween(200))
                                    spin.animateTo(-15f, tween(400))
                                    spin.animateTo(0f, tween(200))
                                }
                                launch {
                                    orbit.snapTo(0f)
                                    orbit.animateTo(360f, tween(700, easing = FastOutSlowInEasing))
                                }
                                launch {
                                    shimmer.snapTo(-1f)
                                    shimmer.animateTo(2f, tween(500))
                                }
                                launch {
                                    burst.snapTo(0f)
                                    burst.animateTo(1f, tween(150))
                                    burst.animateTo(0f, tween(300))
                                }
                            }
                        }
                    }
                }
        ) {
            val isHeritage = theme == NovaTheme.CRIMSON_PULSE
            
            Canvas(
                modifier = Modifier
                    .size(72.dp)
                    .graphicsLayer {
                        if (isHeritage) {
                            rotationZ = spin.value
                        } else {
                            rotationY = spin.value
                        }
                        val s = scaleAnim.value + 0.06f * burst.value
                        scaleX = s
                        scaleY = s
                        cameraDistance = 12f * density
                    }
            ) {
                val cx = size.width / 2f
                val cy = size.height / 2f
                val r = size.minDimension / 2f

                val outerStroke = 2.5.dp.toPx()
                drawArc(
                    color = theme.accent,
                    startAngle = 150f,
                    sweepAngle = 240f,
                    useCenter = false,
                    style = Stroke(width = outerStroke, cap = StrokeCap.Round)
                )

                val innerRadius = r - 8.dp.toPx()
                drawArc(
                    color = theme.accent.copy(alpha = 0.5f),
                    startAngle = 330f,
                    sweepAngle = 120f,
                    useCenter = false,
                    topLeft = Offset(cx - innerRadius, cy - innerRadius),
                    size = Size(innerRadius * 2, innerRadius * 2),
                    style = Stroke(width = outerStroke, cap = StrokeCap.Round)
                )

                val nStroke = 2.dp.toPx()
                val offX = 8.dp.toPx()
                val offY = 10.dp.toPx()

                drawLine(theme.accent, Offset(cx - offX, cy - offY), Offset(cx - offX, cy + offY), strokeWidth = nStroke, cap = StrokeCap.Round)
                drawLine(theme.accent, Offset(cx + offX, cy - offY), Offset(cx + offX, cy + offY), strokeWidth = nStroke, cap = StrokeCap.Round)
                drawLine(theme.accent, Offset(cx - offX, cy - offY), Offset(cx + offX, cy + offY), strokeWidth = nStroke, cap = StrokeCap.Round)
            }

            // Diagonal shimmer sweep across the ring
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
            letterSpacing = 3.sp,
            color = theme.accent,
            modifier = Modifier.padding(top = 0.dp)
        )
    }
}
