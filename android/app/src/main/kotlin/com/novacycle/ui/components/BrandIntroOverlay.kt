package com.novacycle.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.novacycle.R

/**
 * Branded cold-start intro matching the splash storyboard:
 *   Step 1 — gold star ignition
 *   Step 2 — cycle ring (N mark) fade-in
 *   Step 3 — brand text (full logo) fade-in
 * Runs once per process launch, right after the system splash screen, then
 * fades away to reveal the app. Consumes all input while visible.
 */
object BrandIntro {
    /** True once the intro has played in this process. */
    var hasPlayed = false

    /** True while the intro overlay is on screen (used to suppress tap counting). */
    @Volatile
    var isVisible = false
}

private const val INTRO_DURATION_MS = 1600

@Composable
fun BrandIntroOverlay(
    themeGlow: Color = Color(0xFFCBA135),
    themeGlowBright: Color = Color(0xFFFFD700),
    bgColor: Color = Color(0xFF0D0D0D)
) {
    var visible by remember { mutableStateOf(!BrandIntro.hasPlayed) }
    if (!visible) {
        BrandIntro.isVisible = false
        return
    }
    BrandIntro.isVisible = true

    val progress = remember { Animatable(0f) }
    LaunchedEffect(Unit) {
        BrandIntro.hasPlayed = true
        progress.animateTo(1f, tween(INTRO_DURATION_MS, easing = LinearEasing))
        BrandIntro.isVisible = false
        visible = false
    }

    val p = progress.value
    // Phase windows (fractions of the timeline)
    val starAlpha  = ramp(p, 0.00f, 0.18f) * (1f - ramp(p, 0.45f, 0.62f))
    val ringAlpha  = ramp(p, 0.30f, 0.55f) * (1f - ramp(p, 0.60f, 0.72f))
    val logoAlpha  = ramp(p, 0.62f, 0.80f)
    val overlayAlpha = 1f - ramp(p, 0.88f, 1.00f)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .graphicsLayer { alpha = overlayAlpha }
            .background(bgColor)
            .pointerInput(Unit) {
                // Swallow all input while the intro is visible so taps cannot
                // reach the UI underneath (or inflate the tap counter).
                awaitPointerEventScope {
                    while (true) {
                        val event = awaitPointerEvent(PointerEventPass.Initial)
                        event.changes.forEach { it.consume() }
                    }
                }
            },
        contentAlignment = Alignment.Center
    ) {
        // Step 1 — gold star ignition
        Canvas(Modifier.fillMaxSize().graphicsLayer { alpha = starAlpha }) {
            val c = Offset(size.width * 0.5f, size.height * 0.42f)
            val r = size.minDimension * (0.06f + 0.10f * p)
            drawCircle(
                brush = Brush.radialGradient(
                    listOf(Color(0xFFFFF3C0), themeGlowBright, Color.Transparent),
                    center = c, radius = r * 2.2f
                ),
                radius = r * 2.2f, center = c
            )
            val flare = r * 3.4f
            listOf(0f, 90f, 45f, 135f).forEachIndexed { i, deg ->
                val len = if (i < 2) flare else flare * 0.55f
                val rad = Math.toRadians(deg.toDouble())
                val dx = (Math.cos(rad) * len).toFloat()
                val dy = (Math.sin(rad) * len).toFloat()
                drawLine(
                    color = themeGlowBright.copy(alpha = 0.85f),
                    start = c - Offset(dx, dy), end = c + Offset(dx, dy),
                    strokeWidth = r * 0.16f, cap = StrokeCap.Round
                )
            }
        }
        
        // Step 2 — cycle ring fade-in
        Canvas(
            modifier = Modifier
                .fillMaxWidth(0.35f)
                .aspectRatio(1f)
                .graphicsLayer { alpha = ringAlpha }
        ) {
            val cx = size.width / 2f
            val cy = size.height / 2f
            val r = size.minDimension / 2f

            val outerStroke = 2.5.dp.toPx()
            drawArc(
                color = themeGlow,
                startAngle = 150f,
                sweepAngle = 240f,
                useCenter = false,
                style = Stroke(width = outerStroke, cap = StrokeCap.Round)
            )

            val innerRadius = r - 8.dp.toPx()
            drawArc(
                color = themeGlow.copy(alpha = 0.5f),
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

            drawLine(themeGlow, Offset(cx - offX, cy - offY), Offset(cx - offX, cy + offY), strokeWidth = nStroke, cap = StrokeCap.Round)
            drawLine(themeGlow, Offset(cx + offX, cy - offY), Offset(cx + offX, cy + offY), strokeWidth = nStroke, cap = StrokeCap.Round)
            drawLine(themeGlow, Offset(cx - offX, cy - offY), Offset(cx + offX, cy + offY), strokeWidth = nStroke, cap = StrokeCap.Round)
        }

        // Step 3 — brand text fade-in (full lock-up) + shimmer
        Box(contentAlignment = Alignment.Center) {
            Image(
                painter = painterResource(R.drawable.nova_launch),
                contentDescription = "NovaCycle",
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .padding(24.dp)
                    .graphicsLayer { alpha = logoAlpha }
            )
            
            val shimmerProgress = ramp(p, 0.70f, 0.95f)
            if (shimmerProgress > 0f && shimmerProgress < 1f) {
                Canvas(Modifier.fillMaxWidth(0.8f).padding(24.dp)) {
                    val sweep = shimmerProgress * 3f - 1f
                    drawRect(
                        brush = Brush.linearGradient(
                            listOf(Color.Transparent, themeGlowBright.copy(alpha = 0.5f), Color.Transparent),
                            start = Offset(size.width * sweep - size.width * 0.2f, 0f),
                            end = Offset(size.width * sweep + size.width * 0.2f, size.height)
                        )
                    )
                }
            }
        }
    }
}

/** Linear 0→1 ramp of [x] between [from] and [to]. */
private fun ramp(x: Float, from: Float, to: Float): Float =
    ((x - from) / (to - from)).coerceIn(0f, 1f)
