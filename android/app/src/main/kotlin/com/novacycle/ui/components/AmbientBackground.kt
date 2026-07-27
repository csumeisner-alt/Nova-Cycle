package com.novacycle.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.novacycle.ui.theme.AmbientStyle
import com.novacycle.ui.theme.LocalNovaTheme
import com.novacycle.ui.theme.NovaStripeGreen
import com.novacycle.ui.theme.NovaStripeRed
import com.novacycle.ui.theme.spec
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Full-screen ambient layer rendered BEHIND all screen content — this is what
 * makes pages feel alive. One slow infinite phase drives gentle motion; the
 * shared [LocalBreathState] pulse briefly intensifies everything when the
 * logo is tapped.
 *
 * Performance: a single Canvas with a few dozen draw ops per frame and one
 * 24s master transition — cheap enough to sit under trading dashboards.
 */
@Composable
fun AmbientBackground(modifier: Modifier = Modifier) {
    val spec = LocalNovaTheme.current.spec()
    val breath = LocalBreathState.current

    val transition = rememberInfiniteTransition(label = "ambient")
    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            tween(durationMillis = 24_000, easing = LinearEasing),
            RepeatMode.Restart
        ),
        label = "ambientPhase"
    )

    Canvas(modifier.fillMaxSize()) {
        val pulse = breath.pulse.value
        when (spec.ambient) {
            AmbientStyle.EMBER_GLOW       -> drawEmberGlow(phase, pulse, spec.glow, spec.glowBright)
            AmbientStyle.LIGHT_RIBBONS    -> drawLightRibbons(phase, pulse, spec.glow)
            AmbientStyle.AURORA_WISPS     -> drawAuroraWisps(phase, pulse, spec.glow)
            AmbientStyle.HERITAGE_PATTERN -> drawHeritagePattern(phase, pulse, spec.glow)
        }
    }
}

/** Executive Gold — two slow-drifting warm ember glows + drifting gold sparks. */
private fun DrawScope.drawEmberGlow(phase: Float, pulse: Float, glow: Color, bright: Color) {
    val t = phase * 2f * PI.toFloat()
    val boost = 1f + pulse * 1.2f
    val c1 = Offset(
        size.width * (0.25f + 0.10f * sin(t)),
        size.height * (0.20f + 0.06f * cos(t * 0.7f))
    )
    val c2 = Offset(
        size.width * (0.80f - 0.08f * cos(t * 0.9f)),
        size.height * (0.75f + 0.07f * sin(t * 0.6f))
    )
    drawCircle(
        brush = Brush.radialGradient(
            listOf(bright.copy(alpha = 0.10f * boost), glow.copy(alpha = 0.05f * boost), Color.Transparent),
            center = c1, radius = size.minDimension * 0.55f
        ),
        radius = size.minDimension * 0.55f, center = c1
    )
    drawCircle(
        brush = Brush.radialGradient(
            listOf(glow.copy(alpha = 0.08f * boost), Color.Transparent),
            center = c2, radius = size.minDimension * 0.45f
        ),
        radius = size.minDimension * 0.45f, center = c2
    )
    // A few gold sparks rising slowly
    repeat(7) { i ->
        val p = (phase * 2f + i / 7f) % 1f
        val xFrac = (0.11f + 0.13f * i + 0.05f * sin(t + i * 1.9f)).mod(1f)
        drawCircle(
            color = bright.copy(alpha = (0.22f * (1f - p) * boost).coerceAtMost(1f)),
            radius = 2.dp.toPx() * (0.7f + 0.3f * sin(t * 2f + i)),
            center = Offset(size.width * xFrac, size.height * (1f - p))
        )
    }
}

/** Rose Luxe — flowing neon light ribbons snaking down the screen. */
private fun DrawScope.drawLightRibbons(phase: Float, pulse: Float, glow: Color) {
    val t = phase * 2f * PI.toFloat()
    val boost = 1f + pulse * 1.5f
    repeat(3) { i ->
        val amp = size.width * (0.22f + 0.05f * i)
        val path = Path()
        var first = true
        var y = -size.height * 0.1f
        while (y <= size.height * 1.1f) {
            val x = size.width * 0.5f +
                amp * sin(y / size.height * 4.2f + t * (1f + 0.3f * i) + i * 2.1f) +
                size.width * 0.18f * (i - 1)
            if (first) { path.moveTo(x, y); first = false } else path.lineTo(x, y)
            y += size.height / 24f
        }
        val alpha = (0.05f + 0.02f * i) * boost
        // Wide soft glow pass + thin bright core pass
        drawPath(path, glow.copy(alpha = alpha.coerceAtMost(0.4f)), style = Stroke(width = 26.dp.toPx(), cap = StrokeCap.Round))
        drawPath(path, glow.copy(alpha = (alpha * 2.2f).coerceAtMost(0.6f)), style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round))
    }
}

/** Mint Luxe — soft aurora wisps sweeping across the screen. */
private fun DrawScope.drawAuroraWisps(phase: Float, pulse: Float, glow: Color) {
    val t = phase * 2f * PI.toFloat()
    val boost = 1f + pulse * 1.4f
    repeat(3) { i ->
        val path = Path()
        val baseY = size.height * (0.25f + 0.28f * i)
        var first = true
        var x = -size.width * 0.1f
        while (x <= size.width * 1.1f) {
            val y = baseY +
                size.height * 0.06f * sin(x / size.width * 3.5f + t * (0.8f + 0.25f * i) + i * 1.7f)
            if (first) { path.moveTo(x, y); first = false } else path.lineTo(x, y)
            x += size.width / 20f
        }
        val alpha = 0.05f * boost
        drawPath(path, glow.copy(alpha = alpha.coerceAtMost(0.3f)), style = Stroke(width = 42.dp.toPx(), cap = StrokeCap.Round))
        drawPath(path, glow.copy(alpha = (alpha * 1.8f).coerceAtMost(0.45f)), style = Stroke(width = 8.dp.toPx(), cap = StrokeCap.Round))
    }
}

/** Heritage — faint interlocking-ring monogram lattice + central racing stripe with a light sweep. */
private fun DrawScope.drawHeritagePattern(phase: Float, pulse: Float, glow: Color) {
    // Monogram lattice: staggered pairs of interlocking rings
    val ring = Color(0xFF6E5D45).copy(alpha = 0.18f)
    val cell = 96.dp.toPx()
    val r = cell * 0.20f
    val stroke = Stroke(width = 1.5.dp.toPx())
    var row = 0
    var cy = cell * 0.4f
    while (cy < size.height + cell) {
        var cx = if (row % 2 == 0) cell * 0.4f else cell * 0.9f
        while (cx < size.width + cell) {
            drawCircle(ring, r, Offset(cx - r * 0.45f, cy), style = stroke)
            drawCircle(ring, r, Offset(cx + r * 0.45f, cy), style = stroke)
            cx += cell
        }
        cy += cell
        row++
    }
    // Central vertical green/red racing stripe
    val stripeW = size.width * 0.16f
    val left = (size.width - stripeW) / 2f
    drawRect(NovaStripeGreen.copy(alpha = 0.85f), Offset(left, 0f), Size(stripeW, size.height))
    drawRect(NovaStripeRed.copy(alpha = 0.9f), Offset(left + stripeW * 0.335f, 0f), Size(stripeW * 0.33f, size.height))
    // Slow copper light sweep down the stripe (brightens on breath)
    val sweepHalf = 160.dp.toPx() / 2f
    val sweepY = size.height * ((phase * 1.6f) % 1f)
    drawRect(
        brush = Brush.verticalGradient(
            listOf(Color.Transparent, glow.copy(alpha = 0.22f + 0.35f * pulse), Color.Transparent),
            startY = sweepY - sweepHalf, endY = sweepY + sweepHalf
        ),
        topLeft = Offset(left, sweepY - sweepHalf),
        size = Size(stripeW, sweepHalf * 2f)
    )
}
