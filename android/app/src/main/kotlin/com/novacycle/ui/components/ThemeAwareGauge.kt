package com.novacycle.ui.components

import android.provider.Settings
import androidx.compose.animation.animateColor
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.novacycle.domain.model.ConfidenceZone
import com.novacycle.domain.model.GaugeState
import com.novacycle.ui.theme.*
import kotlin.math.cos
import kotlin.math.sin

private data class AnimatedGaugeColors(
    val start: Color, val end: Color, val buy: Color, val sell: Color,
    val hold: Color, val needle: Color, val label: Color, val glow: Color
)

@Composable
private fun reducedMotionEnabled(): Boolean {
    // Android's animator scale is the platform-wide reduced-motion signal.
    // Reading it once per composition keeps rendering cheap and deterministic.
    val context = androidx.compose.ui.platform.LocalView.current.context
    return Settings.Global.getFloat(
        context.contentResolver,
        Settings.Global.ANIMATOR_DURATION_SCALE, 1f
    ) == 0f
}

@Composable
fun ThemeAwareGauge(
    gaugeState: GaugeState,
    label: String,
    modifier: Modifier = Modifier
) {
    val theme = LocalNovaTheme.current
    val palette = remember(theme) { theme.gaugePalette() }
    val reducedMotion = reducedMotionEnabled()
    val transition = androidx.compose.animation.core.updateTransition(
        targetState = palette, label = "gaugeThemeTransition"
    )
    val colors = AnimatedGaugeColors(
        start = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.arcStart }.value,
        end = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.arcEnd }.value,
        buy = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.buy }.value,
        sell = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.sell }.value,
        hold = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.hold }.value,
        needle = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.needle }.value,
        label = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.label }.value,
        glow = transition.animateColor(transitionSpec = { tween(700, easing = FastOutSlowInEasing) }) { it.glow }.value
    )
    val isFallback = gaugeState.isFallback
    val signal = gaugeState.displaySignal.uppercase()
    val signalColor = when {
        isFallback || signal.contains("HOLD") -> colors.hold
        signal.contains("BUY") -> colors.buy
        signal.contains("SELL") -> colors.sell
        else -> colors.hold
    }
    val score = gaugeState.score.coerceIn(-100f, 100f)
    val targetAngle = 180f - ((score + 100f) / 200f) * 180f
    val duration = when (palette.motion) {
        GaugeMotionStyle.CONFIDENT_SWEEP -> 1100
        GaugeMotionStyle.FLUID_WAVE -> 800
        GaugeMotionStyle.ELEGANT_SWIRL -> 950
        GaugeMotionStyle.REFINED_GLIDE -> 1250
    }
    val angle = remember { Animatable(180f) }
    LaunchedEffect(targetAngle, reducedMotion) {
        if (reducedMotion) angle.snapTo(targetAngle)
        else angle.animateTo(targetAngle, tween(duration, easing = FastOutSlowInEasing))
    }
    val pulse = remember { Animatable(0f) }
    LaunchedEffect(reducedMotion, palette.motion) {
        if (reducedMotion) pulse.snapTo(0f)
        else while (true) {
            pulse.animateTo(1f, tween(1500, easing = LinearEasing))
            pulse.animateTo(0f, tween(1500, easing = LinearEasing))
        }
    }
    val zoneColor = if (isFallback) colors.hold else when (gaugeState.confidenceZone) {
        ConfidenceZone.WEAK -> colors.sell
        ConfidenceZone.UNCERTAIN -> colors.hold
        ConfidenceZone.STRONG -> colors.buy
    }
    Column(
        modifier = modifier
            .padding(8.dp)
            .drawBehind {
                drawRect(
                    brush = Brush.radialGradient(
                        listOf(palette.background.copy(alpha = 0.36f), Color.Transparent),
                        center = Offset(size.width / 2f, size.height * 0.55f)
                    )
                )
            },
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(label, style = MaterialTheme.typography.labelMedium,
            color = colors.label, fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center)
        Spacer(Modifier.height(4.dp))
        Canvas(Modifier.fillMaxWidth().aspectRatio(2f)) {
            val cx = size.width / 2f
            val cy = size.height
            val radius = size.width / 2f - 8.dp.toPx()
            val stroke = 18.dp.toPx()
            drawCircle(
                brush = Brush.radialGradient(
                    listOf(colors.glow.copy(alpha = 0.26f + 0.18f * pulse.value), Color.Transparent),
                    center = Offset(cx, cy), radius = radius + stroke * 1.7f
                ),
                radius = radius + stroke * 1.7f, center = Offset(cx, cy)
            )
            drawThemeArc(cx, cy, radius, stroke, colors, isFallback)
            drawTicks(cx, cy, radius, stroke, colors.hold)
            drawThemeNeedle(cx, cy, radius, angle.value, colors.needle, signalColor, isFallback)
        }
        Spacer(Modifier.height(6.dp))
        Text("${gaugeState.confidencePercent.coerceIn(0, 100)}% confidence",
            fontSize = 24.sp, fontWeight = FontWeight.Bold, color = zoneColor,
            textAlign = TextAlign.Center)
        val trend = if (isFallback) "NEUTRAL" else gaugeState.trend.uppercase()
        val display = if (isFallback) "NEUTRAL / HOLD" else gaugeState.displaySignal
        Text("${themeTrendGlyph(trend)} $trend · $display",
            style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold,
            color = zoneColor, textAlign = TextAlign.Center)
        Text(if (isFallback) "No data" else gaugeState.confidenceZone.label,
            style = MaterialTheme.typography.labelSmall,
            color = colors.label.copy(alpha = 0.72f), textAlign = TextAlign.Center)
    }
}

private fun themeTrendGlyph(trend: String): String = when (trend.uppercase()) {
    "UP" -> "▲"
    "DOWN" -> "▼"
    else -> "◆"
}

private fun DrawScope.drawThemeArc(
    cx: Float, cy: Float, radius: Float, stroke: Float,
    colors: AnimatedGaugeColors, muted: Boolean
) {
    val left = cx - radius
    val top = cy - radius
    val size = Size(radius * 2f, radius * 2f)
    drawArc(
        color = Color.Black.copy(alpha = 0.28f),
        startAngle = 180f, sweepAngle = 180f, useCenter = false,
        topLeft = Offset(left, top), size = size,
        style = Stroke(stroke, cap = StrokeCap.Round)
    )
    repeat(60) { i ->
        val t = i / 59f
        val color = if (muted) colors.hold.copy(alpha = 0.5f)
        else Color(
            red = colors.start.red + (colors.end.red - colors.start.red) * t,
            green = colors.start.green + (colors.end.green - colors.start.green) * t,
            blue = colors.start.blue + (colors.end.blue - colors.start.blue) * t
        )
        drawArc(
            color = color,
            startAngle = 180f + i * 3f, sweepAngle = 3.4f, useCenter = false,
            topLeft = Offset(left, top), size = size,
            style = Stroke(stroke, cap = StrokeCap.Butt)
        )
    }
}

private fun DrawScope.drawTicks(cx: Float, cy: Float, radius: Float, stroke: Float, color: Color) {
    val inner = radius - stroke / 2f - 4.dp.toPx()
    val outer = radius + stroke / 2f + 4.dp.toPx()
    listOf(180f, 225f, 270f, 315f, 360f).forEach { degrees ->
        val radians = Math.toRadians(degrees.toDouble())
        drawLine(color.copy(alpha = 0.65f),
            Offset(cx + inner * cos(radians).toFloat(), cy + inner * sin(radians).toFloat()),
            Offset(cx + outer * cos(radians).toFloat(), cy + outer * sin(radians).toFloat()),
            2.dp.toPx(), StrokeCap.Round)
    }
}

private fun DrawScope.drawThemeNeedle(
    cx: Float, cy: Float, radius: Float, angle: Float,
    needle: Color, accent: Color, muted: Boolean
) {
    val radians = Math.toRadians(angle.toDouble())
    val length = radius - 10.dp.toPx()
    val tip = Offset(cx + length * cos(radians).toFloat(), cy + length * sin(radians).toFloat())
    val color = if (muted) Color(0xFF777777) else needle
    drawLine(Color.Black.copy(alpha = 0.42f), Offset(cx + 2f, cy + 2f),
        Offset(tip.x + 2f, tip.y + 2f), 6.dp.toPx(), StrokeCap.Round)
    drawLine(color, Offset(cx, cy), tip, 4.dp.toPx(), StrokeCap.Round)
    if (!muted) drawLine(Color.White.copy(alpha = 0.28f), Offset(cx, cy), tip, 1.dp.toPx(), StrokeCap.Round)
    drawCircle(Color(0xFF171717), 10.dp.toPx(), Offset(cx, cy))
    drawCircle(if (muted) color else accent, 6.dp.toPx(), Offset(cx, cy))
}