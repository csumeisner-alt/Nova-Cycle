package com.novacycle.ui.components

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.novacycle.domain.model.GaugeState
import com.novacycle.ui.theme.spec
import kotlin.math.*

/**
 * DualGaugeWidget — semicircular arc gauge with animated needle.
 *
 * The gauge spans 180° (from left = -100 to right = +100).
 * Color gradient: Red (SELL) → Yellow (neutral) → Green (BUY).
 * Needle animates smoothly using spring physics.
 *
 * @param gaugeState  Signal state containing score, signal, and confidence
 * @param label       Gauge title, e.g. "Long-Trend" or "Short-Trend"
 */
@Composable
fun DualGaugeWidget(
    gaugeState: GaugeState,
    label: String,
    modifier: Modifier = Modifier
) {
    // Theme glow + shared breath: the halo behind the arc swells when the
    // logo is tapped, so gauges pulse with the rest of the page.
    val spec = com.novacycle.ui.theme.LocalNovaTheme.current.spec()
    val breath = LocalBreathState.current
    val score = gaugeState.score
    val signal = gaugeState.signal
    val confidence = gaugeState.confidence
    // Animate needle position: score [-100, +100] maps to angle [180°, 0°]
    // (left side = -100 = SELL = red, right side = +100 = BUY = green)
    val targetAngle = remember(score) {
        // Map [-100, +100] → [180°, 0°]  (needle sweeps left→right)
        180f - ((score + 100f) / 200f) * 180f
    }
    val animatedAngle by animateFloatAsState(
        targetValue = targetAngle,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "gaugeNeedle"
    )

    val signalColor = when (signal.lowercase()) {
        "buy"  -> Color(0xFF00C853)
        "sell" -> Color(0xFFD50000)
        else   -> Color(0xFF9E9E9E)
    }

    Column(
        modifier = modifier.padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            // Theme-driven: gauges sit on the app background, so use
            // onBackground (dark on Heritage taupe, light on dark themes).
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.75f),
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(4.dp))

        // Canvas gauge
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(2f)  // 2:1 = semicircle
        ) {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val cx = canvasWidth / 2f
            val cy = canvasHeight          // Arc center at bottom of canvas
            val radius = canvasWidth / 2f - 8.dp.toPx()
            val strokeWidth = 18.dp.toPx()

            // Themed glow halo behind the arc — breathes with the logo tap.
            // Signal colors on the arc itself stay untinted (BUY green / SELL red).
            val pulse = breath.pulse.value
            drawCircle(
                brush = Brush.radialGradient(
                    listOf(
                        spec.glow.copy(alpha = 0.10f + 0.30f * pulse),
                        Color.Transparent
                    ),
                    center = Offset(cx, cy),
                    radius = radius + strokeWidth * (1.2f + 0.8f * pulse)
                ),
                radius = radius + strokeWidth * (1.2f + 0.8f * pulse),
                center = Offset(cx, cy)
            )

            // Draw gradient arc background (SELL red → neutral yellow → BUY green)
            drawGaugeArc(cx, cy, radius, strokeWidth)

            // Draw tick marks at -100, -50, 0, +50, +100
            drawTicks(cx, cy, radius, strokeWidth)

            // Draw animated needle
            drawNeedle(cx, cy, radius, animatedAngle, signalColor)

            // Draw center hub
            drawCircle(
                color = Color(0xFF1E1E1E),
                radius = 10.dp.toPx(),
                center = Offset(cx, cy)
            )
            drawCircle(
                color = signalColor,
                radius = 6.dp.toPx(),
                center = Offset(cx, cy)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Score number
        Text(
            text = "${score.toInt()}",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = signalColor,
            textAlign = TextAlign.Center
        )

        // Signal label
        Text(
            text = signal.uppercase(),
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = signalColor,
            textAlign = TextAlign.Center
        )

        // Confidence percentage
        Text(
            text = "${(confidence * 100).toInt()}% confidence",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
            textAlign = TextAlign.Center
        )
    }
}

/**
 * Draw the semicircular gradient arc background.
 * Uses a color sweep: Red (0°/left) → Yellow (90°/top) → Green (180°/right).
 */
private fun DrawScope.drawGaugeArc(cx: Float, cy: Float, radius: Float, strokeWidth: Float) {
    val arcLeft = cx - radius
    val arcTop = cy - radius
    val arcSize = radius * 2f

    // Background track (dark)
    drawArc(
        color = Color(0xFF2A2A2A),
        startAngle = 180f,
        sweepAngle = 180f,
        useCenter = false,
        topLeft = Offset(arcLeft, arcTop),
        size = Size(arcSize, arcSize),
        style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
    )

    // Colored gradient segments (approximate with multiple arcs)
    val segments = 60
    val sweepPerSegment = 180f / segments
    for (i in 0 until segments) {
        val t = i.toFloat() / segments.toFloat()
        val color = gaugeColor(t)
        drawArc(
            color = color,
            startAngle = 180f + i * sweepPerSegment,
            sweepAngle = sweepPerSegment + 0.5f,  // slight overlap to avoid gaps
            useCenter = false,
            topLeft = Offset(arcLeft, arcTop),
            size = Size(arcSize, arcSize),
            style = Stroke(width = strokeWidth, cap = StrokeCap.Butt)
        )
    }
}

/**
 * Color interpolation: t=0 → Red (SELL), t=0.5 → Yellow (neutral), t=1 → Green (BUY).
 */
private fun gaugeColor(t: Float): Color {
    return when {
        t < 0.5f -> {
            val f = t / 0.5f
            Color(
                red = 1f,
                green = f,
                blue = 0f,
                alpha = 1f
            )
        }
        else -> {
            val f = (t - 0.5f) / 0.5f
            Color(
                red = 1f - f,
                green = 1f,
                blue = 0f,
                alpha = 1f
            )
        }
    }
}

/** Draw tick marks at 0%, 25%, 50%, 75%, 100% of the arc. */
private fun DrawScope.drawTicks(cx: Float, cy: Float, radius: Float, strokeWidth: Float) {
    val tickAngles = listOf(180f, 225f, 270f, 315f, 360f)
    val innerR = radius - strokeWidth / 2f - 4.dp.toPx()
    val outerR = radius + strokeWidth / 2f + 4.dp.toPx()

    for (angleDeg in tickAngles) {
        val rad = Math.toRadians(angleDeg.toDouble())
        val startX = cx + innerR * cos(rad).toFloat()
        val startY = cy + innerR * sin(rad).toFloat()
        val endX = cx + outerR * cos(rad).toFloat()
        val endY = cy + outerR * sin(rad).toFloat()
        drawLine(
            color = Color(0xFF424242),
            start = Offset(startX, startY),
            end = Offset(endX, endY),
            strokeWidth = 2.dp.toPx(),
            cap = StrokeCap.Round
        )
    }
}

/**
 * Draw the needle from the center hub toward the arc.
 * @param angle Angle in degrees (0° = right, 180° = left).
 */
private fun DrawScope.drawNeedle(cx: Float, cy: Float, radius: Float, angle: Float, color: Color) {
    val rad = Math.toRadians(angle.toDouble())
    val needleLength = radius - 10.dp.toPx()
    val tipX = cx + needleLength * cos(rad).toFloat()
    val tipY = cy + needleLength * sin(rad).toFloat()

    // Shadow
    drawLine(
        color = Color.Black.copy(alpha = 0.4f),
        start = Offset(cx + 2f, cy + 2f),
        end = Offset(tipX + 2f, tipY + 2f),
        strokeWidth = 5.dp.toPx(),
        cap = StrokeCap.Round
    )

    // Needle
    drawLine(
        color = color,
        start = Offset(cx, cy),
        end = Offset(tipX, tipY),
        strokeWidth = 4.dp.toPx(),
        cap = StrokeCap.Round
    )
}
