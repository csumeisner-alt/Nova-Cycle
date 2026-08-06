package com.novacycle.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.runtime.LaunchedEffect
import com.novacycle.domain.model.ConfidenceZone
import com.novacycle.domain.model.GaugeState
import com.novacycle.ui.theme.LocalNovaTheme
import com.novacycle.ui.theme.NovaTheme
import com.novacycle.ui.theme.spec
import kotlin.math.*

/** Zone colors for the normalized 0–100% confidence display. */
internal fun confidenceZoneColor(zone: ConfidenceZone): Color = when (zone) {
    ConfidenceZone.WEAK      -> Color(0xFFD50000)  // red   0–30%
    ConfidenceZone.UNCERTAIN -> Color(0xFFFFC107)  // yellow 31–64%
    ConfidenceZone.STRONG    -> Color(0xFF00C853)  // green 65–100%
}

/** Arrow glyph for the trend direction label. */
internal fun trendGlyph(trend: String): String = when (trend.uppercase()) {
    "UP"   -> "▲"
    "DOWN" -> "▼"
    else   -> "◆"
}

/**
 * DualGaugeWidget — simple directional 0–100% gauge.
 *
 * The gauge spans 180° (from left = -100 to right = +100).
 * Color gradient: Red (SELL) → Yellow (HOLD) → Green (BUY).
 *
 * @param gaugeState  Signal state containing score, signal, and confidence
 * @param label       Gauge title, e.g. "Long-Trend" or "Short-Trend"
 */
@Composable
fun DualGaugeWidget(
    gaugeState: GaugeState,
    label: String,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null
) {
    val theme = LocalNovaTheme.current
    val spec = theme.spec()
    val breath = LocalBreathState.current
    val gaugePercent = gaugeState.gaugePercent.coerceIn(0, 100)
    // 0% is the far-left red end, 100% is the far-right green end.
    val needleAngle = 180f + gaugePercent * 1.8f
    val animatedNeedleAngle = remember { Animatable(needleAngle) }
    LaunchedEffect(needleAngle) {
        animatedNeedleAngle.animateTo(
            targetValue = needleAngle,
            animationSpec = spring(
                dampingRatio = Spring.DampingRatioNoBouncy,
                stiffness = Spring.StiffnessLow
            )
        )
    }

    // Gray "no data" state: fallback gauges render fully muted.
    val isFallback = gaugeState.isFallback
    val zoneColor = if (isFallback) Color(0xFF9E9E9E)
                    else confidenceZoneColor(gaugeState.gaugeZone)

    Column(
        modifier = modifier
            .then(
                if (onClick != null) {
                    Modifier.clickable(
                        onClickLabel = "What does this confidence mean?",
                        onClick = onClick
                    )
                } else Modifier
            )
            .padding(8.dp),
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
            val strokeWidth = spec.gaugeArcStroke.dp.toPx()
            val radius = canvasWidth / 2f - strokeWidth / 2f - 12.dp.toPx()

            // Themed glow halo behind the arc
            val pulse = breath.pulse.value
            val haloAlpha = spec.gaugeGlowIntensity * (0.4f + 0.6f * pulse)
            drawCircle(
                brush = Brush.radialGradient(
                    listOf(
                        spec.glow.copy(alpha = haloAlpha),
                        Color.Transparent
                    ),
                    center = Offset(cx, cy),
                    radius = radius + strokeWidth * (1.2f + 0.8f * pulse)
                ),
                radius = radius + strokeWidth * (1.2f + 0.8f * pulse),
                center = Offset(cx, cy)
            )

            val trackColor = when (theme) {
                NovaTheme.DARK_LUXE -> Color(0xFF1C1C1C)
                NovaTheme.MINT_LUXE -> Color(0xFF1A2A22)
                NovaTheme.AURORA_FLUX -> Color(0xFF200D14)
                NovaTheme.CRIMSON_PULSE -> Color(0xFF3A2E23)
            }

            drawGaugeArc(cx, cy, radius, strokeWidth, muted = isFallback, trackColor = trackColor)
            drawTicks(cx, cy, radius, strokeWidth, spec, theme)
            drawNeedle(cx, cy, radius, animatedNeedleAngle.value, zoneColor, spec, pulse)

            // Draw center hub
            when (spec.gaugeHubStyle) {
                com.novacycle.ui.theme.GaugeHubStyle.GOLD_RING -> {
                    drawCircle(color = Color(0xFF1E1E1E), radius = 10.dp.toPx(), center = Offset(cx, cy))
                    drawCircle(color = spec.glow, radius = 10.dp.toPx(), center = Offset(cx, cy), style = Stroke(width = 6.dp.toPx()))
                    drawCircle(color = zoneColor, radius = 4.dp.toPx(), center = Offset(cx, cy))
                }
                com.novacycle.ui.theme.GaugeHubStyle.TEAL_DOT -> {
                    drawCircle(color = spec.glow, radius = 7.dp.toPx(), center = Offset(cx, cy))
                }
                com.novacycle.ui.theme.GaugeHubStyle.ROSE_GLOW -> {
                    drawCircle(color = spec.glow.copy(alpha = 0.5f + 0.3f * pulse), radius = 14.dp.toPx(), center = Offset(cx, cy))
                    drawCircle(color = zoneColor, radius = 5.dp.toPx(), center = Offset(cx, cy))
                }
                com.novacycle.ui.theme.GaugeHubStyle.COPPER_RING -> {
                    drawCircle(color = Color(0xFF1E1E1E), radius = 12.dp.toPx(), center = Offset(cx, cy))
                    drawCircle(color = spec.rim, radius = 12.dp.toPx(), center = Offset(cx, cy), style = Stroke(width = 2.dp.toPx()))
                    drawCircle(color = spec.glow, radius = 5.dp.toPx(), center = Offset(cx, cy))
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // One clear number: the directional position of the gauge.
        Text(
            text = "$gaugePercent%",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = zoneColor,
            textAlign = TextAlign.Center
        )

        // The action is derived from the same 0–100 value.
        Text(
            text = if (isFallback) "NO DATA" else gaugeState.gaugeAction,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = zoneColor,
            textAlign = TextAlign.Center
        )

        // Secondary context
        if (!isFallback) {
            val isBaseline = gaugeState.modelState == "baseline_mode"
            val confidencePercent = gaugeState.confidencePercent.coerceIn(0, 100)
            val confidenceColor = confidenceZoneColor(gaugeState.confidenceZone)
            Spacer(modifier = Modifier.height(4.dp))
            if (isBaseline) {
                Text(
                    text = "~$confidencePercent% base rate · not a trained signal",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.55f),
                    textAlign = TextAlign.Center
                )
            } else {
                Text(
                    text = "$confidencePercent% confidence · ${gaugeState.confidenceZone.label}",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = confidenceColor,
                    textAlign = TextAlign.Center
                )
            }
            Text(
                text = "${trendGlyph(gaugeState.trend)} ${gaugeState.trend.uppercase()} · ${gaugeState.displaySignal}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.65f),
                textAlign = TextAlign.Center
            )
            gaugeState.convictionTier?.let { tier ->
                Spacer(modifier = Modifier.height(4.dp))
                ConvictionTierBadge(tier)
            }
            if (gaugeState.isCandidate && gaugeState.candidateSignal != null) {
                Spacer(modifier = Modifier.height(4.dp))
                CandidateBadge(direction = gaugeState.candidateSignal)
            }
            if (!gaugeState.predictionReliable) {
                Spacer(modifier = Modifier.height(6.dp))
                ModelDegradedBanner(modelState = gaugeState.modelState)
            }
        }
    }
}

/**
 * Compact inline banner rendered inside the gauge widget when the underlying
 * model is in a degraded state.
 */
@Composable
private fun ModelDegradedBanner(modelState: String?) {
    val isBaseline = modelState == "baseline_mode"
    val message = when (modelState) {
        "baseline_mode"     -> "BASELINE MODE · NO TRAINED EDGE — calibrated base rate (~73% bull bias), not a trained prediction."
        "training_stuck"    -> "⚠ Model degraded — repeated retrain failures. Do not rely on this signal."
        "stale_rolled_back" -> "⚠ Model stale — last retrain failed and was rolled back. Reliability reduced."
        "model_unavailable" -> "⚠ Model unavailable — showing a neutral fallback, not a real prediction."
        else                -> "⚠ Prediction unreliable — model is in a degraded state."
    }
    val bannerColor = if (isBaseline) Color(0xFFE65100) else Color(0xFFD32F2F)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .background(bannerColor.copy(alpha = 0.12f))
            .padding(horizontal = 8.dp, vertical = 6.dp)
    ) {
        Text(
            text      = message,
            style     = MaterialTheme.typography.labelSmall,
            color     = bannerColor,
            fontWeight = if (isBaseline) FontWeight.Bold else FontWeight.Normal,
            textAlign = TextAlign.Center,
            modifier  = Modifier.fillMaxWidth()
        )
    }
}

private fun DrawScope.drawGaugeArc(
    cx: Float, cy: Float, radius: Float, strokeWidth: Float, muted: Boolean = false, trackColor: Color = Color(0xFF2A2A2A)
) {
    val arcLeft = cx - radius
    val arcTop = cy - radius
    val arcSize = radius * 2f

    // Background track
    drawArc(
        color = trackColor,
        startAngle = 180f,
        sweepAngle = 180f,
        useCenter = false,
        topLeft = Offset(arcLeft, arcTop),
        size = Size(arcSize, arcSize),
        style = Stroke(width = strokeWidth, cap = StrokeCap.Round)
    )

    // Colored gradient segments
    val segments = 60
    val sweepPerSegment = 180f / segments
    for (i in 0 until segments) {
        val t = i.toFloat() / segments.toFloat()
        val color = if (muted) Color(0xFF616161) else gaugeColor(t)
        drawArc(
            color = color,
            startAngle = 180f + i * sweepPerSegment,
            sweepAngle = sweepPerSegment + 0.5f,
            useCenter = false,
            topLeft = Offset(arcLeft, arcTop),
            size = Size(arcSize, arcSize),
            style = Stroke(width = strokeWidth, cap = StrokeCap.Butt)
        )
    }
}

private fun gaugeColor(t: Float): Color {
    return when {
        t < 0.5f -> {
            val f = t / 0.5f
            Color(red = 1f, green = f, blue = 0f, alpha = 1f)
        }
        else -> {
            val f = (t - 0.5f) / 0.5f
            Color(red = 1f - f, green = 1f, blue = 0f, alpha = 1f)
        }
    }
}

private fun DrawScope.drawTicks(
    cx: Float, cy: Float, radius: Float, strokeWidth: Float,
    spec: com.novacycle.ui.theme.NovaThemeSpec,
    theme: NovaTheme
) {
    if (!spec.gaugeShowTicks) return

    val tickAngles = if (spec.gaugeTickCount == 3) listOf(180f, 270f, 360f) 
                     else listOf(180f, 225f, 270f, 315f, 360f)
    
    val (innerDp, outerDp) = when (theme) {
        NovaTheme.DARK_LUXE -> 3.dp to 3.dp
        NovaTheme.CRIMSON_PULSE -> 6.dp to 6.dp
        else -> 4.dp to 4.dp
    }
    val innerR = radius - strokeWidth / 2f - innerDp.toPx()
    val outerR = radius + strokeWidth / 2f + outerDp.toPx()

    for (angleDeg in tickAngles) {
        val rad = Math.toRadians(angleDeg.toDouble())
        val startX = cx + innerR * cos(rad).toFloat()
        val startY = cy + innerR * sin(rad).toFloat()
        val endX = cx + outerR * cos(rad).toFloat()
        val endY = cy + outerR * sin(rad).toFloat()
        drawLine(
            color = spec.glow,
            start = Offset(startX, startY),
            end = Offset(endX, endY),
            strokeWidth = 2.dp.toPx(),
            cap = StrokeCap.Round
        )
    }
}

private fun DrawScope.drawNeedle(
    cx: Float, cy: Float, radius: Float, angle: Float, color: Color,
    spec: com.novacycle.ui.theme.NovaThemeSpec, pulse: Float
) {
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

    when (spec.gaugeNeedleTip) {
        com.novacycle.ui.theme.GaugeNeedleTip.SHARP -> {
            val baseWidth = 3.dp.toPx()
            val baseAngle1 = rad + Math.PI / 2
            val baseAngle2 = rad - Math.PI / 2
            
            val path = Path()
            path.moveTo(tipX, tipY)
            path.lineTo(
                cx + baseWidth * cos(baseAngle1).toFloat(),
                cy + baseWidth * sin(baseAngle1).toFloat()
            )
            path.lineTo(
                cx + baseWidth * cos(baseAngle2).toFloat(),
                cy + baseWidth * sin(baseAngle2).toFloat()
            )
            path.close()
            
            drawPath(path, color = color)
            drawLine(
                color = spec.glowBright,
                start = Offset(cx, cy),
                end = Offset(tipX, tipY),
                strokeWidth = 1.dp.toPx(),
                cap = StrokeCap.Round
            )
        }
        com.novacycle.ui.theme.GaugeNeedleTip.GLOW_DOT -> {
            drawLine(
                color = color,
                start = Offset(cx, cy),
                end = Offset(tipX, tipY),
                strokeWidth = 3.dp.toPx(),
                cap = StrokeCap.Round
            )
            drawCircle(
                color = spec.glow,
                radius = 5.dp.toPx(),
                center = Offset(tipX, tipY)
            )
        }
        com.novacycle.ui.theme.GaugeNeedleTip.GLOW_HALO -> {
            drawLine(
                color = color,
                start = Offset(cx, cy),
                end = Offset(tipX, tipY),
                strokeWidth = 5.dp.toPx(),
                cap = StrokeCap.Round
            )
            drawCircle(
                color = spec.glow.copy(alpha = 0.4f + 0.3f * pulse),
                radius = 12.dp.toPx(),
                center = Offset(tipX, tipY)
            )
            drawCircle(
                color = color,
                radius = 4.dp.toPx(),
                center = Offset(tipX, tipY)
            )
        }
        com.novacycle.ui.theme.GaugeNeedleTip.BLUNT -> {
            drawLine(
                color = spec.glow,
                start = Offset(cx, cy),
                end = Offset(tipX, tipY),
                strokeWidth = 6.dp.toPx(),
                cap = StrokeCap.Square
            )
        }
    }
}
