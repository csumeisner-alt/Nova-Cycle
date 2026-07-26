package com.novacycle.ui.components.animations

import androidx.compose.animation.core.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign

/**
 * An integer that animates smoothly when its target value changes.
 *
 * Expo Router and React Native animations often use layout transitions for
 * numbers. In Compose we use [animateIntAsState] to give the same "live"
 * feeling to gauge scores, signal counts, and confidence values without
 * requiring manual interpolation.
 *
 * @param target Target integer value
 * @param modifier Compose modifier
 * @param style Text style; defaults to MaterialTheme.typography.bodyLarge
 * @param color Text color; defaults to MaterialTheme.colorScheme.onBackground
 * @param fontWeight Optional font weight override
 * @param textAlign Alignment of the text
 */
@Composable
fun AnimatedInt(
    target: Int,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.bodyLarge,
    color: Color = MaterialTheme.colorScheme.onBackground,
    fontWeight: FontWeight? = null,
    textAlign: TextAlign? = null
) {
    val animated by animateIntAsState(
        targetValue = target,
        animationSpec = tween(durationMillis = 500, easing = FastOutSlowInEasing),
        label = "animatedInt"
    )

    Text(
        text = animated.toString(),
        modifier = modifier,
        style = style,
        color = color,
        fontWeight = fontWeight,
        textAlign = textAlign
    )
}

/**
 * Animate a float value to a string with a fixed number of decimal places.
 */
@Composable
fun AnimatedFloat(
    target: Float,
    decimals: Int,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.bodyLarge,
    color: Color = MaterialTheme.colorScheme.onBackground,
    fontWeight: FontWeight? = null,
    textAlign: TextAlign? = null
) {
    val animated by animateFloatAsState(
        targetValue = target,
        animationSpec = tween(durationMillis = 500, easing = FastOutSlowInEasing),
        label = "animatedFloat"
    )

    val pattern = "%.${decimals}f"
    Text(
        text = pattern.format(animated),
        modifier = modifier,
        style = style,
        color = color,
        fontWeight = fontWeight,
        textAlign = textAlign
    )
}
