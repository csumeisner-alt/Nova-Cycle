package com.novacycle.ui.components

import androidx.compose.foundation.border
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.dp
import com.novacycle.ui.theme.LocalNovaTheme
import com.novacycle.ui.theme.spec

/**
 * Thin metallic/neon rim on a card edge, in the current theme's rim color.
 * Brightens with the shared breath pulse so cards visibly respond when the
 * logo is tapped.
 */
fun Modifier.luxeRim(shape: Shape): Modifier = composed {
    val spec = LocalNovaTheme.current.spec()
    val breath = LocalBreathState.current
    val pulse = breath.pulse.value
    border(
        width = 1.dp,
        brush = Brush.verticalGradient(
            listOf(
                spec.glowBright.copy(alpha = 0.35f + 0.45f * pulse),
                spec.rim.copy(alpha = (spec.rim.alpha * (1f + pulse)).coerceAtMost(1f)),
                spec.glow.copy(alpha = 0.15f + 0.35f * pulse)
            )
        ),
        shape = shape
    )
}
