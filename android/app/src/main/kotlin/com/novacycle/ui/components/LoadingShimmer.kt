package com.novacycle.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.novacycle.ui.theme.NovaSurface
import com.novacycle.ui.theme.NovaSurfaceVariant

/**
 * Shimmer loading placeholder used while screen data is loading.
 *
 * Expo apps often skeleton-load content before data arrives. This component
 * provides a reusable shimmer brush and pre-built shapes (card, line, circle)
 * so every screen looks consistent during refresh.
 */
@Composable
fun rememberShimmerBrush(): Brush {
    val shimmerColors = listOf(
        NovaSurfaceVariant.copy(alpha = 0.6f),
        NovaSurfaceVariant.copy(alpha = 0.2f),
        NovaSurfaceVariant.copy(alpha = 0.6f)
    )

    val transition = rememberInfiniteTransition(label = "shimmer")
    val translateAnimation = transition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 1200,
                easing = FastOutSlowInEasing
            ),
            repeatMode = RepeatMode.Restart
        ),
        label = "shimmerTranslate"
    )

    return Brush.linearGradient(
        colors = shimmerColors,
        start = Offset.Zero,
        end = Offset(x = translateAnimation.value, y = translateAnimation.value)
    )
}

/**
 * Rounded rectangle shimmer placeholder with fixed size.
 */
@Composable
fun ShimmerCard(
    height: Dp = 120.dp,
    modifier: Modifier = Modifier,
    shape: RoundedCornerShape = RoundedCornerShape(12.dp)
) {
    val brush = rememberShimmerBrush()
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(height)
            .clip(shape)
            .background(brush)
    )
}

/**
 * Horizontal shimmer line placeholder.
 */
@Composable
fun ShimmerLine(
    widthFraction: Float = 1f,
    height: Dp = 16.dp,
    modifier: Modifier = Modifier
) {
    val brush = rememberShimmerBrush()
    Box(
        modifier = modifier
            .fillMaxWidth(widthFraction)
            .height(height)
            .clip(RoundedCornerShape(4.dp))
            .background(brush)
    )
}

/**
 * Circular shimmer placeholder, useful for avatars or gauge hubs.
 */
@Composable
fun ShimmerCircle(
    size: Dp = 48.dp,
    modifier: Modifier = Modifier
) {
    val brush = rememberShimmerBrush()
    Box(
        modifier = modifier
            .size(size)
            .clip(RoundedCornerShape(50))
            .background(brush)
    )
}

/**
 * Full-screen shimmer placeholder for a chart screen.
 */
@Composable
fun ChartShimmerLayout(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        ShimmerLine(widthFraction = 0.4f)
        ShimmerLine(widthFraction = 0.7f, height = 12.dp)
        Spacer(modifier = Modifier.height(16.dp))
        ShimmerCard(height = 240.dp)
        Spacer(modifier = Modifier.height(16.dp))
        ShimmerCard(height = 120.dp)
    }
}
