package com.novacycle.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.navigation.NavDestination
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import com.novacycle.ui.navigation.Destination
import com.novacycle.ui.theme.NovaBackground

/**
 * Animated bottom navigation bar with active indicator pill.
 *
 * Expo Router's native tabs feel responsive because selected state changes
 * animate. We replicate that in Material3 by:
 *   - Scaling up the selected icon
 *   - Showing a subtle pill background behind the selected item
 *   - Tinting selected labels with the primary color
 */
@Composable
fun NovaBottomNav(
    navController: NavHostController,
    currentDestination: NavDestination?,
    modifier: Modifier = Modifier
) {
    NavigationBar(
        modifier = modifier,
        containerColor = NovaBackground,
        tonalElevation = 0.dp
    ) {
        Destination.bottomNavItems.forEach { destination ->
            val selected = currentDestination?.hierarchy?.any { it.route == destination.route } == true
            val iconScale by animateFloatAsState(
                targetValue = if (selected) 1.15f else 1f,
                animationSpec = tween(durationMillis = 200),
                label = "navIconScale"
            )

            NavigationBarItem(
                selected = selected,
                onClick = {
                    navController.navigate(destination.route) {
                        popUpTo(navController.graph.findStartDestination().id) {
                            saveState = true
                        }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                icon = {
                    Box(contentAlignment = Alignment.Center) {
                        // Selected pill scales in/out (AnimatedVisibility can't be
                        // called here — NavigationBar's RowScope shadows it).
                        val pillScale by animateFloatAsState(
                            targetValue = if (selected) 1f else 0f,
                            animationSpec = tween(durationMillis = 200),
                            label = "navPillScale"
                        )
                        Box(
                            modifier = Modifier
                                .size(36.dp)
                                .scale(pillScale)
                                .background(
                                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                                    shape = CircleShape
                                )
                        )
                        Icon(
                            imageVector = destination.icon ?: Icons.Default.ExpandMore,
                            contentDescription = destination.label,
                            modifier = Modifier.scale(iconScale)
                        )
                    }
                },
                label = {
                    Text(
                        text = destination.label ?: "",
                        style = MaterialTheme.typography.labelSmall
                    )
                },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = MaterialTheme.colorScheme.primary,
                    selectedTextColor = MaterialTheme.colorScheme.primary,
                    unselectedIconColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    unselectedTextColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                    indicatorColor = Color.Transparent
                )
            )
        }
    }
}

