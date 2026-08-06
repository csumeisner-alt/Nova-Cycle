package com.novacycle.ui.navigation

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.*
import com.novacycle.ui.components.HealthBanners
import com.novacycle.ui.screens.*
import com.novacycle.viewmodel.HealthViewModel
import com.novacycle.viewmodel.SettingsViewModel
import com.novacycle.viewmodel.ThemeViewModel
import com.novacycle.ui.theme.LocalNovaTheme

/** Route constants — single source of truth for navigation destinations */
object Routes {
    const val DUAL_GAUGE         = "dual_gauge"
    const val RAW_CHART          = "raw_chart"
    const val FILTERED_CHART     = "filtered_chart"
    const val CONFIDENCE_HISTORY = "confidence_history"
    const val INDICATOR_LIST     = "indicator_list"
    const val HOLD_TIME          = "hold_time"
    const val SETTINGS           = "settings"
    const val RELIABILITY        = "reliability"
}

/** Bottom nav items — the 5 main destinations accessible from every screen */
data class BottomNavItem(
    val route: String,
    val label: String,
    val icon: ImageVector
)

private val bottomNavItems = listOf(
    BottomNavItem(Routes.DUAL_GAUGE,         "Gauge",      Icons.Filled.Speed),
    BottomNavItem(Routes.FILTERED_CHART,     "Chart",      Icons.AutoMirrored.Filled.ShowChart),
    BottomNavItem(Routes.CONFIDENCE_HISTORY, "Confidence", Icons.Filled.Timeline),
    BottomNavItem(Routes.INDICATOR_LIST,     "Indicators", Icons.Filled.BarChart),
    BottomNavItem(Routes.SETTINGS,           "Settings",   Icons.Filled.Settings)
)

/**
 * Root navigation host with bottom nav bar.
 * Bottom nav is shown on the 5 main destinations; all others are secondary screens
 * reachable via in-screen navigation (e.g., tapping hold time card → HoldTimeScreen).
 */
@Composable
fun NovaCycleNavHost(
    healthViewModel: HealthViewModel = hiltViewModel(),
    themeViewModel: ThemeViewModel,
    settingsViewModel: SettingsViewModel
) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination
    val healthState by healthViewModel.uiState.collectAsStateWithLifecycle()

    // Keep the health warning visible only where it directly explains the
    // prediction gauges. Other data screens should stay focused; users can
    // still inspect detailed health from the Gauge dashboard.
    val showHealthBanners = currentDestination?.route == Routes.DUAL_GAUGE

    // Destinations that show the bottom nav bar
    val bottomNavRoutes = bottomNavItems.map { it.route }.toSet()
    val showBottomBar = currentDestination?.route in bottomNavRoutes

    // Transparent container so the app-level AmbientBackground (living glows /
    // ribbons / pattern) shows through behind every screen.
    Scaffold(
        containerColor = androidx.compose.ui.graphics.Color.Transparent,
        bottomBar = {
            if (showBottomBar) {
                NavigationBar(
                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f)
                    ,
                    tonalElevation = 8.dp
                ) {
                    bottomNavItems.forEach { item ->
                        val selected = currentDestination?.hierarchy?.any { it.route == item.route } == true
                        val iconScale by animateFloatAsState(
                            targetValue = if (selected) 1.08f else 1f,
                            animationSpec = tween(180),
                            label = "nav-icon-scale-${item.route}"
                        )
                        val theme = LocalNovaTheme.current
                        val indicatorColor by animateColorAsState(
                            targetValue = theme.accent.copy(alpha = if (selected) 0.28f else 0f),
                            animationSpec = tween(180),
                            label = "nav-indicator-${item.route}"
                        )
                        NavigationBarItem(
                            selected = selected,
                            onClick = {
                                navController.navigate(item.route) {
                                    // Pop back to start to avoid deep back stacks
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = {
                                Icon(
                                    imageVector = item.icon,
                                    contentDescription = item.label,
                                    modifier = Modifier.graphicsLayer {
                                        scaleX = iconScale
                                        scaleY = iconScale
                                    }
                                )
                            },
                            label = {
                                Text(
                                    text = item.label,
                                    maxLines = 1,
                                    softWrap = false,
                                    style = TextStyle(
                                        fontSize = 10.sp,
                                        lineHeight = 12.sp,
                                        fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                                        letterSpacing = 0.sp
                                    )
                                )
                            },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = theme.accent,
                                selectedTextColor = theme.accent,
                                indicatorColor = indicatorColor,
                                unselectedIconColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.72f),
                                unselectedTextColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.72f)
                            )
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // The prediction-health banner belongs to the Gauge dashboard,
            // not to every screen that happens to use backend data.
            if (showHealthBanners) {
                HealthBanners(
                    state = healthState,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                )
            }
            NavHost(
                navController = navController,
                startDestination = Routes.DUAL_GAUGE,
                modifier = Modifier.fillMaxSize()
            ) {
            composable(Routes.DUAL_GAUGE) {
                DualGaugeScreen(
                    onNavigateToRawChart      = { navController.navigate(Routes.RAW_CHART) },
                    onNavigateToHoldTime      = { navController.navigate(Routes.HOLD_TIME) },
                    onNavigateToReliability   = { navController.navigate(Routes.RELIABILITY) }
                )
            }
            composable(Routes.RAW_CHART) {
                RawChartScreen(
                    onBack = { navController.popBackStack() }
                )
            }
            composable(Routes.FILTERED_CHART) {
                FilteredChartScreen()
            }
            composable(Routes.CONFIDENCE_HISTORY) {
                ConfidenceHistoryScreen(settingsViewModel = settingsViewModel)
            }
            composable(Routes.INDICATOR_LIST) {
                IndicatorListScreen()
            }
            composable(Routes.HOLD_TIME) {
                HoldTimeScreen(
                    onBack = { navController.popBackStack() }
                )
            }
            composable(Routes.SETTINGS) {
                SettingsScreen(
                    viewModel = settingsViewModel,
                    themeViewModel = themeViewModel
                )
            }
            composable(Routes.RELIABILITY) {
                ReliabilityScreen(
                    onBack = { navController.popBackStack() }
                )
            }
            }
        }
    }
}
