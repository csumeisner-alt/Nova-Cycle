package com.novacycle.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.*
import com.novacycle.ui.screens.*

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
    BottomNavItem(Routes.FILTERED_CHART,     "Chart",      Icons.Filled.ShowChart),
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
fun NovaCycleNavHost() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    // Destinations that show the bottom nav bar
    val bottomNavRoutes = bottomNavItems.map { it.route }.toSet()
    val showBottomBar = currentDestination?.route in bottomNavRoutes

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar(
                    containerColor = MaterialTheme.colorScheme.surface
                ) {
                    bottomNavItems.forEach { item ->
                        val selected = currentDestination?.hierarchy?.any { it.route == item.route } == true
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
                                    contentDescription = item.label
                                )
                            },
                            label = { Text(item.label) }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Routes.DUAL_GAUGE,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Routes.DUAL_GAUGE) {
                DualGaugeScreen(
                    onNavigateToRawChart      = { navController.navigate(Routes.RAW_CHART) },
                    onNavigateToHoldTime      = { navController.navigate(Routes.HOLD_TIME) },
                    onNavigateToReliability   = { navController.navigate(Routes.RELIABILITY) }
                )
            }
            composable(Routes.RAW_CHART) {
                RawChartScreen()
            }
            composable(Routes.FILTERED_CHART) {
                FilteredChartScreen()
            }
            composable(Routes.CONFIDENCE_HISTORY) {
                ConfidenceHistoryScreen()
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
                SettingsScreen()
            }
            composable(Routes.RELIABILITY) {
                ReliabilityScreen(
                    onBack = { navController.popBackStack() }
                )
            }
        }
    }
}
