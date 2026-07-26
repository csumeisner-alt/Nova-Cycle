package com.novacycle.ui.navigation

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavHostController
import androidx.navigation.compose.*
import com.novacycle.ui.components.HealthBanners
import com.novacycle.ui.components.NovaBottomNav
import com.novacycle.ui.screens.*
import com.novacycle.viewmodel.HealthViewModel


/**
 * Root navigation host with bottom nav bar.
 *
 * Maps Expo Router's file-based routing concepts to Compose Navigation:
 *   - Type-safe destinations via [Destination]
 *   - Animated screen transitions (slide + fade)
 *   - Animated bottom nav bar with active indicator pill
 *   - Shared health banners above data screens
 *
 * Bottom nav is shown on the 5 main destinations; all others are secondary screens
 * reachable via in-screen navigation (e.g., tapping hold time card → HoldTimeScreen).
 */
@Composable
fun NovaCycleNavHost(
    healthViewModel: HealthViewModel = hiltViewModel()
) {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination
    val healthState by healthViewModel.uiState.collectAsStateWithLifecycle()

    // Screens that display backend-derived data — the shared health banners
    // appear on all of these. Settings is the only non-data screen.
    val dataScreenRoutes = setOf(
        Destination.DualGauge.route,
        Destination.RawChart.route,
        Destination.FilteredChart.route,
        Destination.ConfidenceHistory.route,
        Destination.IndicatorList.route,
        Destination.HoldTime.route,
        Destination.Reliability.route
    )
    val showHealthBanners = currentDestination?.route in dataScreenRoutes

    // Destinations that show the bottom nav bar
    val showBottomBar = currentDestination?.route in Destination.bottomNavItems.map { it.route }

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NovaBottomNav(
                    navController = navController,
                    currentDestination = currentDestination
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // App-level banner slot: one shared /healthz poll drives the
            // degraded / unreachable banners across every data screen.
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
                startDestination = Destination.DualGauge.route,
                modifier = Modifier.fillMaxSize(),
                enterTransition = {
                    slideIntoContainer(
                        towards = AnimatedContentTransitionScope.SlideDirection.Start,
                        animationSpec = tween(250)
                    ) + fadeIn(animationSpec = tween(250))
                },
                exitTransition = {
                    slideOutOfContainer(
                        towards = AnimatedContentTransitionScope.SlideDirection.Start,
                        animationSpec = tween(250)
                    ) + fadeOut(animationSpec = tween(250))
                },
                popEnterTransition = {
                    slideIntoContainer(
                        towards = AnimatedContentTransitionScope.SlideDirection.End,
                        animationSpec = tween(250)
                    ) + fadeIn(animationSpec = tween(250))
                },
                popExitTransition = {
                    slideOutOfContainer(
                        towards = AnimatedContentTransitionScope.SlideDirection.End,
                        animationSpec = tween(250)
                    ) + fadeOut(animationSpec = tween(250))
                }
            ) {
                composable(Destination.DualGauge.route) {
                    DualGaugeScreen(
                        onNavigateToRawChart    = { navController.navigate(Destination.RawChart.route) },
                        onNavigateToHoldTime    = { navController.navigate(Destination.HoldTime.route) },
                        onNavigateToReliability = { navController.navigate(Destination.Reliability.route) }
                    )
                }
                composable(Destination.RawChart.route) { RawChartScreen() }
                composable(Destination.FilteredChart.route) { FilteredChartScreen() }
                composable(Destination.ConfidenceHistory.route) { ConfidenceHistoryScreen() }
                composable(Destination.IndicatorList.route) { IndicatorListScreen() }
                composable(Destination.HoldTime.route) {
                    HoldTimeScreen(onBack = { navController.popBackStack() })
                }
                composable(Destination.Settings.route) { SettingsScreen() }
                composable(Destination.Reliability.route) {
                    ReliabilityScreen(onBack = { navController.popBackStack() })
                }
            }
        }
    }
}

