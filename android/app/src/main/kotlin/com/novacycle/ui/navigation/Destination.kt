package com.novacycle.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.ui.graphics.vector.ImageVector

/**
 * Type-safe navigation destinations.
 *
 * Mirrors the file-based routing used by Expo Router: each screen is a
 * strongly-typed destination with a stable route string and optional nav icon.
 *
 * Note: This project uses Navigation Compose 2.7.x, which does not yet support
 * Kotlin-serialization type-safe navigation out of the box. We therefore keep
 * the route strings as a separate property and use them with the legacy
 * string-based NavHost DSL. The sealed class still gives us compile-time safety
 * for all call sites.
 */
sealed class Destination(
    val route: String,
    val label: String? = null,
    val icon: ImageVector? = null
) {
    data object DualGauge : Destination(
        route = "dual_gauge",
        label = "Gauge",
        icon = Icons.Filled.Speed
    )

    data object RawChart : Destination(
        route = "raw_chart",
        label = "Chart",
        icon = Icons.Filled.ShowChart
    )

    data object FilteredChart : Destination(
        route = "filtered_chart",
        label = "Chart",
        icon = Icons.Filled.Timeline
    )

    data object ConfidenceHistory : Destination(
        route = "confidence_history",
        label = "Confidence",
        icon = Icons.Filled.Timeline
    )

    data object IndicatorList : Destination(
        route = "indicator_list",
        label = "Indicators",
        icon = Icons.Filled.BarChart
    )

    data object HoldTime : Destination(
        route = "hold_time",
        label = "Hold Time",
        icon = Icons.Filled.Schedule
    )

    data object Settings : Destination(
        route = "settings",
        label = "Settings",
        icon = Icons.Filled.Settings
    )

    data object Reliability : Destination(
        route = "reliability",
        label = "Reliability",
        icon = Icons.Filled.TrendingUp
    )

    companion object {
        /** Main destinations that appear in the bottom navigation bar. */
        val bottomNavItems = listOf(
            DualGauge,
            FilteredChart,
            ConfidenceHistory,
            IndicatorList,
            Settings
        )

        /** All known destinations for back-stack routing checks. */
        val allRoutes = listOf(
            DualGauge, RawChart, FilteredChart, ConfidenceHistory,
            IndicatorList, HoldTime, Settings, Reliability
        ).map { it.route }
    }
}
