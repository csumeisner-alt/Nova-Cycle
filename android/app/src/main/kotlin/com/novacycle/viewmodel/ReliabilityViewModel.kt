package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.ModelPerformanceResponse
import com.novacycle.data.remote.models.ReliabilityMetricsResponse
import com.novacycle.data.remote.models.TradeCycleResponse
import com.novacycle.data.repository.NovaCycleRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import javax.inject.Inject

/**
 * Sortable columns for the trade-cycle table.
 */
enum class CycleSortColumn {
    RETURN_PERCENT, RETURN_DOLLARS, HOLD_TIME, CONFIDENCE, GAP_TYPE, LIQUIDITY_SCORE
}

/**
 * User-facing filters for the Reliability screen.
 * All filters default to "any" so the initial view shows every cycle.
 */
data class CycleFilters(
    val startDateMs: Long? = null,
    val endDateMs: Long? = null,
    val winLoss: WinLossFilter = WinLossFilter.ALL,
    val volatilityClass: String? = null,
    val liquidityClass: String? = null,
    val sessionType: String? = null
)

enum class WinLossFilter { ALL, WIN, LOSS }

/**
 * Time-window filter for the model-performance data.
 * Each option maps to the backend `window` query parameter.
 */
enum class PeriodFilter(val window: String) {
    D1("1d"), D7("7d"), D30("30d")
}

/**
 * Confidence-band filter, mapping to confidence_min/confidence_max query params
 * and to a local cycle filter on [TradeCycleResponse.confidenceAtBuy].
 *
 * Bands: Low 0.0–0.4, Medium 0.4–0.7, High 0.7–1.0. ALL disables both bounds.
 */
enum class ConfidenceBand(val min: Float?, val max: Float?) {
    ALL(null, null),
    LOW(0.0f, 0.4f),
    MEDIUM(0.4f, 0.7f),
    HIGH(0.7f, 1.0f)
}

/**
 * Complete UI state for the Reliability Metrics screen.
 */
data class ReliabilityUiState(
    val cycles: List<TradeCycleResponse> = emptyList(),
    val filteredCycles: List<TradeCycleResponse> = emptyList(),
    val summary: ReliabilityMetricsResponse? = null,
    val filters: CycleFilters = CycleFilters(),
    val sortColumn: CycleSortColumn = CycleSortColumn.RETURN_PERCENT,
    val sortAscending: Boolean = false,
    val expandedCycleId: String? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    /** Epoch millis of the last successful data refresh on this screen; null if none yet */
    val lastUpdatedAtMillis: Long? = null,
    // ── Model-performance additions ──
    val periodFilter: PeriodFilter = PeriodFilter.D30,
    val confidenceBand: ConfidenceBand = ConfidenceBand.ALL,
    val performance: ModelPerformanceResponse? = null,
    val performanceLoading: Boolean = false,
    val performanceError: String? = null
) {
    /** Number of rallies the model failed to enter, from the performance feed. */
    val missedRallyCount: Int get() = performance?.missedRallies?.count ?: 0

    /** Cumulative return over the selected window, in percent. */
    val cumulativeReturnPercent: Float get() = performance?.summary?.cumulativeReturnPercent ?: 0f

    /** Observed win rate for high-confidence calls (0–1), from the confidence buckets. */
    val highConfidenceWinRate: Float?
        get() = performance?.confidenceBuckets?.get("high")?.winRate

    /**
     * The confidence the model *claimed* for high-band calls (0–1). Derived from
     * the average of the high-band calibration midpoints; falls back to the 0.85
     * midpoint of the High band when calibration data is unavailable.
     */
    val highConfidenceClaim: Float
        get() {
            val curve = performance?.calibrationCurve.orEmpty()
            val highPoints = curve.filter { it.confidenceMid >= 0.7f && it.tradeCount > 0 }
            return if (highPoints.isNotEmpty()) {
                highPoints.map { it.confidenceMid }.average().toFloat()
            } else {
                0.85f
            }
        }
}

/**
 * ViewModel for the Reliability Metrics screen.
 *
 * Responsibilities:
 *   1. Load trade cycles + summary from the backend via the repository.
 *   2. Apply filters and sorting to derive the displayed list.
 *   3. Expose a single UI state flow that Compose observes.
 *
 * All heavy work (filtering/sorting) is done in the ViewModel so the UI remains
 * a pure function of state. No existing signal logic is modified.
 */
@HiltViewModel
class ReliabilityViewModel @Inject constructor(
    private val repository: NovaCycleRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(ReliabilityUiState())
    val uiState: StateFlow<ReliabilityUiState> = _uiState.asStateFlow()

    init {
        loadTradeHistory()
        loadModelPerformance()
    }

    /**
     * Fetch the latest trade history from the backend. The backend regenerates
     * cycles from the filtered BUY→SELL timeline and computes metrics, so a
     * single call gives us both the cycle list and the summary panel.
     *
     * The window defaults to the currently selected [PeriodFilter] so trade
     * history stays in step with the model-performance feed.
     */
    fun loadTradeHistory(
        ticker: String = "VOO",
        window: String = _uiState.value.periodFilter.window
    ) {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }

            val result = repository.getTradeHistory(ticker, window)
            result.fold(
                onSuccess = { response ->
                    _uiState.update { state ->
                        state.copy(
                            cycles = response.cycles,
                            summary = response.summary,
                            isLoading = false,
                            error = null,
                            // Shared DataFreshnessTracker is updated by the repository;
                            // this timestamp drives the screen's own "Updated X ago" label.
                            lastUpdatedAtMillis = System.currentTimeMillis()
                        ).applyFiltersAndSort()
                    }
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            error = error.message ?: "Failed to load trade history"
                        )
                    }
                }
            )
        }
    }

    /**
     * Fetch model-performance analytics for the current period + confidence band.
     * Window maps from [PeriodFilter]; confidence_min/max map from [ConfidenceBand].
     */
    fun loadModelPerformance(ticker: String = "VOO") {
        val period = _uiState.value.periodFilter
        val band = _uiState.value.confidenceBand
        viewModelScope.launch {
            _uiState.update { it.copy(performanceLoading = true, performanceError = null) }

            val result = repository.getModelPerformance(
                ticker = ticker,
                window = period.window,
                confidenceMin = band.min,
                confidenceMax = band.max
            )
            result.fold(
                onSuccess = { response ->
                    _uiState.update {
                        it.copy(
                            performance = response,
                            performanceLoading = false,
                            performanceError = null
                        )
                    }
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(
                            performanceLoading = false,
                            performanceError = error.message ?: "Failed to load model performance"
                        )
                    }
                }
            )
        }
    }

    fun refresh() {
        loadTradeHistory()
        loadModelPerformance()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Period + confidence-band controls
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Switch the time window. Re-fetches both /model_performance and
     * /trade_history with the matching window, then re-derives the filtered list.
     */
    fun setPeriodFilter(period: PeriodFilter) {
        if (_uiState.value.periodFilter == period) return
        _uiState.update { it.copy(periodFilter = period) }
        loadTradeHistory()
        loadModelPerformance()
    }

    /**
     * Switch the confidence band. Re-fetches /model_performance with the matching
     * confidence bounds and also filters the local cycle list by confidenceAtBuy.
     */
    fun setConfidenceBand(band: ConfidenceBand) {
        if (_uiState.value.confidenceBand == band) return
        _uiState.update { it.copy(confidenceBand = band).applyFiltersAndSort() }
        loadModelPerformance()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Filter controls
    // ─────────────────────────────────────────────────────────────────────────

    fun setDateRange(startMs: Long?, endMs: Long?) {
        _uiState.update { state ->
            state.copy(filters = state.filters.copy(startDateMs = startMs, endDateMs = endMs))
                .applyFiltersAndSort()
        }
    }

    fun setWinLossFilter(filter: WinLossFilter) {
        _uiState.update { state ->
            state.copy(filters = state.filters.copy(winLoss = filter))
                .applyFiltersAndSort()
        }
    }

    fun setVolatilityClass(value: String?) {
        _uiState.update { state ->
            state.copy(filters = state.filters.copy(volatilityClass = value))
                .applyFiltersAndSort()
        }
    }

    fun setLiquidityClass(value: String?) {
        _uiState.update { state ->
            state.copy(filters = state.filters.copy(liquidityClass = value))
                .applyFiltersAndSort()
        }
    }

    fun setSessionType(value: String?) {
        _uiState.update { state ->
            state.copy(filters = state.filters.copy(sessionType = value))
                .applyFiltersAndSort()
        }
    }

    fun clearFilters() {
        _uiState.update { state ->
            state.copy(filters = CycleFilters()).applyFiltersAndSort()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Sort controls
    // ─────────────────────────────────────────────────────────────────────────

    fun setSortColumn(column: CycleSortColumn) {
        _uiState.update { state ->
            val newAscending = if (state.sortColumn == column) !state.sortAscending else false
            state.copy(sortColumn = column, sortAscending = newAscending)
                .applyFiltersAndSort()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Expand/collapse row details
    // ─────────────────────────────────────────────────────────────────────────

    fun toggleExpanded(cycleId: String) {
        _uiState.update { state ->
            state.copy(expandedCycleId = if (state.expandedCycleId == cycleId) null else cycleId)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Filter + sort logic
    // ─────────────────────────────────────────────────────────────────────────

    private fun ReliabilityUiState.applyFiltersAndSort(): ReliabilityUiState {
        val filtered = cycles
            .filter { it.passes(filters) }
            .filter { it.passesConfidenceBand(confidenceBand) }
        val sorted = filtered.sortedWith(cycleComparator(sortColumn, sortAscending))
        return this.copy(filteredCycles = sorted)
    }

    /**
     * Local confidence-band filter on [TradeCycleResponse.confidenceAtBuy].
     *
     * Bands are half-open [min, max) to match the backend: min is included, max
     * is excluded — EXCEPT at the top of the scale, where max >= 1.0 also
     * includes confidence == 1.0. So MEDIUM = [0.4, 0.7) and HIGH = [0.7, 1.0].
     * A confidence of exactly 0.7 belongs to HIGH only; 0.4 to MEDIUM only.
     * Cycles without a confidence value are excluded from any non-ALL band.
     */
    private fun TradeCycleResponse.passesConfidenceBand(band: ConfidenceBand): Boolean {
        if (band == ConfidenceBand.ALL) return true
        val c = confidenceAtBuy ?: return false
        val min = band.min ?: 0f
        val max = band.max ?: 1f
        val underMax = if (max >= 1.0f) c <= max else c < max
        return c >= min && underMax
    }

    private fun TradeCycleResponse.passes(filters: CycleFilters): Boolean {
        // Date range filter
        val buyMs = buyTimestamp?.toEpochMillis()
        if (buyMs != null) {
            filters.startDateMs?.let { if (buyMs < it) return false }
            filters.endDateMs?.let { if (buyMs > it) return false }
        }

        // Win/loss filter
        when (filters.winLoss) {
            WinLossFilter.WIN -> if ((returnPercent ?: 0f) <= 0f) return false
            WinLossFilter.LOSS -> if ((returnPercent ?: 0f) >= 0f) return false
            WinLossFilter.ALL -> { /* no-op */ }
        }

        // Categorical filters
        filters.volatilityClass?.let { if (volatilityClass != it) return false }
        filters.liquidityClass?.let { if (liquidityClass != it) return false }
        filters.sessionType?.let { if (sessionTypeAtBuy != it) return false }

        return true
    }

    private fun cycleComparator(
        column: CycleSortColumn,
        ascending: Boolean
    ): Comparator<TradeCycleResponse> {
        val typedComparator = Comparator<TradeCycleResponse> { a, b ->
            when (column) {
                CycleSortColumn.RETURN_PERCENT -> compareNullable(a.returnPercent, b.returnPercent)
                CycleSortColumn.RETURN_DOLLARS -> compareNullable(a.returnDollars, b.returnDollars)
                CycleSortColumn.HOLD_TIME -> compareNullable(a.holdTimeMinutes, b.holdTimeMinutes)
                CycleSortColumn.CONFIDENCE -> compareNullable(a.confidenceAtBuy, b.confidenceAtBuy)
                CycleSortColumn.GAP_TYPE -> compareNullable(a.gapTypeAtBuy, b.gapTypeAtBuy)
                CycleSortColumn.LIQUIDITY_SCORE -> compareNullable(a.liquidityScoreAtBuy, b.liquidityScoreAtBuy)
            }
        }
        return if (ascending) typedComparator else typedComparator.reversed()
    }

    /**
     * Type-safe nulls-last comparator helper.
     * Null values are sorted to the end of the list regardless of direction.
     */
    private fun <T : Comparable<T>> compareNullable(a: T?, b: T?): Int {
        return when {
            a == null && b == null -> 0
            a == null -> 1
            b == null -> -1
            else -> a.compareTo(b)
        }
    }

    companion object {
        private val formatter = DateTimeFormatter.ISO_DATE_TIME

        private fun String.toEpochMillis(): Long? = try {
            LocalDateTime.parse(this, formatter)
                .atZone(ZoneId.systemDefault())
                .toInstant()
                .toEpochMilli()
        } catch (e: Exception) {
            try {
                Instant.parse(this).toEpochMilli()
            } catch (e2: Exception) {
                null
            }
        }
    }
}
