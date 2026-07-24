package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
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
    val error: String? = null
)

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
    }

    /**
     * Fetch the latest trade history from the backend. The backend regenerates
     * cycles from the filtered BUY→SELL timeline and computes metrics, so a
     * single call gives us both the cycle list and the summary panel.
     */
    fun loadTradeHistory(ticker: String = "VOO", window: String = "30d") {
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
                            error = null
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

    fun refresh() = loadTradeHistory()

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
        val filtered = cycles.filter { it.passes(filters) }
        val sorted = filtered.sortedWith(cycleComparator(sortColumn, sortAscending))
        return this.copy(filteredCycles = sorted)
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
