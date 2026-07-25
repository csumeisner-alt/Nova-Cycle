package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.HealthzResponse
import com.novacycle.data.remote.models.HoldTimeResponse
import com.novacycle.data.remote.models.IndicatorResponse
import com.novacycle.data.remote.models.PredictionResponse
import com.novacycle.data.repository.NovaCycleRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject

data class DualGaugeUiState(
    val longPrediction: PredictionResponse? = null,
    val shortPrediction: PredictionResponse? = null,
    val holdTime: HoldTimeResponse? = null,
    val indicators: IndicatorResponse? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    /** Latest backend health snapshot; null until the first successful poll */
    val health: HealthzResponse? = null,
    /** True after several consecutive failed /healthz polls — backend unreachable */
    val backendUnreachable: Boolean = false,
    /** Currently selected ticker — only "VOO" supported, placeholder for multi-ticker */
    val selectedTicker: String = "VOO"
)

/**
 * ViewModel for the main Dual Gauge screen.
 * Calls predictLong + predictShort in parallel to minimize latency.
 * Auto-refreshes every 5 minutes while the screen is active.
 */
@HiltViewModel
class DualGaugeViewModel @Inject constructor(
    private val repository: NovaCycleRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(DualGaugeUiState())
    val uiState: StateFlow<DualGaugeUiState> = _uiState.asStateFlow()

    init {
        loadPredictions()
        startAutoRefresh()
        startHealthPolling()
    }

    /** Parallel fetch of all dashboard data */
    fun loadPredictions() {
        viewModelScope.launch {
            val ticker = _uiState.value.selectedTicker
            _uiState.update { it.copy(isLoading = true, error = null) }

            // Launch long and short predictions in parallel
            val longDeferred = async { repository.getPredictionLong(ticker) }
            val shortDeferred = async { repository.getPredictionShort(ticker) }
            val holdDeferred = async { repository.getHoldTime(ticker) }
            val indicatorsDeferred = async { repository.getIndicators(ticker) }

            val longResult = longDeferred.await()
            val shortResult = shortDeferred.await()
            val holdResult = holdDeferred.await()
            val indicatorsResult = indicatorsDeferred.await()

            _uiState.update { state ->
                state.copy(
                    longPrediction = longResult.getOrNull() ?: state.longPrediction,
                    shortPrediction = shortResult.getOrNull() ?: state.shortPrediction,
                    holdTime = holdResult.getOrNull() ?: state.holdTime,
                    indicators = indicatorsResult.getOrNull() ?: state.indicators,
                    isLoading = false,
                    error = if (longResult.isFailure && shortResult.isFailure) {
                        longResult.exceptionOrNull()?.message ?: "Failed to load predictions"
                    } else null
                )
            }
        }
    }

    fun refreshAll() = loadPredictions()

    /**
     * Placeholder: only "VOO" is supported. In a future multi-ticker version,
     * this would trigger a reload with the new ticker symbol.
     */
    fun selectTicker(ticker: String) {
        if (ticker == "VOO") {
            _uiState.update { it.copy(selectedTicker = "VOO") }
        }
        // Silently ignore unsupported tickers
    }

    /** Poll every 5 minutes using a simple delay loop in viewModelScope */
    private fun startAutoRefresh() {
        viewModelScope.launch {
            while (isActive) {
                delay(5 * 60 * 1000L) // 5 minutes
                loadPredictions()
            }
        }
    }

    /**
     * Poll /healthz every 60 seconds so a degraded backend (failed retrain or
     * neutral-fallback model) is surfaced as a warning banner, matching the
     * web status page. A failed poll keeps the last known health rather than
     * flashing/clearing the banner on transient network errors, but after
     * [UNREACHABLE_THRESHOLD] consecutive failures the UI shows a distinct
     * "Backend unreachable" notice. Any successful poll clears it.
     */
    private fun startHealthPolling() {
        viewModelScope.launch {
            var consecutiveFailures = 0
            while (isActive) {
                repository.getHealth()
                    .onSuccess { health ->
                        consecutiveFailures = 0
                        _uiState.update {
                            it.copy(health = health, backendUnreachable = false)
                        }
                    }
                    .onFailure {
                        consecutiveFailures++
                        if (consecutiveFailures >= UNREACHABLE_THRESHOLD) {
                            _uiState.update { it.copy(backendUnreachable = true) }
                        }
                    }
                delay(60 * 1000L) // 60 seconds
            }
        }
    }

    companion object {
        /** Consecutive failed health polls before showing the unreachable notice */
        private const val UNREACHABLE_THRESHOLD = 3
    }
}
