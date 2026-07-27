package com.novacycle.viewmodel

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.ConnectivityErrorMapper
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
    /** Currently selected ticker — only "VOO" supported, placeholder for multi-ticker */
    val selectedTicker: String = "VOO",
    /** Epoch millis of the last successful data refresh on this screen; null if none yet */
    val lastUpdatedAtMillis: Long? = null
)

/**
 * ViewModel for the main Dual Gauge screen.
 * Calls predictLong + predictShort in parallel to minimize latency.
 * Auto-refreshes every 5 minutes while the screen is active.
 */
@HiltViewModel
class DualGaugeViewModel @Inject constructor(
    private val repository: NovaCycleRepository,
    private val connectivityErrorMapper: ConnectivityErrorMapper
) : ViewModel() {

    private val _uiState = MutableStateFlow(DualGaugeUiState())
    val uiState: StateFlow<DualGaugeUiState> = _uiState.asStateFlow()

    init {
        probeBackendReachability()
        loadPredictions()
        startAutoRefresh()
        // NOTE: health polling now lives in the app-level HealthViewModel
        // (one shared /healthz poll for all screens) — do not poll here.
    }

    /**
     * Lightweight startup reachability probe (runs concurrently with the first
     * prediction fetch): if /healthz fails, surface a classified connectivity
     * error immediately instead of waiting for all prediction calls to time out.
     */
    private fun probeBackendReachability() {
        viewModelScope.launch {
            repository.getHealth().onFailure { e ->
                val mapped = connectivityErrorMapper.map(e)
                Log.w(
                    "DualGaugeViewModel",
                    "Health probe failed [${mapped.code}]: ${e.javaClass.name}: ${e.message}"
                )
                _uiState.update { state ->
                    // Don't overwrite a more specific error from a completed fetch
                    if (state.error == null && state.longPrediction == null) {
                        state.copy(error = mapped.userMessage)
                    } else state
                }
            }
        }
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

            // Data freshness is recorded centrally by the repository on remote success;
            // here we only track this screen's own "Updated X ago" header label.
            val anySuccess = listOf(longResult, shortResult, holdResult, indicatorsResult)
                .any { it.isSuccess }

            _uiState.update { state ->
                val bothFailed = longResult.isFailure && shortResult.isFailure
                state.copy(
                    // Safe fallback: keep the last known prediction if this fetch
                    // failed; when there is none at all, show a neutral HOLD gauge
                    // (confidence 0.0) instead of an empty/crashed screen.
                    longPrediction = longResult.getOrNull() ?: state.longPrediction
                        ?: if (bothFailed) NEUTRAL_FALLBACK else null,
                    shortPrediction = shortResult.getOrNull() ?: state.shortPrediction
                        ?: if (bothFailed) NEUTRAL_FALLBACK else null,
                    holdTime = holdResult.getOrNull() ?: state.holdTime,
                    indicators = indicatorsResult.getOrNull() ?: state.indicators,
                    lastUpdatedAtMillis = if (anySuccess) System.currentTimeMillis()
                                          else state.lastUpdatedAtMillis,
                    isLoading = false,
                    error = if (bothFailed) {
                        // Classify the failure so the banner names the real cause
                        // (offline / DNS / unreachable / timeout) instead of "null".
                        val cause = longResult.exceptionOrNull()
                            ?: shortResult.exceptionOrNull()
                        cause?.let { ex ->
                            val mapped = connectivityErrorMapper.map(ex)
                            Log.w(
                                "DualGaugeViewModel",
                                "Predictions failed [${mapped.code}]: ${ex.javaClass.name}: ${ex.message}"
                            )
                            mapped.userMessage
                        } ?: "Failed to load predictions"
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

    companion object {
        /**
         * Neutral gauge state shown when the backend is unreachable and no
         * cached prediction exists — never leaves the gauges blank or crashes.
         */
        val NEUTRAL_FALLBACK = PredictionResponse(
            score = 0f,
            signal = "neutral",
            confidence = 0f,
            note = "Backend unreachable — showing neutral fallback"
        )
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

}
