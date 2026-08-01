package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.remote.models.FilteredSignalResponse
import com.novacycle.data.remote.models.PriceSnapshotResponse
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.ApplyFilteredSignalsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class FilteredChartUiState(
    val candles: List<CandleResponse> = emptyList(),
    val filteredSignals: List<SignalData> = emptyList(),
    val tradeCycles: List<ApplyFilteredSignalsUseCase.TradeCycle> = emptyList(),
    val priceSnapshot: PriceSnapshotResponse? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val selectedWindow: String = "30d",
    /** Candle timeframe: 'daily', '5min', '15min' or '1h' */
    val selectedTimeframe: String = "daily",
    val ticker: String = "VOO",
    /** Epoch millis of the last successful data refresh on this screen; null if none yet */
    val lastUpdatedAtMillis: Long? = null
)

/**
 * ViewModel for the Filtered Chart screen.
 * Fetches backend-filtered signals, then applies the client-side strongest-confidence
 * rule via ApplyFilteredSignalsUseCase with the user's current sensitivity settings.
 * This double-filtering gives the user real-time control over what they see.
 */
@HiltViewModel
class FilteredChartViewModel @Inject constructor(
    private val repository: NovaCycleRepository,
    private val applyFilteredSignalsUseCase: ApplyFilteredSignalsUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(FilteredChartUiState())
    val uiState: StateFlow<FilteredChartUiState> = _uiState.asStateFlow()

    private var currentSettings: SensitivitySettings = SensitivitySettings()

    init {
        loadData()
    }

    fun loadData(
        window: String = _uiState.value.selectedWindow,
        timeframe: String = _uiState.value.selectedTimeframe
    ) {
        viewModelScope.launch {
            val ticker = _uiState.value.ticker
            _uiState.update {
                it.copy(isLoading = true, error = null, selectedWindow = window, selectedTimeframe = timeframe)
            }

            val candlesDeferred = async { repository.getCandles(ticker, window, timeframe) }
            val signalsDeferred = async { repository.getFilteredSignals(ticker, window) }
            val priceSnapshotDeferred = async { repository.getPriceSnapshot(ticker) }

            val candlesResult = candlesDeferred.await()
            val signalsResult = signalsDeferred.await()
            val priceSnapshotResult = priceSnapshotDeferred.await()

            val rawSignals = signalsResult.getOrDefault(emptyList()).map { it.toDomain() }

            // Apply client-side strongest-confidence filtering with user settings
            val filterResult = applyFilteredSignalsUseCase(rawSignals, currentSettings)

            // Freshness is reported to the shared DataFreshnessTracker by the
            // repository on remote success; here we only track this screen's label.
            val anySuccess = candlesResult.isSuccess || signalsResult.isSuccess ||
                priceSnapshotResult.isSuccess

            _uiState.update { state ->
                state.copy(
                    candles = candlesResult.getOrDefault(state.candles),
                    filteredSignals = filterResult.signals,
                    tradeCycles = filterResult.cycles,
                    priceSnapshot = priceSnapshotResult.getOrNull() ?: state.priceSnapshot,
                    lastUpdatedAtMillis = if (anySuccess) System.currentTimeMillis()
                                          else state.lastUpdatedAtMillis,
                    isLoading = false,
                    error = when {
                        candlesResult.isFailure -> candlesResult.exceptionOrNull()?.message
                        signalsResult.isFailure -> signalsResult.exceptionOrNull()?.message
                        else -> null
                    }
                )
            }
        }
    }

    fun setWindow(window: String) {
        if (window != _uiState.value.selectedWindow) loadData(window)
    }

    fun setTimeframe(timeframe: String) {
        if (timeframe != _uiState.value.selectedTimeframe) {
            loadData(timeframe = timeframe)
        }
    }

    fun applySettings(settings: SensitivitySettings) {
        currentSettings = settings
        loadData()
    }

    private fun FilteredSignalResponse.toDomain() = SignalData(
        id = id,
        timestamp = timestamp,
        ticker = ticker,
        cycleId = cycleId,
        signalType = signalType,
        gaugeType = gaugeType,
        confidence = confidence,
        sessionType = sessionType,
        convictionTier = convictionTier,
        convictionReasons = convictionReasons
    )
}
