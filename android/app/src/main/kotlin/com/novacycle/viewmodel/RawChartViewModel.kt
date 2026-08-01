package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.remote.models.PriceSnapshotResponse
import com.novacycle.data.repository.ChartPreferencesRepository
import com.novacycle.data.repository.ChartScreenKey
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.data.repository.CandlesWithSource
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.GetSignalsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class RawChartUiState(
    val candles: List<CandleResponse> = emptyList(),
    val signals: List<SignalData> = emptyList(),
    val priceSnapshot: PriceSnapshotResponse? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val selectedWindow: String = "30d",
    /** Candle timeframe: 'daily', '5min', '15min' or '1h' */
    val selectedTimeframe: String = "daily",
    val ticker: String = "VOO",
    /** Chart render mode name: 'CANDLES' or 'LINE' */
    val renderMode: String = "CANDLES",
    /** Epoch millis of the last successful data refresh on this screen; null if none yet */
    val lastUpdatedAtMillis: Long? = null,
    /** True when the candle data was served from the Room cache (network unavailable). */
    val candlesFromCache: Boolean = false,
    /** ISO-8601 timestamp of the newest cached bar; non-null only when [candlesFromCache]. */
    val cacheNewestBarTimestamp: String? = null
)

/**
 * ViewModel for the Raw Chart screen.
 * Loads candles and ALL raw signals, then applies sensitivity filter client-side.
 */
@HiltViewModel
class RawChartViewModel @Inject constructor(
    private val repository: NovaCycleRepository,
    private val getSignalsUseCase: GetSignalsUseCase,
    private val chartPrefs: ChartPreferencesRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(RawChartUiState())
    val uiState: StateFlow<RawChartUiState> = _uiState.asStateFlow()

    // Current settings cached from SettingsViewModel via screen
    private var currentSettings: SensitivitySettings = SensitivitySettings()

    init {
        // Restore the persisted timeframe + render mode BEFORE the first load,
        // so the saved timeframe is fetched immediately (no daily flash-reload).
        viewModelScope.launch {
            val saved = chartPrefs.prefs(ChartScreenKey.RAW).first()
            _uiState.update { it.copy(renderMode = saved.renderMode) }
            loadData(timeframe = saved.timeframe)
        }
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
            val signalsDeferred = async {
                getSignalsUseCase(ticker, window, currentSettings)
            }
            val priceSnapshotDeferred = async { repository.getPriceSnapshot(ticker) }

            val candlesResult = candlesDeferred.await()
            val signalsResult = signalsDeferred.await()
            val priceSnapshotResult = priceSnapshotDeferred.await()

            val anySuccess = candlesResult.isSuccess || signalsResult.isSuccess ||
                priceSnapshotResult.isSuccess

            val candlesSource = candlesResult.getOrNull()
            val previousSource = CandlesWithSource(
                candles = _uiState.value.candles,
                fromCache = _uiState.value.candlesFromCache,
                newestBarTimestamp = _uiState.value.cacheNewestBarTimestamp
            )
            val resolvedSource = candlesSource ?: previousSource

            _uiState.update { state ->
                state.copy(
                    candles = resolvedSource.candles,
                    candlesFromCache = resolvedSource.fromCache,
                    cacheNewestBarTimestamp = if (resolvedSource.fromCache)
                        resolvedSource.newestBarTimestamp else null,
                    signals = signalsResult.getOrDefault(state.signals),
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
            viewModelScope.launch { chartPrefs.saveTimeframe(ChartScreenKey.RAW, timeframe) }
        }
    }

    /** Toggle candles/line render mode and persist the choice. */
    fun setRenderMode(renderMode: String) {
        if (renderMode != _uiState.value.renderMode) {
            _uiState.update { it.copy(renderMode = renderMode) }
            viewModelScope.launch { chartPrefs.saveRenderMode(ChartScreenKey.RAW, renderMode) }
        }
    }

    /** Called by screen when settings change (passed down from SettingsViewModel) */
    fun applySettings(settings: SensitivitySettings) {
        currentSettings = settings
        loadData()
    }
}
