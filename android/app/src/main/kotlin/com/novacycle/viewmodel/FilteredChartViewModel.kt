package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.NetworkMonitor
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.remote.models.FilteredSignalResponse
import com.novacycle.data.remote.models.PriceSnapshotResponse
import com.novacycle.data.repository.CandlesWithSource
import com.novacycle.data.repository.ChartPreferencesRepository
import com.novacycle.data.repository.ChartScreenKey
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.ApplyFilteredSignalsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
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
 * ViewModel for the Filtered Chart screen.
 * Fetches backend-filtered signals, then applies the client-side strongest-confidence
 * rule via ApplyFilteredSignalsUseCase with the user's current sensitivity settings.
 * This double-filtering gives the user real-time control over what they see.
 *
 * Observes [NetworkMonitor] so that when connectivity is restored after an
 * offline session the cache badge clears automatically — no user interaction
 * required.  Rapid reconnects are debounced to avoid parallel fan-out.
 */
@HiltViewModel
class FilteredChartViewModel @Inject constructor(
    private val repository: NovaCycleRepository,
    private val applyFilteredSignalsUseCase: ApplyFilteredSignalsUseCase,
    private val chartPrefs: ChartPreferencesRepository,
    private val networkMonitor: NetworkMonitor
) : ViewModel() {

    private val _uiState = MutableStateFlow(FilteredChartUiState())
    val uiState: StateFlow<FilteredChartUiState> = _uiState.asStateFlow()

    private var currentSettings: SensitivitySettings = SensitivitySettings()

    init {
        // Restore the persisted timeframe + render mode BEFORE the first load,
        // so the saved timeframe is fetched immediately (no daily flash-reload).
        viewModelScope.launch {
            val saved = chartPrefs.prefs(ChartScreenKey.FILTERED).first()
            _uiState.update { it.copy(renderMode = saved.renderMode) }
            loadData(timeframe = saved.timeframe)
        }

        // Auto-refresh when the network comes back while the badge is showing.
        // debounce(500 ms) collapses rapid cellular ↔ Wi-Fi handoffs into one
        // reload so parallel requests never fan out.
        viewModelScope.launch {
            networkMonitor.isConnected
                .debounce(RECONNECT_DEBOUNCE_MS)
                .filter { connected -> connected && _uiState.value.candlesFromCache }
                .collect { loadData() }
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
            viewModelScope.launch { chartPrefs.saveTimeframe(ChartScreenKey.FILTERED, timeframe) }
        }
    }

    /** Toggle candles/line render mode and persist the choice. */
    fun setRenderMode(renderMode: String) {
        if (renderMode != _uiState.value.renderMode) {
            _uiState.update { it.copy(renderMode = renderMode) }
            viewModelScope.launch { chartPrefs.saveRenderMode(ChartScreenKey.FILTERED, renderMode) }
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
        convictionReasons = convictionReasons,
        modelState = modelState
    )

    companion object {
        /** Rapid reconnect debounce window in milliseconds. */
        const val RECONNECT_DEBOUNCE_MS = 500L
    }
}
