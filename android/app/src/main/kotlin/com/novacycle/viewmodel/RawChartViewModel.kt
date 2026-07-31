package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.remote.models.PriceSnapshotResponse
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.GetSignalsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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
    val ticker: String = "VOO",
    /** Epoch millis of the last successful data refresh on this screen; null if none yet */
    val lastUpdatedAtMillis: Long? = null
)

/**
 * ViewModel for the Raw Chart screen.
 * Loads candles and ALL raw signals, then applies sensitivity filter client-side.
 */
@HiltViewModel
class RawChartViewModel @Inject constructor(
    private val repository: NovaCycleRepository,
    private val getSignalsUseCase: GetSignalsUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(RawChartUiState())
    val uiState: StateFlow<RawChartUiState> = _uiState.asStateFlow()

    // Current settings cached from SettingsViewModel via screen
    private var currentSettings: SensitivitySettings = SensitivitySettings()

    init {
        loadData()
    }

    fun loadData(window: String = _uiState.value.selectedWindow) {
        viewModelScope.launch {
            val ticker = _uiState.value.ticker
            _uiState.update { it.copy(isLoading = true, error = null, selectedWindow = window) }

            val candlesDeferred = async { repository.getCandles(ticker, window) }
            val signalsDeferred = async {
                getSignalsUseCase(ticker, window, currentSettings)
            }
            val priceSnapshotDeferred = async { repository.getPriceSnapshot(ticker) }

            val candlesResult = candlesDeferred.await()
            val signalsResult = signalsDeferred.await()
            val priceSnapshotResult = priceSnapshotDeferred.await()

            val anySuccess = candlesResult.isSuccess || signalsResult.isSuccess ||
                priceSnapshotResult.isSuccess

            _uiState.update { state ->
                state.copy(
                    candles = candlesResult.getOrDefault(state.candles),
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

    /** Called by screen when settings change (passed down from SettingsViewModel) */
    fun applySettings(settings: SensitivitySettings) {
        currentSettings = settings
        loadData()
    }
}
