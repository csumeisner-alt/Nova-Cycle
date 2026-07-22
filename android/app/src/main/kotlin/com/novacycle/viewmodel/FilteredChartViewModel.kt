package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.CandleResponse
import com.novacycle.data.remote.models.FilteredSignalResponse
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
    val isLoading: Boolean = false,
    val error: String? = null,
    val selectedWindow: String = "30d",
    val ticker: String = "VOO"
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

    fun loadData(window: String = _uiState.value.selectedWindow) {
        viewModelScope.launch {
            val ticker = _uiState.value.ticker
            _uiState.update { it.copy(isLoading = true, error = null, selectedWindow = window) }

            val candlesDeferred = async { repository.getCandles(ticker, window) }
            val signalsDeferred = async { repository.getFilteredSignals(ticker, window) }

            val candlesResult = candlesDeferred.await()
            val signalsResult = signalsDeferred.await()

            val rawSignals = signalsResult.getOrDefault(emptyList()).map { it.toDomain() }

            // Apply client-side strongest-confidence filtering with user settings
            val filterResult = applyFilteredSignalsUseCase(rawSignals, currentSettings)

            _uiState.update { state ->
                state.copy(
                    candles = candlesResult.getOrDefault(state.candles),
                    filteredSignals = filterResult.signals,
                    tradeCycles = filterResult.cycles,
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
        sessionType = sessionType
    )
}
