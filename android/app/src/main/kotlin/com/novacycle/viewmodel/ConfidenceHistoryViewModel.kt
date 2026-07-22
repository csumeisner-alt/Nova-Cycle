package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.domain.model.ConfidencePoint
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.usecase.GetConfidenceHistoryUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ConfidenceHistoryUiState(
    val confidencePoints: List<ConfidencePoint> = emptyList(),
    /** Separate list of momentum values for the momentum ribbon chart */
    val longMomentumPoints: List<Float> = emptyList(),
    val shortMomentumPoints: List<Float> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val selectedWindow: String = "7d",
    val ticker: String = "VOO",
    val smoothingEnabled: Boolean = false
)

/**
 * ViewModel for the Confidence History chart screen.
 * Fetches history and computes momentum (delta) between consecutive points.
 * EMA smoothing is applied based on SensitivitySettings.
 */
@HiltViewModel
class ConfidenceHistoryViewModel @Inject constructor(
    private val getConfidenceHistoryUseCase: GetConfidenceHistoryUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(ConfidenceHistoryUiState())
    val uiState: StateFlow<ConfidenceHistoryUiState> = _uiState.asStateFlow()

    private var currentSettings: SensitivitySettings = SensitivitySettings()

    init {
        loadHistory()
    }

    fun loadHistory(window: String = _uiState.value.selectedWindow) {
        viewModelScope.launch {
            val ticker = _uiState.value.ticker
            _uiState.update { it.copy(isLoading = true, error = null, selectedWindow = window) }

            val result = getConfidenceHistoryUseCase(ticker, window, currentSettings)

            result.fold(
                onSuccess = { points ->
                    _uiState.update { state ->
                        state.copy(
                            confidencePoints = points,
                            longMomentumPoints = points.map { it.longMomentum },
                            shortMomentumPoints = points.map { it.shortMomentum },
                            isLoading = false,
                            error = null
                        )
                    }
                },
                onFailure = { error ->
                    _uiState.update { state ->
                        state.copy(
                            isLoading = false,
                            error = error.message ?: "Failed to load confidence history"
                        )
                    }
                }
            )
        }
    }

    fun setWindow(window: String) {
        if (window != _uiState.value.selectedWindow) loadHistory(window)
    }

    fun applySettings(settings: SensitivitySettings) {
        currentSettings = settings
        loadHistory()
    }
}
