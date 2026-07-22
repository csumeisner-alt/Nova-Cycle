package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.IndicatorResponse
import com.novacycle.data.repository.NovaCycleRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class IndicatorUiState(
    val indicators: IndicatorResponse? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val ticker: String = "VOO"
)

/**
 * ViewModel for the Indicator List screen.
 * Loads the current technical indicator snapshot from the API.
 */
@HiltViewModel
class IndicatorViewModel @Inject constructor(
    private val repository: NovaCycleRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(IndicatorUiState())
    val uiState: StateFlow<IndicatorUiState> = _uiState.asStateFlow()

    init {
        loadIndicators()
    }

    fun loadIndicators() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }
            val result = repository.getIndicators(_uiState.value.ticker)
            result.fold(
                onSuccess = { indicators ->
                    _uiState.update { it.copy(indicators = indicators, isLoading = false) }
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            error = error.message ?: "Failed to load indicators"
                        )
                    }
                }
            )
        }
    }
}
