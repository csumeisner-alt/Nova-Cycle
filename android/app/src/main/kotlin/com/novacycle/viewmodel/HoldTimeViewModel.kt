package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.HoldTimeResponse
import com.novacycle.data.repository.NovaCycleRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HoldTimeUiState(
    val holdTime: HoldTimeResponse? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val ticker: String = "VOO"
)

/**
 * ViewModel for the Hold Time screen.
 */
@HiltViewModel
class HoldTimeViewModel @Inject constructor(
    private val repository: NovaCycleRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(HoldTimeUiState())
    val uiState: StateFlow<HoldTimeUiState> = _uiState.asStateFlow()

    init {
        loadHoldTime()
    }

    fun loadHoldTime() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }
            val result = repository.getHoldTime(_uiState.value.ticker)
            result.fold(
                onSuccess = { holdTime ->
                    _uiState.update { it.copy(holdTime = holdTime, isLoading = false) }
                },
                onFailure = { error ->
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            error = error.message ?: "Failed to load hold time estimate"
                        )
                    }
                }
            )
        }
    }
}
