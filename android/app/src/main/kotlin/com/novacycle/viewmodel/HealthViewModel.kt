package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.HealthzResponse
import com.novacycle.data.repository.NovaCycleRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HealthUiState(
    /** Latest backend health snapshot; null until the first successful poll */
    val health: HealthzResponse? = null,
    /** True after several consecutive failed /healthz polls — backend unreachable */
    val backendUnreachable: Boolean = false
)

/**
 * App-level backend health state, shared across every screen.
 *
 * Scoped to the activity (obtained at the NavHost level), so there is exactly
 * ONE /healthz poll for the whole app — individual screens must not poll
 * health themselves.
 *
 * Polls /healthz every 60 seconds so a degraded backend (failed retrain or
 * neutral-fallback model) is surfaced as a warning banner on all data screens,
 * matching the web status page. A failed poll keeps the last known health
 * rather than flashing/clearing the banner on transient network errors, but
 * after [UNREACHABLE_THRESHOLD] consecutive failures the UI shows a distinct
 * "Backend unreachable" notice. Any successful poll clears it.
 */
@HiltViewModel
class HealthViewModel @Inject constructor(
    private val repository: NovaCycleRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(HealthUiState())
    val uiState: StateFlow<HealthUiState> = _uiState.asStateFlow()

    init {
        startHealthPolling()
    }

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
