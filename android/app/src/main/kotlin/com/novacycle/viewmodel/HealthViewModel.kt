package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.remote.models.HealthzResponse
import com.novacycle.data.repository.DataFreshnessTracker
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
    val backendUnreachable: Boolean = false,
    /** Epoch millis of the last successful user-visible DATA fetch (not health polls); null if none yet */
    val lastSuccessAtMillis: Long? = null
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
    private val repository: NovaCycleRepository,
    private val freshnessTracker: DataFreshnessTracker
) : ViewModel() {

    private val _uiState = MutableStateFlow(HealthUiState())
    val uiState: StateFlow<HealthUiState> = _uiState.asStateFlow()

    init {
        startHealthPolling()
        observeFreshness()
    }

    /** Mirror the app-wide last-successful-fetch timestamp into the UI state */
    private fun observeFreshness() {
        viewModelScope.launch {
            freshnessTracker.lastSuccessAtMillis.collect { ts ->
                _uiState.update { it.copy(lastSuccessAtMillis = ts) }
            }
        }
    }

    private fun startHealthPolling() {
        viewModelScope.launch {
            var consecutiveFailures = 0
            while (isActive) {
                repository.getHealth()
                    .onSuccess { health ->
                        // NOTE: a successful /healthz poll proves reachability only —
                        // it must NOT advance the data-freshness timestamp, which
                        // tracks user-visible data fetches (recorded by the repository).
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
                // Fast recovery: once flagged unreachable, re-check every 5 s so the
                // app snaps back the moment the backend comes up. Normal cadence
                // (60 s) otherwise to avoid needless traffic.
                val interval = if (_uiState.value.backendUnreachable) {
                    RECOVERY_POLL_INTERVAL_MS
                } else {
                    NORMAL_POLL_INTERVAL_MS
                }
                delay(interval)
            }
        }
    }

    companion object {
        /** Consecutive failed health polls before showing the unreachable notice */
        private const val UNREACHABLE_THRESHOLD = 3
        /** Poll cadence while the backend is reachable */
        const val NORMAL_POLL_INTERVAL_MS = 60 * 1000L
        /** Fast retry cadence while the backend is flagged unreachable */
        const val RECOVERY_POLL_INTERVAL_MS = 5 * 1000L
    }
}
