package com.novacycle.data.repository

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * App-wide record of the last time ANY backend data was fetched successfully
 * (predictions, indicators, hold-time, or a /healthz poll).
 *
 * Shared singleton so the "Backend unreachable" notice can tell users how
 * stale the on-screen data actually is ("last updated 12 min ago"), no matter
 * which ViewModel performed the last successful fetch.
 */
@Singleton
class DataFreshnessTracker @Inject constructor() {

    private val _lastSuccessAtMillis = MutableStateFlow<Long?>(null)

    /** Epoch millis of the most recent successful backend fetch; null if none yet */
    val lastSuccessAtMillis: StateFlow<Long?> = _lastSuccessAtMillis.asStateFlow()

    /** Call after any successful backend fetch */
    fun recordSuccess(nowMillis: Long = System.currentTimeMillis()) {
        _lastSuccessAtMillis.value = nowMillis
    }
}
