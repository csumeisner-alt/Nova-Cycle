package com.novacycle.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.data.repository.ThemeRepository
import com.novacycle.data.repository.ThemeState
import com.novacycle.ui.theme.NovaTheme
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicInteger
import javax.inject.Inject

/**
 * Drives the luxe theme system: exposes the persisted [ThemeState], batches the
 * global tap counter into periodic DataStore writes (one write per tap would
 * hammer the disk at 10k+ taps), and broadcasts exactly-once unlock events for
 * the celebration (shimmer + ping + haptic).
 */
@HiltViewModel
class ThemeViewModel @Inject constructor(
    private val themeRepository: ThemeRepository
) : ViewModel() {

    companion object {
        /** How often pending taps are flushed to DataStore. */
        const val TAP_FLUSH_INTERVAL_MS = 250L
    }

    val themeState: StateFlow<ThemeState> = themeRepository.themeState
        .stateIn(viewModelScope, SharingStarted.Eagerly, ThemeState())

    private val _unlockEvents = MutableSharedFlow<NovaTheme>(extraBufferCapacity = 8)
    /** Emits each theme exactly once, at the moment it becomes unlocked. */
    val unlockEvents: SharedFlow<NovaTheme> = _unlockEvents

    private val pendingTaps = AtomicInteger(0)

    init {
        // Flush loop — cancelled automatically with viewModelScope.
        viewModelScope.launch {
            while (true) {
                delay(TAP_FLUSH_INTERVAL_MS)
                flushPendingTaps()
            }
        }
    }

    /** Called from the root composable for every pointer-down anywhere in the app. */
    fun registerTap() {
        pendingTaps.incrementAndGet()
    }

    /**
     * Flush any pending taps immediately, without waiting for the next 250ms
     * tick. Called from the activity's onStop so taps registered in the final
     * batch window are persisted before the process becomes killable.
     */
    fun flushNow() {
        viewModelScope.launch { flushPendingTaps() }
    }

    fun selectTheme(theme: NovaTheme) {
        viewModelScope.launch { themeRepository.selectTheme(theme) }
    }

    internal suspend fun flushPendingTaps() {
        val n = pendingTaps.getAndSet(0)
        if (n > 0) {
            themeRepository.addTaps(n).forEach { _unlockEvents.emit(it) }
        }
    }
}
