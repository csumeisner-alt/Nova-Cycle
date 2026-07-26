package com.novacycle.viewmodel

import android.app.Activity
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.novacycle.billing.BillingManager
import com.novacycle.billing.MintBillingState
import com.novacycle.data.theme.ThemePrefs
import com.novacycle.data.theme.ThemeState
import com.novacycle.ui.theme.AppTheme
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import javax.inject.Inject

/**
 * Thin ViewModel over the singleton [ThemePrefs] + [BillingManager].
 * Because both underlying stores are process-wide singletons exposing
 * StateFlows, every screen that instantiates its own ThemeViewModel sees
 * the exact same state — the activity, dashboard, and settings stay in sync.
 */
@HiltViewModel
class ThemeViewModel @Inject constructor(
    private val themePrefs: ThemePrefs,
    private val billingManager: BillingManager
) : ViewModel() {

    val themeState: StateFlow<ThemeState> = themePrefs.state

    val billingState: StateFlow<MintBillingState> = billingManager.state
        .stateIn(viewModelScope, SharingStarted.Eagerly, billingManager.state.value)

    /** One-shot: set to true on the tap that crosses 20,000; UI shows dialog then clears. */
    private val _showUnlockCelebration = MutableStateFlow(false)
    val showUnlockCelebration: StateFlow<Boolean> = _showUnlockCelebration.asStateFlow()

    init {
        // Connect + restore purchases as soon as the theme system is used.
        billingManager.ensureConnected()
    }

    val selectedTheme: AppTheme get() = themePrefs.state.value.selectedTheme

    fun onLogoTap() {
        if (themePrefs.registerTap()) {
            _showUnlockCelebration.value = true
        }
    }

    fun dismissUnlockCelebration() {
        _showUnlockCelebration.value = false
    }

    fun selectTheme(theme: AppTheme) {
        themePrefs.selectTheme(theme)
    }

    /** Returns false when billing isn't ready (UI shows the unavailable reason). */
    fun purchaseMintLuxe(activity: Activity): Boolean =
        billingManager.launchPurchase(activity)
}
