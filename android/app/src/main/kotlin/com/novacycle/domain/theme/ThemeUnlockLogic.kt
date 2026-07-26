package com.novacycle.domain.theme

import com.novacycle.ui.theme.AppTheme

/**
 * Pure, side-effect-free rules for the tap-achievement and theme locking.
 * Kept free of Android dependencies so it is fully covered by JVM unit tests.
 */
object ThemeUnlockLogic {

    /** Taps required to unlock Aurora Flux + Crimson Pulse. */
    const val UNLOCK_TAP_THRESHOLD = 20_000

    /** True exactly on the tap that crosses the achievement threshold. */
    fun isUnlockTap(newTapCount: Int): Boolean = newTapCount == UNLOCK_TAP_THRESHOLD

    /** Whether the achievement is met at [tapCount]. */
    fun achievementReached(tapCount: Int): Boolean = tapCount >= UNLOCK_TAP_THRESHOLD

    /**
     * Whether [theme] can currently be selected.
     * Dark Luxe is always available; Aurora/Crimson need the tap achievement
     * flags; Mint Luxe needs a verified purchase.
     */
    fun isThemeAvailable(
        theme: AppTheme,
        auroraUnlocked: Boolean,
        crimsonUnlocked: Boolean,
        mintUnlocked: Boolean
    ): Boolean = when (theme) {
        AppTheme.DARK_LUXE     -> true
        AppTheme.AURORA_FLUX   -> auroraUnlocked
        AppTheme.CRIMSON_PULSE -> crimsonUnlocked
        AppTheme.MINT_LUXE     -> mintUnlocked
    }

    /**
     * Theme to fall back to if the persisted selection is no longer available
     * (e.g. corrupted prefs). Never returns a locked theme.
     */
    fun sanitizeSelection(
        selected: AppTheme,
        auroraUnlocked: Boolean,
        crimsonUnlocked: Boolean,
        mintUnlocked: Boolean
    ): AppTheme =
        if (isThemeAvailable(selected, auroraUnlocked, crimsonUnlocked, mintUnlocked)) selected
        else AppTheme.DARK_LUXE

    /** "12,345 / 20,000 taps" style progress label. */
    fun progressLabel(tapCount: Int): String {
        val capped = tapCount.coerceAtMost(UNLOCK_TAP_THRESHOLD)
        return "%,d / %,d taps".format(capped, UNLOCK_TAP_THRESHOLD)
    }

    /** Progress fraction in [0, 1] for progress bars. */
    fun progressFraction(tapCount: Int): Float =
        (tapCount.toFloat() / UNLOCK_TAP_THRESHOLD).coerceIn(0f, 1f)
}
