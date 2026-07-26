package com.novacycle.ui.theme

/**
 * The four NovaCycle visual themes.
 *
 * - DARK_LUXE      — default: gold accents on near-black.
 * - AURORA_FLUX    — unlocked at 20,000 logo taps: teal/violet.
 * - CRIMSON_PULSE  — unlocked at 20,000 logo taps: red/ember.
 * - MINT_LUXE      — premium, purchased via Google Play Billing ($1.49).
 *
 * [storageKey] is what gets persisted in SharedPreferences — never rename
 * existing keys or users lose their selected theme on upgrade.
 */
enum class AppTheme(val storageKey: String, val displayName: String, val tagline: String) {
    DARK_LUXE("dark_luxe", "Dark Luxe", "Gold on midnight — the NovaCycle signature"),
    AURORA_FLUX("aurora_flux", "Aurora Flux", "Teal & violet northern-lights glow"),
    CRIMSON_PULSE("crimson_pulse", "Crimson Pulse", "Deep red ember energy"),
    MINT_LUXE("mint_luxe", "Mint Luxe", "Premium mint & emerald calm");

    companion object {
        fun fromStorageKey(key: String?): AppTheme =
            entries.firstOrNull { it.storageKey == key } ?: DARK_LUXE
    }
}
