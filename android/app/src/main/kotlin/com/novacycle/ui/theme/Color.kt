package com.novacycle.ui.theme

import androidx.compose.ui.graphics.Color

// Primary brand colors
val NovaBuyGreen      = Color(0xFF00C853)   // Strong green for BUY signals
val NovaSellRed       = Color(0xFFD50000)   // Strong red for SELL signals
val NovaNeutralGray   = Color(0xFF757575)   // Gray for NEUTRAL signals

// Background hierarchy
val NovaBackground    = Color(0xFF0D0D0D)   // Near-black app background
val NovaSurface       = Color(0xFF1A1A1A)   // Cards / surfaces
val NovaSurfaceVariant= Color(0xFF242424)   // Slightly lighter surface

// Accent colors
val NovaExtendedBlue  = Color(0xFF2196F3)   // Extended-hours signals
val NovaGapPurple     = Color(0xFF9C27B0)   // Gap-driven signals
val NovaWarningYellow = Color(0xFFFFD600)   // Macro-override suppressed
val NovaFadedGray     = Color(0xFF424242)   // Liquidity-filtered signals

// Gauge arc gradient stops
val GaugeColorLeft    = Color(0xFFD50000)   // Leftmost arc (full sell)
val GaugeColorCenter  = Color(0xFFFFD600)   // Center arc (neutral)
val GaugeColorRight   = Color(0xFF00C853)   // Rightmost arc (full buy)

// Text colors
val NovaOnBackground  = Color(0xFFEEEEEE)
val NovaOnSurface     = Color(0xFFCCCCCC)
val NovaOnPrimary     = Color(0xFFFFFFFF)

// ── Theme accent palettes ─────────────────────────────────────────────
// Dark Luxe (default) — bronze-gold on near-black, matching the uploaded logo
val LuxeGold          = Color(0xFFE0B040)
val LuxeGoldDeep      = Color(0xFFC89030)
val LuxeGoldSoft      = Color(0xFFF5E6A8)

// Aurora Flux — teal / violet
val AuroraTeal        = Color(0xFF00E5CF)
val AuroraViolet      = Color(0xFF9C6BFF)
val AuroraSoft        = Color(0xFFB9FFF4)

// Crimson Pulse — red / ember
val CrimsonAccent     = Color(0xFFFF4655)
val CrimsonEmber      = Color(0xFFFF8A50)
val CrimsonSoft       = Color(0xFFFFC9CE)

// Mint Luxe — mint / emerald (premium)
val MintAccent        = Color(0xFF69F0AE)
val MintEmerald       = Color(0xFF00BFA5)
val MintSoft          = Color(0xFFD8FFEF)

// VIX regime badge colors
val VixLow      = Color(0xFF00C853)
val VixNormal   = Color(0xFF2196F3)
val VixHigh     = Color(0xFFFF6D00)
val VixExtreme  = Color(0xFFD50000)
