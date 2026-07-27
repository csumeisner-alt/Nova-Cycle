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

// Luxe theme palette (gold brand + unlockable themes)
val NovaGold                 = Color(0xFFCBA135)   // DarkLuxe primary (matte gold)
val NovaGoldBright           = Color(0xFFFFD700)   // DarkLuxe highlight (bright gold)
val NovaMint                 = Color(0xFFA8F5D1)   // MintLuxe primary
val NovaAurora               = Color(0xFF00D4FF)   // AuroraFlux primary
val NovaAuroraBackground     = Color(0xFF020B24)   // AuroraFlux deep navy background
val NovaAuroraSurface        = Color(0xFF0A1730)
val NovaAuroraSurfaceVariant = Color(0xFF122242)
val NovaCrimson              = Color(0xFFFF0033)   // CrimsonPulse primary
val NovaCrimsonBackground    = Color(0xFF000000)   // CrimsonPulse pure black background
val NovaCrimsonSurface       = Color(0xFF160308)
val NovaCrimsonSurfaceVariant= Color(0xFF220510)

// Rose Luxe (rose-pink neon on black, flowing light ribbons)
val NovaRose                 = Color(0xFFF48FA6)   // Rose Luxe primary
val NovaRoseGlow             = Color(0xFFFF9FB2)   // Rose neon glow
val NovaRoseBackground       = Color(0xFF0C0507)   // Near-black with warm cast
val NovaRoseSurface          = Color(0xFF1A0D12)
val NovaRoseSurfaceVariant   = Color(0xFF241318)

// Heritage Motion (copper/bronze on warm patterned taupe, green/red racing stripe)
val NovaCopper               = Color(0xFFC88A5A)   // Heritage primary (polished copper)
val NovaCopperBright         = Color(0xFFE8B98C)   // Copper highlight
val NovaHeritageBackground   = Color(0xFF9C8D74)   // Warm taupe canvas
val NovaHeritageSurface      = Color(0xFF3A2E23)   // Deep espresso cards
val NovaHeritageSurfaceVariant = Color(0xFF463829)
val NovaHeritageOnSurface    = Color(0xFFF3E5CE)   // Cream text on espresso
val NovaHeritageOnBackground = Color(0xFF2A2119)   // Dark brown text on taupe
val NovaStripeGreen          = Color(0xFF14452F)   // Heritage stripe green
val NovaStripeRed            = Color(0xFFB01E28)   // Heritage stripe red

// VIX regime badge colors
val VixLow      = Color(0xFF00C853)
val VixNormal   = Color(0xFF2196F3)
val VixHigh     = Color(0xFFFF6D00)
val VixExtreme  = Color(0xFFD50000)
