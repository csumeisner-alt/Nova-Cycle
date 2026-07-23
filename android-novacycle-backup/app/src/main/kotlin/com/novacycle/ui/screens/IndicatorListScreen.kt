package com.novacycle.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.novacycle.data.remote.models.IndicatorResponse
import com.novacycle.ui.theme.*
import com.novacycle.viewmodel.IndicatorViewModel

@Composable
fun IndicatorListScreen(viewModel: IndicatorViewModel = hiltViewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Row(Modifier.fillMaxWidth().padding(12.dp), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("Indicators", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            TextButton(onClick = { viewModel.loadIndicators() }) { Text("Refresh") }
        }

        if (uiState.isLoading) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        if (uiState.error != null) Text("⚠️ ${uiState.error}", color = NovaSellRed, modifier = Modifier.padding(12.dp))

        val ind = uiState.indicators
        if (ind != null) {
            RsiCard(ind); StochCard(ind); StochRsiCard(ind); MacdCard(ind)
            SmaCard(ind); BollingerCard(ind); AdxCard(ind); OscillatorCard(ind); VixCard(ind)
        } else if (!uiState.isLoading) {
            Box(Modifier.fillMaxWidth().padding(32.dp), Alignment.Center) {
                Text("No indicator data. Tap Refresh.", color = NovaNeutralGray)
            }
        }
        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable private fun RsiCard(ind: IndicatorResponse) {
    IndicatorCard("RSI — Relative Strength Index") {
        val color = when { ind.rsi > 70 -> NovaSellRed; ind.rsi < 30 -> NovaBuyGreen; else -> NovaNeutralGray }
        val note  = when { ind.rsi > 70 -> "Overbought"; ind.rsi < 30 -> "Oversold"; else -> "Neutral" }
        Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("%.2f".format(ind.rsi), fontWeight = FontWeight.Bold, color = color, style = MaterialTheme.typography.titleLarge)
            Text(note, style = MaterialTheme.typography.bodyMedium, color = color)
        }
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(progress = { (ind.rsi/100f).coerceIn(0f,1f) },
            modifier = Modifier.fillMaxWidth().height(8.dp), color = color,
            trackColor = MaterialTheme.colorScheme.surfaceVariant)
        Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
            Text("0", style = MaterialTheme.typography.labelSmall)
            Text("30", style = MaterialTheme.typography.labelSmall, color = NovaBuyGreen)
            Text("70", style = MaterialTheme.typography.labelSmall, color = NovaSellRed)
            Text("100", style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable private fun StochCard(ind: IndicatorResponse) {
    IndicatorCard("Stochastic Oscillator") {
        IndicatorRow("K", "%.2f".format(ind.stochK))
        IndicatorRow("D", "%.2f".format(ind.stochD))
        val note = when { ind.stochK > 80 -> "Overbought"; ind.stochK < 20 -> "Oversold"; ind.stochK > ind.stochD -> "Bullish crossover"; else -> "Bearish / neutral" }
        Text(note, style = MaterialTheme.typography.bodyMedium, color = NovaNeutralGray)
    }
}

@Composable private fun StochRsiCard(ind: IndicatorResponse) {
    IndicatorCard("Stochastic RSI") {
        IndicatorRow("K", "%.4f".format(ind.stochRsiK))
        IndicatorRow("D", "%.4f".format(ind.stochRsiD))
    }
}

@Composable private fun MacdCard(ind: IndicatorResponse) {
    IndicatorCard("MACD") {
        IndicatorRow("MACD Line",   "%.4f".format(ind.macdLine))
        IndicatorRow("Signal Line", "%.4f".format(ind.macdSignal))
        val histColor = if (ind.macdHistogram >= 0) NovaBuyGreen else NovaSellRed
        Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("Histogram", style = MaterialTheme.typography.bodyMedium)
            Text("%+.4f".format(ind.macdHistogram), fontWeight = FontWeight.Bold, color = histColor)
        }
        Text(if (ind.macdLine > ind.macdSignal) "Bullish — MACD above signal" else "Bearish — MACD below signal",
            style = MaterialTheme.typography.bodyMedium,
            color = if (ind.macdLine > ind.macdSignal) NovaBuyGreen else NovaSellRed)
    }
}

@Composable private fun SmaCard(ind: IndicatorResponse) {
    IndicatorCard("Simple Moving Averages") {
        listOf("SMA 20" to ind.sma20, "SMA 50" to ind.sma50, "SMA 200" to ind.sma200)
            .forEach { (n, v) -> IndicatorRow(n, "$%.2f".format(v)) }
    }
}

@Composable private fun BollingerCard(ind: IndicatorResponse) {
    IndicatorCard("Bollinger Bands") {
        IndicatorRow("Upper", "$%.2f".format(ind.bollingerUpper))
        IndicatorRow("Lower", "$%.2f".format(ind.bollingerLower))
        val percBColor = when { ind.bollingerPercB > 1f -> NovaSellRed; ind.bollingerPercB < 0f -> NovaBuyGreen; else -> NovaNeutralGray }
        Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
            Text("%B", style = MaterialTheme.typography.bodyMedium)
            Text("%.3f".format(ind.bollingerPercB), fontWeight = FontWeight.Bold, color = percBColor)
        }
        val note = when { ind.bollingerPercB > 1f -> "Above upper band — overbought"; ind.bollingerPercB < 0f -> "Below lower band — oversold"; ind.bollingerPercB > 0.8f -> "Near upper band"; ind.bollingerPercB < 0.2f -> "Near lower band"; else -> "Within normal range" }
        Text(note, style = MaterialTheme.typography.bodyMedium, color = percBColor)
    }
}

@Composable private fun AdxCard(ind: IndicatorResponse) {
    IndicatorCard("ADX — Average Directional Index") {
        val strength = when { ind.adx > 50 -> "Very strong"; ind.adx > 25 -> "Strong trend"; ind.adx > 20 -> "Moderate"; else -> "Weak / no trend" }
        val color = if (ind.adx > 25) NovaBuyGreen else NovaNeutralGray
        Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
            Text("%.2f".format(ind.adx), fontWeight = FontWeight.Bold, color = color, style = MaterialTheme.typography.titleLarge)
            Surface(color = color.copy(alpha = 0.2f), shape = MaterialTheme.shapes.small) {
                Text(strength, style = MaterialTheme.typography.labelSmall, color = color, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
            }
        }
    }
}

@Composable private fun OscillatorCard(ind: IndicatorResponse) {
    IndicatorCard("CCI & Williams %R") {
        val cciColor = when { ind.cci > 100 -> NovaSellRed; ind.cci < -100 -> NovaBuyGreen; else -> NovaNeutralGray }
        val wrColor  = when { ind.williamsR > -20 -> NovaSellRed; ind.williamsR < -80 -> NovaBuyGreen; else -> NovaNeutralGray }
        IndicatorRow("CCI",         "%.2f".format(ind.cci), cciColor)
        IndicatorRow("Williams %R", "%.2f".format(ind.williamsR), wrColor)
        IndicatorRow("ATR",         "%.4f".format(ind.atr))
    }
}

@Composable private fun VixCard(ind: IndicatorResponse) {
    val (color, desc) = when (ind.vixRegime.lowercase()) {
        "low"     -> VixLow     to "Low volatility — favorable for trend-following"
        "high"    -> VixHigh    to "High volatility — wider stop-loss zones"
        "extreme" -> VixExtreme to "Extreme volatility — signals may be less reliable"
        else      -> VixNormal  to "Normal volatility regime"
    }
    IndicatorCard("VIX Regime") {
        Surface(color = color.copy(alpha = 0.2f), shape = MaterialTheme.shapes.medium) {
            Text(ind.vixRegime.uppercase(), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold,
                color = color, modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
        }
        Spacer(Modifier.height(6.dp))
        Text(desc, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun IndicatorCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun IndicatorRow(label: String, value: String, color: Color = MaterialTheme.colorScheme.onSurface) {
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium, color = color)
    }
}
