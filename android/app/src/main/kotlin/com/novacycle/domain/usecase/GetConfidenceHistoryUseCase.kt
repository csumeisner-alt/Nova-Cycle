package com.novacycle.domain.usecase

import com.novacycle.data.remote.models.ConfidenceHistoryResponse
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.ConfidencePoint
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SmoothingMode
import javax.inject.Inject

/**
 * Fetches confidence history and optionally applies EMA smoothing based on user settings.
 * Also computes momentum (delta between consecutive points).
 */
class GetConfidenceHistoryUseCase @Inject constructor(
    private val repository: NovaCycleRepository
) {
    suspend operator fun invoke(
        ticker: String = "VOO",
        window: String = "7d",
        settings: SensitivitySettings
    ): Result<List<ConfidencePoint>> {
        return repository.getConfidenceHistory(ticker, window).map { rawResponses ->
            // Backend returns rows newest-first; sort chronologically so the
            // chart's time axis and momentum deltas read left-to-right in time.
            val responses = rawResponses.sortedBy { it.timestamp }
            val points = responses.mapIndexed { index, r ->
                val prev = if (index > 0) responses[index - 1] else null
                // The API stores confidence as a normalized fraction [0, 1].
                // The chart domain uses percentage points [0, 100], matching
                // its axis, tooltip, zones, and trend summaries.
                val longBuy = r.longBuyConfidence * 100f
                val longSell = r.longSellConfidence * 100f
                val shortBuy = r.shortBuyConfidence * 100f
                val shortSell = r.shortSellConfidence * 100f
                val previousLongBuy = prev?.longBuyConfidence?.times(100f)
                val previousShortBuy = prev?.shortBuyConfidence?.times(100f)
                ConfidencePoint(
                    timestamp = r.timestamp,
                    ticker = r.ticker,
                    longBuyConfidence = longBuy,
                    longSellConfidence = longSell,
                    shortBuyConfidence = shortBuy,
                    shortSellConfidence = shortSell,
                    isExtendedHours = r.isExtendedHours,
                    longMomentum = longBuy - (previousLongBuy ?: longBuy),
                    shortMomentum = shortBuy - (previousShortBuy ?: shortBuy)
                )
            }

            // Apply smoothing based on user's preference
            when (settings.smoothingMode) {
                SmoothingMode.RAW -> points
                SmoothingMode.LIGHT -> applyEma(points, alpha = 0.5f)
                SmoothingMode.EMA -> applyEma(points, alpha = 0.3f)
                SmoothingMode.HEAVY -> applyEma(points, alpha = 0.1f)
            }
        }
    }

    /**
     * Exponential Moving Average smoothing applied to confidence values.
     * alpha near 1.0 = more reactive (less smoothing)
     * alpha near 0.0 = more smoothed (slower to react)
     */
    private fun applyEma(points: List<ConfidencePoint>, alpha: Float): List<ConfidencePoint> {
        if (points.isEmpty()) return points
        var emaLongBuy = points.first().longBuyConfidence
        var emaLongSell = points.first().longSellConfidence
        var emaShortBuy = points.first().shortBuyConfidence
        var emaShortSell = points.first().shortSellConfidence

        return points.map { point ->
            emaLongBuy = alpha * point.longBuyConfidence + (1 - alpha) * emaLongBuy
            emaLongSell = alpha * point.longSellConfidence + (1 - alpha) * emaLongSell
            emaShortBuy = alpha * point.shortBuyConfidence + (1 - alpha) * emaShortBuy
            emaShortSell = alpha * point.shortSellConfidence + (1 - alpha) * emaShortSell
            point.copy(
                longBuyConfidence = emaLongBuy,
                longSellConfidence = emaLongSell,
                shortBuyConfidence = emaShortBuy,
                shortSellConfidence = emaShortSell
            )
        }
    }
}
