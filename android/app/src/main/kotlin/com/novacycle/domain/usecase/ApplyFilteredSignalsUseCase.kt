package com.novacycle.domain.usecase

import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import java.util.UUID
import javax.inject.Inject

/**
 * Client-side strongest-confidence filtering rule.
 *
 * Algorithm:
 *  1. Sort all signals by timestamp ascending.
 *  2. Group consecutive signals of the same type (buy/sell).
 *  3. Within each group, keep only the one with the highest confidence.
 *  4. Enforce strict BUY → SELL → BUY alternation (discard duplicates).
 *  5. Assign a UUID cycle_id to each BUY→SELL pair.
 *  6. Apply user's sensitivity threshold: hide signals below threshold.
 *  7. Apply extended-hours filter if disabled in settings.
 *
 * This mirrors the backend logic so the user can see the effect of their
 * sensitivity settings without requiring a backend change.
 */
class ApplyFilteredSignalsUseCase @Inject constructor() {

    data class TradeCycle(
        val cycleId: String,
        val buySignal: SignalData,
        val sellSignal: SignalData?   // null if cycle is still open
    )

    data class FilterResult(
        val signals: List<SignalData>,
        val cycles: List<TradeCycle>
    )

    operator fun invoke(
        rawSignals: List<SignalData>,
        settings: SensitivitySettings
    ): FilterResult {
        if (rawSignals.isEmpty()) return FilterResult(emptyList(), emptyList())

        // Step 1: sort chronologically
        val sorted = rawSignals.sortedBy { it.timestamp }

        // Step 2 & 3: group consecutive same-type signals, keep highest confidence per group
        val deduped = mutableListOf<SignalData>()
        var i = 0
        while (i < sorted.size) {
            val current = sorted[i]
            var best = current
            // Advance while same signal type
            while (i < sorted.size && sorted[i].signalType == current.signalType) {
                if (sorted[i].confidence > best.confidence) {
                    best = sorted[i]
                }
                i++
            }
            deduped.add(best)
        }

        // Step 4: enforce strict alternation BUY → SELL → BUY
        val alternated = mutableListOf<SignalData>()
        var expectingBuy = true  // Start by expecting a BUY signal
        for (signal in deduped) {
            val isBuy = signal.signalType.lowercase() == "buy"
            if (expectingBuy && isBuy) {
                alternated.add(signal)
                expectingBuy = false
            } else if (!expectingBuy && !isBuy) {
                alternated.add(signal)
                expectingBuy = true
            }
            // Otherwise skip: wrong order (e.g., two consecutive BUYs after dedup edge case)
        }

        // Step 5: assign cycle IDs to BUY→SELL pairs
        val withCycles = alternated.toMutableList()
        val cycles = mutableListOf<TradeCycle>()
        var j = 0
        while (j < withCycles.size) {
            val sig = withCycles[j]
            if (sig.signalType.lowercase() == "buy") {
                val cycleId = UUID.randomUUID().toString()
                val buyWithId = sig.copy(cycleId = cycleId)
                withCycles[j] = buyWithId

                val nextSell = if (j + 1 < withCycles.size) {
                    val sell = withCycles[j + 1].copy(cycleId = cycleId)
                    withCycles[j + 1] = sell
                    sell
                } else null

                cycles.add(TradeCycle(cycleId = cycleId, buySignal = buyWithId, sellSignal = nextSell))
            }
            j++
        }

        // Step 6: apply sensitivity threshold
        // Backend confidence is normalized to 0–1; settings are whole
        // percentages in the 50–80 range.
        val buyMin = settings.buyThreshold / 100f
        val sellMin = kotlin.math.abs(settings.sellThreshold) / 100f

        val thresholded = withCycles.filter { signal ->
            when (signal.signalType.lowercase()) {
                "buy" -> signal.confidence >= buyMin
                "sell" -> signal.confidence >= sellMin
                else -> true
            }
        }

        // Step 7: extended-hours filter
        val finalSignals = if (settings.extendedHoursEnabled) {
            thresholded
        } else {
            thresholded.filter { !it.isExtendedHours }
        }

        // Re-filter cycles to only include those with both signals surviving
        val survivingIds = finalSignals.mapNotNull { it.cycleId }.toSet()
        val finalCycles = cycles.filter { it.cycleId in survivingIds }

        return FilterResult(signals = finalSignals, cycles = finalCycles)
    }
}
