package com.novacycle.domain.usecase

import com.novacycle.data.remote.models.SignalResponse
import com.novacycle.data.repository.NovaCycleRepository
import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import javax.inject.Inject

/**
 * Fetches raw signal history and applies client-side sensitivity filtering.
 * Signals below the user's configured threshold are hidden.
 */
class GetSignalsUseCase @Inject constructor(
    private val repository: NovaCycleRepository
) {
    suspend operator fun invoke(
        ticker: String = "VOO",
        window: String = "30d",
        settings: SensitivitySettings
    ): Result<List<SignalData>> {
        return repository.getSignalHistory(ticker, window).map { signals ->
            signals
                .map { it.toDomain() }
                .filter { signal ->
                    // Apply sensitivity threshold filtering
                    when {
                        signal.isBuy && signal.confidence < settings.buyThreshold -> false
                        signal.isSell && signal.confidence < kotlin.math.abs(settings.sellThreshold) -> false
                        // Optionally hide extended-hours signals
                        signal.isExtendedHours && !settings.extendedHoursEnabled -> false
                        else -> true
                    }
                }
        }
    }

    private fun SignalResponse.toDomain() = SignalData(
        id = id,
        timestamp = timestamp,
        ticker = ticker,
        cycleId = cycleId,
        signalType = signalType,
        gaugeType = gaugeType,
        confidence = confidence,
        sessionType = sessionType,
        isExtendedHours = isExtendedHours,
        gapType = gapType,
        liquidityScore = liquidityScore,
        macroOverrideApplied = macroOverrideApplied,
        convictionTier = convictionTier,
        convictionReasons = convictionReasons
    )
}
