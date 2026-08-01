package com.novacycle.domain

import com.novacycle.domain.model.SensitivitySettings
import com.novacycle.domain.model.SignalData
import com.novacycle.domain.usecase.ApplyFilteredSignalsUseCase
import org.junit.Assert.assertEquals
import org.junit.Test

class ApplyFilteredSignalsUseCaseTest {

    private val useCase = ApplyFilteredSignalsUseCase()
    private val settings = SensitivitySettings(
        buyThreshold = 70,
        sellThreshold = -70,
        apiBaseUrl = "https://test.invalid/api/"
    )

    @Test
    fun `percentage settings filter normalized buy and sell confidence`() {
        val result = useCase(
            rawSignals = listOf(
                signal("buy-low", "2026-08-01T10:00:00Z", "buy", 0.69f),
                signal("sell-low", "2026-08-01T11:00:00Z", "sell", 0.69f),
                signal("buy-high", "2026-08-01T12:00:00Z", "buy", 0.71f),
                signal("sell-high", "2026-08-01T13:00:00Z", "sell", 0.71f),
            ),
            settings = settings
        )

        assertEquals(listOf("buy-high", "sell-high"), result.signals.map { it.id })
    }

    private fun signal(
        id: String,
        timestamp: String,
        type: String,
        confidence: Float,
    ) = SignalData(
        id = id,
        timestamp = timestamp,
        signalType = type,
        gaugeType = "long",
        confidence = confidence,
    )
}