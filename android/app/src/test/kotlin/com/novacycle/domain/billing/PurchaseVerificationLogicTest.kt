package com.novacycle.domain.billing

import com.novacycle.domain.billing.PurchaseVerificationLogic.Decision
import com.novacycle.domain.billing.PurchaseVerificationLogic.ServerVerdict
import org.junit.Assert.assertEquals
import org.junit.Test

class PurchaseVerificationLogicTest {

    @Test
    fun `server-confirmed purchase unlocks`() {
        assertEquals(Decision.UNLOCK, PurchaseVerificationLogic.decide(ServerVerdict.Entitled))
    }

    @Test
    fun `fake token is locked even though Play client claimed ownership`() {
        assertEquals(
            Decision.LOCK,
            PurchaseVerificationLogic.decide(ServerVerdict.NotEntitled("invalid"))
        )
    }

    @Test
    fun `server-detected refund locks`() {
        assertEquals(
            Decision.LOCK,
            PurchaseVerificationLogic.decide(ServerVerdict.NotEntitled("revoked"))
        )
    }

    @Test
    fun `pending purchase does not unlock`() {
        assertEquals(
            Decision.LOCK,
            PurchaseVerificationLogic.decide(ServerVerdict.NotEntitled("pending"))
        )
    }

    @Test
    fun `offline or backend-down keeps a real purchase - provisional unlock`() {
        assertEquals(
            Decision.UNLOCK_PROVISIONALLY,
            PurchaseVerificationLogic.decide(ServerVerdict.Unreachable("connection refused"))
        )
    }

    @Test
    fun `502 and 503 map to unreachable, not rejection`() {
        listOf(500, 502, 503).forEach { code ->
            assertEquals(
                Decision.UNLOCK_PROVISIONALLY,
                PurchaseVerificationLogic.decide(
                    PurchaseVerificationLogic.verdictForHttpError(code)
                )
            )
        }
    }
}
