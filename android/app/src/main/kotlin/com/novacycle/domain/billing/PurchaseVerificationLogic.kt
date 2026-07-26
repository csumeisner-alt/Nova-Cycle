package com.novacycle.domain.billing

/**
 * Pure decision logic for server-side Mint Luxe purchase verification.
 *
 * Security model:
 *  - Google Play (client library) says a purchase is owned → we still ask our
 *    backend to verify the purchase token against the Play Developer API
 *    before flipping `mintUnlocked`. A rooted device can edit prefs or spoof
 *    the local billing result, but cannot forge a valid purchase token.
 *  - The server is authoritative when it answers: entitled → unlock,
 *    not entitled (invalid/revoked) → lock.
 *  - When the server is unreachable (offline, backend down, verification not
 *    configured), a Play-verified purchase is honoured provisionally so a
 *    legitimate buyer never loses their purchase; the token is re-verified
 *    on the next app start.
 */
object PurchaseVerificationLogic {

    /** Outcome of a backend verification attempt. */
    sealed interface ServerVerdict {
        /** Backend confirmed the token with Google Play. */
        data object Entitled : ServerVerdict

        /** Backend says the token is fake, refunded, or cancelled. */
        data class NotEntitled(val state: String) : ServerVerdict

        /** Backend unreachable / Play API down / verification unconfigured (HTTP 502/503, IO error). */
        data class Unreachable(val reason: String) : ServerVerdict
    }

    /** What the app should do with the local `mintUnlocked` flag. */
    enum class Decision { UNLOCK, LOCK, UNLOCK_PROVISIONALLY }

    /**
     * Decide the local unlock state after the backend verification attempt for
     * a purchase that Google Play reports as PURCHASED on this device.
     */
    fun decide(verdict: ServerVerdict): Decision = when (verdict) {
        is ServerVerdict.Entitled -> Decision.UNLOCK
        // Authoritative rejection: fake token or refund → lock, even though
        // the local Play client claimed the purchase was owned.
        is ServerVerdict.NotEntitled -> Decision.LOCK
        // Fail-safe for real buyers: keep the Play-verified unlock, retry later.
        is ServerVerdict.Unreachable -> Decision.UNLOCK_PROVISIONALLY
    }

    /**
     * Map an HTTP status code from the verification endpoint to a verdict.
     * 2xx responses are mapped from the body by the caller; this handles errors.
     * 502/503 (and any 5xx) mean "verification infrastructure down" → Unreachable.
     * Any other error code (unexpected 4xx) is treated as Unreachable too:
     * it signals a client/server contract problem, not proof of a fake purchase.
     */
    fun verdictForHttpError(code: Int): ServerVerdict =
        ServerVerdict.Unreachable("HTTP $code from verification endpoint")
}
