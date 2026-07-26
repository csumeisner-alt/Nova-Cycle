package com.novacycle.billing

import android.util.Log
import com.novacycle.data.remote.NovaCycleApiService
import com.novacycle.data.remote.models.VerifyPurchaseRequest
import com.novacycle.domain.billing.PurchaseVerificationLogic.ServerVerdict
import com.novacycle.domain.billing.PurchaseVerificationLogic.verdictForHttpError
import retrofit2.HttpException
import java.io.IOException
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Verifies Play purchase tokens with the NovaCycle backend, which in turn
 * checks them against the Google Play Developer API. Interface exists so
 * [BillingManager] logic can be exercised with a fake in tests.
 */
interface MintEntitlementVerifier {
    /** Verify a fresh purchase token (records the entitlement server-side). */
    suspend fun verifyPurchase(productId: String, purchaseToken: String): ServerVerdict

    /** Re-check an owned token on app start — where refunds are detected. */
    suspend fun checkEntitlement(productId: String, purchaseToken: String): ServerVerdict
}

@Singleton
class BackendMintEntitlementVerifier @Inject constructor(
    private val api: NovaCycleApiService
) : MintEntitlementVerifier {

    private companion object { const val TAG = "MintVerifier" }

    override suspend fun verifyPurchase(productId: String, purchaseToken: String): ServerVerdict =
        call { api.verifyPurchase(VerifyPurchaseRequest(productId, purchaseToken)) }

    override suspend fun checkEntitlement(productId: String, purchaseToken: String): ServerVerdict =
        call { api.checkEntitlement(productId, purchaseToken) }

    private suspend fun call(
        block: suspend () -> com.novacycle.data.remote.models.EntitlementResponse
    ): ServerVerdict = try {
        val resp = block()
        if (resp.entitled) ServerVerdict.Entitled else ServerVerdict.NotEntitled(resp.state)
    } catch (e: HttpException) {
        Log.w(TAG, "Verification endpoint returned HTTP ${e.code()}")
        verdictForHttpError(e.code())
    } catch (e: IOException) {
        Log.w(TAG, "Verification endpoint unreachable: ${e.message}")
        ServerVerdict.Unreachable(e.message ?: "network error")
    }
}
