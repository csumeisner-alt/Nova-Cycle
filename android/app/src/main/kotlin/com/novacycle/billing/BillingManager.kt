package com.novacycle.billing

import android.app.Activity
import android.content.Context
import android.util.Log
import com.android.billingclient.api.AcknowledgePurchaseParams
import com.android.billingclient.api.BillingClient
import com.android.billingclient.api.BillingClientStateListener
import com.android.billingclient.api.BillingFlowParams
import com.android.billingclient.api.BillingResult
import com.android.billingclient.api.ProductDetails
import com.android.billingclient.api.Purchase
import com.android.billingclient.api.PurchasesUpdatedListener
import com.android.billingclient.api.QueryProductDetailsParams
import com.android.billingclient.api.QueryPurchasesParams
import com.novacycle.data.theme.ThemePrefs
import com.novacycle.domain.billing.PurchaseVerificationLogic
import com.novacycle.domain.billing.PurchaseVerificationLogic.Decision
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

/** UI-facing state of the Mint Luxe purchase flow. */
sealed interface MintBillingState {
    /** Play Billing not connected yet (or store unavailable on this device). */
    data class Unavailable(val reason: String) : MintBillingState
    /** Product ready to buy; [formattedPrice] e.g. "$1.49". */
    data class Available(val formattedPrice: String) : MintBillingState
    /** Purchase owned & acknowledged — Mint Luxe is unlocked. */
    data object Purchased : MintBillingState
    /** A purchase flow is in progress. */
    data object Pending : MintBillingState
}

/**
 * Google Play Billing wrapper for the Mint Luxe theme
 * (managed in-app product `mint_luxe_theme`, one-time purchase, $1.49).
 *
 * - Connects lazily on first [ensureConnected] and restores purchases on
 *   every successful connection (`queryPurchasesAsync`), so a reinstall or
 *   new device re-unlocks Mint Luxe automatically.
 * - On a verified purchase, acknowledges it (required within 3 days or Play
 *   refunds automatically) and flips `mintUnlocked` in [ThemePrefs].
 * - Degrades gracefully: if Play is unavailable (e.g. sideloaded build,
 *   emulator without Play), state stays [MintBillingState.Unavailable] and
 *   the Settings UI disables the buy button with an explanation.
 *
 * NOTE: real purchases only work once the app is published on Play Console
 * with the `mint_luxe_theme` product configured — that setup is external.
 */
@Singleton
class BillingManager @Inject constructor(
    @ApplicationContext private val context: Context,
    private val themePrefs: ThemePrefs,
    private val entitlementVerifier: MintEntitlementVerifier
) : PurchasesUpdatedListener {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    companion object {
        const val PRODUCT_ID = "mint_luxe_theme"
        private const val TAG = "BillingManager"
    }

    private val _state = MutableStateFlow<MintBillingState>(
        MintBillingState.Unavailable("Connecting to Google Play…")
    )
    val state: StateFlow<MintBillingState> = _state.asStateFlow()

    private var productDetails: ProductDetails? = null
    private var connecting = false

    private val billingClient: BillingClient = BillingClient.newBuilder(context)
        .setListener(this)
        .enablePendingPurchases()
        .build()

    /** Idempotent: connect if not connected; then refresh product + purchases. */
    fun ensureConnected() {
        if (billingClient.isReady) return
        if (connecting) return
        connecting = true
        billingClient.startConnection(object : BillingClientStateListener {
            override fun onBillingSetupFinished(result: BillingResult) {
                connecting = false
                if (result.responseCode == BillingClient.BillingResponseCode.OK) {
                    queryProduct()
                    restorePurchases()
                } else {
                    _state.value = MintBillingState.Unavailable(
                        "Google Play Billing unavailable (${result.debugMessage.ifBlank { "code ${result.responseCode}" }})"
                    )
                }
            }

            override fun onBillingServiceDisconnected() {
                connecting = false
                _state.value = MintBillingState.Unavailable("Disconnected from Google Play")
            }
        })
    }

    private fun queryProduct() {
        val params = QueryProductDetailsParams.newBuilder()
            .setProductList(
                listOf(
                    QueryProductDetailsParams.Product.newBuilder()
                        .setProductId(PRODUCT_ID)
                        .setProductType(BillingClient.ProductType.INAPP)
                        .build()
                )
            )
            .build()
        billingClient.queryProductDetailsAsync(params) { result, detailsList ->
            if (result.responseCode == BillingClient.BillingResponseCode.OK && detailsList.isNotEmpty()) {
                productDetails = detailsList.first()
                if (_state.value !is MintBillingState.Purchased) {
                    val price = detailsList.first().oneTimePurchaseOfferDetails?.formattedPrice ?: "$1.49"
                    _state.value = MintBillingState.Available(price)
                }
            } else {
                _state.value = MintBillingState.Unavailable(
                    "Mint Luxe not available in the store yet (${result.debugMessage.ifBlank { "code ${result.responseCode}" }})"
                )
            }
        }
    }

    /** Re-check owned purchases (reinstall / new device / app start). */
    fun restorePurchases() {
        if (!billingClient.isReady) {
            ensureConnected()
            return
        }
        val params = QueryPurchasesParams.newBuilder()
            .setProductType(BillingClient.ProductType.INAPP)
            .build()
        billingClient.queryPurchasesAsync(params) { result, purchases ->
            if (result.responseCode != BillingClient.BillingResponseCode.OK) return@queryPurchasesAsync
            val mint = purchases.firstOrNull { PRODUCT_ID in it.products }
            if (mint != null && mint.purchaseState == Purchase.PurchaseState.PURCHASED) {
                // Re-verify the owned token with the backend — this is where a
                // refund issued since purchase is detected and Mint is relocked.
                handlePlayPurchase(mint, isRestore = true)
            } else if (mint == null) {
                // Product no longer owned (refund / revocation) — relock Mint Luxe.
                // Only act on an authoritative OK response with no owned purchase.
                if (themePrefs.state.value.mintUnlocked) {
                    Log.i(TAG, "Mint Luxe no longer owned — revoking unlock")
                    themePrefs.setMintUnlocked(false)
                }
                if (_state.value is MintBillingState.Purchased) {
                    val price = productDetails?.oneTimePurchaseOfferDetails?.formattedPrice
                    _state.value = if (price != null) {
                        MintBillingState.Available(price)
                    } else {
                        // Re-query product details so the buy button recovers.
                        MintBillingState.Unavailable("Checking store availability…").also { queryProduct() }
                    }
                }
            }
        }
    }

    /** Launch the Play purchase sheet. Returns false if billing isn't ready. */
    fun launchPurchase(activity: Activity): Boolean {
        val details = productDetails ?: run {
            ensureConnected()
            return false
        }
        val flowParams = BillingFlowParams.newBuilder()
            .setProductDetailsParamsList(
                listOf(
                    BillingFlowParams.ProductDetailsParams.newBuilder()
                        .setProductDetails(details)
                        .build()
                )
            )
            .build()
        val result = billingClient.launchBillingFlow(activity, flowParams)
        val launched = result.responseCode == BillingClient.BillingResponseCode.OK
        if (launched) _state.value = MintBillingState.Pending
        return launched
    }

    override fun onPurchasesUpdated(result: BillingResult, purchases: MutableList<Purchase>?) {
        when (result.responseCode) {
            BillingClient.BillingResponseCode.OK -> {
                purchases?.filter { PRODUCT_ID in it.products }?.forEach { purchase ->
                    when (purchase.purchaseState) {
                        Purchase.PurchaseState.PURCHASED -> handlePlayPurchase(purchase, isRestore = false)
                        Purchase.PurchaseState.PENDING ->
                            _state.value = MintBillingState.Pending
                        else -> Unit
                    }
                }
            }
            BillingClient.BillingResponseCode.ITEM_ALREADY_OWNED -> restorePurchases()
            BillingClient.BillingResponseCode.USER_CANCELED ->
                revertToAvailable()
            else -> {
                Log.w(TAG, "Purchase failed: ${result.responseCode} ${result.debugMessage}")
                revertToAvailable()
            }
        }
    }

    private fun revertToAvailable() {
        if (_state.value is MintBillingState.Pending) {
            val price = productDetails?.oneTimePurchaseOfferDetails?.formattedPrice ?: "$1.49"
            _state.value = MintBillingState.Available(price)
        }
    }

    /**
     * A purchase Google Play reports as PURCHASED on this device.
     *
     * The local Play result alone no longer unlocks Mint Luxe — the purchase
     * token is first verified server-side against the Play Developer API
     * (a rooted device can spoof local state, but not a valid token):
     *  - server says entitled       → unlock
     *  - server says fake/refunded  → lock (authoritative)
     *  - server unreachable         → unlock provisionally so a real buyer
     *    never loses the purchase offline; re-verified on next app start.
     */
    private fun handlePlayPurchase(purchase: Purchase, isRestore: Boolean) {
        acknowledgeIfNeeded(purchase)
        scope.launch {
            val verdict = if (isRestore) {
                entitlementVerifier.checkEntitlement(PRODUCT_ID, purchase.purchaseToken)
            } else {
                entitlementVerifier.verifyPurchase(PRODUCT_ID, purchase.purchaseToken)
            }
            when (PurchaseVerificationLogic.decide(verdict)) {
                Decision.UNLOCK -> {
                    themePrefs.setMintUnlocked(true)
                    _state.value = MintBillingState.Purchased
                }
                Decision.UNLOCK_PROVISIONALLY -> {
                    Log.w(TAG, "Server verification unavailable — honouring Play purchase provisionally")
                    themePrefs.setMintUnlocked(true)
                    _state.value = MintBillingState.Purchased
                }
                Decision.LOCK -> {
                    Log.w(TAG, "Server rejected purchase token — locking Mint Luxe")
                    if (themePrefs.state.value.mintUnlocked) themePrefs.setMintUnlocked(false)
                    revertToAvailable()
                    val price = productDetails?.oneTimePurchaseOfferDetails?.formattedPrice
                    if (price != null && _state.value is MintBillingState.Purchased) {
                        _state.value = MintBillingState.Available(price)
                    }
                }
            }
        }
    }

    private fun acknowledgeIfNeeded(purchase: Purchase) {
        if (!purchase.isAcknowledged) {
            val ackParams = AcknowledgePurchaseParams.newBuilder()
                .setPurchaseToken(purchase.purchaseToken)
                .build()
            billingClient.acknowledgePurchase(ackParams) { ackResult ->
                if (ackResult.responseCode != BillingClient.BillingResponseCode.OK) {
                    // Will retry via restorePurchases() on next app start —
                    // Play keeps the purchase in queryPurchasesAsync until acked.
                    Log.w(TAG, "Acknowledge failed: ${ackResult.debugMessage}")
                }
            }
        }
    }
}
