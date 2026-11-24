package com.chisnova.mercury.billing

import android.app.Activity
import android.content.Context
import android.util.Base64
import com.android.billingclient.api.*
import com.chisnova.mercury.BuildConfig
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

data class PurchaseState(
    val hasFullAccess: Boolean = false,
    val price: String? = null,
    val busy: Boolean = false,
    val message: String? = null,
)

/** One non-consumable purchase. Cached receipts are signature-checked on every app start. */
class PurchaseManager private constructor(context: Context) : PurchasesUpdatedListener {
    companion object {
        const val PRODUCT_ID = BuildConfig.FULL_UNLOCK_PRODUCT_ID
        @Volatile private var instance: PurchaseManager? = null
        fun get(context: Context): PurchaseManager = instance ?: synchronized(this) {
            instance ?: PurchaseManager(context.applicationContext).also { instance = it }
        }
    }

    private val packageName = context.packageName
    private val preferences = context.getSharedPreferences("verified_purchases", Context.MODE_PRIVATE)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val mutableState = MutableStateFlow(PurchaseState())
    val state = mutableState.asStateFlow()
    private val publicKey = runCatching {
        Base64.decode(BuildConfig.PLAY_BILLING_PUBLIC_KEY, Base64.DEFAULT).also {
            java.security.KeyFactory.getInstance("RSA").generatePublic(java.security.spec.X509EncodedKeySpec(it))
        }
    }.getOrNull()
    private var restoring = false
    private var connecting = false
    private var revision = 0
    private var purchaseInFlight = false
    private val acknowledging = mutableSetOf<String>()
    private val client = BillingClient.newBuilder(context)
        .setListener(this)
        .enablePendingPurchases(PendingPurchasesParams.newBuilder().enableOneTimeProducts().build())
        .enableAutoServiceReconnection()
        .build()

    init {
        val cached = runCatching {
            Purchase(preferences.getString("json", "")!!, preferences.getString("signature", "")!!)
        }.getOrNull()
        mutableState.value = state.value.copy(hasFullAccess = cached != null && verified(cached))
    }

    fun canAccess(lessonId: String?, assetBase: String?) =
        state.value.hasFullAccess || ContentAccess.isFree(lessonId, assetBase)

    fun refresh() {
        if (client.isReady) {
            queryPurchases()
            queryProduct()
        } else if (!connecting) {
            connecting = true
            client.startConnection(object : BillingClientStateListener {
                override fun onBillingSetupFinished(result: BillingResult) {
                    scope.launch {
                        connecting = false
                        if (result.responseCode == BillingClient.BillingResponseCode.OK) {
                            queryPurchases()
                            queryProduct()
                        } else fail("Google Play에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.")
                    }
                }
                override fun onBillingServiceDisconnected() {
                    scope.launch { connecting = false }
                }
            })
        }
    }

    private fun queryProduct() {
        fetchProduct { product ->
            mutableState.value = state.value.copy(price = product?.oneTimePurchaseOfferDetails?.formattedPrice)
        }
    }

    private fun fetchProduct(completion: (ProductDetails?) -> Unit) {
        val params = QueryProductDetailsParams.newBuilder().setProductList(listOf(
            QueryProductDetailsParams.Product.newBuilder().setProductId(PRODUCT_ID)
                .setProductType(BillingClient.ProductType.INAPP).build()
        )).build()
        client.queryProductDetailsAsync(params) { result, response ->
            scope.launch {
                val product = if (result.responseCode == BillingClient.BillingResponseCode.OK)
                    response.productDetailsList.firstOrNull { it.productId == PRODUCT_ID } else null
                if (product?.oneTimePurchaseOfferDetails == null) {
                    fail("구매 상품을 불러올 수 없습니다. 잠시 후 다시 시도해 주세요.")
                    completion(null)
                } else completion(product)
            }
        }
    }

    fun purchase(activity: Activity) {
        if (purchaseInFlight || state.value.hasFullAccess) return
        // Never accept payment when this build cannot verify its receipt.
        if (publicKey == null) {
            fail("현재 구매를 준비 중입니다. 잠시 후 다시 시도해 주세요.")
            return
        }
        if (!client.isReady) { refresh(); return }
        purchaseInFlight = true
        mutableState.value = state.value.copy(busy = true, message = null)
        // Refresh ProductDetails before launching; do not reuse a stale offer.
        fetchProduct { product ->
            if (product == null || activity.isFinishing || activity.isDestroyed) {
                finishFlow()
                return@fetchProduct
            }
            val details = BillingFlowParams.ProductDetailsParams.newBuilder().setProductDetails(product)
            product.oneTimePurchaseOfferDetails?.offerToken?.let { details.setOfferToken(it) }
            val result = client.launchBillingFlow(activity, BillingFlowParams.newBuilder()
                .setProductDetailsParamsList(listOf(details.build())).build())
            if (result.responseCode != BillingClient.BillingResponseCode.OK) handleFailure(result)
        }
    }

    fun restore() {
        if (purchaseInFlight) return
        mutableState.value = state.value.copy(message = "구매 내역을 확인하고 있습니다.")
        restoring = true
        if (client.isReady) queryPurchases() else refresh()
    }

    private fun queryPurchases() {
        val queryRevision = revision
        client.queryPurchasesAsync(QueryPurchasesParams.newBuilder()
            .setProductType(BillingClient.ProductType.INAPP).build()) { result, purchases ->
            scope.launch {
                if (queryRevision != revision) { queryPurchases(); return@launch }
                if (result.responseCode != BillingClient.BillingResponseCode.OK) {
                    // Temporary network errors must not erase a previously verified offline purchase.
                    fail("구매 내역을 확인하지 못했습니다. 인터넷 연결을 확인해 주세요.")
                    return@launch
                }
                val showRestoreResult = restoring
                restoring = false
                val owned = purchases.firstOrNull { verified(it) }
                if (owned != null) grant(owned) else {
                    preferences.edit().clear().apply()
                    mutableState.value = state.value.copy(hasFullAccess = false)
                }
                when {
                    purchases.any { PRODUCT_ID in it.products && it.purchaseState == Purchase.PurchaseState.PENDING } ->
                        fail("결제 승인 대기 중입니다. 승인이 완료되면 전체 콘텐츠가 열립니다.")
                    owned == null && purchases.any { PRODUCT_ID in it.products } ->
                        fail("구매 검증에 실패했습니다. 구매 복원을 다시 시도해 주세요.")
                    showRestoreResult -> mutableState.value = state.value.copy(message =
                        if (owned != null) "구매가 복원되었습니다." else "이 Google 계정에서 복원할 구매 내역이 없습니다.")
                }
            }
        }
    }

    override fun onPurchasesUpdated(result: BillingResult, purchases: MutableList<Purchase>?) {
        scope.launch {
            revision++
            finishFlow()
            if (result.responseCode != BillingClient.BillingResponseCode.OK) {
                handleFailure(result)
                return@launch
            }
            purchases.orEmpty().filter { PRODUCT_ID in it.products }.forEach { purchase ->
                when {
                    purchase.purchaseState == Purchase.PurchaseState.PENDING ->
                        fail("결제 승인 대기 중입니다. 승인이 완료되면 전체 콘텐츠가 열립니다.")
                    verified(purchase) -> grant(purchase)
                    else -> fail("구매 검증에 실패했습니다. 구매 복원을 시도해 주세요.")
                }
            }
            queryPurchases()
        }
    }

    private fun verified(purchase: Purchase): Boolean {
        if (purchase.packageName != packageName || PRODUCT_ID !in purchase.products ||
            purchase.purchaseState != Purchase.PurchaseState.PURCHASED || purchase.purchaseToken.isBlank()) return false
        return runCatching {
            PurchaseSignature.verify(publicKey ?: return false,
                purchase.originalJson, Base64.decode(purchase.signature, Base64.DEFAULT))
        }.getOrDefault(false)
    }

    private fun grant(purchase: Purchase) {
        preferences.edit().putString("json", purchase.originalJson)
            .putString("signature", purchase.signature).apply()
        mutableState.value = state.value.copy(hasFullAccess = true, message = null)
        // Do not consume: this is a permanent unlock. Failed acknowledgements retry on refresh.
        if (!purchase.isAcknowledged && acknowledging.add(purchase.purchaseToken)) {
            client.acknowledgePurchase(AcknowledgePurchaseParams.newBuilder()
                .setPurchaseToken(purchase.purchaseToken).build()) { result ->
                scope.launch {
                    acknowledging.remove(purchase.purchaseToken)
                    if (result.responseCode != BillingClient.BillingResponseCode.OK)
                        fail("구매 확인을 마무리하지 못했습니다. 인터넷 연결 후 구매 복원을 눌러 주세요.")
                }
            }
        }
    }

    private fun finishFlow() {
        purchaseInFlight = false
        mutableState.value = state.value.copy(busy = false)
    }

    private fun handleFailure(result: BillingResult) {
        finishFlow()
        when (result.responseCode) {
            BillingClient.BillingResponseCode.USER_CANCELED -> Unit
            BillingClient.BillingResponseCode.ITEM_ALREADY_OWNED -> { restoring = true; queryPurchases() }
            else -> fail("구매를 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.")
        }
    }

    private fun fail(message: String) { mutableState.value = state.value.copy(message = message) }
}
