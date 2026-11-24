package com.chisnova.mercury

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch

class PurchaseActivity : BaseActivity() {
    override fun getCurrentScreen() = BottomNavScreen.HOME
    private var continued = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_purchase)
        LegalActivity.bindPurchaseLinks(this)
        findViewById<android.widget.ImageView>(R.id.btnBack).setOnClickListener { finish() }
        setupBottomNavigation()
        findViewById<LinearLayout>(R.id.btnNavHome).setOnClickListener {
            startActivity(Intent(this, HomeActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP))
            finish()
        }
        val title = findViewById<TextView>(R.id.purchaseTitle)
        val buy = findViewById<Button>(R.id.purchaseBuy)
        val retry = findViewById<Button>(R.id.purchaseRetry)
        val restore = findViewById<Button>(R.id.purchaseRestore)
        val status = findViewById<TextView>(R.id.purchaseStatus)
        buy.setOnClickListener { purchases.purchase(this) }
        retry.setOnClickListener { purchases.refresh() }
        restore.setOnClickListener { purchases.restore() }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                purchases.state.collect { state ->
                    title.text = if (state.hasFullAccess) "전체 콘텐츠 이용 중" else "한 번 구매로\n전체 콘텐츠 이용"
                    buy.visibility = if (state.hasFullAccess) android.view.View.GONE else android.view.View.VISIBLE
                    buy.text = state.price?.let { "$it · 전체 이용권 구매" } ?: if (state.message == null) "상품 확인 중" else "상품 확인 필요"
                    buy.isEnabled = state.price != null && !state.busy
                    buy.alpha = if (buy.isEnabled) 1f else 0.45f
                    restore.isEnabled = !state.busy
                    retry.visibility = if (state.price == null && !state.hasFullAccess) android.view.View.VISIBLE else android.view.View.GONE
                    retry.isEnabled = !state.busy
                    status.visibility = if (state.busy || state.message != null) android.view.View.VISIBLE else android.view.View.GONE
                    status.text = if (state.busy) "결제 처리 중…" else state.message.orEmpty()
                    if (state.hasFullAccess && intent.hasExtra("continue_lesson") && !continued) {
                        continued = true
                        // Reconstruct a fixed internal destination; never launch a supplied Intent.
                        startActivity(Intent(this@PurchaseActivity, VoiceRecognitionActivity::class.java).apply {
                            putExtra("lesson_id", intent.getStringExtra("lesson_id"))
                            putExtra("lesson_title", intent.getStringExtra("lesson_title"))
                            putExtra("asset_base", intent.getStringExtra("asset_base"))
                            putExtra("lesson_sentences", intent.getStringArrayExtra("lesson_sentences"))
                        })
                        finish()
                    }
                }
            }
        }
    }

}
