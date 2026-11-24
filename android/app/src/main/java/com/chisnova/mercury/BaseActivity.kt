package com.chisnova.mercury

import android.content.Intent
import android.os.Bundle
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.chisnova.mercury.billing.PurchaseManager
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.distinctUntilChanged
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity

abstract class BaseActivity : ComponentActivity() {
    protected val purchases by lazy { PurchaseManager.get(this) }
    private val accessBindings = mutableListOf<() -> Unit>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                purchases.state.map { it.hasFullAccess }.distinctUntilChanged().collect {
                    accessBindings.forEach { it() }
                    onPurchaseAccessChanged()
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        purchases.refresh()
    }

    protected open fun onPurchaseAccessChanged() {}

    protected fun bindPurchaseAccess(update: () -> Unit) {
        accessBindings.add(update)
        update()
    }

    protected fun bindContentIcon(icon: android.widget.ImageView, isFree: Boolean = false) {
        val originalBackground = icon.background
        val originalTint = icon.imageTintList
        bindPurchaseAccess {
            val unlocked = isFree || purchases.state.value.hasFullAccess
            if (unlocked) {
                icon.background = originalBackground
                icon.imageTintList = originalTint
            } else {
                icon.setBackgroundResource(R.drawable.bg_level_icon_muted)
                icon.imageTintList = getColorStateList(R.color.text_tertiary)
            }
        }
    }

    /** 사진과 컬러 이모지는 형태를 유지하면서 잠긴 동안만 흑백으로 표시한다. */
    protected fun bindContentArtwork(view: android.view.View, isFree: Boolean = false) {
        val grayscale = android.graphics.Paint().apply {
            colorFilter = android.graphics.ColorMatrixColorFilter(
                android.graphics.ColorMatrix().apply { setSaturation(0f) }
            )
        }
        bindPurchaseAccess {
            if (isFree || purchases.state.value.hasFullAccess) {
                view.setLayerType(android.view.View.LAYER_TYPE_NONE, null)
            } else {
                view.setLayerType(android.view.View.LAYER_TYPE_HARDWARE, grayscale)
            }
        }
    }

    protected fun addPurchaseLock(container: LinearLayout, isFree: Boolean = false) {
        if (isFree) return
        val lock = android.widget.ImageView(this).apply {
            setImageResource(R.drawable.ic_lock_24)
            contentDescription = "전체 이용권 필요"
            imageTintList = getColorStateList(R.color.text_secondary)
        }
        val size = (20 * resources.displayMetrics.density).toInt()
        container.addView(lock, LinearLayout.LayoutParams(size, size))
        val update: () -> Unit = {
            lock.visibility = if (purchases.state.value.hasFullAccess) android.view.View.GONE else android.view.View.VISIBLE
        }
        accessBindings.add(update)
        update()
    }

    
    // 각 Activity에서 현재 화면을 지정해야 함
    abstract fun getCurrentScreen(): BottomNavScreen
    
    enum class BottomNavScreen {
        HOME, HISTORY, ANALYSIS
    }
    
    /**
     * 하단 네비게이션 설정
     * 각 Activity의 onCreate에서 호출해야 함
     */
    protected fun setupBottomNavigation() {
        updateBottomNavigationState()
        
        // 홈 버튼
        findViewById<LinearLayout>(R.id.btnNavHome)?.setOnClickListener {
            navigateToScreen(BottomNavScreen.HOME)
        }
        
        // 히스토리 버튼
        findViewById<LinearLayout>(R.id.btnNavHistory)?.setOnClickListener {
            navigateToScreen(BottomNavScreen.HISTORY)
        }
        
        // 분석 버튼
        findViewById<LinearLayout>(R.id.btnNavAnalysis)?.setOnClickListener {
            navigateToScreen(BottomNavScreen.ANALYSIS)
        }
    }
    
    /**
     * 화면 전환 처리
     */
    private fun navigateToScreen(targetScreen: BottomNavScreen) {
        val currentScreen = getCurrentScreen()
        
        // 현재 화면과 같으면 아무 동작 안함
        if (currentScreen == targetScreen) {
            return
        }
        
        val intent = when (targetScreen) {
            BottomNavScreen.HOME -> Intent(this, HomeActivity::class.java)
            BottomNavScreen.HISTORY -> Intent(this, HistoryActivity::class.java)
            BottomNavScreen.ANALYSIS -> Intent(this, GraphActivity::class.java)
        }
        
        startActivity(intent)
        
        // 홈 화면이 아닌 경우 현재 Activity 종료 (백스택 관리)
        if (currentScreen != BottomNavScreen.HOME) {
            finish()
        }
    }
    
    /**
     * 하단 네비게이션 버튼 상태 업데이트
     */
    private fun updateBottomNavigationState() {
        val currentScreen = getCurrentScreen()
        
        // 모든 버튼을 비활성 상태로 초기화
        resetAllNavButtons()
        
        // 현재 화면에 해당하는 버튼만 활성화
        when (currentScreen) {
            BottomNavScreen.HOME -> activateNavButton(R.id.btnNavHome, "홈")
            BottomNavScreen.HISTORY -> activateNavButton(R.id.btnNavHistory, "히스토리")
            BottomNavScreen.ANALYSIS -> activateNavButton(R.id.btnNavAnalysis, "분석")
        }
    }
    
    /**
     * 모든 네비게이션 버튼을 비활성 상태로 설정
     */
    private fun resetAllNavButtons() {
        val navButtons = listOf(R.id.btnNavHome, R.id.btnNavHistory, R.id.btnNavAnalysis)
        
        navButtons.forEach { buttonId ->
            val button = findViewById<LinearLayout>(buttonId) ?: return@forEach
            val icon = button.getChildAt(0) as? android.widget.ImageView
            val text = button.getChildAt(1) as? TextView
            
            icon?.setColorFilter(android.graphics.Color.parseColor("#9CA3AF"))
            text?.setTextColor(android.graphics.Color.parseColor("#9CA3AF"))
            text?.typeface = android.graphics.Typeface.DEFAULT
        }
    }
    
    /**
     * 특정 네비게이션 버튼을 활성 상태로 설정
     */
    private fun activateNavButton(buttonId: Int, buttonText: String) {
        val button = findViewById<LinearLayout>(buttonId) ?: return
        val icon = button.getChildAt(0) as? android.widget.ImageView
        val text = button.getChildAt(1) as? TextView
        
        icon?.setColorFilter(android.graphics.Color.parseColor("#3B82F6"))
        text?.setTextColor(android.graphics.Color.parseColor("#3B82F6"))
        text?.typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    
    /**
     * 하단 네비게이션이 포함된 레이아웃인지 확인
     */
    protected fun hasBottomNavigation(): Boolean {
        return findViewById<LinearLayout>(R.id.btnNavHome) != null
    }
}
