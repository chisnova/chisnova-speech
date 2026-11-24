package com.chisnova.mercury

import android.content.Intent
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.chisnova.mercury.data.LessonData

class Level1Activity : BaseActivity() {
    private val lessonProgress by lazy { LessonProgressBinder(this, R.color.ds_primary) }

    override fun onResume() {
        super.onResume()
        lessonProgress.refresh()
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_level1)
        
        setupUI()
        setupLessons()
        
        // 하단 네비게이션 설정 (있는 경우에만)
        if (hasBottomNavigation()) {
            setupBottomNavigation()
        }
    }
    
    private fun setupUI() {
        // 뒤로가기 버튼
        findViewById<ImageView>(R.id.btnBack)?.setOnClickListener {
            finish()
        }
    }
    
    private fun setupLessons() {
        val categories = LessonData.getLevel1Categories()
        val mainContainer = findViewById<LinearLayout>(R.id.lessonsContainer)
        
        categories.forEach { category ->
            val categoryView = createCategoryView(category)
            mainContainer?.addView(categoryView)
        }
    }
    
    private fun createCategoryView(category: com.chisnova.mercury.data.LessonCategory): LinearLayout {
        val categoryLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = resources.getDimensionPixelSize(R.dimen.margin_medium)
                setMargins(
                    resources.getDimensionPixelSize(R.dimen.margin_large),
                    topMargin,
                    resources.getDimensionPixelSize(R.dimen.margin_large),
                    0
                )
            }
        }
        
        // 카테고리 헤더
        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }
        
        val emojiText = TextView(this).apply {
            text = category.emoji
            textSize = 18f
        }
        bindContentArtwork(emojiText, category.lessons.any {
            com.chisnova.mercury.billing.ContentAccess.isFree(it.id, assetBase(it, category))
        })
        
        val titleText = TextView(this).apply {
            text = category.title
            textSize = 16f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(getColor(android.R.color.black))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginStart = resources.getDimensionPixelSize(R.dimen.margin_small)
            }
        }
        
        headerLayout.addView(emojiText)
        headerLayout.addView(titleText)
        headerLayout.addView(android.view.View(this), LinearLayout.LayoutParams(0, 1, 1f))
        val accessLabel = TextView(this).apply {
            textSize = 11f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(getColor(if (category.id == "basic") R.color.ds_primary else R.color.text_secondary))
            setBackgroundResource(if (category.id == "basic") R.drawable.bg_pill_blue else R.drawable.bg_rounded_button)
            val inset = (8 * resources.displayMetrics.density).toInt()
            setPadding(inset, inset / 2, inset, inset / 2)
        }
        headerLayout.addView(accessLabel)
        bindPurchaseAccess {
            accessLabel.text = if (category.id == "basic") "4개 주제 무료" else "전체 이용권 필요"
            accessLabel.visibility = if (category.id == "basic" || !purchases.state.value.hasFullAccess)
                android.view.View.VISIBLE else android.view.View.GONE
        }
        categoryLayout.addView(headerLayout)
        
        // 레슨 목록 컨테이너
        val lessonsContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = resources.getDimensionPixelSize(R.dimen.margin_small)
            }
        }
        
        // 각 레슨 아이템 생성
        category.lessons.forEach { lesson ->
            val lessonView = createLessonView(lesson, category)
            lessonsContainer.addView(lessonView)
        }
        
        categoryLayout.addView(lessonsContainer)
        return categoryLayout
    }
    
    private fun createLessonView(lesson: com.chisnova.mercury.data.Lesson, category: com.chisnova.mercury.data.LessonCategory): LinearLayout {
        val lessonLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setBackgroundResource(R.drawable.bg_card_list)
            setPadding(
                resources.getDimensionPixelSize(R.dimen.padding_medium),
                resources.getDimensionPixelSize(R.dimen.padding_medium),
                resources.getDimensionPixelSize(R.dimen.padding_medium),
                resources.getDimensionPixelSize(R.dimen.padding_medium)
            )
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = resources.getDimensionPixelSize(R.dimen.margin_small)
            }
            
            // 클릭 리스너 추가
            setOnClickListener {
                navigateToVoiceRecognition(lesson, category)
            }
            
            // 클릭 효과 추가
            isClickable = true
            isFocusable = true
            foreground = getDrawable(android.R.drawable.list_selector_background)
        }
        
        // 아이콘
        val iconView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                resources.getDimensionPixelSize(R.dimen.icon_size_large),
                resources.getDimensionPixelSize(R.dimen.icon_size_large)
            )
            setBackgroundResource(category.bgColorResId)
            setImageResource(lesson.iconResId)
            imageTintList = getColorStateList(android.R.color.white)
            setPadding(
                resources.getDimensionPixelSize(R.dimen.padding_small),
                resources.getDimensionPixelSize(R.dimen.padding_small),
                resources.getDimensionPixelSize(R.dimen.padding_small),
                resources.getDimensionPixelSize(R.dimen.padding_small)
            )
        }
        
        // 텍스트 컨테이너
        val textContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            ).apply {
                marginStart = resources.getDimensionPixelSize(R.dimen.margin_medium)
                marginEnd = resources.getDimensionPixelSize(R.dimen.margin_small)
            }
        }
        
        val titleText = TextView(this).apply {
            text = lesson.title
            textSize = 15f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(getColor(R.color.text_primary))
        }
        
        textContainer.addView(titleText)
        // duration 제거: 시간 표시 없이 간결하게 유지
        
        lessonProgress.addTo(textContainer, assetBase(lesson, category))

        lessonLayout.addView(iconView)
        lessonLayout.addView(textContainer)
        val isFree = com.chisnova.mercury.billing.ContentAccess.isFree(lesson.id, assetBase(lesson, category))
        bindContentIcon(iconView, isFree)
        addPurchaseLock(lessonLayout, isFree)
        
        return lessonLayout
    }
    
    private fun assetBase(lesson: com.chisnova.mercury.data.Lesson, category: com.chisnova.mercury.data.LessonCategory): String {
        val baseFolder = when (category.id) {
            "basic" -> "contents/1.basic"
            "real_life" -> "contents/1.basic"
            "communication" -> "contents/1.basic"
            "important" -> "contents/1.basic"
            else -> "contents/1.basic"
        }
        val subFolder = when (lesson.id) {
            "greetings" -> "greeting_basic_expression/greeting_basic_expression"
            "daily_conversation" -> "phone_conversation/phone_conversation"
            "emotions" -> "expressing_opinions/expressing_opinions"
            "time_date" -> "telling_time/telling_time"
            "food_ordering" -> "ordering_food/ordering_food"
            "directions" -> "asking_for_directions/asking_for_directions"
            "asking_help" -> "asking_for_help/asking_for_help"
            "shopping" -> "shopping/shopping"
            "apologies" -> "declining_politely/declining_politely"
            "making_plans" -> "making_appointment/making_appointment"
            "opinions_feelings" -> "expressing_opinions/expressing_opinions"
            "polite_disagreement" -> "declining_politely/declining_politely"
            "small_talk" -> "introducing_people/introducing_people"
            "suggestions_advice" -> "making_suggestion/making_suggestion"
            "emergency" -> "emergency_situation/emergency_situation"
            else -> ""
        }

        return if (subFolder.isNotEmpty()) "$baseFolder/$subFolder" else ""
    }

    private fun navigateToVoiceRecognition(lesson: com.chisnova.mercury.data.Lesson, category: com.chisnova.mercury.data.LessonCategory) {
        val base = assetBase(lesson, category)
        val canStudy = purchases.canAccess(lesson.id, base)
        val destination = if (canStudy) VoiceRecognitionActivity::class.java else PurchaseActivity::class.java
        val intent = Intent(this, destination).apply {
            if (!canStudy) putExtra("continue_lesson", true)
            putExtra("lesson_id", lesson.id)
            putExtra("lesson_title", lesson.title)
            if (base.isNotEmpty()) {
                putExtra("asset_base", base)
            } else {
                putExtra("lesson_sentences", lesson.sentences.toTypedArray())
            }
        }
        startActivity(intent)
    }
    
    override fun getCurrentScreen(): BottomNavScreen {
        // Level Activity는 홈에서 접근하므로 홈으로 분류
        return BottomNavScreen.HOME
    }
}
