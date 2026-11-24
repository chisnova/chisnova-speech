package com.chisnova.mercury

import android.content.Intent
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.chisnova.mercury.data.ArticleLevels
import com.chisnova.mercury.data.LessonData

class Level2Activity : BaseActivity() {
    private val lessonProgress by lazy {
        LessonProgressBinder(this, when (tier) {
            3 -> R.color.ds_accent_orange
            4 -> R.color.ds_danger
            else -> R.color.ds_accent_green
        })
    }

    override fun onResume() {
        super.onResume()
        lessonProgress.refresh()
    }

    /**
     * 이 화면이 보여 줄 난이도 단계(2·3·4).
     * 주제별 콘텐츠는 어휘 밀도로 세 단계로 나뉘어 있고, 홈에서 어느 단계로 들어왔는지 받는다.
     */
    private val tier: Int by lazy { intent.getIntExtra(EXTRA_TIER, 2) }

    companion object {
        const val EXTRA_TIER = "article_tier"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_level2)

        setupUI()
        setupHeader()
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
    
    /** 이 단계의 콘텐츠 폴더 이름. 폴더 번호가 곧 레벨 번호다. */
    private fun levelFolder(): String = when (tier) {
        3 -> "3.intermediate"
        4 -> "4.advanced"
        else -> "2.beginner"
    }

    /** 제목과 부제를 단계에 맞게 바꾼다. */
    private fun setupHeader() {
        val (title, subtitle) = when (tier) {
            3 -> "주제별 콘텐츠 · 중급" to "Level 3 · 과학·경제로 넓히기"
            4 -> "주제별 콘텐츠 · 심화" to "Level 4 · 추상 개념 다루기"
            else -> "주제별 콘텐츠 · 입문" to "Level 2 · 친숙한 주제로 시작하기"
        }
        findViewById<TextView>(R.id.levelTitle)?.text = title
        findViewById<TextView>(R.id.levelSubtitle)?.text = subtitle
    }

    private fun setupLessons() {
        val mainContainer = findViewById<LinearLayout>(R.id.lessonsContainer)

        // 이 단계에 배정된 주제만 남긴다. 빈 분류는 아예 그리지 않는다.
        LessonData.getLevel2Categories().forEach { category ->
            val lessons = category.lessons.filter { ArticleLevels.tierOf(it.id) == tier }
            if (lessons.isEmpty()) return@forEach
            val categoryView = createCategoryView(category.copy(lessons = lessons))
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
        
        val iconView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                resources.getDimensionPixelSize(R.dimen.icon_size_medium),
                resources.getDimensionPixelSize(R.dimen.icon_size_medium)
            )
            setBackgroundResource(category.bgColorResId)
            setImageResource(category.iconResId)
            imageTintList = getColorStateList(android.R.color.white)
            setPadding(
                resources.getDimensionPixelSize(R.dimen.padding_tiny),
                resources.getDimensionPixelSize(R.dimen.padding_tiny),
                resources.getDimensionPixelSize(R.dimen.padding_tiny),
                resources.getDimensionPixelSize(R.dimen.padding_tiny)
            )
        }
        
        val titleText = TextView(this).apply {
            text = category.title
            textSize = 16f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(getColor(android.R.color.black))
            layoutParams = LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            ).apply {
                marginStart = resources.getDimensionPixelSize(R.dimen.margin_small)
            }
        }
        
        val lessonCountText = TextView(this).apply {
            text = "${category.lessons.size}개 레슨"
            textSize = 12f
            setTextColor(getColor(R.color.text_secondary))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                marginStart = resources.getDimensionPixelSize(R.dimen.margin_small)
            }
        }
        
        bindContentIcon(iconView)
        headerLayout.addView(iconView)
        headerLayout.addView(titleText)
        headerLayout.addView(lessonCountText)
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
                topMargin = resources.getDimensionPixelSize(R.dimen.margin_small)
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

        bindContentIcon(iconView)
        lessonLayout.addView(iconView)
        lessonLayout.addView(textContainer)
        addPurchaseLock(lessonLayout)
        
        return lessonLayout
    }
    
    private fun assetBase(lesson: com.chisnova.mercury.data.Lesson, category: com.chisnova.mercury.data.LessonCategory): String {
        val baseFolder = when (category.id) {
            // 난이도 단계별로 폴더가 나뉘어 있다 — 이 화면이 맡은 단계의 폴더를 쓴다
            "culture", "economics", "entertain", "health",
            "politic", "science", "social" -> "contents/${levelFolder()}/${category.id}"
            else -> "contents/${levelFolder()}"
        }
        val subFolder = when (category.id) {
            "culture" -> when (lesson.id) {
                "architecture" -> "architect/architect"
                "art" -> "art/art"
                "city" -> "city/city"
                "festival" -> "festival/festival"
                "instrument" -> "instrument/instrument"
                "local_food" -> "local_food/local_food"
                "museum" -> "museum/museum"
                "performance" -> "performance/performance"
                "unesco" -> "unesco/unesco"
                else -> ""
            }
            "economics" -> when (lesson.id) {
                "behavior" -> "behavior/behavior"
                "bitcoin" -> "bitcoin/bitcoin"
                "currency" -> "currency/currency"
                "digital_economics" -> "digital_economics/digital_economics"
                "exchange_rate" -> "exchange_rate/exchange_rate"
                "macro" -> "macro/macro"
                "micro" -> "micro/micro"
                "stock" -> "stock/stock"
                "supply_demand" -> "supply_demand/supply_demand"
                "trend" -> "trend/trend"
                else -> ""
            }
            "entertain" -> when (lesson.id) {
                "e-sports" -> "e-sports/e-sports"
                "k-pop" -> "k-pop/k-pop"
                "k-webtoon" -> "k-webtoon/k-webtoon"
                "movie_casting" -> "movie_casting/movie_casting"
                "movie_music" -> "movie_music/movie_music"
                "music_genre" -> "music_genre/music_genre"
                "musical"->"musical/musical"
                "ott" -> "ott/ott"
                "soccer" -> "soccer/soccer"
                "swimming" -> "swimming/swimming"
                else -> ""
            }
            "health" -> when (lesson.id) {
                "addiction" -> "addiction/addiction"
                "disease" -> "disease/disease"
                "hormon" -> "hormon/hormon"
                "mental" -> "mental/mental"
                "muscle" -> "muscle/muscle"
                "obesity" -> "obesity/obesity"
                "sleep" -> "sleep/sleep"
                "stress" -> "stress/stress"
                "vitamin" -> "vitamin/vitamin"
                "work_out" -> "work_out/work_out"
                else -> ""
            }
            "politic" -> when (lesson.id) {
                "democracy" -> "democracy/democracy"
                "dictator"->"dictator/dictator"
                "diplomacy"->"diplomacy/diplomacy"
                "election"->"election/election"
                "government"->"government/government"
                "law"->"law/law"
                "parliamentary"->"parliamentary/parliamentary"
                "party"->"party/party"
                "security"->"security/security"
                "social"->"social/social"
                else -> ""
            }
            "science" -> when (lesson.id) {
                "battery" -> "battery/battery"
                "computer" -> "computer/computer"
                "earth" -> "earth/earth"
                "material" -> "material/material"
                "neuro_science" -> "neuro_science/neuro_science"
                "physics" -> "physics/physics"
                "robot" -> "robot/robot"
                "uam" -> "uam/uam"
                "universe" -> "universe/universe"
                else -> ""
            }
            "social" -> when (lesson.id) {
                "aging_society" -> "aging_society/aging_society"
                "crime" -> "crime/crime"
                "elitism" -> "elitism/elitism"
                "gender_equality" -> "gender/gender"
                "generation_gap" -> "generation_gap/generation_gap"
                "house" -> "house/house"
                "lower_birth" -> "lower_birth/lower_birth"
                "poverty" -> "poverty/poverty"
                "racism" -> "racism/racism"
                "welfare" -> "welfare/welfare"
                else -> ""
            }
            else -> ""
        }

        return if (subFolder.isNotEmpty()) "$baseFolder/$subFolder" else ""
    }

    private fun navigateToVoiceRecognition(lesson: com.chisnova.mercury.data.Lesson, category: com.chisnova.mercury.data.LessonCategory) {
        val base = assetBase(lesson, category)
        val intent = Intent(this, VoiceRecognitionActivity::class.java).apply {
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
