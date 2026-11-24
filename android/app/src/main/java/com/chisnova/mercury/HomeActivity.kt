package com.chisnova.mercury

import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.chisnova.mercury.asr.AsrController
import com.chisnova.mercury.data.LearningStats
import com.chisnova.mercury.database.DatabaseHelper

class HomeActivity : BaseActivity() {

    private var dbHelper: DatabaseHelper? = null

    /**
     * 화면에 세울 다섯 단계.
     * 주제별 콘텐츠는 한 화면(Level2Activity)을 단계만 달리해 재사용한다.
     */
    private data class LevelCard(
        val level: Int,
        val label: String,
        val title: String,
        val caption: String,
        val iconRes: Int,
        val tintRes: Int,
        val surfaceRes: Int,
        val open: () -> Unit,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)
        findViewById<View>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, LegalActivity::class.java))
        }

        dbHelper = DatabaseHelper(this)

        if (hasBottomNavigation()) {
            setupBottomNavigation()
        }
        Thread { initAsr() }.start()
    }

    override fun onResume() {
        super.onResume()
        // 연습하고 돌아오면 진도가 달라져 있다 — 돌아올 때마다 다시 그린다
        bindHome()
    }

    override fun onPurchaseAccessChanged() {
        if (dbHelper != null) bindHome()
    }

    private fun bindHome() {
        findViewById<TextView>(R.id.purchaseBanner)?.apply {
            text = if (purchases.state.value.hasFullAccess) "전체 콘텐츠 이용 중 · 구매 복원" else "전체 이용권 · 한 번 구매로 전체 이용"
            setOnClickListener { startActivity(Intent(this@HomeActivity, PurchaseActivity::class.java)) }
        }
        val stats = LearningStats(dbHelper)
        bindStreak(stats)
        bindWeek(stats)
        bindLevels(stats)
    }

    private fun bindStreak(stats: LearningStats) {
        val days = stats.streakDays()
        findViewById<TextView>(R.id.streakBadge)?.apply {
            if (days > 0) {
                text = "🔥 ${days}일"
                visibility = View.VISIBLE
            } else {
                visibility = View.GONE
            }
        }
    }

    private fun bindWeek(stats: LearningStats) {
        val container = findViewById<LinearLayout>(R.id.weekChart) ?: return
        container.removeAllViews()
        val week = stats.thisWeek()
        val peak = maxOf(1, week.maxOfOrNull { it.second } ?: 1)
        findViewById<TextView>(R.id.weekSummary)?.text = "문장 ${stats.weekTotal()}개"

        week.forEach { (name, count, isToday) ->
            val column = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.MATCH_PARENT, 1f)
            }
            val bar = View(this).apply {
                val height = if (count > 0) maxOf(dp(8), (count.toFloat() / peak * dp(52)).toInt()) else dp(3)
                layoutParams = LinearLayout.LayoutParams(dp(10), height)
                setBackgroundResource(
                    when {
                        count == 0 -> R.drawable.bg_week_bar_empty
                        isToday -> R.drawable.bg_week_bar_today
                        else -> R.drawable.bg_week_bar
                    }
                )
            }
            val label = TextView(this).apply {
                text = name
                textSize = 11f
                setPadding(0, dp(6), 0, 0)
                setTextColor(getColor(if (isToday) R.color.ds_accent_orange else R.color.text_tertiary))
                if (isToday) setTypeface(typeface, android.graphics.Typeface.BOLD)
            }
            column.addView(bar)
            column.addView(label)
            container.addView(column)
        }
    }

    private fun bindLevels(stats: LearningStats) {
        val container = findViewById<LinearLayout>(R.id.levelContainer) ?: return
        container.removeAllViews()

        val cards = listOf(
            LevelCard(1, "LEVEL 1", "빈출표현 정리", "자주 쓰는 표현을 구 단위로 연습",
                R.drawable.ic_bookmark_added_24, R.color.ds_primary, R.drawable.bg_level_icon_blue) {
                startActivity(Intent(this, Level1Activity::class.java))
            },
            LevelCard(2, "LEVEL 2", "주제별 콘텐츠 · 입문", "스포츠·음악·문화 등 친숙한 주제",
                R.drawable.ic_message_circle_24, R.color.ds_accent_green, R.drawable.bg_level_icon_green) {
                openArticles(2)
            },
            LevelCard(3, "LEVEL 3", "주제별 콘텐츠 · 중급", "과학·의학·경제 · 어휘 넓히기",
                R.drawable.ic_book_open_24, R.color.ds_accent_orange, R.drawable.bg_level_icon_orange) {
                openArticles(3)
            },
            LevelCard(4, "LEVEL 4", "주제별 콘텐츠 · 심화", "사회·정치·경제 이론 · 추상 표현",
                R.drawable.ic_book_24, R.color.ds_danger, R.drawable.bg_level_icon_red) {
                openArticles(4)
            },
            LevelCard(5, "LEVEL 5", "연설문", "연속 발화 · 억양과 전달력까지",
                R.drawable.ic_mic_24, R.color.ds_accent_purple, R.drawable.bg_level_icon_purple) {
                startActivity(Intent(this, Level3Activity::class.java))
            },
        )

        cards.forEach { card ->
            container.addView(createLevelCard(card, stats))
        }
    }

    private fun createLevelCard(card: LevelCard, stats: LearningStats): View {
        val view = layoutInflater.inflate(R.layout.item_home_level, null)
        val unlocked = (card.level == 1 || purchases.state.value.hasFullAccess)
        val ratio = stats.progress(card.level)

        view.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(10) }

        val tint = getColor(card.tintRes)
        view.findViewById<ImageView>(R.id.levelIcon).apply {
            setImageResource(card.iconRes)
            setBackgroundResource(if (unlocked) card.surfaceRes else R.drawable.bg_level_icon_muted)
            imageTintList = android.content.res.ColorStateList.valueOf(
                if (unlocked) tint else getColor(R.color.text_tertiary))
        }
        view.findViewById<TextView>(R.id.levelLabel).apply {
            text = if (card.level == 1 && !purchases.state.value.hasFullAccess) "${card.label} · 일부 무료" else card.label
            setTextColor(if (unlocked) tint else getColor(R.color.text_tertiary))
        }
        view.findViewById<TextView>(R.id.levelTitle).apply {
            text = card.title
            setTextColor(getColor(R.color.text_primary))
        }
        view.findViewById<TextView>(R.id.levelCaption).text =
            if (card.level == 1 && !purchases.state.value.hasFullAccess) "기초 4개 주제 무료 · 나머지는 전체 이용권" else card.caption

        view.findViewById<ProgressBar>(R.id.levelProgress).apply {
            progress = (ratio * 100).toInt()
            progressTintList = android.content.res.ColorStateList.valueOf(tint)
        }
        view.findViewById<TextView>(R.id.levelPercent).apply {
            visibility = if (unlocked) View.VISIBLE else View.GONE
            text = "${(ratio * 100).toInt()}%"
        }
        view.findViewById<ImageView>(R.id.levelLock).visibility =
            if (purchases.state.value.hasFullAccess) View.GONE else View.VISIBLE

        view.alpha = 1f
        view.setOnClickListener { card.open() }
        return view
    }

    private fun titleOf(level: Int) = when (level) {
        2 -> "주제별 콘텐츠 · 입문"
        3 -> "주제별 콘텐츠 · 중급"
        4 -> "주제별 콘텐츠 · 심화"
        5 -> "연설문"
        else -> "빈출표현 정리"
    }

    private fun openLevel(level: Int) = when (level) {
        1 -> startActivity(Intent(this, Level1Activity::class.java))
        5 -> startActivity(Intent(this, Level3Activity::class.java))
        else -> openArticles(level)
    }

    /** 주제별 콘텐츠는 화면 하나를 단계만 달리해 재사용한다 */
    private fun openArticles(tier: Int) {
        startActivity(Intent(this, Level2Activity::class.java).apply {
            putExtra(Level2Activity.EXTRA_TIER, tier)
        })
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun initAsr() {
        val asrController = AsrController.getInstance()
        asrController.initAsync(this)
    }

    override fun getCurrentScreen(): BottomNavScreen {
        return BottomNavScreen.HOME
    }
}
