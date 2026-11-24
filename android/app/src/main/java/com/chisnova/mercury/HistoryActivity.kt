package com.chisnova.mercury

import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.graphics.Color
import com.chisnova.mercury.database.DatabaseHelper

class HistoryActivity : BaseActivity() {
    
    private var selectedFilter = "all" // "all", "recent", "level1", "level2", "level3"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)
        
        setupUI()
        updateStatistics()
        
        // Initial load
        updateHistoryList("all")
        
        // 하단 네비게이션 설정
        if (hasBottomNavigation()) {
            setupBottomNavigation()
        }
    }
    
    private fun setupUI() {
        // 뒤로가기 버튼
        findViewById<ImageView>(R.id.btnBack)?.setOnClickListener {
            finish()
        }
        
        // 필터 버튼들 설정
        setupFilterButtons()
    }
    
    private fun setupFilterButtons() {
        val filterButtons = listOf(
            findViewById<Button>(R.id.btnFilterAll) to "all",
            findViewById<Button>(R.id.btnFilterLevel1) to "level1",
            findViewById<Button>(R.id.btnFilterLevel2) to "level2",
            findViewById<Button>(R.id.btnFilterLevel3) to "level3"
        )
        
        filterButtons.forEach { (button, filter) ->
            button?.setOnClickListener {
                selectFilter(filter)
                updateFilterButtonsState()
                updateHistoryList(filter)
            }
        }
    }
    
    private fun selectFilter(filter: String) {
        selectedFilter = filter
    }
    
    private fun updateFilterButtonsState() {
        val filterButtons = mapOf(
            "all" to findViewById<Button>(R.id.btnFilterAll),
            "level1" to findViewById<Button>(R.id.btnFilterLevel1),
            "level2" to findViewById<Button>(R.id.btnFilterLevel2),
            "level3" to findViewById<Button>(R.id.btnFilterLevel3)
        )
        
        filterButtons.forEach { (filter, button) ->
            if (filter == selectedFilter) {
                button?.setBackgroundResource(R.drawable.bg_filter_button_selected)
                button?.setTextColor(getColor(android.R.color.white))
            } else {
                button?.setBackgroundResource(R.drawable.bg_filter_button)
                button?.setTextColor(getColor(android.R.color.darker_gray))
            }
        }
    }
    
    private fun updateHistoryList(filter: String) {
        val container = findViewById<LinearLayout>(R.id.historyListContainer) ?: return
        container.removeAllViews()
        
        val items = DatabaseHelper.getInstance(this).getHistoryUiItems(filter)
        val inflater = android.view.LayoutInflater.from(this)
        
        for (item in items) {
            val view = inflater.inflate(R.layout.item_history, container, false)
            
            view.findViewById<TextView>(R.id.tvDate)?.text = item.date
            view.findViewById<TextView>(R.id.tvTitle)?.text = item.lessonTitle
            view.findViewById<TextView>(R.id.tvSentence)?.text = item.sentence
            
            val scoreText = view.findViewById<TextView>(R.id.tvScore)
            scoreText?.text = "${item.score}점"
            
            val score = item.score
            when {
                score >= 85 -> {
                    scoreText?.setTextColor(Color.parseColor("#3B82F6"))
                    scoreText?.setBackgroundResource(R.drawable.bg_score_badge_blue)
                }
                score >= 75 -> {
                    scoreText?.setTextColor(Color.parseColor("#F97316"))
                    scoreText?.setBackgroundResource(R.drawable.bg_score_badge_orange)
                }
                else -> {
                    scoreText?.setTextColor(Color.parseColor("#EF4444"))
                    scoreText?.setBackgroundResource(R.drawable.bg_score_badge_red)
                }
            }
            
            view.setOnClickListener {
                android.widget.Toast.makeText(this, "히스토리 상세 보기 (ID: ${item.aid})", android.widget.Toast.LENGTH_SHORT).show()
            }
            
            container.addView(view)
        }
    }
    
    private fun updateStatistics() {
        val dbHelper = DatabaseHelper.getInstance(this)
        
        // 평균 점수 계산
        val averageScore = dbHelper.getAverageAttemptScore()
        val averageScoreText = findViewById<TextView>(R.id.averageScore)
        averageScoreText?.text = "${averageScore}점"

        // 점수에 따른 색상 변경
        val colorRes = when {
            averageScore >= 85 -> android.R.color.holo_blue_bright
            averageScore >= 75 -> android.R.color.holo_orange_dark
            else -> android.R.color.holo_red_dark
        }
        averageScoreText?.setTextColor(getColor(colorRes))
        
        // 총 도전 횟수
        val totalAttempts = dbHelper.getTotalAttemptsCount()
        findViewById<TextView>(R.id.totalAttempts)?.text = "${totalAttempts}회"
        
        // 총 학습 단어 수
        val totalWords = dbHelper.getTotalLearnedWordsCount()
        findViewById<TextView>(R.id.totalWords)?.text = "${totalWords}개"
        
        // 개선 필요 개수
        val improvementNeeded = dbHelper.getImprovementNeededCount()
        findViewById<TextView>(R.id.improvementNeeded)?.text = "${improvementNeeded}개"
    }
    
    override fun getCurrentScreen(): BottomNavScreen {
        return BottomNavScreen.HISTORY
    }
}
