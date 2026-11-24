package com.chisnova.mercury

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.os.Bundle
import android.util.AttributeSet
import android.view.View
import android.view.MotionEvent
import android.graphics.Rect
import android.graphics.RectF
import android.app.AlertDialog
import android.widget.Button
import android.widget.LinearLayout
import android.graphics.Typeface
import android.view.Gravity
import android.util.TypedValue
import com.google.android.material.bottomsheet.BottomSheetDialog
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import com.chisnova.mercury.database.DatabaseHelper
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

class GraphActivity : BaseActivity() {
    
    private var selectedPeriod = "month" // "week", "month", "all"
    private lateinit var radarChartView: RadarChartView
    private lateinit var dbHelper: DatabaseHelper
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_graph)
        
        dbHelper = DatabaseHelper.getInstance(this)

        setupUI()
        setupRadarChart()
        loadData()
        
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
        
        // 기간 선택 버튼들
        setupPeriodButtons()
    }
    
    private fun setupPeriodButtons() {
        val btnWeek = findViewById<Button>(R.id.btnWeek)
        val btnMonth = findViewById<Button>(R.id.btnMonth)
        val btnAll = findViewById<Button>(R.id.btnAll)
        
        btnWeek?.setOnClickListener {
            selectPeriod("week")
            updatePeriodButtons()
            loadData()
        }
        
        btnMonth?.setOnClickListener {
            selectPeriod("month")
            updatePeriodButtons()
            loadData()
        }
        
        btnAll?.setOnClickListener {
            selectPeriod("all")
            updatePeriodButtons()
            loadData()
        }
        
        // 초기 상태 설정
        updatePeriodButtons()
    }
    
    private fun selectPeriod(period: String) {
        selectedPeriod = period
    }
    
    private fun updatePeriodButtons() {
        val btnWeek = findViewById<Button>(R.id.btnWeek)
        val btnMonth = findViewById<Button>(R.id.btnMonth)
        val btnAll = findViewById<Button>(R.id.btnAll)
        
        // 모든 버튼 기본 상태로 리셋
        btnWeek?.setBackgroundResource(R.drawable.bg_period_button)
        btnWeek?.setTextColor(getColor(android.R.color.black))
        btnMonth?.setBackgroundResource(R.drawable.bg_period_button)
        btnMonth?.setTextColor(getColor(android.R.color.black))
        btnAll?.setBackgroundResource(R.drawable.bg_period_button)
        btnAll?.setTextColor(getColor(android.R.color.black))
        
        // 선택된 버튼 하이라이트
        when (selectedPeriod) {
            "week" -> {
                btnWeek?.setBackgroundResource(R.drawable.bg_period_button_selected)
                btnWeek?.setTextColor(getColor(android.R.color.white))
            }
            "month" -> {
                btnMonth?.setBackgroundResource(R.drawable.bg_period_button_selected)
                btnMonth?.setTextColor(getColor(android.R.color.white))
            }
            "all" -> {
                btnAll?.setBackgroundResource(R.drawable.bg_period_button_selected)
                btnAll?.setTextColor(getColor(android.R.color.white))
            }
        }
    }
    
    private fun setupRadarChart() {
        val chartContainer = findViewById<FrameLayout>(R.id.radarChartContainer)
        radarChartView = RadarChartView(this)
        chartContainer?.addView(radarChartView)
    }

    private fun getStartDate(period: String): String? {
        if (period == "all") return null
        val cal = Calendar.getInstance()
        when (period) {
            "week" -> cal.add(Calendar.DAY_OF_YEAR, -7)
            "month" -> cal.add(Calendar.MONTH, -1)
        }
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(cal.time)
    }
    
    private fun loadData() {
        val startDate = getStartDate(selectedPeriod)
        Thread {
            val scores = dbHelper.queryCategoryScores(startDate)
            val intScores = scores.mapValues { it.value.toInt() }
            val weakPhones = dbHelper.getWeakestPhones(startDate)
            
            runOnUiThread {
                updateScoreDisplay(intScores)
                updateWeakPhonesList(weakPhones)
            }
        }.start()
    }
    
    private fun updateScoreDisplay(scores: Map<String, Int>) {
        val totalScoreText = findViewById<TextView>(R.id.totalScore)
        val textStrong = findViewById<TextView>(R.id.textStrongType)
        val textWeak = findViewById<TextView>(R.id.textWeakType)
        
        // 전체 평균 점수 계산 (데이터가 없으면 0)
        val avgScore = if (scores.isNotEmpty()) scores.values.average().toInt() else 0
        
        totalScoreText?.text = "${avgScore}점"
        
        // 점수에 따른 색상 변경
        val colorRes = when {
            avgScore >= 90 -> android.R.color.holo_green_dark
            avgScore >= 80 -> android.R.color.holo_blue_dark
            avgScore >= 70 -> android.R.color.holo_orange_dark
            else -> android.R.color.holo_red_dark
        }
        totalScoreText?.setTextColor(getColor(colorRes))
        
        // 강점/약점 분석
        if (scores.isNotEmpty()) {
            val maxEntry = scores.maxByOrNull { it.value }
            val minEntry = scores.minByOrNull { it.value }
            
            textStrong?.text = maxEntry?.let { "${it.key}" } ?: "-"
            textWeak?.text = minEntry?.let { "${it.key}" } ?: "-"
        } else {
            textStrong?.text = "-"
            textWeak?.text = "-"
        }
        
        // 레이더 차트 업데이트
        radarChartView.updateData(scores)
        
        // 토스트로 기간 변경 알림
        val periodText = when (selectedPeriod) {
            "week" -> "최근 1주"
            "month" -> "최근 1달"
            "all" -> "전체 기간"
            else -> "최근 1달"
        }
        
        android.widget.Toast.makeText(
            this, 
            "$periodText 데이터로 업데이트됨 (평균: ${avgScore}점)", 
            android.widget.Toast.LENGTH_SHORT
        ).show()
    }
    
    private fun updateWeakPhonesList(weakPhones: List<DatabaseHelper.WeakPhoneItem>) {
        val container = findViewById<LinearLayout>(R.id.weakPronunciationListContainer)
        container?.removeAllViews()
        
        if (weakPhones.isEmpty()) {
            val emptyView = TextView(this).apply {
                text = "데이터가 부족합니다."
                setPadding(0, 32, 0, 32)
                gravity = Gravity.CENTER
                setTextColor(getColor(android.R.color.darker_gray))
            }
            container?.addView(emptyView)
            return
        }

        weakPhones.forEach { item ->
            val itemView = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    bottomMargin = dpToPx(12)
                }
                gravity = Gravity.CENTER_VERTICAL
                setBackgroundResource(R.drawable.bg_stat_card)
                setPadding(dpToPx(12), dpToPx(12), dpToPx(12), dpToPx(12))
                
                setOnClickListener {
                    showDescriptionDialog(item.category, item.phone)
                }
            }

            // Score Badge
            val scoreView = TextView(this).apply {
                text = "${item.score}점"
                textSize = 18f
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
                
                val colorRes = when {
                    item.score >= 80 -> android.R.color.holo_blue_dark
                    item.score >= 60 -> android.R.color.holo_orange_dark
                    else -> android.R.color.holo_red_dark
                }
                setTextColor(getColor(colorRes))
                
                layoutParams = LinearLayout.LayoutParams(
                    dpToPx(60), 
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    rightMargin = dpToPx(12)
                }
            }
            itemView.addView(scoreView)

            // Phone & Category
            val infoLayout = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    0,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    1f
                )
            }
            
            val phoneText = TextView(this).apply {
                text = "/${item.phone}/"
                textSize = 16f
                typeface = Typeface.DEFAULT_BOLD
                setTextColor(android.graphics.Color.parseColor("#1F2937"))
            }
            infoLayout.addView(phoneText)
            
            val categoryText = TextView(this).apply {
                text = item.category
                textSize = 12f
                setTextColor(android.graphics.Color.parseColor("#6B7280"))
            }
            infoLayout.addView(categoryText)
            
            itemView.addView(infoLayout)
            
            // Arrow
            val arrowView = ImageView(this).apply {
                setImageResource(R.drawable.ic_chevron_right_24)
                setColorFilter(android.graphics.Color.parseColor("#9CA3AF"))
            }
            itemView.addView(arrowView)

            container?.addView(itemView)
        }
    }

    private fun showDescriptionDialog(category: String, specificPhone: String? = null) {
        val (description, examples) = categoryDescriptions[category] ?: Pair("설명이 없습니다.", "-")
        
        val dialog = BottomSheetDialog(this)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(24), dpToPx(24), dpToPx(24), dpToPx(48))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        
        // Title
        val titleView = TextView(this).apply {
            text = if (specificPhone != null) "$category (발음: /$specificPhone/)" else category
            textSize = 24f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(0xFF111827.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(8)
            }
        }
        container.addView(titleView)
        
        val divider = View(this).apply {
            setBackgroundColor(0xFFE5E7EB.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dpToPx(1)
            ).apply {
                bottomMargin = dpToPx(16)
            }
        }
        container.addView(divider)

        val descLabel = TextView(this).apply {
            text = "설명"
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(0xFF6B7280.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(4)
            }
        }
        container.addView(descLabel)
        
        val descView = TextView(this).apply {
            text = description
            textSize = 16f
            setTextColor(0xFF374151.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(20)
            }
        }
        container.addView(descView)
        
        val exampleLabel = TextView(this).apply {
            text = "해당 발음 (ARPABET)"
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(0xFF6B7280.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(4)
            }
        }
        container.addView(exampleLabel)
        
        val exampleView = TextView(this).apply {
            text = examples
            textSize = 16f
            setTextColor(0xFF8B5CF6.toInt())
            typeface = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        container.addView(exampleView)
        
        dialog.setContentView(container)
        dialog.show()
    }
    
    private fun dpToPx(dp: Int): Int {
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            dp.toFloat(),
            resources.displayMetrics
        ).toInt()
    }

    private val categoryDescriptions = mapOf(
        "모음" to Pair(
            "성대의 진동을 받은 소리가 목, 입, 코를 거쳐 나오면서 그 통로가 좁아지거나 완전히 막히는 등의 장애를 받지 않고 나는 소리입니다.",
            "AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW"
        ),
        "파열음" to Pair(
            "폐에서 나오는 공기를 막았다가 터뜨리면서 내는 소리입니다.",
            "B, D, G, K, P, T"
        ),
        "마찰음" to Pair(
            "입안의 통로를 좁히고 공기를 그 틈 사이로 내보내어 마찰을 일으키며 내는 소리입니다.",
            "DH, F, S, SH, TH, V, Z, ZH"
        ),
        "비음" to Pair(
            "입안의 통로를 막고 코로 공기를 내보내면서 내는 소리입니다.",
            "M, N, NG"
        ),
        "유음" to Pair(
            "혀끝을 잇몸에 가볍게 대었다가 떼거나, 혀끝을 윗잇몸에 댄 채 공기를 그 양옆으로 흘려 내보내면서 내는 소리입니다.",
            "L, R"
        ),
        "반모음" to Pair(
            "모음과 성질이 비슷하지만 홀로 발음되지 못하고 다른 모음에 붙어서 발음되는 소리입니다.",
            "W, Y"
        ),
        "파찰음" to Pair(
            "파열음과 마찰음의 성질을 가지는 소리로, 공기를 막았다가 서서히 터뜨리면서 마찰을 일으키는 소리입니다.",
            "CH, JH"
        )
    )

    override fun getCurrentScreen(): BottomNavScreen {
        return BottomNavScreen.ANALYSIS
    }
}

class RadarChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    private val backgroundPaint = Paint().apply {
        color = 0xFFE5E7EB.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }
    
    private val dataPaint = Paint().apply {
        color = 0xFF8B5CF6.toInt()
        style = Paint.Style.FILL
        alpha = 80
        isAntiAlias = true
    }
    
    private val dataStrokePaint = Paint().apply {
        color = 0xFF8B5CF6.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    
    private val pointPaint = Paint().apply {
        color = 0xFF8B5CF6.toInt()
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = 0xFF6B7280.toInt()
        textSize = 32f
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }
    
    private var data = mapOf<String, Int>()
    private val categories = listOf("모음", "파열음", "마찰음", "비음", "유음", "반모음", "파찰음")
    
    // 터치 영역 저장을 위한 리스트
    private val labelRects = mutableListOf<RectF>()
    
    // 발음 설명 데이터 (FinalScoreSaver.kt 기반 예시 적용)
    private val categoryDescriptions = mapOf(
        "모음" to Pair(
            "성대의 진동을 받은 소리가 목, 입, 코를 거쳐 나오면서 그 통로가 좁아지거나 완전히 막히는 등의 장애를 받지 않고 나는 소리입니다.",
            "AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW"
        ),
        "파열음" to Pair(
            "폐에서 나오는 공기를 막았다가 터뜨리면서 내는 소리입니다.",
            "B, D, G, K, P, T"
        ),
        "마찰음" to Pair(
            "입안의 통로를 좁히고 공기를 그 틈 사이로 내보내어 마찰을 일으키며 내는 소리입니다.",
            "DH, F, S, SH, TH, V, Z, ZH"
        ),
        "비음" to Pair(
            "입안의 통로를 막고 코로 공기를 내보내면서 내는 소리입니다.",
            "M, N, NG"
        ),
        "유음" to Pair(
            "혀끝을 잇몸에 가볍게 대었다가 떼거나, 혀끝을 윗잇몸에 댄 채 공기를 그 양옆으로 흘려 내보내면서 내는 소리입니다.",
            "L, R"
        ),
        "반모음" to Pair(
            "모음과 성질이 비슷하지만 홀로 발음되지 못하고 다른 모음에 붙어서 발음되는 소리입니다.",
            "W, Y"
        ),
        "파찰음" to Pair(
            "파열음과 마찰음의 성질을 가지는 소리로, 공기를 막았다가 서서히 터뜨리면서 마찰을 일으키는 소리입니다.",
            "CH, JH"
        )
    )
    
    fun updateData(newData: Map<String, Int>) {
        data = newData
        invalidate()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP) {
            val x = event.x
            val y = event.y
            
            // 터치된 영역 확인
            for (i in labelRects.indices) {
                if (labelRects[i].contains(x, y)) {
                    val category = categories[i]
                    showDescriptionDialog(category)
                    return true
                }
            }
        }
        return true
    }
    
    private fun showDescriptionDialog(category: String) {
        val (description, examples) = categoryDescriptions[category] ?: Pair("설명이 없습니다.", "-")
        
        val dialog = BottomSheetDialog(context)
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(24), dpToPx(24), dpToPx(24), dpToPx(48))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        
        // 제목
        val titleView = TextView(context).apply {
            text = category
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(0xFF111827.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(8)
            }
        }
        container.addView(titleView)
        
        // 구분선
        val divider = View(context).apply {
            setBackgroundColor(0xFFE5E7EB.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dpToPx(1)
            ).apply {
                bottomMargin = dpToPx(16)
            }
        }
        container.addView(divider)
        
        // 설명
        val descLabel = TextView(context).apply {
            text = "설명"
            textSize = 14f
            setTypeface(null, Typeface.BOLD)
            setTextColor(0xFF6B7280.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(4)
            }
        }
        container.addView(descLabel)
        
        val descView = TextView(context).apply {
            text = description
            textSize = 16f
            setTextColor(0xFF374151.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(20)
            }
        }
        container.addView(descView)
        
        // 예시
        val exampleLabel = TextView(context).apply {
            text = "해당 발음 (ARPABET)"
            textSize = 14f
            setTypeface(null, Typeface.BOLD)
            setTextColor(0xFF6B7280.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dpToPx(4)
            }
        }
        container.addView(exampleLabel)
        
        val exampleView = TextView(context).apply {
            text = examples
            textSize = 16f
            setTextColor(0xFF8B5CF6.toInt()) // 포인트 컬러
            setTypeface(null, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        container.addView(exampleView)
        
        dialog.setContentView(container)
        dialog.show()
    }
    
    private fun dpToPx(dp: Int): Int {
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            dp.toFloat(),
            resources.displayMetrics
        ).toInt()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val centerX = width / 2f
        val centerY = height / 2f
        val maxRadius = minOf(centerX, centerY) * 0.8f
        
        // 배경 격자 그리기
        drawBackground(canvas, centerX, centerY, maxRadius)
        
        // 데이터가 있을 때만 차트 그리기
        if (data.isNotEmpty()) {
            drawDataArea(canvas, centerX, centerY, maxRadius)
            drawDataPoints(canvas, centerX, centerY, maxRadius)
        }
        
        // 레이블 그리기
        drawLabels(canvas, centerX, centerY, maxRadius)
    }
    
    private fun drawBackground(canvas: Canvas, centerX: Float, centerY: Float, maxRadius: Float) {
        // 동심원 그리기 (25, 50, 75, 100%)
        for (i in 1..4) {
            val radius = maxRadius * i / 4f
            canvas.drawCircle(centerX, centerY, radius, backgroundPaint)
        }
        
        // 방사선 그리기
        val angleStep = 2 * PI / categories.size
        for (i in categories.indices) {
            val angle = i * angleStep - PI / 2 // -90도부터 시작 (12시 방향)
            val endX = centerX + maxRadius * cos(angle).toFloat()
            val endY = centerY + maxRadius * sin(angle).toFloat()
            canvas.drawLine(centerX, centerY, endX, endY, backgroundPaint)
        }
    }
    
    private fun drawDataArea(canvas: Canvas, centerX: Float, centerY: Float, maxRadius: Float) {
        val path = Path()
        val angleStep = 2 * PI / categories.size
        
        for (i in categories.indices) {
            val category = categories[i]
            val score = data[category] ?: 0
            val radius = maxRadius * score / 100f
            val angle = i * angleStep - PI / 2
            
            val x = centerX + radius * cos(angle).toFloat()
            val y = centerY + radius * sin(angle).toFloat()
            
            if (i == 0) {
                path.moveTo(x, y)
            } else {
                path.lineTo(x, y)
            }
        }
        path.close()
        
        // 채워진 영역 그리기
        canvas.drawPath(path, dataPaint)
        
        // 테두리 그리기
        canvas.drawPath(path, dataStrokePaint)
    }
    
    private fun drawDataPoints(canvas: Canvas, centerX: Float, centerY: Float, maxRadius: Float) {
        val angleStep = 2 * PI / categories.size
        
        for (i in categories.indices) {
            val category = categories[i]
            val score = data[category] ?: 0
            val radius = maxRadius * score / 100f
            val angle = i * angleStep - PI / 2
            
            val x = centerX + radius * cos(angle).toFloat()
            val y = centerY + radius * sin(angle).toFloat()
            
            canvas.drawCircle(x, y, 8f, pointPaint)
        }
    }
    
    private val labelBgPaint = Paint().apply {
        color = 0xFFF3F4F6.toInt()
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val labelBorderPaint = Paint().apply {
        color = 0xFFE5E7EB.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 2f
        isAntiAlias = true
    }

    private fun drawLabels(canvas: Canvas, centerX: Float, centerY: Float, maxRadius: Float) {
        labelRects.clear() // 기존 영역 초기화
        
        val angleStep = 2 * PI / categories.size
        val labelRadius = maxRadius * 1.15f
        
        for (i in categories.indices) {
            val category = categories[i]
            val angle = i * angleStep - PI / 2
            
            val x = centerX + labelRadius * cos(angle).toFloat()
            val y = centerY + labelRadius * sin(angle).toFloat()
            
            val textWidth = textPaint.measureText(category)
            val textHeight = textPaint.textSize
            
            // 버튼 배경 영역 계산 (텍스트보다 약간 크게)
            val paddingH = 30f
            val paddingV = 15f
            val bgRect = RectF(
                x - textWidth / 2 - paddingH,
                y - textHeight / 2 - paddingV,
                x + textWidth / 2 + paddingH,
                y + textHeight / 2 + paddingV
            )
            
            // 둥근 사각형 배경 그리기
            canvas.drawRoundRect(bgRect, 40f, 40f, labelBgPaint)
            canvas.drawRoundRect(bgRect, 40f, 40f, labelBorderPaint)
            
            // 텍스트 그리기 (y 좌표 조정하여 수직 중앙 정렬)
            val textY = y - (textPaint.descent() + textPaint.ascent()) / 2
            canvas.drawText(category, x, textY, textPaint)
            
            // 터치 영역 저장
            labelRects.add(bgRect)
        }
    }
    
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val size = minOf(
            MeasureSpec.getSize(widthMeasureSpec),
            MeasureSpec.getSize(heightMeasureSpec)
        )
        setMeasuredDimension(size, size)
    }
}
