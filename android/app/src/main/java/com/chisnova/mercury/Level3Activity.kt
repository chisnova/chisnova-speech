package com.chisnova.mercury

import android.content.Intent
import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout

class Level3Activity : BaseActivity() {
    private val lessonProgress by lazy { LessonProgressBinder(this, R.color.ds_accent_purple) }

    override fun onResume() {
        super.onResume()
        lessonProgress.refresh()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_level3)
        
        setupUI()
        setupSpeechClickListeners()
        
        // 하단 네비게이션 설정 (있는 경우에만)
        if (hasBottomNavigation()) {
            setupBottomNavigation()
        }
    }
    
    private fun setupUI() {
        // 뒤로가기 버튼 클릭 리스너
        findViewById<ImageView>(R.id.btnBack)?.setOnClickListener {
            finish()
        }
    }
    
    private fun setupSpeechClickListeners() {
        bindSpeech(R.id.speechLincoln, "Abraham Lincoln's Speech", "contents/5.speech/lincoln/lincoln")
        bindSpeech(R.id.speechMandela, "Nelson Mandela's Speech", "contents/5.speech/mandela/mandela")
        bindSpeech(R.id.speechReagan, "Ronald Reagan's Speech", "contents/5.speech/reagan/reagan")

        // Conan O'Brien 연설
        bindSpeech(R.id.speechConan, "Conan O'Brien's Harvard Speech", "contents/5.speech/conan/conan")
        
        // Obama 연설
        bindSpeech(R.id.speechObama, "Barack Obama's Speech", "contents/5.speech/barackobama/barackobama")

        // George Bush 연설
        bindSpeech(R.id.speechBush, "George Bush's Speech", "contents/5.speech/gergebush/gergebush")

        // JFK 연설
        bindSpeech(R.id.speechJFK, "John F. Kennedy's Speech", "contents/5.speech/jfkennedy/jfkennedy")

        // Martin Luther King Jr 연설
        bindSpeech(R.id.speechKing, "Martin Luther King Jr.'s Speech", "contents/5.speech/martinLutherkingjr/martinLutherkingjr")

    }
    
    private fun bindSpeech(viewId: Int, title: String, assetBase: String) {
        val card = findViewById<LinearLayout>(viewId) ?: return
        (card.getChildAt(0) as? ImageView)?.let { bindContentArtwork(it) }
        val textContainer = (0 until card.childCount)
            .map { card.getChildAt(it) }.filterIsInstance<LinearLayout>().firstOrNull()
        textContainer?.let {
            lessonProgress.addTo(it, assetBase)
            val source = runCatching {
                assets.open("$assetBase.metadata.json").bufferedReader().use { reader ->
                    org.json.JSONObject(reader.readText())
                }
            }.getOrNull()
            if (source != null) {
                val sourceLink = android.widget.TextView(this).apply {
                    text = "연설 정보·출처"
                    textSize = 12f
                    setTextColor(getColor(R.color.ds_accent_purple))
                    setPadding(0, 8, 0, 8)
                    setOnClickListener {
                        android.app.AlertDialog.Builder(this@Level3Activity)
                            .setTitle(source.getString("title"))
                            .setMessage(listOf(source.getString("author"), source.getString("date"),
                                source.getString("sourceName"), source.getString("notice")).joinToString("\n\n"))
                            .setPositiveButton("원문 보기") { _, _ ->
                                startActivity(Intent(Intent.ACTION_VIEW, android.net.Uri.parse(source.getString("sourceURL"))))
                            }
                            .setNegativeButton("닫기", null)
                            .show()
                    }
                }
                it.addView(sourceLink)
            }
        }
        addPurchaseLock(card)
        card.setOnClickListener { navigateToVoiceRecognitionWithAssets(title, assetBase) }
    }

    private fun navigateToVoiceRecognitionWithAssets(title: String, assetBase: String) {
        val intent = Intent(this, VoiceRecognitionActivity::class.java).apply {
            putExtra("lesson_title", title)
            putExtra("asset_base", assetBase)
        }
        startActivity(intent)
    }
    
    override fun getCurrentScreen(): BottomNavScreen {
        // Level Activity는 홈에서 접근하므로 홈으로 분류
        return BottomNavScreen.HOME
    }
}
