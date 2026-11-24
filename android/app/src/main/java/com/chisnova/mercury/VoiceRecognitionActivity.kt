package com.chisnova.mercury

import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.widget.Toast
import com.chisnova.mercury.data.AssetContentLoader
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import kotlin.math.min
import com.chisnova.mercury.asr.AsrController
import com.chisnova.mercury.asr.AsrController.FinalScore
import com.chisnova.mercury.asr.AudioController
import androidx.core.content.ContextCompat
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import android.util.Log
import android.text.SpannableString
import android.text.Spanned
import android.text.method.LinkMovementMethod
import android.text.style.ClickableSpan
import android.text.TextPaint
import android.view.View
import java.util.Locale
import android.media.MediaPlayer
import android.media.AudioAttributes
import com.google.gson.Gson
import com.chisnova.mercury.database.DatabaseHelper
import com.chisnova.mercury.database.DatabaseHelper.ContentItem
import com.chisnova.mercury.database.FinalScoreSaver
import android.view.Gravity
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import java.io.File
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class VoiceRecognitionActivity : BaseActivity() {
    private var selectedSentenceId: Int? = null
    /** 지금 녹음 중인 문장(1-based). 버튼 라벨이 여기서 나온다 — null 이면 아무것도 녹음하지 않는다. */
    private var recordingSentenceId: Int? = null
    private var lessonTitle: String = ""
    private var assetBasePath: String? = null
    private var player: ExoPlayer? = null
    companion object {
        const val BEAM_SIZE = 1
        var asrController: AsrController? = null
        var audioController: AudioController? = null
        var dbHelper: DatabaseHelper? = null
        var finalScoreSaver: FinalScoreSaver? = null
    }

    // ActivityResult API 기반 권한 요청 런처
    private val permissionLauncher: ActivityResultLauncher<String> =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                ensureAudioPermissionThenInit()
            } else {
                handlePermissionDenied()
            }
        }

    // 실제 사용할 문장 데이터
    private var sentences: List<Pair<String, String>> = emptyList()
    private var contents: List<ContentItem> = emptyList()
    private var lastOverallScore: Float? = null
    private var lastFinalDetail: FinalScore? = null

    /**
     * 문장의 단어별 채점 상세. 주제를 열 때 DB 에서 복원한 것(sentenceDetails)을 먼저 보고,
     * 없으면 이번 세션의 마지막 채점을 쓴다. 예전에는 lastFinalDetail 만 봐서 주제를 다시
     * 열거나 다른 문장을 누르면 단어 점수가 보이지 않았다.
     */
    private fun detailFor(sentenceId: Int?): FinalScore? {
        val idx = (sentenceId ?: return lastFinalDetail) - 1
        return sentenceDetails[idx] ?: lastFinalDetail.takeIf { sentenceId == recordingSentenceId }
    }

    // Store scores/details per sentence index (0-based)
    private val sentenceScores = mutableMapOf<Int, Float>()
    private val sentenceDetails = mutableMapOf<Int, FinalScore>()
    /** 문장별 마지막 인식 결과. 연습이 끝나도 지우지 않아 화면에 계속 남는다. */
    private val sentenceTranscripts = mutableMapOf<Int, String>()

    // 녹음 파일 존재 여부 캐시: 매 updateActionButtons() 호출 시 File.exists() 304회 방지
    // 녹음 완료 시 uid를 추가, resetHistory() 시 전체 초기화
    private val recordedUids = mutableSetOf<Int>()

    private var myVoicePlayer: MediaPlayer? = null
    // Lazy ASR init
    private var asrInitStarted: Boolean = false
    private var asrReady: Boolean = false
    private val pendingAsrCallbacks = mutableListOf<() -> Unit>()

    // RecyclerView + Adapter
    private lateinit var sentenceAdapter: SentenceAdapter
    private lateinit var sentencesRecyclerView: RecyclerView

    data class SentenceJson(
        val org: List<String>,
        val word_idx: List<Int>,
        val word_idx_group: List<Int>,
        val phn_idx: List<Int>,
        val phn_idx_group: List<Int>,
        /** 문장별 최종 점수 비중 {"pronunciation": 0.9, "recognition": 0.1}. 대부분 없다. */
        val score_weights: Map<String, Double>? = null
    )
    private var sentenceMetadata: List<SentenceJson> = emptyList()

    data class PhoneInfo(
        val description: String,
        val points: String,
        val tips: String,
        val example: String
    )
    private var phoneInfoMap: Map<String, PhoneInfo> = emptyMap()


    override fun onPurchaseAccessChanged() {
        if (learningAuthorized && !purchases.canAccess(intent.getStringExtra("lesson_id"), intent.getStringExtra("asset_base"))) {
            learningAuthorized = false
            audioController?.stopRecording()
            asrController?.cancelDecoding()
            finish()
        }
    }

    private var learningAuthorized = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!purchases.canAccess(intent.getStringExtra("lesson_id"), intent.getStringExtra("asset_base"))) {
            startActivity(android.content.Intent(this, PurchaseActivity::class.java).apply {
                intent.extras?.let { putExtras(it) }
                putExtra("continue_lesson", true)
            })
            finish()
            return
        }
        learningAuthorized = true
        setContentView(R.layout.activity_voice_recognition)

        // Intent에서 레슨 데이터 받기
        lessonTitle = intent.getStringExtra("lesson_title") ?: "음성 인식 연습"
        assetBasePath = intent.getStringExtra("asset_base")

        // RecyclerView 초기화 — UI는 즉시 구성, 데이터는 백그라운드 로딩
        setupRecyclerView()
        setupUI()

        if (hasBottomNavigation()) {
            setupBottomNavigation()
        }

        dbHelper = DatabaseHelper.getInstance(this)
        asrController = AsrController.getInstance()
        audioController = AudioController()
        ensureAudioPermissionThenInit()
        finalScoreSaver = FinalScoreSaver.getInstance()

        // 파일 I/O(eng/kor/jsonl/phone_info)를 IO 스레드에서 읽고 완료 후 UI 바인딩
        loadAssetDataAsync()
    }

    // ──────────────────────────────────────────────────────────────
    // RecyclerView 초기화
    // ──────────────────────────────────────────────────────────────

    private fun setupRecyclerView() {
        sentencesRecyclerView = findViewById(R.id.sentencesRecyclerView)
        sentenceAdapter = SentenceAdapter(
            formatScore = ::formatScore2,
            onItemClick = { index ->
                selectSentence(index + 1)
                val firstWord = sentences.getOrNull(index)?.first
                    ?.split(Regex("\\s+"))
                    ?.firstOrNull()
                    ?.replace(Regex("[^A-Za-z']"), "")
                if (!firstWord.isNullOrBlank()) showWordInfo(firstWord)
            },
            onPracticeClick = { index ->
                // 녹음 중이면 같은 버튼이 "완료"로 동작한다 — 사용자가 직접 끊을 수 있어야
                // 무음 판정을 기다리지 않고 바로 채점으로 넘어간다.
                // 판정 기준은 라벨을 그리는 값과 같아야 한다(iOS 는 state 하나로 둘 다 정한다).
                if (recordingSentenceId == index + 1) {
                    Log.i("[AsrAPI]", "사용자가 녹음 종료를 눌렀다 (index=$index)")
                    if (audioController?.isRecording() == true) {
                        audioController?.stopRecording()
                    } else {
                        // 아직 마이크가 열리기 전(모델 로딩 중)이라 끊을 녹음이 없다.
                        // 표기만 되돌린다 — 아니면 "완료" 가 그대로 남는다.
                        recordingSentenceId = null
                        buildAndSubmitItems()
                    }
                } else {
                    val pair = sentences.getOrNull(index) ?: return@SentenceAdapter
                    startPracticeFor(
                        sentenceId = index + 1,
                        sentenceText = pair.first,
                    )
                }
            },
            onPlayClick = { index -> selectSentence(index + 1); playAudioForSentence(index + 1) },
            onPlayMyVoiceClick = { index -> selectSentence(index + 1); handlePlayMyVoice() },
            onScoreClick = { index -> showScoreDialog(index) },
            // bind 직후 Span 색상 적용 — 스크롤로 새로 나타나는 항목도 누락 없이 처리
            onItemBound = { position, _ -> applyScoreSpanAt(position) },
        )
        sentencesRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@VoiceRecognitionActivity)
            adapter = sentenceAdapter
            // RecyclerView 자체 크기(layout_weight=1)는 항목 변경에 영향받지 않음
            setHasFixedSize(true)
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 어댑터에 전달할 SentenceItem 목록 빌더
    // ──────────────────────────────────────────────────────────────

    /**
     * 현재 상태(sentences, sentenceScores, recordedUids, selectedSentenceId)로부터
     * 불변 SentenceItem 리스트를 생성하여 Adapter에 제출한다.
     * DiffUtil이 변경된 항목만 rebind하므로 전체 notifyDataSetChanged 불필요.
     */
    private fun buildAndSubmitItems() {
        val selectedIdx = (selectedSentenceId ?: 0) - 1
        val items = sentences.mapIndexed { index, pair ->
            val uid = contents.getOrNull(index)?.uid ?: 0
            SentenceItem(
                index = index,
                engText = pair.first,
                korText = pair.second,
                score = sentenceScores[index],
                hasAudio = recordedUids.contains(uid),
                isSelected = index == selectedIdx,
                transcript = sentenceTranscripts[index].orEmpty(),
                isRecording = (index + 1) == recordingSentenceId,
            )
        }
        sentenceAdapter.submitList(items)
    }

    /**
     * 문장 목록 전체를 처음 바인딩하거나 sentences가 교체되었을 때 호출.
     * Span 적용(colorSentenceByScores)은 Adapter의 onBindViewHolder 이후
     * notifyItemChanged 를 통해 각 항목별로 적용된다.
     */
    private fun bindSentencesToViews() {
        buildAndSubmitItems()
        // 점수가 이미 로드된 항목에 Span 색상 적용
        applyScoreSpansToVisibleItems()
    }

    /**
     * DB 로드 완료 후 점수/색상만 갱신 — 재inflate 없이 DiffUtil로 변경 항목만 rebind.
     */
    private fun refreshScoresOnExistingViews() {
        if (sentences.isEmpty()) {
            return
        }
        buildAndSubmitItems()
        applyScoreSpansToVisibleItems()
    }

    /**
     * 현재 화면에 보이는 ViewHolder에 한해 colorSentenceByScores Span을 적용한다.
     * RecyclerView가 스크롤되면 onBindViewHolder에서 재바인딩 시 Span이 없으므로,
     * 스크롤 이후 화면 진입 항목도 처리하기 위해 addOnScrollListener에서도 호출한다.
     */
    private fun applyScoreSpansToVisibleItems() {
        val lm = sentencesRecyclerView.layoutManager as? LinearLayoutManager ?: return
        val first = lm.findFirstVisibleItemPosition()
        val last = lm.findLastVisibleItemPosition()
        if (first < 0) return
        for (pos in first..last) {
            applyScoreSpanAt(pos)
        }
    }

    private fun applyScoreSpanAt(position: Int) {
        val pair = sentences.getOrNull(position) ?: return
        val detail = sentenceDetails[position] ?: return
        val holder = sentencesRecyclerView.findViewHolderForAdapterPosition(position)
            as? SentenceAdapter.SentenceViewHolder ?: return
        try {
            colorSentenceByScores(holder.engText, pair.first, detail)
        } catch (_: Exception) {}
    }

    // ──────────────────────────────────────────────────────────────
    // 선택 상태 관리
    // ──────────────────────────────────────────────────────────────

    private fun selectSentence(sentenceId: Int) {
        val prev = selectedSentenceId
        selectedSentenceId = sentenceId

        // 변경된 항목(이전 선택 + 새 선택)만 DiffUtil로 rebind
        buildAndSubmitItems()

        // 선택된 항목이 화면 밖이면 스크롤
        val targetPos = sentenceId - 1
        sentencesRecyclerView.post {
            (sentencesRecyclerView.layoutManager as? LinearLayoutManager)
                ?.scrollToPositionWithOffset(targetPos, dpToPx(32))
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 단일 항목 점수 갱신 (녹음/채점 완료 후)
    // ──────────────────────────────────────────────────────────────

    // ──────────────────────────────────────────────────────────────
    // 액션 버튼 상태 갱신 (내 녹음 활성화 여부)
    // ──────────────────────────────────────────────────────────────

    private fun updateActionButtons() {
        // Adapter가 SentenceItem.hasAudio를 보고 알파/활성화를 처리하므로
        // 상태 변경 시 buildAndSubmitItems()만 호출하면 DiffUtil이 처리한다.
        buildAndSubmitItems()
    }

    // ──────────────────────────────────────────────────────────────
    // 비즈니스 로직 (기존 코드 유지)
    // ──────────────────────────────────────────────────────────────

    /**
     * assets에서 문장 데이터와 메타데이터를 IO 스레드에서 읽는다.
     * 완료 후에만 bindSentencesToViews()를 호출하여 첫 프레임 블로킹을 없앤다.
     */
    private fun loadAssetDataAsync() {
        val basePath = assetBasePath
        lifecycleScope.launch {
            val (loadedSentences, loadedMeta, loadedPhoneInfo) = withContext(Dispatchers.IO) {
                // 1) .eng / .kor 읽기
                val loadedSentences: List<Pair<String, String>> = if (!basePath.isNullOrBlank()) {
                    val content = AssetContentLoader.loadBilingualContent(this@VoiceRecognitionActivity, basePath)
                    content.englishLines.zip(
                        if (content.koreanLines.isNotEmpty()) content.koreanLines
                        else List(content.englishLines.size) { "" }
                    )
                } else emptyList()

                // 2) .jsonl 읽기
                val loadedMeta: List<SentenceJson> = if (!basePath.isNullOrBlank()) {
                    val jsonlPath = "$basePath.jsonl"
                    try {
                        assets.open(jsonlPath).use { stream ->
                            val reader = java.io.BufferedReader(java.io.InputStreamReader(stream))
                            val metaList = mutableListOf<SentenceJson>()
                            val gson = Gson()
                            reader.forEachLine { line ->
                                if (line.isNotBlank()) {
                                    try {
                                        metaList.add(gson.fromJson(line, SentenceJson::class.java))
                                    } catch (e: Exception) {
                                        Log.e("VoiceRecognitionActivity", "JSONL parse error: $line", e)
                                    }
                                }
                            }
                            metaList
                        }
                    } catch (e: java.io.FileNotFoundException) {
                        Log.i("VoiceRecognitionActivity", "No JSONL at $jsonlPath")
                        emptyList()
                    } catch (e: Exception) {
                        Log.w("VoiceRecognitionActivity", "Failed to load JSONL: ${e.message}")
                        emptyList()
                    }
                } else emptyList()

                // 3) phone_info.json 읽기 (12KB — 매번 파싱 비용 방지)
                val loadedPhoneInfo: Map<String, PhoneInfo> = try {
                    assets.open("contents/phone_info.json").use { stream ->
                        val type = object : com.google.gson.reflect.TypeToken<Map<String, PhoneInfo>>() {}.type
                        Gson().fromJson(java.io.InputStreamReader(stream), type)
                    }
                } catch (e: Exception) {
                    Log.e("VoiceRecognitionActivity", "Failed to load phone_info.json", e)
                    emptyMap()
                }

                Triple(loadedSentences, loadedMeta, loadedPhoneInfo)
            }

            // IO 완료 후 메인 스레드에서 상태 갱신 및 뷰 바인딩
            sentences = loadedSentences
            sentenceMetadata = loadedMeta
            phoneInfoMap = loadedPhoneInfo
            bindSentencesToViews()
        }
    }

    // onStart()가 매번 호출될 때 중복 DB 로딩을 막는 플래그
    private var historyLoaded = false
    private var historyOperationInProgress = false

    override fun onStart() {
        super.onStart()
        if (!learningAuthorized) return
        // 최초 1회만 DB 로드. 이후 onStart()는 이미 메모리에 있는 상태를 재사용.
        // 명시적 새로고침은 헤더의 '불러오기' 버튼으로만 트리거.
        if (!historyLoaded) {
            historyLoaded = true
            loadContentAndHistory(showMessage = false)
        }
    }

    private data class HistoryLoadResult(
        val contents: List<DatabaseHelper.ContentItem>,
        val scores: Map<Int, Float>,
        val details: Map<Int, FinalScore>,
        val recordedUids: Set<Int>,
        /** 문장별 마지막 인식 결과 */
        val transcripts: Map<Int, String>
    )

    private fun loadContentAndHistory(showMessage: Boolean = true) {
        if (historyOperationInProgress || recordingSentenceId != null) return
        historyOperationInProgress = true
        lifecycleScope.launch {
            try {
                val loaded = withContext(Dispatchers.IO) {
                    val categoryName = assetBasePath?.substringAfterLast("/") ?: lessonTitle
                    // 명시적인 불러오기만 초기화 이전 기록까지 복원한다.
                    if (showMessage) dbHelper?.setProgressHidden(categoryName, hidden = false)
                    val result = dbHelper?.queryContents(categoryName = categoryName) ?: emptyList()

                    val newScores = mutableMapOf<Int, Float>()
                    val newDetails = mutableMapOf<Int, FinalScore>()
                    val newTranscripts = mutableMapOf<Int, String>()

                    // 배치 쿼리: N×3회 루프 → 3번 쿼리 (Attempt 1회, WordHistory 1회, PhoneHistory 1회)
                    val uids = result.map { it.uid }
                    val latestAttempts = dbHelper?.queryLatestAttemptsForUids(uids, visibleOnly = true) ?: emptyMap()

                    val aids = latestAttempts.values.map { it.aid }
                    val allWordHistories = dbHelper?.queryWordHistoriesForAids(aids) ?: emptyMap()
                    val allPhoneHistories = dbHelper?.queryPhoneHistoriesForAids(aids) ?: emptyMap()

                    result.forEachIndexed { index, content ->
                        val attempt = latestAttempts[content.uid] ?: return@forEachIndexed
                        newScores[index] = attempt.pronounce
                        if (attempt.transcript.isNotBlank()) {
                            newTranscripts[index] = attempt.transcript.trim()
                        }

                        val wordHistory = allWordHistories[attempt.aid]
                        val phoneHistory = allPhoneHistories[attempt.aid]
                        if (wordHistory != null && phoneHistory != null) {
                            try {
                                val wordScores = wordHistory.sortedBy { it.wordIdx }.map { wh ->
                                    val phones = phoneHistory.filter { ph -> ph.wordIdx == wh.wordIdx }
                                        .sortedBy { ph -> ph.phoneIdx }
                                        .map { ph -> AsrController.PhoneScore(ph.phone, ph.score) }
                                    AsrController.WordScore(wh.word, wh.score, phones)
                                }
                                newDetails[index] = AsrController.FinalScore(attempt.pronounce, wordScores)
                            } catch (e: Exception) {
                                Log.e("VoiceRecognitionActivity", "Error reconstructing score details", e)
                            }
                        }
                    }

                    // 앱 재시작 후 recordedUids 캐시 복원: IO 스레드에서 파일 존재 여부 일괄 스캔
                    val extDir = getExternalFilesDir(null)?.absolutePath
                    val scannedUids = if (extDir != null) {
                        result.mapNotNull { content ->
                            if (content.uid in latestAttempts && File("$extDir/${content.uid}.wav").exists()) content.uid else null
                        }.toSet()
                    } else emptySet()

                    HistoryLoadResult(result, newScores, newDetails, scannedUids, newTranscripts)
                }

                // 메인 스레드: 상태 교체
                contents = loaded.contents
                sentenceScores.clear(); sentenceScores.putAll(loaded.scores)
                sentenceDetails.clear(); sentenceDetails.putAll(loaded.details)
                sentenceTranscripts.clear(); sentenceTranscripts.putAll(loaded.transcripts)
                recordedUids.clear()
                recordedUids.addAll(loaded.recordedUids)

                // View가 이미 inflate된 경우 점수/색상만 갱신 (inflate 생략)
                if (sentences.isNotEmpty()) {
                    refreshScoresOnExistingViews()
                }
                updateActionButtons()
                if (showMessage) {
                    Toast.makeText(this@VoiceRecognitionActivity, "저장된 점수와 진행률을 불러왔습니다.", Toast.LENGTH_SHORT).show()
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                Toast.makeText(this@VoiceRecognitionActivity, "기록을 불러오지 못했습니다.", Toast.LENGTH_SHORT).show()
                Log.e("VoiceRecognitionActivity", "Failed to load history", e)
            } finally {
                historyOperationInProgress = false
            }
        }
    }

    private fun resetHistory() {
        if (historyOperationInProgress || recordingSentenceId != null) return
        historyOperationInProgress = true
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val category = assetBasePath?.substringAfterLast("/") ?: lessonTitle
                    dbHelper?.setProgressHidden(category, hidden = true)
                }
                sentenceScores.clear()
                sentenceDetails.clear()
                sentenceTranscripts.clear()
                recordedUids.clear()
                lastOverallScore = null
                lastFinalDetail = null
                myVoicePlayer?.pause()
                refreshScoresOnExistingViews()
                updateActionButtons()
                Toast.makeText(this@VoiceRecognitionActivity,
                    "화면과 진행률을 초기화했습니다. 불러오기로 복원할 수 있습니다.", Toast.LENGTH_SHORT).show()
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                Toast.makeText(this@VoiceRecognitionActivity, "초기화하지 못했습니다.", Toast.LENGTH_SHORT).show()
                Log.e("VoiceRecognitionActivity", "Failed to reset progress", e)
            } finally {
                historyOperationInProgress = false
            }
        }
    }

    private fun ensureAudioPermissionThenInit() {
        val hasPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (hasPermission) {
            Log.i("[AsrAPI]", "RECORD_AUDIO permission already granted")
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    /**
     * 권한 거부 시 처리: rationale 확인 후 재요청 또는 설정 화면 안내
     */
    private fun handlePermissionDenied() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO)) {
            android.app.AlertDialog.Builder(this)
                .setTitle("마이크 권한 필요")
                .setMessage("음성 인식 연습을 위해 마이크 권한이 필요합니다. 권한을 허용해 주세요.")
                .setPositiveButton("다시 요청") { _, _ ->
                    permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
                .setNegativeButton("취소", null)
                .show()
        } else {
            android.app.AlertDialog.Builder(this)
                .setTitle("마이크 권한 필요")
                .setMessage("마이크 권한이 거부되었습니다. 음성 인식을 사용하려면 앱 설정에서 마이크 권한을 직접 허용해 주세요.")
                .setPositiveButton("설정으로 이동") { _, _ ->
                    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                        data = Uri.fromParts("package", packageName, null)
                    }
                    startActivity(intent)
                }
                .setNegativeButton("취소", null)
                .show()
        }
    }

    private fun ensureAsrReady(onReady: () -> Unit) {
        if (asrReady) {
            onReady()
            return
        }
        pendingAsrCallbacks += onReady
        if (asrInitStarted) return
        asrInitStarted = true
        asrController?.initAsync(this,
            onReady = {
                asrReady = true
                val callbacks = ArrayList(pendingAsrCallbacks)
                pendingAsrCallbacks.clear()
                callbacks.forEach { it.invoke() }
            },
            onError = { errorMsg ->
                Log.e("[VoiceRecognition]", "ASR init failed: $errorMsg")
                pendingAsrCallbacks.clear()
                showAsrInitErrorDialog(errorMsg)
            }
        )
    }

    /**
     * ASR 엔진 초기화 실패 시 사용자에게 오류를 안내하고 무음 크래시를 방지한다.
     */
    private fun showAsrInitErrorDialog(errorMsg: String) {
        if (isFinishing || isDestroyed) return
        android.app.AlertDialog.Builder(this)
            .setTitle("음성 인식 초기화 실패")
            .setMessage("음성 인식 엔진을 시작할 수 없습니다.\n\n원인: $errorMsg")
            .setPositiveButton("확인") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }

    override fun getCurrentScreen(): BottomNavScreen {
        return BottomNavScreen.HOME
    }

    private fun setupUI() {
        // 제목 업데이트
        findViewById<TextView>(R.id.titleText)?.text = lessonTitle

        // 뒤로가기 버튼
        findViewById<ImageView>(R.id.btnBack)?.setOnClickListener {
            finish()
        }

        updateActionButtons()

        // 상단 버튼 연결
        findViewById<TextView>(R.id.btnLoadData)?.setOnClickListener {
            loadContentAndHistory(showMessage = true)
        }
        findViewById<TextView>(R.id.btnResetData)?.setOnClickListener {
            resetHistory()
        }

        // 단어 정보 오버레이 닫기
        findViewById<ImageView>(R.id.btnCloseWordInfo)?.setOnClickListener {
            findViewById<android.widget.FrameLayout>(R.id.wordInfoOverlay)?.visibility = View.GONE
        }
        // 오버레이 바깥 클릭 시 닫기
        findViewById<android.widget.FrameLayout>(R.id.wordInfoOverlay)?.setOnClickListener {
            findViewById<android.widget.FrameLayout>(R.id.wordInfoOverlay)?.visibility = View.GONE
        }

        // 스크롤 후 화면에 진입하는 항목에도 Span 색상을 적용
        sentencesRecyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                if (newState == RecyclerView.SCROLL_STATE_IDLE) {
                    applyScoreSpansToVisibleItems()
                }
            }
        })
    }

    private fun playAudioForSentence(sentenceId: Int) {
        val sentence = sentences.getOrNull(sentenceId - 1)
        sentence?.let {
            val base = assetBasePath
            if (!base.isNullOrBlank()) {
                val lastSlash = base.lastIndexOf('/')
                val folder = if (lastSlash >= 0) base.substring(0, lastSlash) else base
                val opusDir = "$folder/opus"
                val files = try { assets.list(opusDir)?.toList() ?: emptyList() } catch (_: Exception) { emptyList() }
                val opusFiles = files.filter { it.endsWith(".opus") }.sorted()
                if (opusFiles.isEmpty()) {
                    android.widget.Toast.makeText(this, "오디오 파일을 찾을 수 없습니다", android.widget.Toast.LENGTH_SHORT).show()
                    return
                }
                val idx = min(sentenceId - 1, opusFiles.size - 1)
                val assetPath = "$opusDir/${opusFiles[idx]}"
                val uri = android.net.Uri.parse("asset:///" + assetPath)

                if (player == null) {
                    player = ExoPlayer.Builder(this).build()
                }
                player?.apply {
                    stop()
                    clearMediaItems()
                    setMediaItem(MediaItem.fromUri(uri))
                    prepare()
                    play()
                }
            } else {
                android.widget.Toast.makeText(
                    this,
                    "재생 중: ${it.first}",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun handlePlayMyVoice() {
        val sid = selectedSentenceId
        if (sid == null) {
            android.widget.Toast.makeText(this, "먼저 문장을 선택하세요", android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        val uid = contents.getOrNull(sid - 1)?.uid?.toInt() ?: 0
        val audioFile = File(getExternalFilesDir(null), "${uid}.wav")
        if (audioFile.exists()) {
            try {
                myVoicePlayer?.release()
                myVoicePlayer = MediaPlayer().apply {
                    setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    setDataSource(audioFile.absolutePath)
                    prepare()
                    start()
                }
            } catch (e: Exception) {
                Log.e("[AsrAPI]", "My voice playback failed", e)
                android.widget.Toast.makeText(this, "재생 실패", android.widget.Toast.LENGTH_SHORT).show()
            }
            return
        }
    }

    private fun startPracticeFor(
        sentenceId: Int,
        sentenceText: String,
    ) {
        if (historyOperationInProgress || !learningAuthorized ||
            !purchases.canAccess(intent.getStringExtra("lesson_id"), assetBasePath)) return
        selectSentence(sentenceId)

        // position 기반으로 현재 ViewHolder 조회하여 UI 갱신
        val targetPos = sentenceId - 1
        val holder = sentencesRecyclerView.findViewHolderForAdapterPosition(targetPos)
            as? SentenceAdapter.SentenceViewHolder
        holder?.let {
            // 인식 결과는 칸 하나(itemHintText)에서만 보여 준다. 예전에는 스트리밍 중간
            // 결과와 최종 결과를 다른 칸에 나눠 찍어 같은 내용이 두 군데 떠 있었다.
            it.itemHintText.text = "인식 중…"
            it.itemHintText.visibility = View.VISIBLE
            it.itemStatus.visibility = View.GONE
            it.itemProgress.visibility = View.VISIBLE
            // 녹음 중에도 버튼을 살려 둔다 — 다시 누르면 그 자리에서 녹음을 끝낸다
            it.btnPractice.isEnabled = true
            it.btnPractice.alpha = 1.0f
        }

        // 라벨("완료")은 어댑터가 상태를 보고 그린다. 여기서 ViewHolder 를 직접 고치면
        // 화면 밖이거나 스크롤로 재바인딩될 때 표기가 되돌아간다.
        recordingSentenceId = sentenceId
        buildAndSubmitItems()

        val sentence = sentences.getOrNull(sentenceId - 1)
        val content = contents.getOrNull(sentenceId - 1)
        sentence?.let {
            ensureAsrReady {
                asrController?.cancelDecoding()

                if (audioController == null) {
                    audioController = AudioController()
                }
                audioController?.init(this)
                audioController?.startRecording()

                asrController?.clear()
                // ASR 힌트용 TextView: 현재 화면의 ViewHolder를 position으로 조회
                val currentHolder = sentencesRecyclerView.findViewHolderForAdapterPosition(targetPos)
                    as? SentenceAdapter.SentenceViewHolder
                val targetHint = currentHolder?.itemHintText
                    ?: findViewById<TextView>(R.id.hintText)
                if (targetHint != null) {
                    asrController?.startBackgroundEngine(BEAM_SIZE, targetHint)
                } else {
                    val dummy = TextView(this)
                    asrController?.startBackgroundEngine(BEAM_SIZE, dummy)
                }

                audioController?.onRecordingFinished = {
                    handleRecordingFinished(sentenceId, sentence, content, sentenceText)
                }
                Log.i("[AsrAPI]", "startPracticeFor() — about to call audioController.sendAudio(), asrController.isReady=${asrController?.isReady()}")
                asrController?.let { ctrl -> audioController?.sendAudio(ctrl) }
            }
        }
    }

    private fun handleRecordingFinished(
        sentenceId: Int,
        sentence: Pair<String, String>?,
        content: DatabaseHelper.ContentItem?,
        sentenceText: String,
    ) {
        recordingSentenceId = null
        buildAndSubmitItems()

        asrController?.finishDecoding()
        val recordAudio = audioController?.getRecordAudio()

        val meta = sentenceMetadata.getOrNull(sentenceId - 1)
        // 오디오가 한 샘플도 안 들어왔으면 네이티브가 2패스 인코딩을 만들 수 없다.
        // 그 상태로 채점을 부르면 널을 역참조하므로(버튼을 누르자마자 완료한 경우 등) 건너뛴다.
        val hasAudio = audioController?.hasRecordedAudio() == true
        val finalScore = if (meta != null && hasAudio) {
            val orgStr = meta.org.joinToString(" ")
            // .jsonl 의 문장별 점수 비중을 채점 직전에 넘긴다. 없으면 모델 기본으로 되돌린다 —
            // 이전 문장의 비중이 남지 않게 항상 지정한다.
            val pronunciation = meta.score_weights?.get("pronunciation")
            val recognition = meta.score_weights?.get("recognition")
            if (pronunciation != null && recognition != null) {
                asrController?.setSentenceScoreWeights(pronunciation.toFloat(), recognition.toFloat())
            } else {
                asrController?.setSentenceScoreWeights(-1f, -1f)
            }
            asrController?.getPronounceScore(
                orgStr,
                meta.word_idx.toIntArray(),
                meta.word_idx_group.toIntArray(),
                meta.phn_idx.toIntArray(),
                meta.phn_idx_group.toIntArray()
            )
        } else {
            // 채점에 필요한 서브워드·음소 인덱스는 콘텐츠의 .jsonl 에만 있다.
            // 그것이 없으면 채점을 건너뛴다 — 예전에는 라벨만 넘기는 경로를 탔는데
            // 그 오버로드는 네이티브 구현이 없어 앱이 SIGSEGV 로 죽었다.
            Log.w("[AsrAPI]", "발음 채점 메타(.jsonl)가 없어 점수를 건너뛴다 — sentenceId=$sentenceId")
            null
        }

        Log.i("[AsrAPI]", "practice finalScore: $finalScore")
        try {
            recordAudio?.let { audioData ->
                try {
                    val uid = contents.getOrNull(sentenceId - 1)?.uid?.toInt() ?: 0
                    val audioFileName = "${uid}.wav"
                    val audioFile = File(getExternalFilesDir(null), audioFileName)
                    writeWavFile(audioFile, audioData, 16000, 1, 16)
                    Log.i("[AsrAPI]", "Audio saved to: ${audioFile.absolutePath}")
                    recordedUids.add(uid)
                } catch (e: Exception) {
                    Log.e("[AsrAPI]", "Error saving audio: ${e.message}", e)
                }
            }

            // 2패스 인식 결과를 화면과 DB 양쪽에 남긴다. 예전에는 빈 문자열을 넣어
            // 무엇이라 발음했는지 되돌아볼 수 없었다.
            val transcript = asrController?.getCtcAsrResult()
                ?.trim()
                .orEmpty()
            if (transcript.isNotEmpty()) {
                sentenceTranscripts[sentenceId - 1] = transcript
            }
            var saved = finalScoreSaver?.saveFinalScoreToDatabase(finalScore, content, transcript, dbHelper)
            lastOverallScore = saved?.toFloat()

            val sIdx = sentenceId - 1
            if (saved != null) {
                sentenceScores[sIdx] = saved.toFloat()
            }

            finalScore?.let {
                try {
                    lastFinalDetail = Gson().fromJson(it, FinalScore::class.java)
                    sentenceDetails[sIdx] = lastFinalDetail!!
                } catch (_: Exception) { lastFinalDetail = null }
            }

            // 상태 갱신 완료 후 한 번만 DiffUtil 제출 (점수 + hasAudio 동시 반영)
            buildAndSubmitItems()

            // Span 색상 반영 — ViewHolder가 재활용되었을 수 있으므로 position 기반으로 조회
            val targetPos = sentenceId - 1
            val currentHolder = sentencesRecyclerView.findViewHolderForAdapterPosition(targetPos)
                as? SentenceAdapter.SentenceViewHolder
            if (currentHolder != null) {
                try { colorSentenceByScores(currentHolder.engText, sentenceText, lastFinalDetail) } catch (_: Exception) {}
                currentHolder.itemStatus.visibility = View.GONE
                currentHolder.itemProgress.visibility = View.GONE
                currentHolder.btnPractice.isEnabled = true
                currentHolder.btnPractice.alpha = 1.0f
                // 최종 결과는 buildAndSubmitItems -> bindTranscript 가 같은 칸에 남긴다
            } else {
                // ViewHolder가 화면 밖 — UI 상태는 다음 bind()에서 기본값으로 복원됨
                Log.w("[AsrAPI]", "handleRecordingFinished: ViewHolder for pos $targetPos not visible, skipping UI update")
            }
        } catch (e: Exception) {
            Log.e("[AsrAPI]", "Error saving final score to database: ${e.message}", e)
            // 실패 시에도 position 기반으로 ViewHolder 조회
            val targetPos = sentenceId - 1
            val currentHolder = sentencesRecyclerView.findViewHolderForAdapterPosition(targetPos)
                as? SentenceAdapter.SentenceViewHolder
            if (currentHolder != null) {
                currentHolder.itemStatus.text = "실패"
                currentHolder.itemProgress.visibility = View.GONE
                currentHolder.btnPractice.isEnabled = true
                currentHolder.btnPractice.alpha = 1.0f
                Handler(Looper.getMainLooper()).postDelayed({
                    currentHolder.itemStatus.visibility = View.GONE
                }, 1500)
            }
        }
    }

    private fun colorSentenceByScores(textView: TextView?, sentence: String, detail: FinalScore?) {
        textView ?: return
        if (sentence.isBlank()) {
            textView.text = ""
            return
        }
        val spannable = SpannableString(sentence)
        val norm: (String) -> String = { t -> t.replace(Regex("[^\\p{L}']"), "").lowercase() }
        val list = detail?.final_score ?: emptyList()
        val byName = list.associateBy { norm(it.word) }

        // 문장을 공백 단위로 끊는다. 글자만 찾는 정규식을 쓰면 "5%," 나 "$32,000." 처럼
        // 글자가 없는 낱말이 통째로 빠져 색이 안 칠해진다.
        //
        // 짝은 자리(순서)로 맞춘다 — 채점 결과의 낱말 차례는 문장의 낱말 차례와 같다.
        // 이름으로 찾으면 위 같은 낱말은 열쇠가 비고, 같은 낱말이 두 번 나오면
        // 앞의 점수만 쓰게 된다. 자리가 어긋난 경우를 대비해 이름도 확인한다.
        Regex("\\S+").findAll(sentence).forEachIndexed { index, match ->
            val start = match.range.first
            val end = match.range.last + 1
            val word = match.value
            val key = norm(word)

            var scored = list.getOrNull(index)?.takeIf { key.isEmpty() || norm(it.word) == key }
            if (scored == null && key.isNotEmpty()) scored = byName[key]
            val s = scored?.score

            var color = android.graphics.Color.parseColor("#000000")
            if (scored != null && !scored.isGraded) {
                // 채점 제외 낱말 — 점수로 색을 칠하지 않고 회색으로 둔다
                color = android.graphics.Color.parseColor("#6B7280")
            } else s?.let {
                color = when {
                    it >= 80f -> android.graphics.Color.parseColor("#2563EB")
                    it >= 50f -> android.graphics.Color.parseColor("#F59E0B")
                    else -> android.graphics.Color.parseColor("#DC2626")
                }
            }

            val infoWord = scored?.word ?: word
            val span = object : ClickableSpan() {
                override fun onClick(widget: View) {
                    showWordInfo(infoWord)
                }
                override fun updateDrawState(ds: TextPaint) {
                    super.updateDrawState(ds)
                    ds.isUnderlineText = false
                    ds.color = color
                }
            }
            spannable.setSpan(span, start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }

        textView.text = spannable
        textView.movementMethod = LinkMovementMethod.getInstance()
        textView.highlightColor = android.graphics.Color.TRANSPARENT
    }

    /** 점수 배지를 눌렀을 때 — 그 문장의 점수와 단어별 점수를 보여 준다. */
    private fun showScoreDialog(index: Int) {
        val sentenceId = index + 1
        val score = sentenceScores[index]
            ?: lastOverallScore.takeIf { sentenceId == recordingSentenceId || sentenceScores.isEmpty() }
        val message = if (score == null) {
            "아직 점수가 없습니다. 마이크로 발음해보세요."
        } else {
            val words = detailFor(sentenceId)?.final_score.orEmpty()
            buildString {
                append("문장 점수: ${formatScore2(score)}점")
                if (words.isNotEmpty()) {
                    append("\n\n단어별 점수")
                    for (w in words) {
                        val s = if (w.isGraded) "${formatScore2(w.score)}점" else "채점 제외"
                        append("\n  ${w.word}  $s")
                    }
                }
            }
        }
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("발음 분석")
            .setMessage(message)
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    private fun formatScore2(value: Float): String {
        return String.format(Locale.getDefault(), "%.2f", value)
    }

    private fun writeWavFile(target: File, pcm: ShortArray, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val byteRate = sampleRate * channels * (bitsPerSample / 8)
        val dataSize = pcm.size * 2
        val totalDataLen = 36 + dataSize
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        writeLE(header, 4, totalDataLen)
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        writeLE(header, 16, 16)
        writeLE(header, 20, 1)
        writeLE(header, 22, channels)
        writeLE(header, 24, sampleRate)
        writeLE(header, 28, byteRate)
        writeLE(header, 32, (channels * (bitsPerSample / 8)))
        writeLE(header, 34, bitsPerSample)
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        writeLE(header, 40, dataSize)

        val out = java.io.BufferedOutputStream(java.io.FileOutputStream(target))
        out.write(header, 0, 44)
        for (s in pcm) {
            out.write((s.toInt() and 0xFF))
            out.write(((s.toInt() shr 8) and 0xFF))
        }
        out.flush()
        out.close()
    }

    private fun writeLE(buf: ByteArray, offset: Int, value: Int) {
        buf[offset] = (value and 0xFF).toByte()
        buf[offset + 1] = ((value shr 8) and 0xFF).toByte()
        buf[offset + 2] = ((value shr 16) and 0xFF).toByte()
        buf[offset + 3] = ((value shr 24) and 0xFF).toByte()
    }

    private fun showWordInfo(word: String) {
        val normalizedClick = word.replace(Regex("[^A-Za-z']"), "").lowercase()
        findViewById<TextView>(R.id.selectedWord)?.text = word
        findViewById<TextView>(R.id.wordPhonetic)?.text = "—"

        findViewById<TextView>(R.id.wordOverallScore)?.text = "—"
        val container = findViewById<LinearLayout>(R.id.phoneScoresContainer)
        container?.removeAllViews()
        findViewById<android.widget.FrameLayout>(R.id.wordInfoOverlay)?.visibility = View.VISIBLE
        val wordScore = detailFor(selectedSentenceId)?.final_score?.firstOrNull {
            it.word.replace(Regex("[^A-Za-z']"), "").equals(normalizedClick, ignoreCase = true)
        }
        if (wordScore != null) {
            findViewById<TextView>(R.id.wordPhonetic)?.text =
                wordScore.phone_score.joinToString(" ") { it.word }.ifBlank { "—" }
            val scoreStr = if (wordScore.isGraded) "${formatScore2(wordScore.score)}점" else "채점 제외"
            findViewById<TextView>(R.id.wordOverallScore)?.text = scoreStr
            findViewById<TextView>(R.id.selectedWord)?.text = "$word  •  $scoreStr"

            container?.let { c ->
                c.removeAllViews()

                fun createFeedbackView(phone: String, score: Float): View {
                    val feedbackContainer = LinearLayout(this)
                    val fbLp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                    fbLp.topMargin = dpToPx(16)
                    feedbackContainer.layoutParams = fbLp
                    feedbackContainer.orientation = LinearLayout.VERTICAL

                    val feedbackLayout = LinearLayout(this)
                    feedbackLayout.orientation = LinearLayout.VERTICAL
                    feedbackLayout.setPadding(dpToPx(16), dpToPx(16), dpToPx(16), dpToPx(16))

                    val fbBg = android.graphics.drawable.GradientDrawable()
                    fbBg.setColor(android.graphics.Color.parseColor("#F3F4F6"))
                    fbBg.cornerRadius = dpToPx(12).toFloat()
                    feedbackLayout.background = fbBg

                    val pInfo = phoneInfoMap[phone.uppercase()]

                    if (pInfo != null) {
                        val title = TextView(this)
                        title.text = "발음 가이드: '$phone' (${score.toInt()}점)"
                        title.textSize = 15f
                        title.setTypeface(null, android.graphics.Typeface.BOLD)
                        title.setTextColor(android.graphics.Color.parseColor("#111827"))
                        title.setPadding(0, 0, 0, dpToPx(12))
                        feedbackLayout.addView(title)

                        fun addRow(label: String, content: String) {
                            val row = LinearLayout(this)
                            row.orientation = LinearLayout.HORIZONTAL
                            row.layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                                bottomMargin = dpToPx(8)
                            }

                            val labelView = TextView(this)
                            labelView.text = label
                            labelView.textSize = 13f
                            labelView.setTypeface(null, android.graphics.Typeface.BOLD)
                            labelView.setTextColor(android.graphics.Color.parseColor("#4B5563"))
                            labelView.layoutParams = LinearLayout.LayoutParams(dpToPx(60), LinearLayout.LayoutParams.WRAP_CONTENT)

                            val contentView = TextView(this)
                            contentView.text = content
                            contentView.textSize = 13f
                            contentView.setTextColor(android.graphics.Color.parseColor("#1F2937"))
                            contentView.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)

                            row.addView(labelView)
                            row.addView(contentView)
                            feedbackLayout.addView(row)
                        }

                        addRow("설명", pInfo.description)
                        addRow("포인트", pInfo.points)
                        addRow("팁", pInfo.tips)
                    } else {
                        val msg = TextView(this)
                        msg.text = "발음 '$phone'에 대한 정보가 없습니다."
                        msg.textSize = 13f
                        msg.setTextColor(android.graphics.Color.parseColor("#4B5563"))
                        feedbackLayout.addView(msg)
                    }
                    feedbackContainer.addView(feedbackLayout)
                    return feedbackContainer
                }

                fun renderList(selectedPhone: String, selectedScore: Float) {
                    c.removeAllViews()
                    c.orientation = LinearLayout.VERTICAL

                    wordScore.phone_score.forEach { ps ->
                        val isSelected = ps.word == selectedPhone

                        val row = LinearLayout(this)
                        row.orientation = LinearLayout.HORIZONTAL
                        row.gravity = Gravity.CENTER_VERTICAL
                        val rowLp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                        rowLp.bottomMargin = dpToPx(8)
                        row.layoutParams = rowLp
                        row.setPadding(dpToPx(8), dpToPx(8), dpToPx(8), dpToPx(8))

                        val bgDrawable = android.graphics.drawable.GradientDrawable()
                        bgDrawable.cornerRadius = dpToPx(8).toFloat()
                        if (isSelected) {
                            bgDrawable.setColor(android.graphics.Color.parseColor("#EFF6FF"))
                            bgDrawable.setStroke(dpToPx(1), android.graphics.Color.parseColor("#BFDBFE"))
                        } else {
                            bgDrawable.setColor(android.graphics.Color.TRANSPARENT)
                        }
                        row.background = bgDrawable

                        row.isClickable = true
                        row.setOnClickListener {
                            if (!isSelected) {
                                renderList(ps.word, ps.score)
                            }
                        }

                        val tvPhone = TextView(this)
                        tvPhone.text = ps.word
                        tvPhone.setTextColor(if (isSelected) android.graphics.Color.parseColor("#1D4ED8") else android.graphics.Color.parseColor("#374151"))
                        tvPhone.textSize = 15f
                        tvPhone.setTypeface(null, if (isSelected) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
                        tvPhone.width = dpToPx(40)
                        tvPhone.gravity = Gravity.CENTER_VERTICAL
                        row.addView(tvPhone)

                        val barContainer = LinearLayout(this)
                        val barLp = LinearLayout.LayoutParams(0, dpToPx(10), 1f)
                        barLp.marginStart = dpToPx(8)
                        barLp.marginEnd = dpToPx(12)
                        barContainer.layoutParams = barLp
                        barContainer.orientation = LinearLayout.HORIZONTAL
                        barContainer.weightSum = 100f

                        val trackDrawable = android.graphics.drawable.GradientDrawable()
                        trackDrawable.cornerRadius = dpToPx(5).toFloat()
                        trackDrawable.setColor(android.graphics.Color.parseColor("#E5E7EB"))
                        barContainer.background = trackDrawable

                        val barFill = View(this)
                        val scoreVal = ps.score.coerceIn(0f, 100f)
                        val fillLp = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, scoreVal)
                        barFill.layoutParams = fillLp

                        val fillDrawable = android.graphics.drawable.GradientDrawable()
                        fillDrawable.cornerRadius = dpToPx(5).toFloat()
                        fillDrawable.setColor(getScoreColor(scoreVal))
                        barFill.background = fillDrawable

                        barContainer.addView(barFill)
                        row.addView(barContainer)

                        val tvScore = TextView(this)
                        tvScore.text = "${ps.score.toInt()}"
                        tvScore.setTextColor(if (isSelected) android.graphics.Color.parseColor("#1D4ED8") else android.graphics.Color.parseColor("#6B7280"))
                        tvScore.textSize = 13f
                        tvScore.setTypeface(null, if (isSelected) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
                        tvScore.width = dpToPx(24)
                        tvScore.gravity = Gravity.END
                        row.addView(tvScore)

                        val chevron = ImageView(this)
                        chevron.setImageResource(R.drawable.ic_chevron_right_24)
                        chevron.setColorFilter(if (isSelected) android.graphics.Color.parseColor("#2563EB") else android.graphics.Color.parseColor("#9CA3AF"))
                        val chevronLp = LinearLayout.LayoutParams(dpToPx(20), dpToPx(20))
                        chevronLp.marginStart = dpToPx(4)
                        chevron.layoutParams = chevronLp
                        row.addView(chevron)

                        c.addView(row)
                    }

                    c.addView(createFeedbackView(selectedPhone, selectedScore))
                }

                var minScore = 101f
                var minPhone: String? = null
                wordScore.phone_score.forEach {
                    if (it.score < minScore) {
                        minScore = it.score
                        minPhone = it.word
                    }
                }

                minPhone?.let { phone ->
                    renderList(phone, minScore)
                }
            }
        }
    }

    private fun getScoreColor(score: Float): Int {
        return when {
            score >= 75f -> android.graphics.Color.parseColor("#2563EB")
            score >= 65f -> android.graphics.Color.parseColor("#F59E0B")
            else -> android.graphics.Color.parseColor("#DC2626")
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    override fun onStop() {
        super.onStop()
        recordingSentenceId = null
        audioController?.stopRecording()
        asrController?.cancelDecoding()
        player?.release()
        player = null
        myVoicePlayer?.release()
        myVoicePlayer = null
    }

    override fun onDestroy() {
        super.onDestroy()
        audioController?.release()
        audioController = null
    }
}
