package com.chisnova.mercury.asr

import android.content.Intent
import android.util.Log
import androidx.recyclerview.widget.RecyclerView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.chisnova.mercury.R
import com.chisnova.mercury.SentenceAdapter
import com.chisnova.mercury.VoiceRecognitionActivity
import com.chisnova.mercury.billing.ContentAccess
import com.chisnova.mercury.billing.PurchaseManager
import com.chisnova.mercury.data.LessonData
import com.chisnova.mercury.support.AsrTestSupport
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * 실기기 음향 루프백 전수검사.
 *
 * 앱 화면을 실제로 조작한다: 문장의 [발음연습]을 눌러 마이크 녹음(디코딩)을 시작하고,
 * 이어서 [예시듣기]를 눌러 예시 음원을 스피커로 재생한다. 그 소리가 단말 마이크로 들어가
 * 인식·무음 판정·채점까지 앱의 실제 경로를 그대로 탄다. 화면의 점수 칸에 점수가 뜨면 기록한다.
 *
 * [ContentSweepTest] 가 음원을 엔진에 직접 넣어 "엔진·자산"을 보는 것이라면, 이 테스트는
 * "스피커 → 마이크 → 앱 화면"까지 포함한 끝단 검증이다. 방 소음·볼륨에 영향을 받으므로
 * 점수는 참고값이고, 문장이 인식되지 않거나 점수가 안 나오는 것을 잡는 데 쓴다.
 *
 * 액티비티는 exported=false 라 shell 에서 못 열지만, 같은 패키지의 계측 테스트는 열 수 있다.
 * 학습 화면에서도 구매 권한을 검사한다. 유료 주제는 테스트 전에 구매 복원을 완료해야 한다.
 *
 * 실행 (연결된 단말, 미디어 볼륨 최대, 화면 켜짐 권장):
 *   adb shell am instrument -w -e class com.chisnova.mercury.asr.LoopbackSweepTest \
 *     -e sweepPrefix 1.basic com.chisnova.mercury.test/androidx.test.runner.AndroidJUnitRunner
 *   adb pull /sdcard/Android/data/com.chisnova.mercury/files/loopback_sweep.csv
 *
 * 문장당 10초 안팎이라 전체(4,476문장)는 십수 시간이다. sweepPrefix 로 나눠 돌리고,
 * CSV 에 이미 있는 문장은 건너뛰어 이어서 돌릴 수 있다(resume=false 로 끄면 다시 한다).
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class LoopbackSweepTest {

    @get:Rule
    val timeout: Timeout = Timeout(20, TimeUnit.HOURS)

    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private val context get() = instrumentation.targetContext

    private data class Row(
        val topic: String,
        val index: Int,             // 1부터
        val sentence: String,
        val scoreText: String,      // 화면 점수 칸 (예: "99.77점")
        val recognised: String,     // 화면 인식 결과 칸
        val status: String,         // ok | timeout | no_holder
        val elapsedMs: Long,
    )

    @Test
    fun loopbackAllContent() {
        val args = InstrumentationRegistry.getArguments()
        val prefix = args.getString("sweepPrefix") ?: ""
        // 디코더 시작 확인 대기(첫 문장은 마이크 열기까지 몇 초 걸린다) / 종료 확인 대기
        val startWaitMs = args.getString("startWaitMs")?.toLongOrNull() ?: 15_000L
        val maxWaitMs = args.getString("maxWaitMs")?.toLongOrNull() ?: 45_000L
        val resume = args.getString("resume")?.toBoolean() ?: true
        // 특정 문장만 다시 볼 때: "topic:index,topic:index,…" (index 는 1부터). 비우면 전체.
        val only: Set<Pair<String, Int>> = (args.getString("only") ?: "")
            .split(',').filter { it.contains(':') }
            .mapNotNull { s ->
                val i = s.lastIndexOf(':')
                s.substring(i + 1).toIntOrNull()?.let { s.substring(0, i) to it }
            }.toSet()

        var topics = findTopics(prefix)
        if (only.isNotEmpty()) {
            val wanted = only.map { it.first }.toSet()
            topics = topics.filter { it in wanted }
        }
        assertTrue("검사할 주제가 없다 — 접두사 '$prefix' 를 확인하라", topics.isNotEmpty())

        // 앱은 홈 화면에서 엔진을 미리 올리고, 연습 화면은 첫 [발음연습] 때 지연 초기화한다.
        // 계측 프로세스에는 홈이 없으므로 여기서 먼저 올려 두고 준비될 때까지 기다린다.
        instrumentation.runOnMainSync { AsrController.getInstance().initAsync(context) }
        val engineDeadline = System.currentTimeMillis() + 180_000
        while (!AsrController.getInstance().isReady()) {
            assertTrue("엔진이 3분 안에 준비되지 않았다", System.currentTimeMillis() < engineDeadline)
            Thread.sleep(500)
        }
        Log.i(TAG, "[loopback] 엔진 준비 완료")

        val csv = File(context.getExternalFilesDir(null), "loopback_sweep.csv")
        val done = if (resume) alreadyDone(csv) else emptySet()
        if (!resume || !csv.exists()) {
            csv.writeText("topic,index,sentence,score,recognised,status,elapsed_ms\n")
        }
        Log.i(TAG, "[loopback] 주제 ${topics.size}개, 이미 완료 ${done.size}문장, CSV=${csv.absolutePath}")

        var total = 0
        var okCount = 0
        val started = System.currentTimeMillis()
        for (topic in topics) {
            val name = topic.substringAfterLast('/')
            val assetBase = "$CONTENTS_DIR/$topic/$name"
            val lessonId = LessonData.getLevel1Categories().flatMap { it.lessons }
                .firstOrNull { ContentAccess.isFree(it.id, assetBase) }?.id ?: name
            instrumentation.runOnMainSync {
                assertTrue("$topic: 유료 주제는 전체 이용권 구매 복원 후 검사할 수 있다",
                    PurchaseManager.get(context).canAccess(lessonId, assetBase))
            }
            val intent = Intent(context, VoiceRecognitionActivity::class.java).apply {
                putExtra("asset_base", assetBase)
                putExtra("lesson_id", lessonId)
                putExtra("lesson_title", name)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            val scenario = ActivityScenario.launch<VoiceRecognitionActivity>(intent)
            try {
                val count = waitForList(scenario)
                if (count <= 0) {
                    Log.w(TAG, "[loopback] $topic: 문장 목록이 뜨지 않았다")
                    continue
                }
                // 잠금 화면(Bouncer)이나 알림창이 앞을 덮고 있으면 액티비티는 떠 있어도 프레임이
                // 그려지지 않는다. 그러면 스크롤·라벨 갱신이 전부 멈춰 no_holder/no_start 만 쌓인다 —
                // 가짜 결과를 남기지 말고 여기서 바로 멈춘다.
                assertTrue("화면이 그려지지 않는다 — 단말 잠금을 풀고 알림창을 닫은 뒤 다시 실행하라",
                    framesAreRendering(scenario))
                for (position in 0 until count) {
                    if ((topic to position + 1) in done) continue
                    if (only.isNotEmpty() && (topic to position + 1) !in only) continue
                    val row = runOne(scenario, topic, position, startWaitMs, maxWaitMs)
                    csv.appendText(row.toCsv())
                    total++
                    if (row.status == "ok") okCount++
                    Log.i(TAG, "[loopback] $topic:${position + 1} ${row.status} ${row.scoreText} " +
                        "'${row.recognised.take(60)}' (${row.elapsedMs}ms, 누적 $total)")
                }
            } finally {
                scenario.close()
                // 다음 주제로 넘어가기 전에 오디오 세션이 완전히 정리되게 잠시 둔다
                Thread.sleep(1500)
            }
            Log.i(TAG, "[loopback] $topic 완료 (누적 $total, ${(System.currentTimeMillis() - started) / 1000}초)")
        }
        Log.i(TAG, "[loopback] 끝 — 이번 실행 $total 문장, 채점 성공 $okCount")
    }

    /**
     * 문장 하나. 각 단계는 고정 대기가 아니라 화면 상태 전환으로 확인한다.
     *
     * 1. [발음연습] 클릭 → 버튼 라벨이 "완료" 가 될 때까지 기다린다 = 디코더(녹음) 시작 확인
     * 2. [예시듣기] 클릭 → TTS 재생
     * 3. 라벨이 "완료" 에서 돌아올 때까지 기다린다 = 디코더 종료 확인 → 점수·인식 결과 읽기
     */
    private fun runOne(
        scenario: ActivityScenario<VoiceRecognitionActivity>,
        topic: String, position: Int, startWaitMs: Long, maxWaitMs: Long,
    ): Row {
        val start = System.currentTimeMillis()

        val sentence = holderText(scenario, position) { it.engText.text.toString() }
            ?: return Row(topic, position + 1, "", "", "", "no_holder",
                System.currentTimeMillis() - start)

        // 1. 발음연습 → 디코더 시작 확인. 첫 문장은 마이크 열기까지 시간이 걸릴 수 있다.
        clickHolder(scenario, position) { it.btnPractice }
        if (!waitLabel(scenario, position, startWaitMs) { it == RECORDING }) {
            return Row(topic, position + 1, sentence, "", "", "no_start",
                System.currentTimeMillis() - start)
        }
        Thread.sleep(300)   // 마이크가 열린 직후 잡음 바닥이 잡히도록 잠깐 둔다

        // 2. 예시듣기 → 예시 음원이 스피커로 나가고, 그 소리가 마이크로 들어간다
        clickHolder(scenario, position) { it.btnPlay }

        // 3. 디코더 종료 확인: 무음 판정으로 녹음이 끝나면 라벨이 "완료" 에서 돌아온다.
        val ended = waitLabel(scenario, position, maxWaitMs) { it != RECORDING }
        val status: String
        if (!ended) {
            // 안 끝났다 — 같은 버튼(완료)으로 끊고 정리될 때까지 기다린다
            status = "timeout"
            clickHolder(scenario, position) { it.btnPractice }
            waitLabel(scenario, position, 10_000) { it != RECORDING }
        } else {
            status = "ok"
        }
        // 종료 직후 채점·화면 갱신이 한 박자 늦을 수 있다 — 인식 칸이 "인식 중…" 을 벗어날 때까지
        val deadline = System.currentTimeMillis() + 5_000
        var recognised = ""
        while (System.currentTimeMillis() < deadline) {
            recognised = holderText(scenario, position) { it.itemHintText.text.toString() } ?: ""
            if (recognised != RECOGNISING) break
            Thread.sleep(200)
        }
        val scoreText = holderText(scenario, position) { it.itemScore.text.toString() }
            ?.takeIf { SCORE_PATTERN.matches(it) } ?: ""
        return Row(topic, position + 1, sentence, scoreText, recognised, status,
            System.currentTimeMillis() - start)
    }

    /** 발음연습 버튼 라벨이 조건을 만족할 때까지 기다린다. 시간 안에 만족하면 true. */
    private fun waitLabel(
        scenario: ActivityScenario<VoiceRecognitionActivity>, position: Int,
        timeoutMs: Long, ok: (String) -> Boolean,
    ): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            val label = holderText(scenario, position) { it.btnPracticeLabel.text.toString() } ?: ""
            if (ok(label)) return true
            Thread.sleep(250)
        }
        return false
    }

    // ──────────────────────────────────────────────────────────────
    // 화면 조작 — 전부 UI 스레드에서, ViewHolder 는 쓸 때마다 다시 찾는다(재활용되므로)
    // ──────────────────────────────────────────────────────────────

    private fun recycler(activity: VoiceRecognitionActivity): RecyclerView =
        activity.findViewById(R.id.sentencesRecyclerView)

    /**
     * 화면이 실제로 그려지는지 — 레이아웃을 요청하고 다음 프레임 콜백이 3초 안에 오는지 본다.
     * 잠금 화면·알림창 뒤에 있으면 Choreographer 프레임이 오지 않는다.
     */
    private fun framesAreRendering(scenario: ActivityScenario<VoiceRecognitionActivity>): Boolean {
        val latch = java.util.concurrent.CountDownLatch(1)
        scenario.onActivity { activity ->
            val rv = recycler(activity)
            rv.requestLayout()
            android.view.Choreographer.getInstance().postFrameCallback { latch.countDown() }
        }
        return latch.await(3, TimeUnit.SECONDS)
    }

    /** 문장 목록이 뜰 때까지 기다린다. 문장 수를 돌려준다(0이면 실패). */
    private fun waitForList(scenario: ActivityScenario<VoiceRecognitionActivity>): Int {
        val deadline = System.currentTimeMillis() + 30_000
        var count = 0
        while (System.currentTimeMillis() < deadline) {
            scenario.onActivity { activity ->
                count = recycler(activity).adapter?.itemCount ?: 0
            }
            if (count > 0) {
                Thread.sleep(1000)   // 첫 바인딩이 끝나도록
                return count
            }
            Thread.sleep(500)
        }
        return 0
    }

    private fun <T> holderText(
        scenario: ActivityScenario<VoiceRecognitionActivity>, position: Int,
        read: (SentenceAdapter.SentenceViewHolder) -> T,
    ): T? {
        var result: T? = null
        repeat(6) { attempt ->
            scenario.onActivity { activity ->
                val rv = recycler(activity)
                val holder = rv.findViewHolderForAdapterPosition(position)
                    as? SentenceAdapter.SentenceViewHolder
                if (holder != null) {
                    result = read(holder)
                } else {
                    val lm = rv.layoutManager as? androidx.recyclerview.widget.LinearLayoutManager
                    Log.w(TAG, "[loopback] holder 없음 pos=$position attempt=$attempt " +
                        "children=${rv.childCount} h=${rv.height} items=${rv.adapter?.itemCount} " +
                        "visible=${lm?.findFirstVisibleItemPosition()}..${lm?.findLastVisibleItemPosition()} " +
                        "pendingLayout=${rv.isLayoutRequested} computing=${rv.isComputingLayout}")
                    rv.scrollToPosition(position)
                }
            }
            if (result != null) return result
            Thread.sleep(300L * (attempt + 1))
        }
        return result
    }

    private fun clickHolder(
        scenario: ActivityScenario<VoiceRecognitionActivity>, position: Int,
        button: (SentenceAdapter.SentenceViewHolder) -> android.view.View,
    ) {
        repeat(6) { attempt ->
            var clicked = false
            scenario.onActivity { activity ->
                val rv = recycler(activity)
                val holder = rv.findViewHolderForAdapterPosition(position)
                    as? SentenceAdapter.SentenceViewHolder
                if (holder != null) {
                    // performClick 은 클릭 효과음을 내고, 마이크가 그 소리를 발화로 잡아
                    // 예시 음원이 나오기 전에 무음 판정으로 끝나 버린다. 리스너만 부른다.
                    button(holder).callOnClick()
                    clicked = true
                } else {
                    rv.scrollToPosition(position)
                }
            }
            if (clicked) return
            Thread.sleep(300L * (attempt + 1))
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 주제 열거·CSV
    // ──────────────────────────────────────────────────────────────

    /** 주제 = <이름>.jsonl 을 가진 폴더의 contents 기준 상대 경로 (ContentSweepTest 와 같다). */
    private fun findTopics(prefix: String): List<String> {
        val found = mutableListOf<String>()
        fun walk(relative: String) {
            val entries = AsrTestSupport.assetNames(
                if (relative.isEmpty()) CONTENTS_DIR else "$CONTENTS_DIR/$relative")
            val name = relative.substringAfterLast('/')
            if (entries.contains("$name.jsonl")) {
                if (prefix.isEmpty() || relative.startsWith(prefix)) found.add(relative)
                return
            }
            for (entry in entries) {
                val child = if (relative.isEmpty()) entry else "$relative/$entry"
                if (AsrTestSupport.assetNames("$CONTENTS_DIR/$child").isNotEmpty()) walk(child)
            }
        }
        walk("")
        return found.sorted()
    }

    private fun alreadyDone(csv: File): Set<Pair<String, Int>> {
        if (!csv.exists()) return emptySet()
        return csv.readLines().drop(1).mapNotNull { line ->
            val parts = line.split(',', limit = 3)
            val index = parts.getOrNull(1)?.toIntOrNull() ?: return@mapNotNull null
            parts[0] to index
        }.toSet()
    }

    private fun Row.toCsv(): String {
        fun q(s: String) = "\"" + s.replace("\"", "\"\"") + "\""
        return "$topic,$index,${q(sentence)},$scoreText,${q(recognised)},$status,$elapsedMs\n"
    }

    companion object {
        private const val TAG = "[AsrTest]"
        private const val CONTENTS_DIR = "contents"
        private val SCORE_PATTERN = Regex("""\d+(\.\d+)?점""")
        /** VoiceRecognitionActivity.startPracticeFor 가 인식 칸에 띄우는 문구. */
        private const val RECOGNISING = "인식 중…"
        /** SentenceAdapter.bindPracticeLabel — 녹음(디코딩) 중일 때의 발음연습 버튼 라벨. */
        private const val RECORDING = "완료"
    }
}
