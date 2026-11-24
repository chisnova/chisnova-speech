package com.chisnova.mercury.asr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import com.chisnova.mercury.support.AsrTestSupport
import com.chisnova.mercury.support.OpusAssets
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * APK 에 실린 콘텐츠 음원을 실제 엔진에 흘려 전수검사한다.
 *
 * 데스크톱 auditV2(contents/run_audit_v2.sh)가 점수 기준의 원천이고, 여기서 보는 것은
 * "실제 앱에 들어간 자산 + MediaCodec 디코딩 + JNI 브리지 + 단말 엔진"의 결합이다.
 * iOS `ContentSweepTests.swift` 와 같은 판정 기준을 쓴다.
 *
 * ## 판정
 * - 구조 문제(음원 수·JSONL 줄 수 불일치, 디코딩 실패, 빈 인식 결과, 점수 JSON 오류)는 실패.
 * - 문장 점수가 failBelow(기본 85) 미만이면 실패 — auditV2 반영 후 85점 미만은 0건이 기대치다.
 * - warnBelow(기본 90) 미만은 CSV 와 로그에만 남긴다.
 *
 * ## 실행 범위와 회수
 * 전체는 4,476문장이라 레벨 폴더로 샤딩할 수 있다:
 *
 *   ./gradlew connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.chisnova.mercury.asr.ContentSweepTest \
 *     -Pandroid.testInstrumentationRunnerArguments.sweepPrefix=1.basic
 *
 * 결과 CSV:
 *   adb pull /sdcard/Android/data/com.chisnova.mercury/files/content_sweep.csv
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class ContentSweepTest {

    @get:Rule
    val timeout: Timeout = Timeout(8, TimeUnit.HOURS)

    companion object {
        private const val TAG = "[AsrTest]"
        private const val CONTENTS_DIR = "contents"

        @BeforeClass
        @JvmStatic
        fun prepare() {
            AsrTestSupport.engine()
        }
    }

    @Before
    fun setUp() {
        AsrTestSupport.reset()
    }

    @After
    fun tearDown() {
        AsrTestSupport.reset()
    }

    /** 한 문장의 검사 결과 한 줄. */
    private data class Row(
        val topic: String,
        val index: Int,          // JSONL 줄 번호(0부터) = 음원 번호
        val file: String,
        val status: String,      // ok | decode_fail | empty_result | score_fail
        val avgScore: Double?,
        val minWordScore: Double?,
        val recognised: String
    )

    @Test
    fun sweepAllContentAudio() {
        val args = InstrumentationRegistry.getArguments()
        val prefix = args.getString("sweepPrefix") ?: ""
        val failBelow = args.getString("failBelow")?.toDoubleOrNull() ?: 85.0
        val warnBelow = args.getString("warnBelow")?.toDoubleOrNull() ?: 90.0

        val topics = findTopics(prefix)
        assertTrue("검사할 주제가 없다 — 접두사 '$prefix' 와 APK 자산을 확인하라", topics.isNotEmpty())

        val rows = mutableListOf<Row>()
        val structural = mutableListOf<String>()
        val started = System.currentTimeMillis()

        for (topic in topics) {
            val sentences = readJsonl("$CONTENTS_DIR/$topic/${topic.substringAfterLast('/')}.jsonl")
            val opusFiles = AsrTestSupport.assetNames("$CONTENTS_DIR/$topic/opus")
                .filter { it.endsWith(".opus") }.sorted()
            if (sentences.size != opusFiles.size) {
                structural.add("$topic: JSONL ${sentences.size}줄 vs 음원 ${opusFiles.size}개")
                continue
            }
            sentences.forEachIndexed { index, sentence ->
                rows.add(sweepOne(topic, index, opusFiles[index], sentence))
            }
            Log.i(TAG, "[sweep] $topic ${sentences.size}문장 완료 (누적 ${rows.size}, " +
                "${(System.currentTimeMillis() - started) / 1000}초)")
        }

        writeCsv(rows)

        val broken = rows.filter { it.status != "ok" }
        val low = rows.filter { it.status == "ok" && (it.avgScore ?: 0.0) < failBelow }
        val warned = rows.filter {
            it.status == "ok" && (it.avgScore ?: 0.0) >= failBelow && (it.avgScore ?: 0.0) < warnBelow
        }
        Log.i(TAG, "[sweep] 총 ${rows.size}문장 — 구조 문제 ${structural.size}, 오류 ${broken.size}, " +
            "${failBelow.toInt()}점 미만 ${low.size}, ${warnBelow.toInt()}점 미만(경고) ${warned.size}")
        for (warn in warned) {
            Log.i(TAG, "[sweep] 경고 ${warn.topic}:${warn.index + 1} %.2f점".format(warn.avgScore))
        }

        assertTrue("구조 문제: ${structural.joinToString(" / ")}", structural.isEmpty())
        assertTrue("검사 오류 ${broken.size}건: " + broken.take(10)
            .joinToString(", ") { "${it.topic}:${it.index + 1}=${it.status}" }, broken.isEmpty())
        assertTrue("${failBelow.toInt()}점 미만 ${low.size}건: " + low.take(10)
            .joinToString(", ") { "%s:%d=%.2f".format(it.topic, it.index + 1, it.avgScore) },
            low.isEmpty())
    }

    /** 한 문장을 엔진에 흘리고 채점까지 마친다. */
    private fun sweepOne(topic: String, index: Int, opusName: String,
                         sentence: AsrTestSupport.PronounceMeta): Row {
        val samples = try {
            OpusAssets.readOpusAsset("$CONTENTS_DIR/$topic/opus/$opusName")
        } catch (e: Exception) {
            Log.w(TAG, "[sweep] $topic:$opusName 디코딩 실패: $e")
            null
        }
        if (samples == null || samples.isEmpty()) {
            return Row(topic, index, opusName, "decode_fail", null, null, "")
        }

        // 앱 마이크 경로(AudioController)와 같은 순서로 보낸다: 데이터 청크는 전부
        // isFinal=false, 끝나면 빈 최종 청크. 빔도 앱의 BEAM_SIZE=1 을 쓴다.
        val api = AsrTestSupport.engine()
        api.setCallBack { }
        api.startBackGroundEngine(1)
        AsrTestSupport.stream(api, samples, markFinal = false)
        api.sendAudio(FloatArray(0), 0, true)
        api.finishDecoding()
        // 문장별 점수 비중도 앱과 같이 채점 직전에 지정한다(없으면 기본으로 되돌린다).
        val weights = sentence.scoreWeights
        if (weights != null) {
            api.setSentenceScoreWeights(weights.first, weights.second)
        } else {
            api.setSentenceScoreWeights(-1f, -1f)
        }
        val ctc = api.ctcAsrResult ?: ""

        val row = try {
            if (ctc.isBlank()) {
                Row(topic, index, opusName, "empty_result", null, null, "")
            } else {
                score(topic, index, opusName, sentence, ctc)
            }
        } finally {
            AsrTestSupport.reset()
        }
        return row
    }

    private fun score(topic: String, index: Int, opusName: String,
                      sentence: AsrTestSupport.PronounceMeta, recognised: String): Row {
        val json = AsrTestSupport.pronounceScore(sentence)
        val root = try {
            JSONObject(json)
        } catch (_: Exception) {
            return Row(topic, index, opusName, "score_fail", null, null, recognised)
        }
        if (!root.has("avg_score") || !root.getDouble("avg_score").isFinite()) {
            return Row(topic, index, opusName, "score_fail", null, null, recognised)
        }

        // 채점 제외(graded=false) 단어는 최저 단어 점수에서 뺀다 — 앱 화면과 같은 기준.
        var minWord: Double? = null
        val words = root.optJSONArray("final_score")
        if (words != null) {
            for (i in 0 until words.length()) {
                val word = words.getJSONObject(i)
                if (!word.optBoolean("graded", true)) continue
                val score = word.optDouble("score")
                if (score.isFinite() && (minWord == null || score < minWord!!)) minWord = score
            }
        }
        return Row(topic, index, opusName, "ok", root.getDouble("avg_score"), minWord, recognised)
    }

    /** 주제 = <이름>.jsonl 을 가진 폴더의 contents 기준 상대 경로. */
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
            // 디렉터리만 내려간다. 레벨 폴더 이름에 점이 있어(1.basic) 파일 판별에
            // 확장자를 쓸 수 없으므로, 하위 항목이 있으면 디렉터리로 본다.
            for (entry in entries) {
                val child = if (relative.isEmpty()) entry else "$relative/$entry"
                if (AsrTestSupport.assetNames("$CONTENTS_DIR/$child").isNotEmpty()) {
                    walk(child)
                }
            }
        }
        walk("")
        return found.sorted()
    }

    private fun readJsonl(assetPath: String): List<AsrTestSupport.PronounceMeta> =
        AsrTestSupport.assets.open(assetPath).bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() }.map { line ->
                val record = JSONObject(line)
                fun ints(key: String): IntArray {
                    val array = record.getJSONArray(key)
                    return IntArray(array.length()) { array.getInt(it) }
                }
                val org = record.getJSONArray("org")
                val sw = record.optJSONObject("score_weights")
                val weights = if (sw != null && sw.has("pronunciation") && sw.has("recognition")) {
                    Pair(sw.getDouble("pronunciation").toFloat(),
                         sw.getDouble("recognition").toFloat())
                } else null
                AsrTestSupport.PronounceMeta(
                    label = (0 until org.length()).joinToString(" ") { org.getString(it) },
                    wordIdx = ints("word_idx"),
                    wordGroup = ints("word_idx_group"),
                    phoneIdx = ints("phn_idx"),
                    phoneGroup = ints("phn_idx_group"),
                    scoreWeights = weights
                )
            }.toList()
        }

    private fun writeCsv(rows: List<Row>) {
        val out = File(AsrTestSupport.context.getExternalFilesDir(null), "content_sweep.csv")
        out.printWriter().use { writer ->
            writer.println("topic,index,file,status,avg_score,min_word_score,recognised")
            for (row in rows) {
                val avg = row.avgScore?.let { "%.2f".format(it) } ?: ""
                val minWord = row.minWordScore?.let { "%.2f".format(it) } ?: ""
                val recognised = row.recognised.replace("\"", "\"\"")
                writer.println("${row.topic},${row.index + 1},${row.file},${row.status}," +
                    "$avg,$minWord,\"$recognised\"")
            }
        }
        Log.i(TAG, "[sweep] CSV: ${out.absolutePath}")
    }
}
