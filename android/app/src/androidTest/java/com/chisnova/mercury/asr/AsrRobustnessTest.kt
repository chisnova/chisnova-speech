package com.chisnova.mercury.asr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import com.chisnova.mercury.jni.AsrAPI
import com.chisnova.mercury.support.AsrTestSupport
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import org.junit.runner.RunWith
import java.util.concurrent.TimeUnit

/**
 * 비정상 입력에 대한 방어 시험.
 *
 * 마이크 경로에서는 길이 0 버퍼, 완전한 무음, 클리핑된 샘플, 심지어 NaN 이 섞인 버퍼까지
 * 실제로 들어온다. 특징 추출과 빔 서치는 이런 값에서 조용히 NaN 을 퍼뜨리거나
 * 빈 빔을 만들어 네이티브에서 죽을 수 있으므로, 각 케이스마다
 * (1) 죽지 않는지 (2) 다음 세션이 멀쩡한지 두 가지를 함께 본다.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class AsrRobustnessTest {

    @get:Rule
    val timeout: Timeout = Timeout(5, TimeUnit.MINUTES)

    private lateinit var api: AsrAPI

    companion object {
        private const val TAG = "[AsrTest]"
        private lateinit var reference: FloatArray

        @BeforeClass
        @JvmStatic
        fun prepare() {
            reference = AsrTestSupport.readWavAsset("google_tts003.wav")
            AsrTestSupport.engine()
        }
    }

    @Before
    fun setUp() {
        api = AsrTestSupport.engine()
        AsrTestSupport.reset()
    }

    @After
    fun tearDown() {
        AsrTestSupport.reset()
    }

    /** 오디오를 한 샘플도 넣지 않고 끝낸 세션 — 앱이 녹음 실패 시 그대로 이 경로를 탄다. */
    @Test
    fun emptyFinalChunkYieldsEmptyResult() {
        val decoded = AsrTestSupport.decode(FloatArray(0), withSecondPass = false)
        assertEquals("빈 세션에서 텍스트가 나왔다", "", AsrTestSupport.normalize(decoded.ctc))
    }

    @Test
    fun silenceProducesNoSpuriousText() {
        val decoded = AsrTestSupport.decode(AsrTestSupport.silence(1500), withSecondPass = false)
        Log.i(TAG, "silence -> '${decoded.ctc}'")
        assertTrue(
            "무음에서 단어가 만들어졌다: '${decoded.ctc}'",
            AsrTestSupport.words(decoded.ctc).isEmpty()
        )
    }

    /** 저진폭 잡음에서 문장이 통째로 생성되면(환청) 사용자에게 엉뚱한 점수가 나간다. */
    @Test
    fun lowLevelNoiseDoesNotHallucinateSentence() {
        val decoded = AsrTestSupport.decode(AsrTestSupport.noise(1500), withSecondPass = false)
        val words = AsrTestSupport.words(decoded.ctc)
        Log.i(TAG, "noise -> '${decoded.ctc}' (${words.size} words)")
        assertTrue("잡음에서 ${words.size}개 단어가 생성됐다: '${decoded.ctc}'", words.size <= 3)
    }

    /** 한 프레임도 채우지 못하는 길이 — 특징 추출이 음수 길이 프레임을 만들지 않아야 한다. */
    @Test
    fun subFrameAudioIsHandled() {
        val decoded = AsrTestSupport.decode(FloatArray(10) { 0.01f }, withSecondPass = false)
        assertNotNull(decoded.ctc)
    }

    @Test
    fun singleSampleAudioIsHandled() {
        val decoded = AsrTestSupport.decode(FloatArray(1) { 0.5f }, withSecondPass = false)
        assertNotNull(decoded.ctc)
    }

    /** 청크 크기가 앱과 달라도(버퍼 정책 변경) 결과가 흔들리면 안 된다. */
    @Test
    fun chunkSizeDoesNotChangeResult() {
        val small = AsrTestSupport.decode(reference, chunk = 160, withSecondPass = false).ctc
        AsrTestSupport.reset()
        val large = AsrTestSupport.decode(reference, chunk = 16000, withSecondPass = false).ctc
        Log.i(TAG, "chunk160='$small' chunk16000='$large'")
        assertEquals("청크 크기에 따라 결과가 달라진다", small, large)
    }

    /** 엔진 내부 버퍼 임계(3초)를 한 번에 넘기는 큰 청크. */
    @Test
    fun chunkLargerThanInternalBufferThresholdIsHandled() {
        val decoded = AsrTestSupport.decode(reference, chunk = 16000 * 4, withSecondPass = false)
        assertTrue(decoded.ctc.isNotBlank())
    }

    @Test
    fun clippedSamplesDoNotBreakEngine() {
        val clipped = FloatArray(reference.size) { i -> reference[i] * 8f }
        val decoded = AsrTestSupport.decode(clipped, withSecondPass = false)
        Log.i(TAG, "clipped -> '${decoded.ctc}'")
        assertNotNull(decoded.ctc)
        assertNextSessionIsHealthy()
    }

    /** 마이크 DC 오프셋 고장 — 상수 입력에서 로그 스펙트럼이 -inf 가 되기 쉽다. */
    @Test
    fun constantDcInputDoesNotBreakEngine() {
        val decoded = AsrTestSupport.decode(FloatArray(16000) { 1.0f }, withSecondPass = false)
        assertNotNull(decoded.ctc)
        assertNextSessionIsHealthy()
    }

    /**
     * NaN/Inf 가 섞인 버퍼.
     * 방어가 없으면 빔 점수가 전부 NaN 이 되어 정렬이 무너지고, 발음 점수 JSON 에
     * "nan" 이 그대로 찍혀 파싱 불가능한 문자열이 UI 로 흘러간다.
     */
    @Test
    fun nonFiniteSamplesAreContained() {
        val poisoned = reference.copyOf()
        poisoned[100] = Float.NaN
        poisoned[200] = Float.POSITIVE_INFINITY
        poisoned[300] = Float.NEGATIVE_INFINITY
        val decoded = AsrTestSupport.decode(poisoned, withSecondPass = false)
        Log.i(TAG, "non-finite -> '${decoded.ctc}'")
        assertNotNull(decoded.ctc)
        assertNextSessionIsHealthy()
    }

    // ---------- 발음 점수 ----------
    //
    // 라벨만 넘기는 API 는 네이티브 구현이 없어 제거됐다. 채점 인자는 콘텐츠 .jsonl 이 주므로
    // 여기서는 인자를 그대로 두고 "어떤 오디오가 들어와도 점수 JSON 이 온전한가"를 본다.
    // 라벨과 인덱스가 어긋난 조합은 네이티브가 배열 범위를 넘으므로 AsrKnownHazardTest 로 옮겼다.

    @Test
    fun pronounceScoreIsWellFormedJson() {
        AsrTestSupport.decode(reference, withSecondPass = false)
        val json = AsrTestSupport.pronounceScore(AsrTestSupport.readPronounceMeta())
        Log.i(TAG, "score json head=${json.take(120)}")
        assertScoreJsonIsFinite(json)
    }

    /** 무음만 들어온 뒤의 채점 — 정렬할 게 없어도 NaN 이 새어 나오면 안 된다. */
    @Test
    fun pronounceScoreAfterSilenceIsFinite() {
        AsrTestSupport.decode(AsrTestSupport.silence(1500), withSecondPass = false)
        val json = AsrTestSupport.pronounceScore(AsrTestSupport.readPronounceMeta())
        Log.i(TAG, "silence score head=${json.take(120)}")
        assertScoreJsonIsFinite(json)
    }

    /** 잡음만 들어온 뒤의 채점. */
    @Test
    fun pronounceScoreAfterNoiseIsFinite() {
        AsrTestSupport.decode(AsrTestSupport.noise(1500), withSecondPass = false)
        assertScoreJsonIsFinite(AsrTestSupport.pronounceScore(AsrTestSupport.readPronounceMeta()))
    }

    /** NaN/Inf 가 섞인 오디오 뒤의 채점 — 점수까지 오염되면 UI 로 깨진 문자열이 간다. */
    @Test
    fun pronounceScoreAfterNonFiniteAudioIsFinite() {
        val poisoned = reference.copyOf()
        poisoned[100] = Float.NaN
        poisoned[200] = Float.POSITIVE_INFINITY
        AsrTestSupport.decode(poisoned, withSecondPass = false)
        assertScoreJsonIsFinite(AsrTestSupport.pronounceScore(AsrTestSupport.readPronounceMeta()))
    }

    // ---------- 헬퍼 ----------

    private fun assertNextSessionIsHealthy() {
        AsrTestSupport.reset()
        val decoded = AsrTestSupport.decode(reference, withSecondPass = false)
        assertTrue("비정상 입력 이후 정상 오디오가 인식되지 않는다", decoded.ctc.isNotBlank())
    }

    private fun assertScoreJsonIsFinite(json: String) {
        assertTrue("점수 JSON 이 비어 있다", json.isNotBlank())
        assertTrue(
            "JSON 에 nan/inf 가 그대로 직렬화됐다: ${json.take(160)}",
            !json.contains("nan", ignoreCase = true) && !json.contains("inf", ignoreCase = true)
        )
        val root = JSONObject(json)   // 파싱 실패는 곧 UI 로 깨진 문자열이 간다는 뜻이다
        val avg = root.getDouble("avg_score")
        assertTrue("avg_score 가 유한하지 않다: $avg", avg.isFinite())
        val words = root.getJSONArray("final_score")
        for (i in 0 until words.length()) {
            val word = words.getJSONObject(i)
            assertTrue("단어 점수가 유한하지 않다", word.getDouble("score").isFinite())
            val phones = word.getJSONArray("phone_score")
            for (j in 0 until phones.length()) {
                assertTrue(
                    "음소 점수가 유한하지 않다",
                    phones.getJSONObject(j).getDouble("score").isFinite()
                )
            }
        }
    }
}
