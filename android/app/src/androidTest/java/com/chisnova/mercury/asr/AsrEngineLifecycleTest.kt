package com.chisnova.mercury.asr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import com.chisnova.mercury.jni.AsrAPI
import com.chisnova.mercury.support.AsrTestSupport
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
 * 네이티브 상태 머신(Idle → Init → Running → Finishing)의 계약 시험.
 *
 * 앱은 화면 전환·백그라운드 진입·연속 연습 때문에 API 를 순서대로만 부르지 않는다.
 * 허용되지 않은 순서의 호출은 "무시되고 로그만 남는다"가 계약이므로,
 * 그 결과 엔진이 못 쓰게 되거나 값이 새어 나오지 않는지 확인한다.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class AsrEngineLifecycleTest {

    @get:Rule
    val timeout: Timeout = Timeout(5, TimeUnit.MINUTES)

    private lateinit var api: AsrAPI

    companion object {
        private const val TAG = "[AsrTest]"

        /**
         * 같은 오디오를 다시 넣었을 때 허용하는 결과 차이(WER).
         * 청크 경계가 타이밍에 따라 달라지므로 완전 일치는 보장되지 않는다.
         */
        private const val SESSION_DRIFT_BUDGET = 0.34
        private lateinit var shortUtterance: FloatArray

        @BeforeClass
        @JvmStatic
        fun prepare() {
            // 가장 짧은 기준 발화 — 상태 전이만 확인하면 되므로 굳이 긴 문장을 쓰지 않는다.
            shortUtterance = AsrTestSupport.readWavAsset("google_tts003.wav")
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

    // ---------- 세션 시작 전(InitState) ----------

    @Test
    fun resultIsEmptyBeforeAnySession() {
        assertEquals("", api.ctcAsrResult)
    }

    @Test
    fun pronounceScoreIsEmptyBeforeAnySession() {
        val meta = AsrTestSupport.readPronounceMeta()
        assertEquals("", AsrTestSupport.pronounceScore(meta))
    }

    /** 녹음 스레드가 세션 시작보다 먼저 오디오를 밀어 넣는 경합 — 무시되고 이후 세션은 정상이어야 한다. */
    @Test
    fun audioSentBeforeStartIsDroppedAndEngineStaysUsable() {
        AsrTestSupport.stream(api, shortUtterance.copyOf(16000), markFinal = false)
        assertEquals("세션 밖 오디오가 결과로 새어 나왔다", "", api.ctcAsrResult)

        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        assertTrue("세션 밖 오디오 이후 정상 세션이 실패했다", decoded.ctc.isNotBlank())
    }

    @Test
    fun finishWithoutSessionIsIgnored() {
        api.finishDecoding()
        assertEquals("", api.ctcAsrResult)
        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        assertTrue(decoded.ctc.isNotBlank())
    }

    /** 화면을 벗어날 때 항상 호출되는 경로 — 세션이 없어도 안전해야 한다. */
    @Test
    fun cancelWithoutSessionIsIgnored() {
        api.cancelDecoding()
        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        assertTrue(decoded.ctc.isNotBlank())
    }

    @Test
    fun clearWithoutSessionIsIgnored() {
        api.clear()
        api.clear()
        assertEquals("", api.ctcAsrResult)
    }

    // ---------- 세션 진행 중(RunningState) ----------

    @Test
    fun resultIsWithheldWhileSessionIsRunning() {
        val split = shortUtterance.size / 2
        api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
        AsrTestSupport.stream(api, shortUtterance.copyOf(split), markFinal = false)
        assertEquals("종료 전에 확정 결과가 노출됐다", "", api.ctcAsrResult)
        assertEquals(
            "종료 전에 발음 점수가 노출됐다",
            "",
            AsrTestSupport.pronounceScore(AsrTestSupport.readPronounceMeta())
        )

        AsrTestSupport.stream(api, shortUtterance.copyOfRange(split, shortUtterance.size))
        api.finishDecoding()
        assertNotNull(api.ctcAsrResult)
    }

    /** UI 이벤트 중복(더블 탭)으로 start 가 두 번 불릴 수 있다. 두 번째는 무시돼야 한다. */
    @Test
    fun duplicateStartIsIgnored() {
        api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
        api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
        AsrTestSupport.stream(api, shortUtterance)
        api.finishDecoding()
        assertTrue("중복 start 이후 결과가 비었다", api.ctcAsrResult.isNotBlank())
    }

    // ---------- 세션 종료 후(FinishingState) ----------

    @Test
    fun audioSentAfterFinishDoesNotChangeResult() {
        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        AsrTestSupport.stream(api, shortUtterance)
        assertEquals("종료 후 오디오가 결과를 바꿨다", decoded.ctc, api.ctcAsrResult)
    }

    @Test
    fun finishTwiceKeepsResult() {
        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        api.finishDecoding()
        assertEquals(decoded.ctc, api.ctcAsrResult)
    }

    /** 결과 조회는 여러 번 해도 같은 값이어야 한다(UI 가 재조회하는 경로). */
    @Test
    fun resultIsStableAcrossRepeatedReads() {
        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        repeat(3) { assertEquals(decoded.ctc, api.ctcAsrResult) }
    }

    @Test
    fun clearDiscardsPreviousResult() {
        val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        assertTrue(decoded.ctc.isNotBlank())
        api.clear()
        assertEquals("clear 후에도 이전 결과가 남아 있다", "", api.ctcAsrResult)
    }

    /**
     * 연속 연습 경로: 종료 → clear → 새 세션.
     * 이전 세션의 오디오나 빔이 남아 결과가 누적되면 안 된다.
     */
    @Test
    fun sessionsAreIndependentAfterClear() {
        val first = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        AsrTestSupport.reset()
        val second = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
        Log.i(TAG, "session1='${first.ctc}' session2='${second.ctc}'")
        assertTrue("두 번째 세션 결과가 비었다", second.ctc.isNotBlank())
        val drift = AsrTestSupport.wordErrorRate(first.ctc, second.ctc)
        assertTrue(
            "clear 후 새 세션 결과가 크게 다르다 (drift=%.3f): '%s' vs '%s'".format(drift, first.ctc, second.ctc),
            drift <= SESSION_DRIFT_BUDGET
        )
    }

    /**
     * 반복 세션의 결과가 같은 발화를 가리키는지 본다.
     *
     * 글자까지 같기를 요구하지는 않는다. sendAudio 는 3초가 쌓이면 엔진을 깨우지만 엔진이
     * 처리 중이면 오디오가 계속 쌓였다가 한꺼번에 넘어가므로, 실제 청크 경계가 CPU 타이밍에
     * 따라 달라진다. 청크 크기는 스트리밍 attention 의 right context 이기도 해서
     * (cpp/module/conformer_stream_head.cc — 청크 내부는 마스킹 없는 양방향) 결과가 조금씩
     * 흔들린다. 여기서 잡으려는 것은 그 흔들림이 아니라 세션 간 상태가 새는 경우다.
     */
    @Test
    fun repeatedSessionsRemainStable() {
        val results = (1..3).map {
            val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
            AsrTestSupport.reset()
            decoded.ctc
        }
        Log.i(TAG, "repeated results=$results")
        assertTrue("반복 세션 결과가 비었다", results.all { it.isNotBlank() })
        val drift = results.drop(1).map { AsrTestSupport.wordErrorRate(results[0], it) }
        Log.i(TAG, "repeat drift=$drift")
        assertTrue(
            "반복 세션에서 결과가 크게 흔들린다 (drift=$drift): $results",
            drift.all { it <= SESSION_DRIFT_BUDGET }
        )
    }

    /** 콜백은 세션마다 다시 등록된다 — 이전 세션 콜백이 남아 두 번 호출되면 안 된다. */
    @Test
    fun callbackIsDeliveredForEverySession() {
        repeat(2) {
            val decoded = AsrTestSupport.decode(shortUtterance, withSecondPass = false)
            assertTrue("부분 결과 콜백이 한 번도 오지 않았다", decoded.partials.isNotEmpty())
            AsrTestSupport.reset()
        }
    }
}
