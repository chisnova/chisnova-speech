package com.chisnova.mercury.asr

import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.chisnova.mercury.support.AsrTestSupport
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 세션을 반복해도 메모리가 계속 늘지 않고, 비정상 호출에도 죽지 않는지 본다.
 *
 * 이 앱의 메모리는 대부분 네이티브다(실측 PSS 445MB 중 네이티브 힙 365MB). 그래서
 * 자바 힙만 보는 도구로는 누수를 잡을 수 없고 [Debug.getNativeHeapAllocatedSize] 를 본다.
 *
 * ## 어떻게 누수라고 판정하는가
 *
 * "N 회 돌린 뒤 메모리가 늘었다" 는 판정 기준이 될 수 없다. 네이티브 할당자는 해제한
 * 블록을 OS 에 곧바로 돌려주지 않고, 첫 세션은 캐시·버퍼를 채우느라 원래 늘어난다.
 * 그래서 **구간을 둘로 나눠 증가 속도를 비교한다.** 진짜 누수라면 뒤 구간도 앞 구간과
 * 비슷하게 계속 늘고, 단순한 예열이라면 뒤 구간은 평평해진다.
 *
 * 실패하면 실제 수치를 메시지에 담는다 — 임계를 넘겼다는 사실만으로는 고칠 수가 없다.
 */
@RunWith(AndroidJUnit4::class)
class AsrMemoryStressTest {

    companion object {
        /** 예열 구간. 이 동안의 증가는 누수로 세지 않는다. */
        private const val WARMUP_ROUNDS = 3

        /** 비교할 두 구간 각각의 반복 횟수. */
        private const val ROUNDS_PER_PHASE = 8

        /**
         * 뒤 구간 증가량이 앞 구간의 이 배수를 넘으면 누수로 본다.
         *
         * 1.0 이 아니라 여유를 두는 이유: 할당자 단편화와 다른 프로세스의 영향으로
         * 측정에 잡음이 섞인다. 누수는 배수로 커지므로 이 정도로도 충분히 잡힌다.
         */
        private const val GROWTH_RATIO_LIMIT = 1.6

        /**
         * 뒤 구간 증가가 이보다 작으면 비율을 따지지 않고 통과시킨다.
         * 측정 잡음(수백 KB)이 작은 절대값에서 비율을 크게 흔들기 때문이다.
         */
        private const val NOISE_FLOOR_BYTES = 4L * 1024 * 1024

        private lateinit var utterance: FloatArray

        @BeforeClass
        @JvmStatic
        fun prepare() {
            // 짧은 발화면 충분하다 — 여기서 보는 것은 인식 품질이 아니라 반복 후의 메모리다.
            utterance = AsrTestSupport.readWavAsset("google_tts003.wav")
            AsrTestSupport.engine()
        }
    }

    @Before
    fun setUp() = AsrTestSupport.reset()

    @After
    fun tearDown() = AsrTestSupport.reset()

    // ──────────────────────────────────────────────────────────────
    // 측정 도구
    // ──────────────────────────────────────────────────────────────

    /**
     * 지금 잡혀 있는 네이티브 힙 크기.
     *
     * 자바 힙의 참조가 남아 네이티브 해제가 미뤄지는 경우가 있어 GC 를 한 번 돌린다.
     * System.gc() 는 요청일 뿐 보장이 아니지만, 두 구간에 똑같이 적용하므로
     * 비교의 공정성은 유지된다.
     */
    private fun nativeHeapBytes(): Long {
        System.gc()
        System.runFinalization()
        Thread.sleep(120)
        return Debug.getNativeHeapAllocatedSize()
    }

    /**
     * [work] 를 [WARMUP_ROUNDS] + 두 구간만큼 돌리고, 두 구간의 증가량을 견준다.
     *
     * @param label 실패 메시지에 쓸 이름
     */
    private fun assertNoGrowth(label: String, work: (round: Int) -> Unit) {
        repeat(WARMUP_ROUNDS) { work(it) }

        val start = nativeHeapBytes()
        repeat(ROUNDS_PER_PHASE) { work(WARMUP_ROUNDS + it) }
        val mid = nativeHeapBytes()
        repeat(ROUNDS_PER_PHASE) { work(WARMUP_ROUNDS + ROUNDS_PER_PHASE + it) }
        val end = nativeHeapBytes()

        val firstPhase = mid - start
        val secondPhase = end - mid
        val detail = "$label — 예열 $WARMUP_ROUNDS 회 뒤 " +
            "1구간(${ROUNDS_PER_PHASE}회) ${firstPhase / 1024}KB, " +
            "2구간(${ROUNDS_PER_PHASE}회) ${secondPhase / 1024}KB " +
            "(기준 ${start / 1024 / 1024}MB -> ${end / 1024 / 1024}MB)"

        if (secondPhase <= NOISE_FLOOR_BYTES) return          // 사실상 늘지 않았다

        // 앞 구간이 줄었거나 0 이면 비율을 낼 수 없다 — 뒤 구간 절대값으로만 본다.
        if (firstPhase <= 0) {
            assertTrue(
                "$detail — 1구간은 늘지 않았는데 2구간만 늘었다",
                secondPhase <= NOISE_FLOOR_BYTES
            )
            return
        }
        val ratio = secondPhase.toDouble() / firstPhase.toDouble()
        assertTrue(
            "$detail, 증가비 ${"%.2f".format(ratio)} — 반복해도 줄지 않으면 누수다",
            ratio <= GROWTH_RATIO_LIMIT
        )
    }

    // ──────────────────────────────────────────────────────────────
    // 메모리
    // ──────────────────────────────────────────────────────────────

    /** 앱에서 가장 자주 도는 경로 — 문장 하나를 연습할 때마다 이 한 바퀴가 돈다. */
    @Test
    fun repeatedDecodeSessionsDoNotLeak() {
        assertNoGrowth("디코딩 세션 반복") {
            AsrTestSupport.decode(utterance, withSecondPass = false)
            AsrTestSupport.reset()
        }
    }

    /** 2패스 빔서치는 후보 큐를 크게 잡는다 — 세션마다 되돌려주는지 본다. */
    @Test
    fun repeatedSecondPassDoesNotLeak() {
        assertNoGrowth("2패스 포함 세션 반복") {
            AsrTestSupport.decode(utterance, withSecondPass = true)
            AsrTestSupport.reset()
        }
    }

    /** 발음 채점은 낱말·음소 배열을 새로 만든다. */
    @Test
    fun repeatedPronounceScoringDoesNotLeak() {
        val meta = AsrTestSupport.readPronounceMeta()
        assertNoGrowth("채점 반복") {
            AsrTestSupport.decode(utterance, withSecondPass = false)
            AsrTestSupport.pronounceScore(meta)
            AsrTestSupport.reset()
        }
    }

    /**
     * 사용자가 녹음을 중간에 끊는 경우. 끝까지 간 세션과 취소한 세션을 번갈아 돌린다 —
     * 취소 경로가 버퍼를 놓아주지 않으면 여기서만 늘어난다.
     */
    @Test
    fun interleavedCancelAndFinishDoNotLeak() {
        val api = AsrTestSupport.engine()
        assertNoGrowth("취소·완료 교대") { round ->
            if (round % 2 == 0) {
                AsrTestSupport.decode(utterance, withSecondPass = false)
            } else {
                api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
                AsrTestSupport.stream(api, utterance.copyOf(utterance.size / 2), markFinal = false)
                api.cancelDecoding()
            }
            AsrTestSupport.reset()
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 크래시
    // ──────────────────────────────────────────────────────────────

    /**
     * 상태 머신을 어긋난 순서로 두드린다.
     *
     * 개별 오류 호출은 AsrEngineLifecycleTest 가 이미 하나씩 확인한다. 여기서는 그것들을
     * 여러 번 겹쳐 부른 뒤에도 엔진이 살아서 정상 세션을 처리하는지를 본다 — 실제 앱에서
     * 화면을 빠르게 오가면 이런 순서가 만들어진다.
     */
    @Test
    fun abnormalCallSequencesDoNotCrash() {
        val api = AsrTestSupport.engine()
        repeat(30) { round ->
            when (round % 6) {
                0 -> api.finishDecoding()                                  // 세션 없이 종료
                1 -> api.cancelDecoding()                                  // 세션 없이 취소
                2 -> api.clear()                                           // 세션 없이 초기화
                3 -> {                                                     // 중복 시작
                    api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
                    api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
                    api.cancelDecoding()
                }
                4 -> AsrTestSupport.stream(api, utterance, markFinal = true)  // 세션 밖 오디오
                5 -> {                                                     // 시작하자마자 종료
                    api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
                    api.finishDecoding()
                }
            }
            api.clear()
        }

        val decoded = AsrTestSupport.decode(utterance, withSecondPass = false)
        assertTrue(
            "비정상 호출을 겪은 뒤 정상 세션이 결과를 내지 못했다",
            decoded.ctc.isNotBlank()
        )
    }

    /**
     * 빈 세션을 연달아 연다. 오디오가 한 번도 안 들어온 채 끝나면 네이티브가 2패스 인코딩을
     * 만들 수 없는데(앱도 hasRecordedAudio 로 막고 있다), 그 상태가 쌓여도 죽지 않아야 한다.
     */
    @Test
    fun repeatedEmptySessionsDoNotCrash() {
        val api = AsrTestSupport.engine()
        repeat(20) {
            api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
            AsrTestSupport.stream(api, FloatArray(0), markFinal = true)
            api.finishDecoding()
            api.clear()
        }
        val decoded = AsrTestSupport.decode(utterance, withSecondPass = false)
        assertTrue("빈 세션 반복 뒤 정상 세션이 실패했다", decoded.ctc.isNotBlank())
    }

    /** 무음만 넣는 세션 반복 — EPD 가 무음으로 끊었을 때 실제로 일어나는 경로다. */
    @Test
    fun repeatedSilenceSessionsDoNotLeak() {
        val silence = AsrTestSupport.silence(1000)
        assertNoGrowth("무음 세션 반복") {
            AsrTestSupport.decode(silence, withSecondPass = false)
            AsrTestSupport.reset()
        }
    }
}
