package com.chisnova.mercury.asr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import com.chisnova.mercury.jni.AsrAPI
import com.chisnova.mercury.support.AsrTestSupport
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import org.junit.runner.RunWith
import java.util.concurrent.TimeUnit

/**
 * 기준 발화에 대한 인식 품질 회귀 시험.
 *
 * 모델을 갈아끼우거나(assets/copy.sh) 디코더를 손보면 "죽지는 않지만 품질만 무너지는" 회귀가
 * 가장 잡기 어렵다. 기준 오디오 4개의 WER 과 발음 점수의 상대 순서로 그것을 잡는다.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class AsrDecodingRegressionTest {

    @get:Rule
    val timeout: Timeout = Timeout(15, TimeUnit.MINUTES)

    private lateinit var api: AsrAPI

    companion object {
        private const val TAG = "[AsrTest]"

        /**
         * 발화당 허용 WER. 모델 교체 시 로그의 실측값을 보고 조정한다.
         * 목적은 "완벽한 인식"이 아니라 "품질이 무너진 모델이 실려 나가는 것"을 막는 것이다.
         */
        private const val PER_UTTERANCE_WER_BUDGET = 0.50
        private const val AVERAGE_WER_BUDGET = 0.35

        /** 2패스 디코딩이 1패스보다 이만큼 넘게 나빠지면 회귀로 본다. */
        private const val SECOND_PASS_TOLERANCE = 0.10

        /** 마지막 중간 결과와 최종 결과의 허용 괴리(서로 다른 디코딩 경로라 완전 일치하지는 않는다). */
        private const val PARTIAL_CONVERGENCE_BUDGET = 0.50

        @BeforeClass
        @JvmStatic
        fun prepare() {
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

    @Test
    fun decodesReferenceUtterancesWithinWerBudget() {
        val werPerFile = linkedMapOf<String, Double>()
        for ((fileName, transcript) in AsrTestSupport.REFERENCE_TRANSCRIPTS) {
            val decoded = AsrTestSupport.decode(
                AsrTestSupport.readWavAsset(fileName),
                withSecondPass = false
            )
            AsrTestSupport.reset()
            val wer = AsrTestSupport.wordErrorRate(transcript, decoded.ctc)
            werPerFile[fileName] = wer
            Log.i(TAG, "$fileName wer=%.3f hyp='%s'".format(wer, decoded.ctc))
        }

        val failures = werPerFile.filterValues { it > PER_UTTERANCE_WER_BUDGET }
        assertTrue("WER 예산을 넘긴 발화: $failures", failures.isEmpty())

        val average = werPerFile.values.average()
        Log.i(TAG, "average wer=%.3f".format(average))
        assertTrue("평균 WER %.3f 가 예산 %.2f 를 넘었다".format(average, AVERAGE_WER_BUDGET),
            average <= AVERAGE_WER_BUDGET)
    }

    /** 2패스는 1패스 결과를 다듬는 단계다. 더 나빠졌다면 빔/스코어링 회귀다. */
    @Test
    fun secondPassDoesNotRegressAgainstCtc() {
        val regressions = mutableListOf<String>()
        for ((fileName, transcript) in AsrTestSupport.REFERENCE_TRANSCRIPTS) {
            val decoded = AsrTestSupport.decode(AsrTestSupport.readWavAsset(fileName))
            AsrTestSupport.reset()
            val ctcWer = AsrTestSupport.wordErrorRate(transcript, decoded.ctc)
            val secondWer = AsrTestSupport.wordErrorRate(transcript, decoded.secondPass)
            Log.i(TAG, "$fileName ctc=%.3f second=%.3f second='%s'"
                .format(ctcWer, secondWer, decoded.secondPass))
            assertTrue("$fileName 의 2패스 결과가 비었다", decoded.secondPass.isNotBlank())
            if (secondWer > ctcWer + SECOND_PASS_TOLERANCE) {
                regressions.add("$fileName ctc=%.3f second=%.3f".format(ctcWer, secondWer))
            }
        }
        assertTrue("2패스가 1패스보다 나빠진 발화: $regressions", regressions.isEmpty())
    }

    /**
     * 콜백 계약: 스트리밍 중에는 1패스 CTC(ctc_linear1) 중간 결과가 흐르고,
     * 마지막 청크를 처리할 때 2패스 CTC(ctc_linear2) 최종 결과가 한 번 더 온다.
     * 따라서 마지막 콜백은 getCtcAsrResult() 와 같은 값이어야 한다.
     */
    @Test
    fun lastPartialIsTheFinalCtcResult() {
        val decoded = AsrTestSupport.decode(
            AsrTestSupport.readWavAsset("google_tts001.wav"),
            withSecondPass = false
        )
        assertTrue("중간 결과 콜백이 없었다", decoded.partials.isNotEmpty())
        val last = decoded.partials.last()
        Log.i(TAG, "partials=${decoded.partials.size} last='$last' final='${decoded.ctc}'")
        assertTrue("최종 결과가 비었다", decoded.ctc.isNotBlank())
        assertEquals("마지막 콜백이 2패스 CTC 최종 결과와 다르다", decoded.ctc, last)
    }

    /** 중간 결과는 1패스라 최종과 완전히 같지는 않지만, 같은 발화를 가리켜야 한다. */
    @Test
    fun intermediatePartialsTrackTheSameUtterance() {
        val decoded = AsrTestSupport.decode(
            AsrTestSupport.readWavAsset("google_tts001.wav"),
            withSecondPass = false
        )
        val intermediates = decoded.partials.dropLast(1).filter { it.isNotBlank() }
        assertTrue("1패스 중간 결과가 하나도 없었다", intermediates.isNotEmpty())
        val lastIntermediate = intermediates.last()
        val divergence = AsrTestSupport.wordErrorRate(decoded.ctc, lastIntermediate)
        Log.i(TAG, "intermediates=${intermediates.size} last1pass='$lastIntermediate' wer=%.3f".format(divergence))
        assertTrue(
            "마지막 1패스 결과가 최종과 너무 다르다 (%.3f): '%s' vs '%s'"
                .format(divergence, lastIntermediate, decoded.ctc),
            divergence <= PARTIAL_CONVERGENCE_BUDGET
        )
    }

    /** 특수 토큰이 그대로 새어 나오면 화면에 <pad>/<s> 가 그대로 보인다. */
    @Test
    fun resultsCarryNoSpecialTokens() {
        val decoded = AsrTestSupport.decode(AsrTestSupport.readWavAsset("google_tts003.wav"))
        for ((name, text) in listOf(
            "ctc" to decoded.ctc,
            "secondPass" to decoded.secondPass,
            "partial" to decoded.partials.joinToString(" ")
        )) {
            for (token in listOf("<pad>", "<unk>", "<s>", "</s>")) {
                assertTrue("$name 결과에 $token 이 남아 있다: '${text.take(80)}'", !text.contains(token))
            }
        }
    }

    /** 같은 오디오는 항상 같은 결과여야 한다 — 스레드 경합이나 상태 잔류를 잡는다. */
    @Test
    fun decodingIsDeterministic() {
        val samples = AsrTestSupport.readWavAsset("google_tts002.wav")
        val first = AsrTestSupport.decode(samples)
        AsrTestSupport.reset()
        val second = AsrTestSupport.decode(samples)
        assertEquals("1패스 결과가 재현되지 않는다", first.ctc, second.ctc)
        assertEquals("2패스 결과가 재현되지 않는다", first.secondPass, second.secondPass)
    }

    /** 점수 JSON 구조가 UI(FinalScore/WordScore/PhoneScore)와 맞는지. */
    @Test
    fun pronounceScoreJsonMatchesUiSchema() {
        AsrTestSupport.decode(AsrTestSupport.readWavAsset("google_tts003.wav"), withSecondPass = false)

        // 채점 인자는 앱과 동일하게 콘텐츠 .jsonl 에서 가져온다
        val meta = AsrTestSupport.readPronounceMeta()
        val root = JSONObject(AsrTestSupport.pronounceScore(meta))
        assertTrue("avg_score 가 없다", root.has("avg_score"))
        assertTrue("avg_score 가 유한하지 않다", root.getDouble("avg_score").isFinite())

        val words = root.getJSONArray("final_score")
        assertTrue("단어별 점수가 비었다", words.length() > 0)

        val labelWords = AsrTestSupport.words(meta.label).toSet()
        for (i in 0 until words.length()) {
            val word = words.getJSONObject(i)
            val surface = word.getString("word")
            assertTrue("단어 표기가 비었다", surface.isNotEmpty())
            assertTrue(
                "점수 항목 '$surface' 이 라벨에 없는 단어다",
                AsrTestSupport.normalize(surface) in labelWords
            )
            assertTrue("'$surface' 의 점수가 유한하지 않다", word.getDouble("score").isFinite())
        }
    }

    private fun averageScore(json: String): Double {
        assertTrue("점수 JSON 이 비어 있다", json.isNotBlank())
        return JSONObject(json).getDouble("avg_score")
    }
}
