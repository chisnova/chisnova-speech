package com.chisnova.mercury.asr

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import com.chisnova.mercury.jni.AsrAPI
import com.chisnova.mercury.support.AsrTestSupport
import org.junit.Assert.assertNotNull
import org.junit.Ignore
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 네이티브 프로세스를 통째로 죽이는 것으로 확인된 경로들.
 *
 * 이 클래스의 시험은 실패(assert)로 끝나지 않고 SIGABRT/SIGSEGV 로 앱 프로세스를 내리므로,
 * 같은 실행에 포함되면 뒤따르는 모든 시험이 함께 무너진다. 그래서 기본적으로 @Ignore 이며,
 * 네이티브를 고친 뒤 하나씩 어노테이션을 떼고 개별 실행해 확인한다.
 *
 *   ./gradlew :app:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=\
 *     com.chisnova.mercury.asr.AsrKnownHazardTest#cancelThenRestartTerminatesProcess
 *
 * 각 시험 주석에 원인이 되는 네이티브 코드 위치를 적어 두었다.
 */
@RunWith(AndroidJUnit4::class)
@LargeTest
class AsrKnownHazardTest {

    /**
     * cancelDecoding() 뒤에 다시 세션을 시작하면 std::terminate 로 죽는다.
     *
     * - cpp/decoder/asr_async_decoder.cc: cancelDecoding() 은 is_canceled 플래그만 세우고
     *   engineThread 를 join/detach 하지 않는다. 조건변수에 notify 도 하지 않으므로
     *   엔진 스레드는 data_cond.wait 에서 계속 잠들어 있을 수 있다.
     * - 같은 파일 startBackGroundEngine(): `engineThread = std::thread(...)` 로 이동 대입한다.
     *   std::thread 의 이동 대입은 대상이 joinable 이면 std::terminate 를 부른다.
     *
     * 앱에서 이미 밟는 경로다: VoiceRecognitionActivity.onStop() 이 cancelDecoding() 을 부르고,
     * 화면에 돌아와 다시 연습을 시작하면 startBackgroundEngine() 이 호출된다.
     */
    @Test
    @Ignore("네이티브 프로세스가 종료된다 — asr_async_decoder 의 스레드 수명 관리 수정 후 활성화")
    fun cancelThenRestartTerminatesProcess() {
        val api = AsrTestSupport.engine()
        val samples = AsrTestSupport.readWavAsset("google_tts003.wav")

        api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
        AsrTestSupport.stream(api, samples.copyOf(8000), markFinal = false)
        api.cancelDecoding()

        // 여기서 죽는다.
        api.startBackGroundEngine(AsrTestSupport.DEFAULT_NBEST)
        AsrTestSupport.stream(api, samples)
        api.finishDecoding()
        assertNotNull(api.ctcAsrResult)
    }

    /**
     * getSecondPassDecodingResult(0) — 빈 벡터의 첫 원소를 읽는다.
     *
     * cpp/decoder/asr_stream_conformer.cc secondPassDecoding():
     *   while (nbest.size() < num_nbest) { ... }   // num_nbest<=0 이면 한 번도 돌지 않는다
     *   ...
     *   return ret[0].word_ids;                    // ret 이 비어 있다
     */
    @Test
    @Ignore("빈 std::vector 의 [0] 접근 — secondPassDecoding 의 인자 검증 추가 후 활성화")
    fun secondPassWithZeroNbestReadsEmptyVector() {
        val api = AsrTestSupport.engine()
        AsrTestSupport.decode(AsrTestSupport.readWavAsset("google_tts003.wav"), withSecondPass = false)
        assertNotNull(api.getSecondPassDecodingResult(0))
    }

    @Test
    @Ignore("음수 nbest — 위 zero nbest 와 같은 경로")
    fun secondPassWithNegativeNbestReadsEmptyVector() {
        val api = AsrTestSupport.engine()
        AsrTestSupport.decode(AsrTestSupport.readWavAsset("google_tts003.wav"), withSecondPass = false)
        assertNotNull(api.getSecondPassDecodingResult(-1))
    }

    /**
     * 아주 큰 nbest — 빈 우선순위 큐의 top() 을 읽는다.
     *
     * cpp/decoder/asr_stream_conformer.cc secondPassDecoding():
     *   while (count < num_nbest) { Beam beam = running.top(); running.pop(); ... }
     * running 에 쌓이는 후보 수는 어휘 크기로 제한되므로 num_nbest 가 그보다 크면 비어 버린다.
     */
    @Test
    @Ignore("빈 priority_queue 의 top() 접근 — secondPassDecoding 의 큐 고갈 처리 추가 후 활성화")
    fun secondPassWithHugeNbestExhaustsCandidateQueue() {
        val api = AsrTestSupport.engine()
        AsrTestSupport.decode(AsrTestSupport.readWavAsset("google_tts003.wav"), withSecondPass = false)
        assertNotNull(api.getSecondPassDecodingResult(100_000))
    }

    /**
     * 라벨의 단어 수보다 큰 그룹 인덱스를 넘기면 std::vector 범위 밖에 쓴다.
     *
     * cpp/score/scorer.cc getWordScore()/getPhoneScore():
     *   word_score 는 label 을 공백으로 나눈 개수(lab_words.size())만큼만 만들어 두고,
     *   그 뒤 word_idx_group 의 값을 그대로 첨자로 쓴다.
     *     word_score[group].score += s;   // group >= lab_words.size() 면 범위 밖
     *   그룹 인덱스가 라벨 길이 안에 있는지 확인하지 않는다.
     *
     * 앱에서는 라벨과 인덱스가 같은 .jsonl 에서 함께 오므로 정상 경로에서는 생기지 않지만,
     * 콘텐츠와 메타가 어긋난 채 배포되면 조용히 힙을 망가뜨린다.
     */
    @Test
    @Ignore("std::vector 범위 밖 쓰기 — Scorer 의 group 인덱스 검증 추가 후 활성화")
    fun pronounceScoreWithFewerLabelWordsThanGroupsWritesOutOfBounds() {
        val meta = AsrTestSupport.readPronounceMeta()
        AsrTestSupport.decode(AsrTestSupport.readWavAsset("google_tts003.wav"), withSecondPass = false)
        // 인덱스는 원문(12단어 이상)의 것이고 라벨만 두 단어로 줄인다
        assertNotNull(AsrTestSupport.pronounceScore(meta, label = "two words"))
    }

    /**
     * 모델 경로 없이 만든 엔진에 clear() — 널 포인터 역참조.
     *
     * cpp/state/context_manager.cc clear(): IdleState::clear() 가 true 를 돌려주므로
     * 아직 만들어지지 않은 asr_async_decoder(nullptr)->clear() 를 호출한다.
     * (다른 API 는 IdleState 에서 false 라 막히는데 clear 만 열려 있다.)
     */
    @Test
    @Ignore("널 포인터 역참조 — IdleState 에서의 clear 처리 수정 후 활성화")
    fun clearOnUninitializedEngineDereferencesNull() {
        val uninitialized = AsrAPI()
        uninitialized.clear()
    }
}
