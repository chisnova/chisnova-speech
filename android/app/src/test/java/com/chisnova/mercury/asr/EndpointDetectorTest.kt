package com.chisnova.mercury.asr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

/**
 * 엔드포인트 검출기 시험.
 *
 * 실제 증상("작게 말하면 잘리거나 끝나지 않는다")을 합성 오디오로 재현해 고정한다.
 * 순수 로직이라 기기 없이 JVM 에서 돌아간다.
 */
class EndpointDetectorTest {

    private val sampleRate = 16000
    private val chunkMs = 20
    private val chunkSize = sampleRate * chunkMs / 1000

    /** 잡음 바닥. 실제 마이크는 완전한 0 을 주지 않는다. */
    private fun noiseChunk(amplitude: Float, seed: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(chunkSize) { (random.nextFloat() * 2f - 1f) * amplitude }
    }

    /** 잡음 위에 얹은 음성 — 진폭으로 목소리 크기를 흉내낸다. */
    private fun speechChunk(amplitude: Float, seed: Int, phase: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(chunkSize) { i ->
            val tone = sin(2.0 * PI * 220.0 * (phase * chunkSize + i) / sampleRate).toFloat()
            tone * amplitude + (random.nextFloat() * 2f - 1f) * amplitude * 0.1f
        }
    }

    /**
     * 한 세션을 돌린다.
     * @return 종료 사유와 그때까지 흐른 시간(ms)
     */
    private fun run(
        detector: EndpointDetector,
        noiseAmp: Float,
        speechAmp: Float,
        leadingSilenceMs: Int,
        speechMs: Int,
        trailingSilenceMs: Int,
        partialsAt: Map<Int, String> = emptyMap()
    ): Pair<EndpointDetector.Reason, Int> {
        var elapsed = 0
        var phase = 0
        val totalMs = leadingSilenceMs + speechMs + trailingSilenceMs
        while (elapsed < totalMs) {
            val isSpeech = elapsed >= leadingSilenceMs && elapsed < leadingSilenceMs + speechMs
            val chunk = if (isSpeech) speechChunk(speechAmp, phase, phase) else noiseChunk(noiseAmp, phase)
            phase++

            partialsAt[elapsed]?.let { detector.onPartialTranscript(it) }

            val reason = detector.process(chunk, chunkSize)
            elapsed += chunkMs
            if (reason != EndpointDetector.Reason.NONE) return reason to elapsed
        }
        return EndpointDetector.Reason.NONE to elapsed
    }

    // ---- 잡음 바닥 기준 판정 ----

    /** 크게 말하든 작게 말하든 SNR 이 같으면 같은 시점에 끊겨야 한다. */
    @Test
    fun loudAndQuietSpeechEndAtTheSameTime() {
        val loud = EndpointDetector()
        loud.reset(null)
        val (loudReason, loudMs) = run(loud, noiseAmp = 0.002f, speechAmp = 0.20f,
            leadingSilenceMs = 400, speechMs = 1200, trailingSilenceMs = 2000)

        val quiet = EndpointDetector()
        quiet.reset(null)
        val (quietReason, quietMs) = run(quiet, noiseAmp = 0.002f, speechAmp = 0.02f,
            leadingSilenceMs = 400, speechMs = 1200, trailingSilenceMs = 2000)

        assertEquals(EndpointDetector.Reason.SILENCE_AFTER_SPEECH, loudReason)
        assertEquals("작은 목소리가 종료로 이어지지 않았다", EndpointDetector.Reason.SILENCE_AFTER_SPEECH, quietReason)
        assertTrue(
            "음량에 따라 종료 시점이 달라진다: loud=${loudMs}ms quiet=${quietMs}ms",
            Math.abs(loudMs - quietMs) <= 2 * chunkMs
        )
    }

    /** 잡음이 큰 환경에서도 발화만 잡아야 한다 — 잡음이 음성으로 오인되면 끝나지 않는다. */
    @Test
    fun endsEvenWhenBackgroundNoiseIsHigh() {
        val detector = EndpointDetector()
        detector.reset(null)
        val (reason, _) = run(detector, noiseAmp = 0.02f, speechAmp = 0.20f,
            leadingSilenceMs = 400, speechMs = 1200, trailingSilenceMs = 2500)
        assertEquals(EndpointDetector.Reason.SILENCE_AFTER_SPEECH, reason)
    }

    /** 발화가 이어지는 동안에는 끊기지 않아야 한다. */
    @Test
    fun doesNotEndWhileSpeechContinues() {
        val detector = EndpointDetector()
        detector.reset(null)
        val (reason, _) = run(detector, noiseAmp = 0.002f, speechAmp = 0.1f,
            leadingSilenceMs = 300, speechMs = 4000, trailingSilenceMs = 0)
        assertEquals("발화 중에 종료됐다", EndpointDetector.Reason.NONE, reason)
    }

    /** 단어 사이 짧은 휴지에서 잘리면 안 된다. */
    @Test
    fun shortPauseBetweenWordsDoesNotEndSession() {
        val detector = EndpointDetector()
        detector.reset(null)
        var elapsed = 0
        var phase = 0
        // 말 500ms → 쉼 300ms → 말 800ms
        val plan = listOf(500 to true, 300 to false, 800 to true)
        for ((durationMs, isSpeech) in plan) {
            var spent = 0
            while (spent < durationMs) {
                val chunk = if (isSpeech) speechChunk(0.1f, phase, phase) else noiseChunk(0.002f, phase)
                phase++
                val reason = detector.process(chunk, chunkSize)
                assertEquals("단어 사이 휴지에서 끊겼다 (t=${elapsed}ms)", EndpointDetector.Reason.NONE, reason)
                spent += chunkMs
                elapsed += chunkMs
            }
        }
    }

    /** 아무 말도 하지 않으면 타임아웃으로 끝나야 한다. */
    @Test
    fun silenceOnlyEndsByTimeout() {
        val detector = EndpointDetector(noSpeechTimeoutMs = 1000)
        detector.reset(null)
        val (reason, elapsed) = run(detector, noiseAmp = 0.002f, speechAmp = 0f,
            leadingSilenceMs = 3000, speechMs = 0, trailingSilenceMs = 0)
        assertEquals(EndpointDetector.Reason.NO_SPEECH_TIMEOUT, reason)
        assertTrue("타임아웃보다 이르게 끝났다: ${elapsed}ms", elapsed >= 1000)
    }

    /** 최대 길이를 넘기지 않아야 한다 — 계속 말해도 언젠가는 끊어야 한다. */
    @Test
    fun stopsAtMaxDuration() {
        val detector = EndpointDetector(maxRecordingMs = 1000)
        detector.reset(null)
        val (reason, elapsed) = run(detector, noiseAmp = 0.002f, speechAmp = 0.1f,
            leadingSilenceMs = 100, speechMs = 5000, trailingSilenceMs = 0)
        assertEquals(EndpointDetector.Reason.MAX_DURATION, reason)
        assertTrue(elapsed <= 1000 + chunkMs)
    }

    /** 첫 음절만 듣고 끊으면 안 된다. */
    @Test
    fun veryShortBurstDoesNotEndImmediately() {
        val detector = EndpointDetector(minSpeechMs = 400)
        detector.reset(null)
        var phase = 0
        var reason = EndpointDetector.Reason.NONE
        // 100ms 만 소리내고 바로 조용해진 경우 — minSpeechMs 를 못 채워 계속 기다린다
        repeat(400 / chunkMs) { detector.process(noiseChunk(0.002f, phase++), chunkSize) }
        repeat(100 / chunkMs) { detector.process(speechChunk(0.1f, phase, phase++), chunkSize) }
        repeat(600 / chunkMs) { reason = detector.process(noiseChunk(0.002f, phase++), chunkSize) }
        assertEquals(EndpointDetector.Reason.NONE, reason)
    }

    // ---- 목표 문장 진행률 ----

    @Test
    fun progressTracksTargetSentence() {
        val detector = EndpointDetector()
        detector.reset("today i want to tell you three stories")
        assertEquals(0f, detector.progress, 1e-6f)

        detector.onPartialTranscript("today i want to")
        assertEquals(0.5f, detector.progress, 1e-6f)

        detector.onPartialTranscript("today i want to tell you three stories")
        assertEquals(1.0f, detector.progress, 1e-6f)
    }

    /** 중간 결과가 흔들려도 진행률은 되돌아가지 않는다. */
    @Test
    fun progressNeverGoesBackwards() {
        val detector = EndpointDetector()
        detector.reset("today i want to tell you three stories")
        detector.onPartialTranscript("today i want to tell you three stories")
        detector.onPartialTranscript("today i")
        assertEquals(1.0f, detector.progress, 1e-6f)
    }

    /** 인식이 한 단어를 놓쳐도 따라잡아야 한다. */
    @Test
    fun progressToleratesMissedWords() {
        val detector = EndpointDetector()
        detector.reset("today i want to tell you three stories")
        detector.onPartialTranscript("today want to tell")   // "i" 를 놓침
        assertTrue("놓친 단어 때문에 진행률이 멈췄다: ${detector.progress}", detector.progress >= 0.5f)
    }

    /** 대소문자·구두점은 무시한다. */
    @Test
    fun progressIgnoresPunctuationAndCase() {
        val detector = EndpointDetector()
        detector.reset("That's it, no big deal!")
        detector.onPartialTranscript("THAT'S IT NO BIG DEAL")
        assertEquals(1.0f, detector.progress, 1e-6f)
    }

    /** 목표 문장을 다 읽으면 hangover 가 짧아진다. */
    @Test
    fun hangoverShrinksWhenSentenceIsComplete() {
        val detector = EndpointDetector()
        detector.reset("today i want to tell you three stories")
        val early = detector.currentHangoverMs()

        detector.onPartialTranscript("today i want to tell you three stories")
        val complete = detector.currentHangoverMs()

        assertTrue("완독 후 hangover 가 줄지 않았다: $early -> $complete", complete < early)
    }

    /** 목표 문장이 없으면 기본 hangover 를 쓴다. */
    @Test
    fun hangoverIsConstantWithoutTarget() {
        val detector = EndpointDetector(defaultHangoverMs = 900)
        detector.reset(null)
        assertEquals(900, detector.currentHangoverMs())
        detector.onPartialTranscript("anything at all")
        assertEquals(900, detector.currentHangoverMs())
    }

    /** 끝까지 읽은 세션이 절반만 읽은 세션보다 빨리 끝나야 한다. */
    @Test
    fun completedSentenceEndsSoonerThanPartialOne() {
        val target = "today i want to tell you three stories"

        val completed = EndpointDetector()
        completed.reset(target)
        val (completedReason, completedMs) = run(
            completed, noiseAmp = 0.002f, speechAmp = 0.1f,
            leadingSilenceMs = 400, speechMs = 1200, trailingSilenceMs = 2500,
            partialsAt = mapOf(1000 to target)
        )

        val partial = EndpointDetector()
        partial.reset(target)
        val (partialReason, partialMs) = run(
            partial, noiseAmp = 0.002f, speechAmp = 0.1f,
            leadingSilenceMs = 400, speechMs = 1200, trailingSilenceMs = 2500,
            partialsAt = mapOf(1000 to "today i want")
        )

        assertEquals(EndpointDetector.Reason.SILENCE_AFTER_SPEECH, completedReason)
        assertEquals(EndpointDetector.Reason.SILENCE_AFTER_SPEECH, partialReason)
        assertTrue(
            "완독 세션이 더 빨리 끝나지 않았다: completed=${completedMs}ms partial=${partialMs}ms",
            completedMs < partialMs
        )
    }

    /** reset 은 이전 세션의 상태를 남기지 않는다. */
    @Test
    fun resetClearsPreviousSession() {
        val detector = EndpointDetector()
        detector.reset("today i want to tell you three stories")
        detector.onPartialTranscript("today i want to tell you three stories")
        assertEquals(1.0f, detector.progress, 1e-6f)

        detector.reset("that's it no big deal")
        assertEquals(0f, detector.progress, 1e-6f)
        assertTrue(!detector.detectedSpeech)
    }

    /** NaN/Inf 가 섞여도 죽지 않고 판정을 이어간다. */
    @Test
    fun nonFiniteSamplesDoNotBreakDetection() {
        val detector = EndpointDetector()
        detector.reset(null)
        val poisoned = speechChunk(0.1f, 1, 1)
        poisoned[0] = Float.NaN
        poisoned[1] = Float.POSITIVE_INFINITY
        val reason = detector.process(poisoned, chunkSize)
        assertEquals(EndpointDetector.Reason.NONE, reason)
        assertTrue("레벨이 유한하지 않다", detector.lastLevelDb.isFinite())
    }

    /** 크기 0 인 덩어리는 무시한다. */
    @Test
    fun emptyChunkIsIgnored() {
        val detector = EndpointDetector()
        detector.reset(null)
        assertEquals(EndpointDetector.Reason.NONE, detector.process(FloatArray(0), 0))
    }
}
