package com.chisnova.mercury.asr

import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 잡음 바닥(noise floor) 기준 발화 종료(endpoint) 검출기.
 *
 * 이전 구현은 "지금까지 누적된 전체 RMS 의 10%"를 무음 기준으로 삼았다. 그러면 기준이 발화 음량을
 * 따라 움직여서, 앞부분을 크게 시작하면 뒤에서 잘리고 작게 말하면 잡음 바닥과 구분이 안 돼
 * 끝나지 않았다. 여기서는 기준을 잡음 바닥에 두고 SNR(dB)로 판정하므로 목소리 크기가 달라져도
 * 판정 기준이 흔들리지 않는다.
 *
 * 판정은 오디오 스레드에서만 하고, 상태 조회는 UI/로그용이라 @Volatile 로만 노출한다.
 */
class EndpointDetector(
    private val sampleRate: Int = 16000,

    /** 잡음 바닥보다 이만큼(dB) 높으면 음성으로 본다. */
    private val snrThresholdDb: Double = 8.0,

    /**
     * 잡음 바닥의 최대 상승 속도(dB/초). 하강은 즉시 허용한다.
     * 느리게 올려야 발화가 길어져도 기준이 발화 레벨로 끌려 올라가지 않는다.
     */
    private val noiseFloorRiseDbPerSec: Double = 0.5,

    /** 이 구간은 잡음 바닥만 추정하고 종료 판정을 하지 않는다(버튼을 누른 직후). */
    private val warmupMs: Int = 300,

    /** 발화가 잡힌 뒤 최소 이만큼은 녹음한다 — 첫 음절만 듣고 끊는 것을 막는다. */
    private val minSpeechMs: Int = 400,

    /**
     * 발화 뒤 이만큼 무음이 이어지면 끝난 것으로 본다.
     * 단어 사이 휴지보다는 길어야 하고, 길수록 사용자가 기다리는 시간이 늘어난다.
     */
    private val hangoverMs: Int = 900,

    /** 발화가 한 번도 안 잡힌 채 이만큼 지나면 종료한다. */
    private val noSpeechTimeoutMs: Int = 5000,

    /** 무슨 일이 있어도 이만큼을 넘기지 않는다. */
    private val maxRecordingMs: Int = 30000,
    /**
     * 스트리밍 CTC 가 이 시간 안에 발화를 봤으면 "아직 말하는 중" 으로 본다.
     *
     * 실측: 발화 도중 단어 사이 쉼에서 CTC 무음이 최대 320ms 까지 올라간다.
     * 그보다 여유를 두되, 진짜 종료(900ms 이상)와는 확실히 구분되는 값이다.
     */
    private val ctcSpeechRecentMs: Int = 400,
    /**
     * CTC 때문에 미룰 수 있는 시간의 총량.
     *
     * 상한이 없으면 모델이 계속 발화라고 우기는 동안 녹음이 끝나지 않는다.
     * CTC 값은 최대 청크 하나(1초)만큼 오래된 것이라 틀릴 수 있으므로 반드시 막아 둔다.
     */
    private val ctcGraceLimitMs: Int = 600,
    /**
     * 스트리밍 CTC 가 이만큼 발화를 못 봤으면 에너지 판정과 무관하게 끊는다.
     *
     * 에너지 기준은 조용해지면 잡음 바닥이 따라 내려가(level -45.2 / floor -45.2 → SNR 0)
     * 그 뒤 작은 소리에도 SNR 이 8 을 넘어 "발화" 로 오판하고 무음 카운터를 되돌린다.
     * 실측에서 CTC 가 1.3초 만에 종료를 알렸는데 에너지는 8초를 더 끌었다.
     *
     * 다만 모델도 말을 놓친다. 실사용 로그에서 한창 말하는 중에도 CTC 무음이 520ms 까지
     * 올라갔다(발음이 흐리거나 모델이 못 잡는 구간). 그래서 이 값만으로 끊으면 말하는
     * 도중에 잘린다 — 아래 process() 에서 "지금 조용한가" 를 함께 본다.
     */
    private val ctcEndpointMs: Int = 900
) {
    /** 종료 사유. NONE 이면 계속 녹음한다. */
    enum class Reason {
        NONE,
        /** 발화 후 무음이 hangover 를 넘김 */
        SILENCE_AFTER_SPEECH,
        /** 발화가 한 번도 잡히지 않은 채 타임아웃 */
        NO_SPEECH_TIMEOUT,
        /** 최대 녹음 길이 도달 */
        MAX_DURATION
    }

    private companion object {
        /** log10(0) 방지. */
        const val MIN_RMS = 1e-7

        /** CTC 값을 모를 때 [process] 에 넘기는 값 */
        const val CTC_UNKNOWN = -1
    }

    private var noiseFloorDb = Double.NaN
    private var elapsedMs = 0
    private var silenceMs = 0
    private var speechMs = 0
    private var hasSpeech = false
    /** CTC 판단 때문에 종료를 미룬 누적 시간 */
    private var ctcGraceUsedMs = 0

    /** 마지막으로 계산한 값들 — 로그·시험용. */
    @Volatile var lastLevelDb = 0.0
        private set
    @Volatile var lastNoiseFloorDb = 0.0
    /** 마지막으로 받은 CTC 무음 값(ms). 모르면 [CTC_UNKNOWN] */
    @Volatile var lastCtcSilenceMs = CTC_UNKNOWN
    /** CTC 때문에 미룬 누적 시간(ms) — 상한은 ctcGraceLimitMs */
    @Volatile var lastCtcGraceUsedMs = 0
    /** 직전 판정에서 CTC 가 종료를 막았는지 */
    @Volatile var lastDeferredByCtc = false
    /** 직전 종료가 CTC 판단으로 난 것인지 — 로그에서 사유를 가른다 */
    @Volatile var lastEndedByCtc = false
        private set

    val detectedSpeech: Boolean get() = hasSpeech
    val currentHangoverMs: Int get() = hangoverMs

    /** 새 녹음을 시작하기 전에 호출한다. */
    fun reset() {
        noiseFloorDb = Double.NaN
        elapsedMs = 0
        silenceMs = 0
        speechMs = 0
        hasSpeech = false
        ctcGraceUsedMs = 0
        lastCtcSilenceMs = CTC_UNKNOWN
        lastCtcGraceUsedMs = 0
        lastDeferredByCtc = false
        lastEndedByCtc = false
        lastLevelDb = 0.0
        lastNoiseFloorDb = 0.0
    }

    /**
     * 오디오 한 덩어리를 판정한다. 녹음 루프에서 읽은 즉시 호출한다.
     * @return 종료 사유. [Reason.NONE] 이면 계속 녹음한다.
     */
    /**
     * @param ctcSilenceMs 스트리밍 CTC 가 본 "말이 멎은 뒤 흐른 시간"(ms).
     *   아직 발화를 잡지 못했거나 값을 모르면 [CTC_UNKNOWN].
     *   에너지만으로는 무성 자음(/s/, /f/, 파열음 앞 폐쇄)이 무음과 구별되지 않아
     *   어말 자음이 잘린다. 모델이 방금까지 말을 봤다면 종료를 잠시 미룬다.
     */
    @JvmOverloads
    fun process(samples: FloatArray, size: Int, ctcSilenceMs: Int = CTC_UNKNOWN): Reason {
        if (size <= 0) return Reason.NONE

        val chunkMs = size * 1000 / sampleRate
        elapsedMs += chunkMs

        lastCtcSilenceMs = ctcSilenceMs
        lastDeferredByCtc = false
        lastEndedByCtc = false
        val levelDb = rmsDb(samples, size)
        updateNoiseFloor(levelDb, chunkMs)
        lastLevelDb = levelDb
        lastNoiseFloorDb = noiseFloorDb

        if (elapsedMs >= maxRecordingMs) return Reason.MAX_DURATION

        // 시작 직후는 잡음 바닥을 잡는 데만 쓴다
        if (elapsedMs < warmupMs) return Reason.NONE

        val snr = levelDb - noiseFloorDb

        // 모델이 "말이 끝났다" 고 보고, 지금 이 순간도 조용하면 끊는다.
        //
        // 둘을 함께 봐야 한다. CTC 만 보면 말하는 도중 모델이 놓친 구간에서 잘리고,
        // 에너지만 보면 잡음 바닥이 따라 내려가 SNR 이 튀는 바람에 영영 못 끊는다.
        // 지금 소리가 나고 있으면(=SNR 이 임계를 넘으면) 아래 발화 판정으로 넘어간다.
        if (hasSpeech && ctcSilenceMs >= ctcEndpointMs && snr <= snrThresholdDb) {
            lastEndedByCtc = true
            return Reason.SILENCE_AFTER_SPEECH
        }

        if (snr > snrThresholdDb) {
            hasSpeech = true
            speechMs += chunkMs
            silenceMs = 0
            return Reason.NONE
        }

        silenceMs += chunkMs

        if (!hasSpeech) {
            return if (elapsedMs >= noSpeechTimeoutMs) Reason.NO_SPEECH_TIMEOUT else Reason.NONE
        }
        if (speechMs < minSpeechMs) return Reason.NONE
        if (silenceMs < hangoverMs) return Reason.NONE

        // 에너지로는 끝난 것 같다. 모델이 방금까지 말을 보고 있었다면 한 번 더 기다린다.
        // ctcSilenceMs 가 음수면 CTC 가 아직 발화를 못 잡았다는 뜻이라 믿을 근거가 없다.
        if (ctcSilenceMs in 0 until ctcSpeechRecentMs && ctcGraceUsedMs < ctcGraceLimitMs) {
            ctcGraceUsedMs += chunkMs
            lastCtcGraceUsedMs = ctcGraceUsedMs
            lastDeferredByCtc = true
            return Reason.NONE
        }
        return Reason.SILENCE_AFTER_SPEECH
    }

    private fun updateNoiseFloor(levelDb: Double, chunkMs: Int) {
        noiseFloorDb = when {
            noiseFloorDb.isNaN() -> levelDb
            // 더 조용해졌으면 즉시 따라 내려간다
            levelDb < noiseFloorDb -> levelDb
            // 올라갈 때는 천천히 — 발화 레벨로 끌려 올라가면 SNR 기준이 무너진다
            else -> min(levelDb, noiseFloorDb + noiseFloorRiseDbPerSec * chunkMs / 1000.0)
        }
    }

    private fun rmsDb(samples: FloatArray, size: Int): Double {
        var sumSquare = 0.0
        for (i in 0 until size) {
            val v = samples[i].toDouble()
            if (v.isFinite()) sumSquare += v * v
        }
        val rms = sqrt(sumSquare / size)
        return 20.0 * log10(max(rms, MIN_RMS))
    }
}
