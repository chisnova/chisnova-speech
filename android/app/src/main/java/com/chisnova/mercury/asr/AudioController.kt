package com.chisnova.mercury.asr

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.util.Log
import android.widget.TextView
import androidx.core.app.ActivityCompat
import com.chisnova.mercury.asr.AsrController
import kotlin.math.sqrt

class AudioController {
    // 녹음 완료 시 호출되는 콜백
    var onRecordingFinished: (() -> Unit)? = null

    // 인스턴스 변수로 이동 — Activity 재생성 시 상태 꼬임 방지
    @Volatile private var recorderStop = false
    private var audioRecorder: AudioRecord? = null
    private var mediaPlayer: MediaPlayer? = null
    private var hasRecordedAudioFlag: Boolean = false
    private var bufferRecord: ShortArray? = null
    private var recordThread: Thread? = null
    var saveAudioIdx: Int = 0
    var saveAudioBuffer: ShortArray? = null

    // 마이크 입력 전처리 이펙트. 기기가 지원하지 않으면 null 이며,
    // AudioRecord 와 수명을 함께한다(같이 만들고 같이 해제).
    private var automaticGainControl: AutomaticGainControl? = null
    private var noiseSuppressor: NoiseSuppressor? = null

    /** 녹음 스레드가 도는 동안 true. UI 스레드가 읽으므로 @Volatile. */
    @Volatile private var recordingActive = false

    /** 발화 종료 판정기(잡음 바닥 기준 SNR). 세션마다 [sendAudio] 에서 초기화한다. */
    private val endpointDetector = EndpointDetector()

    companion object {
        /**
         * 음성 인식용 입력 소스. MIC 는 원시 입력에 가까워 작게 말하면 레벨이 그대로 낮게
         * 들어오지만, VOICE_RECOGNITION 은 제조사가 ASR 용으로 튜닝한 경로(대개 AGC/NS 포함)를 탄다.
         * 통화용 VOICE_COMMUNICATION 은 AEC 까지 걸려 스펙트럼이 더 많이 바뀌므로 쓰지 않는다.
         */
        private const val AUDIO_SOURCE = MediaRecorder.AudioSource.VOICE_RECOGNITION

        // 상수만 companion object에 유지
        private val bufferRecordSize = AudioRecord.getMinBufferSize(
            16000,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        private val audioAttributes: AudioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()

        private val audioFormat = AudioFormat.Builder()
            .setSampleRate(16000)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
    }

    fun hasRecordedAudio(): Boolean {
        return hasRecordedAudioFlag
    }

    fun init(context: Context) {
        Log.i("[AsrAPI]", "AudioController init")
        bufferRecord = ShortArray(bufferRecordSize)
        // 기존 AudioRecord가 있으면 해제 후 재생성하여 상태 문제 방지
        releaseRecorder()
        Log.i("[AsrAPI]", "AudioController init audioRecorder")
        if (ActivityCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            Log.w("[AsrAPI]", "RECORD_AUDIO permission not granted")
            return
        }
        try {
            audioRecorder = AudioRecord(AUDIO_SOURCE,
                                            16000,
                                            AudioFormat.CHANNEL_IN_MONO,
                                            AudioFormat.ENCODING_PCM_16BIT,
                                            bufferRecordSize)
            if (audioRecorder?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("[AsrAPI]", "AudioRecord failed to initialize, state: ${audioRecorder?.state}")
                audioRecorder?.release()
                audioRecorder = null
            } else {
                Log.i("[AsrAPI]", "AudioController init audioRecorder success")
                // 이펙트는 세션이 살아 있는 동안에만 유효하므로 초기화 성공 후에 붙인다
                attachAudioEffects(audioRecorder!!.audioSessionId)
                // init()은 AudioRecord 객체 초기화만 수행
                // 실제 녹음은 startRecording()을 명시적으로 호출할 때 시작
            }
        } catch (e: Exception) {
            Log.e("[AsrAPI]", "AudioRecord creation failed", e)
            audioRecorder = null
        }
        hasRecordedAudioFlag = false
    }

    /**
     * 마이크 녹음을 시작합니다.
     * init() 호출 후, 실제 녹음이 필요한 시점에 명시적으로 호출하세요.
     */
    fun startRecording() {
        if (audioRecorder == null || audioRecorder?.state != AudioRecord.STATE_INITIALIZED) {
            Log.w("[AsrAPI]", "startRecording() called but AudioRecord is not initialized")
            return
        }
        // 이미 녹음 중이면 중복 시작 방지
        if (audioRecorder?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            Log.i("[AsrAPI]", "AudioRecord is already recording, skipping startRecording()")
            return
        }
        try {
            audioRecorder?.startRecording()
            if (audioRecorder?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                Log.i("[AsrAPI]", "AudioRecord started recording")
            } else {
                Log.w("[AsrAPI]", "AudioRecord startRecording failed, state: ${audioRecorder?.recordingState}")
            }
        } catch (e: Exception) {
            Log.e("[AsrAPI]", "AudioRecord startRecording failed", e)
        }
    }

    /**
     * 자동 이득 조절(AGC)과 잡음 억제(NS)를 마이크 세션에 건다.
     *
     * 작게 말할 때 입력 레벨 자체가 낮게 들어오는 것을 기기 전처리로 끌어올리는 것이 목적이다.
     * 둘 다 기기가 지원해야 동작하는 선택적 기능이라, 없으면 로그만 남기고 그대로 진행한다.
     */
    private fun attachAudioEffects(sessionId: Int) {
        releaseAudioEffects()

        if (AutomaticGainControl.isAvailable()) {
            automaticGainControl = try {
                AutomaticGainControl.create(sessionId)?.apply { enabled = true }
            } catch (e: Exception) {
                Log.w("[AsrAPI]", "AutomaticGainControl 생성 실패", e)
                null
            }
            Log.i("[AsrAPI]", "AGC attached — enabled=${automaticGainControl?.enabled}")
        } else {
            Log.i("[AsrAPI]", "AGC 미지원 기기")
        }

        if (NoiseSuppressor.isAvailable()) {
            noiseSuppressor = try {
                NoiseSuppressor.create(sessionId)?.apply { enabled = true }
            } catch (e: Exception) {
                Log.w("[AsrAPI]", "NoiseSuppressor 생성 실패", e)
                null
            }
            Log.i("[AsrAPI]", "NS attached — enabled=${noiseSuppressor?.enabled}")
        } else {
            Log.i("[AsrAPI]", "NS 미지원 기기")
        }
    }

    /** 이펙트는 AudioRecord 보다 먼저 해제해야 세션이 깨끗하게 정리된다. */
    private fun releaseAudioEffects() {
        try {
            automaticGainControl?.release()
        } catch (e: Exception) {
            Log.w("[AsrAPI]", "AGC 해제 실패", e)
        }
        try {
            noiseSuppressor?.release()
        } catch (e: Exception) {
            Log.w("[AsrAPI]", "NS 해제 실패", e)
        }
        automaticGainControl = null
        noiseSuppressor = null
    }

    private fun releaseRecorder() {
        releaseAudioEffects()
        try {
            audioRecorder?.let { recorder ->
                if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    recorder.stop()
                }
                recorder.release()
                Log.i("[AsrAPI]", "AudioRecord released")
            }
        } catch (e: Exception) {
            Log.e("[AsrAPI]", "Error releasing AudioRecord", e)
        }
        audioRecorder = null
    }

    fun getRecordAudio(): ShortArray? {
        return saveAudioBuffer
    }

    /** 사용자가 직접 녹음을 끝낼 때(완료 버튼) 호출한다. 녹음 중이 아니면 아무 일도 없다. */
    fun stopRecording() {
        if (recordingActive) {
            Log.i("[AsrAPI]", "stopRecording() — 사용자 요청으로 녹음 종료")
        }
        recorderStop = true
    }

    /** 녹음 스레드가 도는 중인지 — UI 가 버튼 상태를 정하는 데 쓴다. */
    fun isRecording(): Boolean = recordingActive

    fun sendAudio(asr_controller: AsrController) {
        Log.i("[AsrAPI]", "sendAudio() 진입 — recorderStop=$recorderStop, audioRecorder=${if (audioRecorder != null) "non-null" else "null"}")

        // 이전 녹음 스레드를 완전히 종료한 후 새 녹음 시작
        stopAndWaitForPreviousThread()
        // 이전 스레드 종료 과정에서 recorderStop=true가 설정되므로, 새 녹음 시작 전 반드시 리셋
        recorderStop = false
        Log.i("[AsrAPI]", "sendAudio() — stopAndWaitForPreviousThread() 완료, recorderStop 리셋=$recorderStop")

        // AudioRecord 상태 확인 - 초기화되지 않았으면 녹음 불가
        if (audioRecorder == null || audioRecorder?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e("[AsrAPI]", "AudioRecord is not initialized, cannot start recording")
            val mainHandler = Handler(Looper.getMainLooper())
            mainHandler.post { onRecordingFinished?.invoke() }
            return
        }
        saveAudioIdx = 0
        saveAudioBuffer = ShortArray(16000 * 30)

        // 이번 세션의 판정 상태 초기화
        endpointDetector.reset()

        val callback = onRecordingFinished
        val thread = Thread(Runnable {
            Log.i("[AsrAPI]", "recordThread 시작 — threadId=${Thread.currentThread().id}")
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            recordingActive = true
            val shorAudio = ShortArray(16000 * 30)
            val audio = FloatArray(16000 * 30)
            var audioIdx = 0

            // startRecording()이 이미 호출되어 녹음 중인지 확인
            val recState = audioRecorder?.recordingState
            Log.i("[AsrAPI]", "recordThread — audioRecorder.recordingState=$recState, RECORDSTATE_RECORDING=${AudioRecord.RECORDSTATE_RECORDING}")
            if (recState == AudioRecord.RECORDSTATE_RECORDING) {
                Log.i("[AsrAPI]", "recordThread — flush loop 진입")
                // 잔여 버퍼 플러시: startRecording()~sendAudio() 사이에 쌓인 오래된 오디오 데이터 제거
                // 최대 3회로 제한하여 무한 블로킹 방지
                var flushedSamples = 0
                var flushCount = 0
                while (flushCount < 3) {
                    val numFlushed = audioRecorder?.read(bufferRecord!!, 0, bufferRecordSize) ?: 0
                    Log.i("[AsrAPI]", "flush #$flushCount — numFlushed=$numFlushed")
                    if (numFlushed <= 0) break
                    flushedSamples += numFlushed
                    flushCount++
                    if (numFlushed < bufferRecordSize) break
                }
                Log.i("[AsrAPI]", "flush 완료 — count=$flushCount, flushedSamples=$flushedSamples")
            } else {
                Log.i("[AsrAPI]", "recordThread — RECORDING 아님 (state=$recState), fallback startRecording 시도")
                // startRecording()이 아직 호출되지 않은 경우 여기서 시작
                try {
                    audioRecorder?.startRecording()
                    if (audioRecorder?.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                        Log.e("[AsrAPI]", "AudioRecord failed to start recording, state: ${audioRecorder?.recordingState}")
                        recordingActive = false
                        val mainHandler = Handler(Looper.getMainLooper())
                        mainHandler.post { callback?.invoke() }
                        return@Runnable
                    }
                    Log.i("[AsrAPI]", "AudioRecord started recording in sendAudio() (fallback)")
                } catch (e: Exception) {
                    Log.e("[AsrAPI]", "Failed to start recording", e)
                    recordingActive = false
                    val mainHandler = Handler(Looper.getMainLooper())
                    mainHandler.post { callback?.invoke() }
                    return@Runnable
                }
            }
            Log.i("[AsrAPI]", "AudioRecord ready for data collection")

            var totalChunksSent = 0
            var meanSquare = 0.0
            // xRT 측정용: 마지막으로 유효 데이터를 수신한 시각
            var lastDataReceivedTimeNs = System.nanoTime()
            while (true) {
                if (recorderStop) break

                val numRead = audioRecorder?.read(bufferRecord!!, 0, bufferRecordSize) ?: 0
                if (numRead <= 0) {
                    continue
                }
                // 유효 데이터를 받을 때마다 시각 갱신
                lastDataReceivedTimeNs = System.nanoTime()

                // 남은 버퍼 용량만큼만 처리하여 out-of-bounds 방지
                val samplesToProcess = minOf(numRead, shorAudio.size - audioIdx)
                if (samplesToProcess <= 0) {
                    recorderStop = true
                    Log.w("[AsrAPI]", "Reached max recording buffer length; stopping.")
                    break
                }
                val chunkFloatAudio = FloatArray(samplesToProcess)
                for (i in 0 until samplesToProcess) {
                    shorAudio[audioIdx] = bufferRecord!![i]
                    saveAudioBuffer!![saveAudioIdx] = bufferRecord!![i]
                    saveAudioIdx++
                    val tmp = (bufferRecord!![i].toFloat() / 32768.0f)
                    audio[audioIdx] = tmp
                    chunkFloatAudio[i] = tmp
                    meanSquare += (1 / (audioIdx + 1).toDouble()) * ((audio[audioIdx] * audio[audioIdx]).toDouble() - meanSquare)
                    audioIdx++
                }
                if (samplesToProcess < numRead) {
                    recorderStop = true
                    Log.w("[AsrAPI]", "Reached max recording buffer length; stopping after partial chunk.")
                }
                totalChunksSent++
                Log.d("[AsrAPI]", "sendAudio chunk #$totalChunksSent — numRead=$numRead, samplesToProcess=$samplesToProcess, audioIdx=$audioIdx")
                asr_controller.sendAudio(chunkFloatAudio, samplesToProcess, false)

                // 엔드포인트 판정: 잡음 바닥 기준 SNR + 목표 문장 진행률.
                // 이전 구현은 "누적 전체 RMS 의 10%"를 기준으로 삼아, 기준이 발화 음량을 따라
                // 움직이는 바람에 작게 말하면 잘리거나 끝나지 않았다.
                // 에너지만 보면 무성 자음(/s/, /f/, 파열음 앞 폐쇄)이 무음과 구별되지 않아
                // 어말 자음이 잘린다. 모델이 방금까지 말을 봤는지 함께 넘긴다.
                val ctcSilenceMs = asr_controller.getCtcTrailingSilenceMs()
                val decision = endpointDetector.process(chunkFloatAudio, samplesToProcess, ctcSilenceMs)
                // 유예는 드물게 일어나므로 그 순간을 남겨도 로그가 시끄러워지지 않는다.
                // 이 줄이 안 보이면 거부권이 한 번도 작동하지 않은 것이다.
                // 판정이 나기 전에도 상태를 남긴다. 예전에는 decision != NONE 일 때만 찍어서
                // "끊기지 않는다" 는 증상이 나와도 왜 그런지(레벨인지 바닥인지 CTC 인지)
                // 알 길이 없었다. iOS AsrSession 이 남기는 주기 로그와 같은 값이다.
                if (totalChunksSent % 20 == 0) {
                    Log.i("[AsrAPI]", "판정: level=%.1fdB floor=%.1fdB SNR=%.1f speech=%s ctcSilence=%dms 유예=%dms 경과=%.1fs"
                        .format(
                            endpointDetector.lastLevelDb,
                            endpointDetector.lastNoiseFloorDb,
                            endpointDetector.lastLevelDb - endpointDetector.lastNoiseFloorDb,
                            if (endpointDetector.detectedSpeech) "yes" else "no",
                            endpointDetector.lastCtcSilenceMs,
                            endpointDetector.lastCtcGraceUsedMs,
                            audioIdx / 16000.0
                        ))
                }
                if (endpointDetector.lastDeferredByCtc) {
                    Log.i("[AsrAPI]", "endpoint 유예 — CTC 가 발화를 봤다: " +
                        "ctcSilence=${ctcSilenceMs}ms 누적유예=${endpointDetector.lastCtcGraceUsedMs}ms")
                }
                if (decision != EndpointDetector.Reason.NONE) {
                    Log.i(
                        "[AsrAPI]",
                        "endpoint=$decision(${if (endpointDetector.lastEndedByCtc) "CTC" else "에너지"}) — " +
                        "level=%.1fdB floor=%.1fdB hangover=%dms ctcSilence=%dms 유예=%dms audioIdx=%d"
                            .format(
                                endpointDetector.lastLevelDb,
                                endpointDetector.lastNoiseFloorDb,
                                endpointDetector.currentHangoverMs,
                                endpointDetector.lastCtcSilenceMs,
                                endpointDetector.lastCtcGraceUsedMs,
                                audioIdx
                            )
                    )
                    break
                }
            }
            Log.i("[AsrAPI]", "Recording loop ended — totalChunksSent=$totalChunksSent, finalAudioIdx=$audioIdx, recorderStop=$recorderStop")
            // 루프를 어떻게 빠져나왔든(판정·수동 종료·버퍼 한계) 여기서 상태를 내린다
            recordingActive = false

            // 녹음 종료 시각 기록 (xRT 계산용)
            val recordingEndTimeNs = System.nanoTime()

            val empty = FloatArray(0)
            Log.i("[AsrAPI]", "Sending final (isFinal=true) empty chunk to ASR engine")
            asr_controller.sendAudio(empty, 0, true)
            try {
                audioRecorder?.stop()
            } catch (e: Exception) {
                Log.e("[AsrAPI]", "Error stopping AudioRecord", e)
            }
            hasRecordedAudioFlag = audioIdx > 0

            // xRT 로그 출력
            val audioLenSec = audioIdx / 16000.0
            val lastChunkToEndSec = (recordingEndTimeNs - lastDataReceivedTimeNs) / 1_000_000_000.0
            val xRT = if (lastChunkToEndSec > 0.0) audioLenSec / lastChunkToEndSec else 0.0
            Log.i("[AsrAPI]", "xRT: %.2f | audio_len=%.2fs | last_chunk_to_end=%.3fs".format(xRT, audioLenSec, lastChunkToEndSec))

            Log.i("[AsrAPI]", "Recording finished, audioIdx: $audioIdx")
            // 메인 스레드에서 녹음 완료 콜백 호출
            val mainHandler = Handler(Looper.getMainLooper())
            mainHandler.post { callback?.invoke() }
        })
        recordThread = thread
        thread.start()
    }

    private fun stopAndWaitForPreviousThread() {
        recordThread?.let { prev ->
            if (prev.isAlive) {
                recorderStop = true
                try { prev.join(1000) } catch (_: InterruptedException) {}
                if (prev.isAlive) {
                    Log.w("[AsrAPI]", "Previous recording thread did not stop in time")
                }
            }
        }
        recordThread = null
    }

    fun release() {
        stopAndWaitForPreviousThread()
        releaseRecorder()
        onRecordingFinished = null
        Log.i("[AsrAPI]", "AudioController released")
    }
}
