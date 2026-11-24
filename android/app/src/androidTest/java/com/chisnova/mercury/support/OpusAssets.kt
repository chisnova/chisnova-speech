package com.chisnova.mercury.support

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.sin

/**
 * APK assets 의 opus 음원을 16kHz mono float PCM 으로 디코딩한다.
 *
 * 콘텐츠 opus 는 AAPT 가 압축해 넣을 수 있어 AssetFileDescriptor 로 바로 열지 못한다.
 * 캐시 파일로 복사한 뒤 MediaExtractor + MediaCodec 으로 푼다. 안드로이드 opus 디코더는
 * 48kHz 로 출력하므로 저역통과 후 3:1 로 솎아 16kHz 를 만든다.
 */
object OpusAssets {

    private const val TARGET_RATE = 16_000

    fun readOpusAsset(assetPath: String): FloatArray {
        val cache = File.createTempFile("sweep", ".opus", AsrTestSupport.context.cacheDir)
        try {
            AsrTestSupport.assets.open(assetPath).use { input ->
                cache.outputStream().use { output -> input.copyTo(output, 64 * 1024) }
            }
            return decodeFile(cache.absolutePath)
        } finally {
            cache.delete()
        }
    }

    private fun decodeFile(path: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                trackIndex = i
                format = f
                break
            }
        }
        check(trackIndex >= 0 && format != null) { "$path 에 오디오 트랙이 없다" }
        extractor.selectTrack(trackIndex)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcm = ArrayList<ShortArray>()
        var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val info = MediaCodec.BufferInfo()
        var sawInputEnd = false
        var sawOutputEnd = false
        try {
            while (!sawOutputEnd) {
                if (!sawInputEnd) {
                    val inIndex = codec.dequeueInputBuffer(10_000)
                    if (inIndex >= 0) {
                        val buffer = codec.getInputBuffer(inIndex)!!
                        val read = extractor.readSampleData(buffer, 0)
                        if (read < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEnd = true
                        } else {
                            codec.queueInputBuffer(inIndex, 0, read, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                when (val outIndex = codec.dequeueOutputBuffer(info, 10_000)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val out = codec.outputFormat
                        sampleRate = out.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = out.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outIndex >= 0) {
                        if (info.size > 0) {
                            val buffer = codec.getOutputBuffer(outIndex)!!
                            val shorts = ShortArray(info.size / 2)
                            buffer.order(ByteOrder.nativeOrder()).asShortBuffer().get(shorts)
                            pcm.add(shorts)
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            sawOutputEnd = true
                        }
                    }
                }
            }
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }

        val mono = toMonoFloat(pcm, channels)
        return resampleTo16k(mono, sampleRate)
    }

    private fun toMonoFloat(chunks: List<ShortArray>, channels: Int): FloatArray {
        val total = chunks.sumOf { it.size } / channels
        val out = FloatArray(total)
        var w = 0
        for (chunk in chunks) {
            var i = 0
            while (i + channels <= chunk.size) {
                out[w++] = chunk[i] / 32768f      // 다채널이면 첫 채널만 — readWavAsset 과 같다
                i += channels
            }
        }
        return out
    }

    /**
     * 16kHz 로 리샘플. 정수배(48k -> 16k)는 저역통과 FIR 뒤 솎아내고,
     * 그 밖의 배율은 선형 보간으로 처리한다.
     */
    private fun resampleTo16k(samples: FloatArray, rate: Int): FloatArray {
        if (rate == TARGET_RATE || samples.isEmpty()) return samples
        if (rate % TARGET_RATE == 0) {
            val factor = rate / TARGET_RATE
            val filtered = lowpass(samples, cutoff = 0.45 * TARGET_RATE, rate = rate)
            val out = FloatArray(filtered.size / factor)
            for (i in out.indices) out[i] = filtered[i * factor]
            return out
        }
        val ratio = rate.toDouble() / TARGET_RATE
        val out = FloatArray((samples.size / ratio).toInt())
        for (i in out.indices) {
            val pos = i * ratio
            val left = pos.toInt().coerceAtMost(samples.size - 1)
            val right = (left + 1).coerceAtMost(samples.size - 1)
            val frac = (pos - left).toFloat()
            out[i] = samples[left] * (1 - frac) + samples[right] * frac
        }
        return out
    }

    /** 63탭 해밍창 sinc 저역통과 — 데시메이션 전 앨리어싱 제거용. */
    private fun lowpass(samples: FloatArray, cutoff: Double, rate: Int): FloatArray {
        val taps = 63
        val half = taps / 2
        val fc = cutoff / rate
        val kernel = DoubleArray(taps)
        var sum = 0.0
        for (i in 0 until taps) {
            val n = i - half
            val sinc = if (n == 0) 2 * fc else sin(2 * PI * fc * n) / (PI * n)
            val window = 0.54 - 0.46 * kotlin.math.cos(2 * PI * i / (taps - 1))
            kernel[i] = sinc * window
            sum += kernel[i]
        }
        for (i in 0 until taps) kernel[i] /= sum

        val out = FloatArray(samples.size)
        for (i in samples.indices) {
            var acc = 0.0
            for (k in 0 until taps) {
                val j = i + k - half
                if (j in samples.indices) acc += samples[j] * kernel[k]
            }
            out[i] = acc.toFloat()
        }
        return out
    }
}
