package com.chisnova.mercury.jni

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import com.chisnova.mercury.support.AsrTestSupport
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayInputStream

/**
 * 테스트용 WAV 파서의 경계 동작 시험.
 *
 * 이 파서는 기준 오디오를 읽어 ASR 시험에 먹이는 통로다. 여기서 조용히 잘못된 PCM 이 나오면
 * ASR 시험 결과 전체가 의미를 잃으므로, 정상 자산이 기대한 형식인지와
 * 깨진 입력을 예외로 걸러 내는지를 함께 확인한다.
 */
@RunWith(AndroidJUnit4::class)
@SmallTest
class WavFileParsingTest {

    private fun assetBytes(name: String): ByteArray =
        AsrTestSupport.audioAssets.open("${AsrTestSupport.AUDIO_ASSET_DIR}/$name").use { it.readBytes() }

    private fun openFrom(bytes: ByteArray): WavFile =
        WavFile.openWavFile(ByteArrayInputStream(bytes))

    /** fmt 청크 선두 오프셋(청크 ID 위치). */
    private fun fmtChunkOffset(bytes: ByteArray): Int {
        for (i in 12 until bytes.size - 8) {
            if (bytes[i] == 'f'.code.toByte() && bytes[i + 1] == 'm'.code.toByte() &&
                bytes[i + 2] == 't'.code.toByte() && bytes[i + 3] == ' '.code.toByte()
            ) return i
        }
        fail("fmt 청크를 찾지 못했다")
        return -1
    }

    @Test
    fun referenceAssetsAre16kHzMonoPcm() {
        for (name in AsrTestSupport.REFERENCE_TRANSCRIPTS.keys) {
            val wav = openFrom(assetBytes(name))
            assertEquals("$name 이 모노가 아니다", 1, wav.numChannels)
            assertEquals("$name 의 표본율이 다르다", AsrTestSupport.SAMPLE_RATE.toLong(), wav.sampleRate)
            assertEquals("$name 의 비트깊이가 다르다", 16, wav.validBits)
            assertTrue("$name 에 프레임이 없다", wav.numFrames > 0)
        }
    }

    @Test
    fun decodedSamplesStayWithinUnitRange() {
        val samples = AsrTestSupport.readWavAsset("google_tts003.wav")
        assertTrue("샘플이 하나도 없다", samples.isNotEmpty())
        val outOfRange = samples.count { !it.isFinite() || it < -1.5f || it > 1.5f }
        assertEquals("정규화 범위를 벗어난 샘플이 있다", 0, outOfRange)
        assertTrue("전 구간이 무음이다 — 자산이나 파서가 잘못됐다", samples.any { kotlin.math.abs(it) > 0.01f })
    }

    @Test
    fun readFramesStopsAtEndOfStream() {
        val wav = openFrom(assetBytes("google_tts003.wav"))
        val frames = wav.numFrames.toInt()
        val buffer = DoubleArray(frames)
        assertEquals(frames, wav.readFrames(buffer, frames))
        assertEquals("스트림 끝에서 0 이 아닌 값을 돌려줬다", 0, wav.readFrames(buffer, 1))
    }

    /** 요청 프레임이 버퍼보다 크면 조용히 잘리는 대신 예외가 나야 한다. */
    @Test
    fun readFramesBeyondBufferCapacityThrows() {
        val wav = openFrom(assetBytes("google_tts003.wav"))
        try {
            wav.readFrames(DoubleArray(8), 64)
            fail("버퍼보다 많은 프레임을 요구했는데 예외가 없다")
        } catch (expected: IndexOutOfBoundsException) {
            // 기대한 동작
        }
    }

    @Test
    fun rejectsEmptyStream() {
        assertThrowsWavFileException { openFrom(ByteArray(0)) }
    }

    @Test
    fun rejectsTruncatedHeader() {
        assertThrowsWavFileException { openFrom(assetBytes("google_tts003.wav").copyOf(8)) }
    }

    @Test
    fun rejectsNonRiffData() {
        val bytes = assetBytes("google_tts003.wav").copyOf()
        bytes[0] = 'X'.code.toByte()
        assertThrowsWavFileException { openFrom(bytes) }
    }

    /** RIFF 이지만 WAVE 가 아닌 컨테이너(예: AVI). */
    @Test
    fun rejectsNonWaveRiffType() {
        val bytes = assetBytes("google_tts003.wav").copyOf()
        bytes[8] = 'A'.code.toByte()
        bytes[9] = 'V'.code.toByte()
        bytes[10] = 'I'.code.toByte()
        bytes[11] = ' '.code.toByte()
        assertThrowsWavFileException { openFrom(bytes) }
    }

    /** 압축 코드가 1(PCM)이 아닌 파일 — 그대로 읽으면 잡음이 된다. */
    @Test
    fun rejectsNonPcmCompressionCode() {
        val bytes = assetBytes("google_tts003.wav").copyOf()
        val fmt = fmtChunkOffset(bytes)
        bytes[fmt + 8] = 3        // IEEE float
        bytes[fmt + 9] = 0
        assertThrowsWavFileException { openFrom(bytes) }
    }

    @Test
    fun rejectsZeroChannelHeader() {
        val bytes = assetBytes("google_tts003.wav").copyOf()
        val fmt = fmtChunkOffset(bytes)
        bytes[fmt + 10] = 0
        bytes[fmt + 11] = 0
        assertThrowsWavFileException { openFrom(bytes) }
    }

    /** blockAlign 이 비트깊이·채널수와 어긋난 헤더. */
    @Test
    fun rejectsInconsistentBlockAlign() {
        val bytes = assetBytes("google_tts003.wav").copyOf()
        val fmt = fmtChunkOffset(bytes)
        bytes[fmt + 20] = 7       // blockAlign
        bytes[fmt + 21] = 0
        assertThrowsWavFileException { openFrom(bytes) }
    }

    /** 데이터 청크가 헤더 선언보다 짧은 파일 — 다운로드가 끊긴 자산이 이 모양이 된다. */
    @Test
    fun rejectsTruncatedDataChunk() {
        val bytes = assetBytes("google_tts003.wav")
        val wav = openFrom(bytes.copyOf(bytes.size / 2))
        val frames = wav.numFrames.toInt()
        assertThrowsWavFileException { wav.readFrames(DoubleArray(frames), frames) }
    }

    private fun assertThrowsWavFileException(block: () -> Unit) {
        try {
            block()
            fail("WavFileException 이 발생하지 않았다")
        } catch (expected: WavFileException) {
            // 기대한 동작
        }
    }
}
