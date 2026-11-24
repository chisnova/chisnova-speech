package com.chisnova.mercury.support

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import com.chisnova.mercury.jni.AsrAPI
import com.chisnova.mercury.jni.WavFile
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.min

/**
 * ASR 계측 테스트 공용 지원 코드.
 *
 * 모델 설치(약 400MB)와 네이티브 엔진 생성은 매우 무거우므로 프로세스당 한 번만 수행하고
 * 모든 테스트가 공유한다. 각 테스트는 [reset] 으로 엔진을 InitState 로 되돌린 뒤 시작한다.
 */
object AsrTestSupport {
    const val MODEL_ASSET_DIR = "model_state"
    const val AUDIO_ASSET_DIR = "google_steve"
    const val SAMPLE_RATE = 16000
    const val CHUNK_SAMPLES = 160          // 10ms @ 16kHz — 앱 녹음 경로와 동일한 단위
    const val DEFAULT_NBEST = 3

    private const val TAG = "[AsrTest]"

    /** google_steve 자산의 기준 전사. 파일명 순서와 1:1 대응한다. */
    val REFERENCE_TRANSCRIPTS = linkedMapOf(
        "google_tts001.wav" to
            "i am honored to be with you today at your commencement from one of the finest universities in the world",
        "google_tts002.wav" to
            "truth be told I never graduated from college and ah this is the closest I've ever gotten to a college graduation",
        "google_tts003.wav" to
            "today i want to tell you three stories",
        "google_tts004.wav" to
            "that's it no big deal just three stories"
    )

    @Volatile private var installedDir: File? = null
    @Volatile private var sharedEngine: AsrAPI? = null

    val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    val assets: AssetManager
        get() = context.assets

    val audioAssets: AssetManager
        get() = InstrumentationRegistry.getInstrumentation().context.assets

    fun assetNames(dir: String): List<String> = (assets.list(dir) ?: emptyArray()).toList()

    fun readAssetText(path: String): String =
        assets.open(path).bufferedReader().use { it.readText() }

    /** 자산 하나의 실제 바이트 수. 압축 자산은 available() 이 부정확하므로 끝까지 읽는다. */
    fun assetByteSize(path: String): Long {
        var total = 0L
        assets.open(path).use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val read = input.read(buf)
                if (read <= 0) break
                total += read
            }
        }
        return total
    }

    /** assets/model_state 를 filesDir 로 설치한다. 이미 설치돼 있으면 재사용한다. */
    @Synchronized
    fun installedModelDir(): File {
        installedDir?.let { return it }
        val names = assetNames(MODEL_ASSET_DIR)
        check(names.isNotEmpty()) {
            "assets/$MODEL_ASSET_DIR 가 비어 있다 — assets/copy.sh 로 모델을 내려받았는지 확인하라"
        }
        val dest = File(context.filesDir, MODEL_ASSET_DIR)
        if (!dest.exists()) dest.mkdirs()
        var copied = 0
        for (name in names) {
            val out = File(dest, name)
            if (out.exists() && out.length() > 0L) continue
            assets.open("$MODEL_ASSET_DIR/$name").use { input ->
                out.outputStream().use { output -> input.copyTo(output, 64 * 1024) }
            }
            copied++
        }
        Log.i(TAG, "model installed at ${dest.absolutePath} (assets=${names.size}, copied=$copied)")
        installedDir = dest
        return dest
    }

    /** 프로세스당 하나의 네이티브 엔진. */
    @Synchronized
    fun engine(): AsrAPI {
        sharedEngine?.let { return it }
        val dir = installedModelDir()
        Log.i(TAG, "native engine init start")
        val api = AsrAPI(dir.absolutePath)
        Log.i(TAG, "native engine init done")
        sharedEngine = api
        return api
    }

    /**
     * 엔진을 InitState 로 되돌린다.
     * clear() 는 Running/Finishing 어느 상태에서도 허용되므로 테스트 간 격리에 쓸 수 있다.
     */
    fun reset() {
        engine().clear()
    }

    /** wav 자산을 모노 float PCM 으로 읽는다. 다채널이면 첫 채널만 사용한다. */
    fun readWavAsset(fileName: String): FloatArray {
        audioAssets.open("$AUDIO_ASSET_DIR/$fileName").use { stream ->
            val wav = WavFile.openWavFile(stream)
            val channels = wav.numChannels
            val frames = wav.numFrames.toInt()
            val interleaved = DoubleArray(frames * channels)
            val framesRead = wav.readFrames(interleaved, frames)
            return FloatArray(framesRead) { i -> interleaved[i * channels].toFloat() }
        }
    }

    /**
     * 오디오를 청크 단위로 흘려보낸다.
     *
     * @param markFinal 마지막 청크에 isFinal=true 를 실을지. false 면 세션이 계속 열려 있어
     *                  "디코딩 진행 중" 상태를 만들 수 있다.
     */
    fun stream(
        api: AsrAPI,
        samples: FloatArray,
        chunk: Int = CHUNK_SAMPLES,
        sleepMillis: Long = 0L,
        markFinal: Boolean = true
    ) {
        if (samples.isEmpty()) {
            // 보낼 샘플이 없으면 size=0 인 최종 청크만 보낸다(앱의 무음 종료 경로와 동일).
            if (markFinal) api.sendAudio(FloatArray(chunk), 0, true)
            return
        }
        val buffer = FloatArray(chunk)
        var offset = 0
        while (offset < samples.size) {
            val n = min(chunk, samples.size - offset)
            System.arraycopy(samples, offset, buffer, 0, n)
            offset += n
            api.sendAudio(buffer, n, markFinal && offset >= samples.size)
            if (sleepMillis > 0) Thread.sleep(sleepMillis)
        }
    }

    /**
     * 한 세션을 끝까지 돌린다. 반환 후 엔진은 FinishingState 이므로
     * 호출자는 getPronounceScore 를 이어서 부를 수 있고, 마친 뒤 [reset] 해야 한다.
     */
    fun decode(
        samples: FloatArray,
        nbest: Int = DEFAULT_NBEST,
        chunk: Int = CHUNK_SAMPLES,
        withSecondPass: Boolean = true
    ): Decoded {
        val api = engine()
        val partials = CopyOnWriteArrayList<String>()
        api.setCallBack { result -> partials.add(result ?: "") }
        api.startBackGroundEngine(nbest)
        stream(api, samples, chunk)
        api.finishDecoding()
        val ctc = api.ctcAsrResult ?: ""
        val secondPass = if (withSecondPass) api.getSecondPassDecodingResult(nbest) ?: "" else ""
        return Decoded(ctc, secondPass, partials.toList())
    }

    data class Decoded(
        val ctc: String,
        val secondPass: String,
        val partials: List<String>
    )

    /**
     * 발음 채점에 필요한 인덱스 묶음.
     *
     * 라벨만 넘기는 API 는 네이티브 구현이 없어 제거됐다(같은 이름의 7-arg 함수에 바인딩되어
     * SIGSEGV). 실제 앱과 마찬가지로 콘텐츠의 .jsonl 이 갖고 있는 인덱스를 그대로 쓴다.
     */
    data class PronounceMeta(
        val label: String,
        val wordIdx: IntArray,
        val wordGroup: IntArray,
        val phoneIdx: IntArray,
        val phoneGroup: IntArray,
        /** 문장별 최종 점수 비중(발음, 인식). 콘텐츠 .jsonl 의 score_weights — 대부분 없다. */
        val scoreWeights: Pair<Float, Float>? = null
    )

    /** 콘텐츠 .jsonl 의 한 줄을 읽어 채점 인자로 만든다. */
    fun readPronounceMeta(
        assetPath: String = "contents/1.basic/greeting_basic_expression/greeting_basic_expression.jsonl",
        lineIndex: Int = 0
    ): PronounceMeta {
        val line = assets.open(assetPath).bufferedReader().use { reader ->
            reader.lineSequence().filter { it.isNotBlank() }.drop(lineIndex).first()
        }
        val record = org.json.JSONObject(line)
        fun ints(key: String): IntArray {
            val array = record.getJSONArray(key)
            return IntArray(array.length()) { array.getInt(it) }
        }
        val org = record.getJSONArray("org")
        val label = (0 until org.length()).joinToString(" ") { org.getString(it) }
        return PronounceMeta(
            label = label,
            wordIdx = ints("word_idx"),
            wordGroup = ints("word_idx_group"),
            phoneIdx = ints("phn_idx"),
            phoneGroup = ints("phn_idx_group")
        )
    }

    /** 메타 그대로 채점을 호출한다. */
    fun pronounceScore(meta: PronounceMeta, label: String = meta.label): String =
        engine().getPronounceScore(
            label,
            meta.wordIdx, meta.wordGroup, meta.wordIdx.size,
            meta.phoneIdx, meta.phoneGroup, meta.phoneIdx.size
        ) ?: ""

    /** 무음 PCM. */
    fun silence(durationMillis: Int): FloatArray =
        FloatArray(SAMPLE_RATE * durationMillis / 1000)

    /** 재현 가능한 저진폭 화이트 노이즈 — 무음과 달리 VAD/특징 추출 경로를 실제로 태운다. */
    fun noise(durationMillis: Int, seed: Long = 20260820L, amplitude: Float = 0.02f): FloatArray {
        val random = java.util.Random(seed)
        return FloatArray(SAMPLE_RATE * durationMillis / 1000) {
            (random.nextGaussian().toFloat() * amplitude).coerceIn(-1f, 1f)
        }
    }

    /** 대소문자·구두점을 지운 비교용 표기. */
    fun normalize(text: String): String = text
        .lowercase()
        .replace(Regex("[^a-z0-9' ]"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()

    fun words(text: String): List<String> =
        normalize(text).split(" ").filter { it.isNotEmpty() }

    /** 단어 단위 편집거리 기반 WER. reference 가 비어 있으면 0 또는 1 을 돌려준다. */
    fun wordErrorRate(reference: String, hypothesis: String): Double {
        val ref = words(reference)
        val hyp = words(hypothesis)
        if (ref.isEmpty()) return if (hyp.isEmpty()) 0.0 else 1.0
        var prev = IntArray(hyp.size + 1) { it }
        for (i in 1..ref.size) {
            val cur = IntArray(hyp.size + 1)
            cur[0] = i
            for (j in 1..hyp.size) {
                val substitution = prev[j - 1] + if (ref[i - 1] == hyp[j - 1]) 0 else 1
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, substitution)
            }
            prev = cur
        }
        return prev[hyp.size].toDouble() / ref.size
    }
}
