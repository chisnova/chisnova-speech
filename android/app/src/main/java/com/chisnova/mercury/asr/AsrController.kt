package com.chisnova.mercury.asr

import android.content.Context
import android.util.Log
import android.content.res.AssetManager
import com.chisnova.mercury.jni.AsrAPI
import java.io.InputStream
import java.io.FileOutputStream
import java.io.OutputStream
import java.io.File
import java.io.IOException
import com.chisnova.mercury.itf.DecoderCallback
import android.widget.TextView
import android.os.Handler
import android.os.Looper
import org.json.JSONArray
import org.json.JSONObject

class AsrController private constructor() {
    private var asr_api: AsrAPI? = null
    private var isInitialized = false
    private var isInitializing = false
    @Volatile
    private var isDecodingActive = false
    @Volatile
    private var decodingFailed = false
    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * asset 파일 하나를 filesDir로 복사한다.
     * @return 복사 성공(또는 이미 존재) 시 true, 실패 시 false
     */
    public fun copyFile(srcFile: String, assetMgr: AssetManager, context: Context): Boolean {
        try {
            val destFile = context.filesDir.toString() + File.separator + srcFile;
            val destFileObj = File(destFile);

            // 파일이 이미 존재하면 건너뛰기.
            // 설치 완료 표시(INSTALL_MARKER)가 없으면 호출자가 디렉터리를 통째로 지우고
            // 오므로, 여기 남아 있는 파일은 완결된 것이다.
            if (destFileObj.exists()) {
                return true;
            }

            // 먼저 임시 이름으로 받고 다 받은 뒤에만 진짜 이름을 준다.
            // 곧바로 최종 이름에 쓰면 도중에 끊겼을 때 잘린 파일이 그 이름으로 남고,
            // 다음 실행이 그것을 "이미 있다" 며 재사용해 앱이 영영 못 뜬다.
            // (Galaxy S9 에서 텐서 하나가 786432 대신 472064 바이트로 남아 SIGSEGV 가 났다.)
            val partFileObj = File("$destFile.part");
            var written = 0L
            assetMgr.open(srcFile).use { inputStream ->
                FileOutputStream(partFileObj).use { outputStream ->
                    val buffer = ByteArray(64 * 1024);
                    var read = inputStream.read(buffer);
                    while (read != -1) {
                        outputStream.write(buffer, 0, read);
                        written += read
                        read = inputStream.read(buffer);
                    }
                    outputStream.flush();
                    outputStream.fd.sync();      // 이름을 바꾸기 전에 실제로 디스크에 내린다
                }
            }
            if (!partFileObj.renameTo(destFileObj)) {
                Log.e("[AsrAPI]", "Failed to finalize asset: $srcFile");
                partFileObj.delete();
                return false;
            }
            return true;
        }
        catch (e: IOException) {
            // 저장공간이 모자라 도중에 끊기는 경우가 여기로 온다. 남은 조각을 지운다.
            Log.e("[AsrAPI]", "Failed to copy asset: $srcFile", e);
            File(context.filesDir.toString() + File.separator + srcFile + ".part").delete();
            return false;
        }
    }

    /**
     * asset 디렉토리를 재귀적으로 filesDir에 복사한다.
     * @return 모든 파일 복사 성공 시 true, 하나라도 실패 시 false
     */
    public fun copyAssetAll(srcPath: String, assetMgr: AssetManager, curContext: Context): Boolean {
        try {
            val assets = assetMgr.list(srcPath);
            if (assets != null) {
                if (assets.isEmpty()) {
                    return copyFile(srcPath, assetMgr, curContext)
                } else {
                    val destPath = curContext.filesDir.toString() + File.separator + srcPath;
                    val dir = File(destPath);
                    if (!dir.exists()) {
                        dir.mkdirs();
                    }
                    var allSuccess = true
                    for (element in assets) {
                        if (!copyAssetAll(srcPath + File.separator + element, assetMgr, curContext)) {
                            allSuccess = false
                        }
                    }
                    return allSuccess
                }
            }
            return true
        }
        catch (e: IOException) {
            Log.e("[AsrAPI]", "Failed to copy asset directory: $srcPath", e)
            return false
        }
    }

    // 마지막 초기화 오류 메시지 — UI에서 조회 가능
    var lastInitError: String? = null
        private set

    fun init(curContext: Context) {
        if (isInitialized) return
        ensureInstalledAndInit(curContext, null, null)
    }

    /**
     * 모델 파일이 올바르게 설치되었는지 검증한다.
     * 파일명을 하드코딩하지 않고 model.json(config)이 선언한 이름을 그대로 읽어 확인하므로
     * 모델이 교체되어도(vocab/model 파일명 변경 등) 이 코드를 고칠 필요가 없다.
     */
    private fun validateModelFiles(modelDir: File): Boolean {
        val configFile = File(modelDir, MODEL_CONFIG_NAME)
        if (!isReadable(configFile)) return false

        val modelPath = try {
            JSONObject(configFile.readText()).getJSONObject(MODEL_PATH_KEY)
        } catch (e: Exception) {
            Log.e("[AsrAPI]", "Failed to parse ${configFile.absolutePath}", e)
            return false
        }

        // config가 선언한 보조 파일(vocab, sentencepiece model, phone, dict) 검증
        for (key in AUX_FILE_KEYS) {
            val fileName = modelPath.optString(key)
            if (fileName.isEmpty()) {
                Log.w("[AsrAPI]", "$MODEL_CONFIG_NAME has no '$key' entry — skipping")
                continue
            }
            if (!isReadable(File(modelDir, fileName))) return false
        }

        // config가 참조하는 .pt 텐서를 균등 간격으로 샘플링해 확인
        // (수천 개 전부 확인하면 느리므로 대표 파일만 확인)
        val tensorPaths = collectTensorPaths(modelPath)
        if (tensorPaths.isEmpty()) {
            Log.e("[AsrAPI]", "$MODEL_CONFIG_NAME references no .pt tensor files")
            return false
        }
        val step = maxOf(1, tensorPaths.size / TENSOR_SAMPLE_COUNT)
        for (i in tensorPaths.indices step step) {
            if (!isReadable(File(modelDir, tensorPaths[i]))) {
                Log.e("[AsrAPI]", "The .pt model weight files are not included in APK assets. " +
                    "Please run copy.sh or ensure model files are bundled in assets/model_state/.")
                return false
            }
        }
        return true
    }

    /** 파일이 존재하고 비어 있지 않은지 확인한다. */
    private fun isReadable(f: File): Boolean {
        if (!f.exists() || f.length() == 0L) {
            Log.e("[AsrAPI]", "Required model file missing or empty: ${f.absolutePath}")
            return false
        }
        return true
    }

    /**
     * config 트리를 순회하며 참조된 .pt 파일 이름을 모은다.
     * bias가 0인 레이어의 bias_path는 네이티브에서도 로드하지 않으므로 제외한다.
     */
    private fun collectTensorPaths(
        node: Any?,
        out: MutableList<String> = mutableListOf()
    ): MutableList<String> {
        when (node) {
            is JSONObject -> {
                val skipBias = node.has(BIAS_PATH_KEY) && node.optInt(BIAS_FLAG_KEY, 1) == 0
                for (key in node.keys()) {
                    if (skipBias && key == BIAS_PATH_KEY) continue
                    collectTensorPaths(node.opt(key), out)
                }
            }
            is JSONArray -> for (i in 0 until node.length()) collectTensorPaths(node.opt(i), out)
            is String -> if (node.endsWith(TENSOR_SUFFIX)) out.add(node)
        }
        return out
    }

    private fun ensureInstalledAndInit(curContext: Context, onReady: (() -> Unit)?, onError: ((String) -> Unit)?) {
        if (isInitialized || isInitializing) {
            if (isInitialized) onReady?.let { Handler(Looper.getMainLooper()).post(it) }
            return
        }
        isInitializing = true
        lastInitError = null
        val mainHandler = Handler(Looper.getMainLooper())
        Thread {
            try {
                val destPath = curContext.filesDir.toString() + File.separator + "model_state"
                val destDir = File(destPath)

                // 지난번 설치가 끝까지 갔는지 표시로 확인한다. 표시가 없으면 어디까지
                // 복사됐는지 알 수 없으므로 통째로 지우고 처음부터 받는다 — 반쯤 남은
                // 파일을 재사용하면 네이티브가 그 파일을 읽다 죽는다.
                val marker = File(destDir, INSTALL_MARKER)
                if (destDir.exists() && !marker.exists()) {
                    Log.w("[AsrAPI]", "이전 설치가 끝나지 않았다 — 모델을 다시 받는다")
                    destDir.deleteRecursively()
                }
                if (!destDir.exists()) destDir.mkdirs()

                val copySuccess = copyAssetAll("model_state", curContext.assets, curContext)
                if (!copySuccess) {
                    // 예전에는 여기서 로그만 남기고 진행했다. 그러면 잘린 파일이 그대로
                    // 남아 네이티브가 널을 역참조하며 프로세스째 죽는다.
                    marker.delete()
                    val errorMsg = "모델 파일을 모두 내려받지 못했습니다. " +
                        "저장 공간을 확보한 뒤 다시 실행해 주세요."
                    Log.e("[AsrAPI]", errorMsg)
                    lastInitError = errorMsg
                    onError?.let { mainHandler.post { it(errorMsg) } }
                    return@Thread
                }

                // 필수 모델 파일 존재 여부 검증 — 없으면 네이티브 크래시 방지를 위해 초기화 중단
                if (!validateModelFiles(destDir)) {
                    val errorMsg = "ASR 모델 파일이 누락되었습니다. " +
                        "assets/model_state/ 에 .pt 텐서 파일이 포함되어 있는지 확인하세요."
                    Log.e("[AsrAPI]", errorMsg)
                    lastInitError = errorMsg
                    onError?.let { mainHandler.post { it(errorMsg) } }
                    return@Thread
                }

                Log.i("[AsrAPI]", "Engine Init Start")
                asr_api = AsrAPI(destPath)
                Log.i("[AsrAPI]", "Engine Init Done")
                isInitialized = true

                // 엔진이 실제로 떴다 — 파일이 온전하다는 가장 확실한 증거다.
                // 이 표시가 있어야 다음 실행이 재복사 없이 그대로 쓴다.
                File(destDir, INSTALL_MARKER).writeText("ok")
            } catch (e: Exception) {
                // 모델이 깨져 실패했을 수 있다. 표시를 지워 두면 다음 실행이 다시 받는다.
                File(File(curContext.filesDir, "model_state"), INSTALL_MARKER).delete()
                val errorMsg = "ASR 엔진 초기화 실패: ${e.message}"
                Log.e("[AsrAPI]", errorMsg, e)
                lastInitError = errorMsg
                onError?.let { mainHandler.post { it(errorMsg) } }
            } finally {
                isInitializing = false
                if (isInitialized) {
                    onReady?.let { mainHandler.post(it) }
                }
                Log.i("[AsrAPI]", "ASR Controller initialized: $isInitialized")
            }
        }.start()
    }

    fun initAsync(curContext: Context, onReady: (() -> Unit)? = null, onError: ((String) -> Unit)? = null) {
        ensureInstalledAndInit(curContext, onReady, onError)
    }

    // 기존 호출부 호환용 오버로드
    fun initAsync(curContext: Context, onReady: (() -> Unit)? = null) {
        ensureInstalledAndInit(curContext, onReady, null)
    }

    fun startBackgroundEngine(beamSize: Int, textView: TextView) {
        // Ensure UI updates happen on the main thread

        asr_api?.setCallBack { asrResult ->
            Log.i("[AsrAPI]", "Callback ASR RESULT: $asrResult");
            // UI 업데이트를 즉시 실행하여 응답성 향상
            if (Looper.myLooper() == Looper.getMainLooper()) {
                textView.text = asrResult
            } else {
                mainHandler.post { textView.text = asrResult }
            }
        }
        asr_api?.startBackGroundEngine(beamSize)
        isDecodingActive = true
        Log.i("[AsrAPI]", "startBackgroundEngine() — decoding session started (beamSize=$beamSize)")
    }

    fun sendAudio(floatBuffer: FloatArray, audioSize: Int) {
        asr_api?.sendAudio(floatBuffer, audioSize)
    }

    fun sendAudio(floatBuffer: FloatArray, audioSize: Int, isFinal: Boolean) {
        asr_api?.sendAudio(floatBuffer, audioSize, isFinal)
    }

    fun clear(){
        asr_api?.clear()
        decodingFailed = false
        isDecodingActive = false
    }

    fun finishDecoding(){
        try {
            asr_api?.finishDecoding()
        } catch (e: RuntimeException) {
            decodingFailed = true
            Log.e("[AsrAPI]", "Decoding failed", e)
            asr_api?.cancelDecoding()
        } finally {
            isDecodingActive = false
        }
        Log.i("[AsrAPI]", "finishDecoding() — decoding session finished")
    }

    // 디코딩이 활성 상태일 때만 cancelDecoding 호출 — InitState에서 호출 시 네이티브 에러 방지
    fun cancelDecoding() {
        if (!isInitialized) {
            Log.w("[AsrAPI]", "cancelDecoding() skipped: ASR not initialized")
            return
        }
        if (!isDecodingActive) {
            Log.w("[AsrAPI]", "cancelDecoding() skipped: no active decoding session")
            return
        }
        Log.i("[AsrAPI]", "cancelDecoding() called — cancelling active decoding session")
        asr_api?.cancelDecoding()
        isDecodingActive = false
    }


    /**
     * 발음 점수. 라벨과 함께 서브워드·음소 인덱스를 넘겨야 한다.
     *
     * 라벨만 받는 오버로드가 있었지만 네이티브 구현이 없어, JNI 가 같은 이름의 7-arg 함수에
     * 바인딩하면서 배열 인자를 쓰레기로 읽어 SIGSEGV 로 죽었다. 인덱스는 콘텐츠의 .jsonl 이
     * 갖고 있으므로 그것이 없으면 채점 자체를 건너뛴다.
     */
    /**
     * 스트리밍 CTC 가 본 "말이 멎은 뒤 흐른 시간"(ms). 엔진이 없거나 발화를 못 잡았으면 -1.
     * 발화 종료 판정에서 에너지 기준을 확인하는 데 쓴다([EndpointDetector.process]).
     */
    fun getCtcTrailingSilenceMs(): Int = asr_api?.getCtcTrailingSilenceMs() ?: -1

    /**
     * 문장별 최종 점수 비중(발음/인식, 합 1). 콘텐츠 .jsonl 의 score_weights 를
     * getPronounceScore 직전에 넘긴다. 음수(예: -1f, -1f)면 모델 기본 채점으로 되돌린다.
     */
    fun setSentenceScoreWeights(pronunciation: Float, recognition: Float) {
        if (!isInitialized) return
        asr_api?.setSentenceScoreWeights(pronunciation, recognition)
    }

    fun getPronounceScore(
        label: String,
        wordIdx: IntArray,
        wordIdxGroup: IntArray,
        phnIdx: IntArray,
        phnIdxGroup: IntArray
    ): String? {
        if (!isInitialized) {
            Log.e("[AsrAPI]", "ASR Controller not initialized")
            return null
        }
        if (decodingFailed) return null
        return try { asr_api?.getPronounceScore(
            label,
            wordIdx,
            wordIdxGroup,
            wordIdx.size,
            phnIdx,
            phnIdxGroup,
            phnIdx.size
        ) } catch (e: RuntimeException) {
            Log.e("[AsrAPI]", "Pronunciation scoring rejected", e)
            null
        }
    }

    /**
     * 발화가 끝난 뒤의 최종 인식 결과(2패스 CTC).
     *
     * 스트리밍 중에는 네이티브가 1패스 CTC 결과를 콜백으로 흘려보내고,
     * 마지막 청크를 처리할 때 이 값과 같은 2패스 결과를 콜백으로 한 번 더 보낸다.
     * 콜백을 쓰지 않고 직접 최종 결과가 필요할 때(저장·비교 등) 쓴다.
     */
    fun getCtcAsrResult(): String? {
        if (!isInitialized) {
            Log.e("[AsrAPI]", "ASR Controller not initialized")
            return null
        }
        if (decodingFailed) return null
        return try { asr_api?.getCtcAsrResult() } catch (e: RuntimeException) {
            Log.e("[AsrAPI]", "Could not read recognition result", e)
            null
        }
    }

    fun getSecondPassDecodingResult(num_nbest: Int, textView: TextView): String? {
        if (!isInitialized) {
            Log.e("[AsrAPI]", "ASR Controller not initialized")
            return null
        }
        val result = asr_api?.getSecondPassDecodingResult(num_nbest)
        Log.i("[AsrAPI]", "Second pass decoding result: $result")
        if (Looper.myLooper() == Looper.getMainLooper()) {
            textView.text = result
        } else {    
            mainHandler.post {
                Log.i("[AsrAPI]", "Second pass decoding result: $result")
                textView.text = result
            }
        }
        return result
    }

    fun isReady(): Boolean = isInitialized && asr_api != null

    companion object {
        // 전역 인스턴스
        private val instance = AsrController()

        // 전역 접근 메서드
        fun getInstance() = instance

        // 모델 구성 파일 및 그 안에서 파일명을 찾을 때 쓰는 키
        private const val MODEL_CONFIG_NAME = "model.json"
        private const val MODEL_PATH_KEY = "model_path"
        private const val BIAS_PATH_KEY = "bias_path"
        private const val BIAS_FLAG_KEY = "bias"
        private const val TENSOR_SUFFIX = ".pt"
        private val AUX_FILE_KEYS = listOf("vocab_name", "vocab_model", "phone_name", "dict_name")

        // 존재 여부를 확인할 .pt 텐서 샘플 개수
        private const val TENSOR_SAMPLE_COUNT = 8

        /** 모델 설치가 끝까지 갔음을 남기는 표시. 이 파일이 없으면 설치를 처음부터 다시 한다. */
        private const val INSTALL_MARKER = ".install_complete"
    }

    data class FinalScore(
        val avg_score: Float,
        val final_score: List<WordScore>
    )

    data class WordScore(
        val word: String,
        val score: Float,
        val phone_score: List<PhoneScore>,
        /** 엔진의 채점 제외 낱말(ungraded_words.txt)이면 false. 구형 JSON 에는 없으므로 null 은 채점됨으로 본다. */
        val graded: Boolean? = null
    ) {
        val isGraded: Boolean get() = graded != false
    }

    data class PhoneScore(
        val word: String,
        val score: Float
    )
}
