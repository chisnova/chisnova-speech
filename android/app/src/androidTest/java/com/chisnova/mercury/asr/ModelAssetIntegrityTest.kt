package com.chisnova.mercury.asr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.filters.MediumTest
import com.chisnova.mercury.support.AsrTestSupport
import com.chisnova.mercury.support.ModelConfigReader
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * APK 에 실제로 담긴 모델 자산이 model.json 과 앞뒤가 맞는지 검사한다.
 *
 * 모델은 assets/copy.sh 로 통째로 갈아끼우기 때문에, 옛 파일이 남거나 새 파일이 빠지는 사고가
 * 가장 흔한 고장 원인이다. 그런 불일치는 네이티브 초기화 중 크래시로만 드러나므로
 * 자산 단계에서 먼저 잡는다.
 */
@RunWith(AndroidJUnit4::class)
@MediumTest
class ModelAssetIntegrityTest {

    companion object {
        private const val TAG = "[AsrTest]"
        private const val TENSOR_SIZE_SAMPLES = 12
        private const val DICT_LINES_TO_CHECK = 500

        private lateinit var modelPath: JSONObject
        private lateinit var packagedNames: Set<String>

        @BeforeClass
        @JvmStatic
        fun readConfig() {
            val names = AsrTestSupport.assetNames(AsrTestSupport.MODEL_ASSET_DIR)
            assertTrue(
                "assets/model_state 가 비어 있다 — assets/copy.sh 로 모델을 받아야 한다",
                names.isNotEmpty()
            )
            packagedNames = names.toSet()
            modelPath = ModelConfigReader.parse(
                AsrTestSupport.readAssetText(
                    "${AsrTestSupport.MODEL_ASSET_DIR}/${ModelConfigReader.CONFIG_NAME}"
                )
            )
        }
    }

    @Test
    fun configDeclaresDenseFloat32Model() {
        assertEquals("float32", modelPath.optString("dtype"))
        assertTrue("model_type 이 비어 있다", modelPath.optString("model_type").isNotEmpty())
        assertTrue("vocab_size 가 없다", modelPath.optInt("vocab_size", -1) > 0)
        assertTrue("phone_size 가 없다", modelPath.optInt("phone_size", -1) > 0)
        assertTrue("model 트리가 없다", modelPath.optJSONObject("model") != null)
    }

    @Test
    fun auxiliaryFilesDeclaredByConfigArePackaged() {
        for (key in ModelConfigReader.AUX_FILE_KEYS) {
            val name = modelPath.optString(key)
            assertTrue("config 에 $key 선언이 없다", name.isNotEmpty())
            assertTrue("config 가 선언한 $key='$name' 이 assets 에 없다", name in packagedNames)
            assertTrue(
                "$name 이 0바이트다",
                AsrTestSupport.assetByteSize("${AsrTestSupport.MODEL_ASSET_DIR}/$name") > 0
            )
        }
    }

    @Test
    fun everyTensorReferencedByConfigIsPackaged() {
        val required = ModelConfigReader.requiredTensors(modelPath)
        assertTrue("config 가 텐서를 하나도 참조하지 않는다", required.isNotEmpty())
        val missing = required.filterNot { it in packagedNames }
        assertTrue(
            "config 가 참조하지만 APK 에 없는 텐서 ${missing.size}개: ${missing.take(10)}",
            missing.isEmpty()
        )
        Log.i(TAG, "referenced tensors=${required.size}, packaged files=${packagedNames.size}")
    }

    /** 모델 교체 후 옛 텐서가 남으면 APK 만 수백 MB 커지고 어느 모델이 실린 건지 알 수 없게 된다. */
    @Test
    fun noStaleTensorRemainsInAssets() {
        val referenced = ModelConfigReader.requiredTensors(modelPath).toSet() +
            ModelConfigReader.skippedBiasTensors(modelPath).toSet()
        val stale = packagedNames
            .filter { it.endsWith(".pt") }
            .filterNot { it in referenced }
        assertTrue(
            "config 가 참조하지 않는 잔여 텐서 ${stale.size}개: ${stale.take(10)}",
            stale.isEmpty()
        )
    }

    /** config 가 선언한 이름과 실제 파일이 이름만 맞고 내용이 다른 경우를 잡는다. */
    @Test
    fun sampledTensorSizesMatchDeclaredShapes() {
        val shapes = ModelConfigReader.weightShapes(modelPath).filterKeys { it in packagedNames }
        assertTrue("shape 가 선언된 weight 가 없다", shapes.isNotEmpty())
        val entries = shapes.entries.toList()
        val step = maxOf(1, entries.size / TENSOR_SIZE_SAMPLES)
        var checked = 0
        for (i in entries.indices step step) {
            val (name, shape) = entries[i]
            val expected = shape.fold(1L) { acc, dim -> acc * dim } * Float.SIZE_BYTES
            val actual = AsrTestSupport.assetByteSize("${AsrTestSupport.MODEL_ASSET_DIR}/$name")
            assertEquals("$name 크기가 shape $shape 와 다르다", expected, actual)
            checked++
        }
        Log.i(TAG, "tensor size sampled: $checked / ${entries.size}")
    }

    @Test
    fun vocabularyFileMatchesDeclaredVocabSize() {
        val name = modelPath.getString("vocab_name")
        val lines = AsrTestSupport
            .readAssetText("${AsrTestSupport.MODEL_ASSET_DIR}/$name")
            .lines()
            .filter { it.isNotBlank() }
        assertEquals("vocab_size 와 $name 의 항목 수가 다르다", modelPath.getInt("vocab_size"), lines.size)
    }

    @Test
    fun phoneFileMatchesDeclaredPhoneSize() {
        val name = modelPath.getString("phone_name")
        val phones = AsrTestSupport
            .readAssetText("${AsrTestSupport.MODEL_ASSET_DIR}/$name")
            .lines()
            .filter { it.isNotBlank() }
        assertEquals("phone_size 와 $name 의 항목 수가 다르다", modelPath.getInt("phone_size"), phones.size)
    }

    /** 발음 사전이 이 모델의 음소 집합으로만 쓰였는지 — 사전만 옛 것이 남는 사고를 잡는다. */
    @Test
    fun dictionaryUsesOnlyDeclaredPhones() {
        val phones = AsrTestSupport
            .readAssetText("${AsrTestSupport.MODEL_ASSET_DIR}/${modelPath.getString("phone_name")}")
            .lines()
            .filter { it.isNotBlank() }
            .toSet()

        val unknown = linkedSetOf<String>()
        var lineCount = 0
        AsrTestSupport.assets
            .open("${AsrTestSupport.MODEL_ASSET_DIR}/${modelPath.getString("dict_name")}")
            .bufferedReader()
            .use { reader ->
                while (lineCount < DICT_LINES_TO_CHECK) {
                    val line = reader.readLine() ?: break
                    lineCount++
                    val parts = line.split("\t", limit = 2)
                    if (parts.size < 2) continue
                    parts[1].split(" ")
                        .filter { it.isNotBlank() }
                        .filterNot { it in phones }
                        .forEach { unknown.add(it) }
                }
            }
        assertTrue("사전에서 $lineCount 줄을 읽지 못했다", lineCount > 0)
        assertTrue("phone.txt 에 없는 음소가 사전에 있다: ${unknown.take(10)}", unknown.isEmpty())
    }

    @Test
    fun referenceAudioAssetsArePackaged() {
        val packagedAudio = (AsrTestSupport.audioAssets.list(AsrTestSupport.AUDIO_ASSET_DIR) ?: emptyArray()).toSet()
        for (name in AsrTestSupport.REFERENCE_TRANSCRIPTS.keys) {
            assertTrue("기준 오디오 $name 이 assets 에 없다", name in packagedAudio)
        }
    }

    @Test
    fun nativeLibraryLoads() {
        System.loadLibrary("asr_api")
    }

    /**
     * 설치본 전수 검사 — filesDir 로 복사된 뒤에는 File.length() 로 빠르게 전부 확인할 수 있다.
     * 자산 압축/복사 중 잘림까지 여기서 걸린다.
     */
    @Test
    @LargeTest
    fun installedTensorSizesMatchDeclaredShapes() {
        val dir = AsrTestSupport.installedModelDir()
        val shapes = ModelConfigReader.weightShapes(modelPath)
        val mismatches = mutableListOf<String>()
        for ((name, shape) in shapes) {
            val expected = shape.fold(1L) { acc, dim -> acc * dim } * Float.SIZE_BYTES
            val actual = File(dir, name).length()
            if (actual != expected) mismatches.add("$name: expected=$expected actual=$actual shape=$shape")
        }
        assertTrue("크기가 어긋난 텐서 ${mismatches.size}개: ${mismatches.take(5)}", mismatches.isEmpty())
        Log.i(TAG, "installed tensor size verified: ${shapes.size}")
    }
}
