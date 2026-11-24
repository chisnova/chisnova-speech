package com.chisnova.mercury.support

import org.json.JSONArray
import org.json.JSONObject

/**
 * model.json(config)을 읽는 공용 헬퍼.
 *
 * 테스트가 파일명을 하드코딩하지 않도록, 앱의 AsrController 및 네이티브 ContextManager 와
 * 동일하게 config 가 선언한 이름만 사용한다.
 */
object ModelConfigReader {
    const val CONFIG_NAME = "model.json"
    const val MODEL_PATH_KEY = "model_path"

    /** config 가 선언하는 보조 파일 키 — 네이티브 ContextManager::init 이 읽는 것과 동일하다. */
    val AUX_FILE_KEYS = listOf("vocab_name", "vocab_model", "phone_name", "dict_name")

    private const val BIAS_PATH_KEY = "bias_path"
    private const val BIAS_FLAG_KEY = "bias"
    private const val TENSOR_SUFFIX = ".pt"

    fun parse(rawJson: String): JSONObject = JSONObject(rawJson).getJSONObject(MODEL_PATH_KEY)

    /**
     * 네이티브가 실제로 읽는 텐서 목록.
     * bias 플래그가 0 인 레이어의 bias_path 는 Convolution1D/2D 로더가 건너뛰므로 제외한다.
     */
    fun requiredTensors(modelPath: JSONObject): List<String> {
        val out = mutableListOf<String>()
        collect(modelPath, out, includeSkippedBias = false)
        return out
    }

    /** bias 플래그가 0 이라 로드되지 않는 bias_path 목록. */
    fun skippedBiasTensors(modelPath: JSONObject): List<String> {
        val all = mutableListOf<String>()
        collect(modelPath, all, includeSkippedBias = true)
        return all - requiredTensors(modelPath).toSet()
    }

    /**
     * weight 텐서 파일명 -> 선언된 shape.
     * bias 는 레이어 종류마다 shape 해석이 달라(out_channels vs shape[1]) 대상에서 제외한다.
     */
    fun weightShapes(modelPath: JSONObject): Map<String, List<Int>> {
        val out = mutableMapOf<String, List<Int>>()
        collectWeightShapes(modelPath, out)
        return out
    }

    private fun collect(node: Any?, out: MutableList<String>, includeSkippedBias: Boolean) {
        when (node) {
            is JSONObject -> {
                val skipBias = !includeSkippedBias &&
                    node.has(BIAS_PATH_KEY) && node.optInt(BIAS_FLAG_KEY, 1) == 0
                for (key in node.keys()) {
                    if (skipBias && key == BIAS_PATH_KEY) continue
                    collect(node.opt(key), out, includeSkippedBias)
                }
            }
            is JSONArray -> for (i in 0 until node.length()) collect(node.opt(i), out, includeSkippedBias)
            is String -> if (node.endsWith(TENSOR_SUFFIX)) out.add(node)
        }
    }

    private fun collectWeightShapes(node: Any?, out: MutableMap<String, List<Int>>) {
        when (node) {
            is JSONObject -> {
                val shape = node.optJSONArray("shape")?.let { arr ->
                    (0 until arr.length()).map { arr.optInt(it, -1) }
                }
                if (shape != null && shape.none { it <= 0 }) {
                    for (key in listOf("weight", "weight_path")) {
                        val name = node.optString(key)
                        if (name.endsWith(TENSOR_SUFFIX)) out[name] = shape
                    }
                }
                for (key in node.keys()) collectWeightShapes(node.opt(key), out)
            }
            is JSONArray -> for (i in 0 until node.length()) collectWeightShapes(node.opt(i), out)
        }
    }
}
