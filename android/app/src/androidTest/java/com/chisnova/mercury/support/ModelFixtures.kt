package com.chisnova.mercury.support

import com.chisnova.mercury.asr.AsrController
import java.io.File

/**
 * AsrController 의 모델 검증 로직을 조작 가능한 소형 모델 디렉터리로 시험하기 위한 픽스처.
 *
 * 실제 모델(1000여 개 파일, 400MB)을 손상시켜 가며 시험할 수는 없으므로,
 * 구조만 동일하고 크기는 작은 가짜 모델을 만들어 쓴다.
 */
object ModelFixtures {

    const val VOCAB_NAME = "fixture.vocab"
    const val VOCAB_MODEL = "fixture.model"
    const val PHONE_NAME = "phone.txt"
    const val DICT_NAME = "full.dict"

    /** bias 플래그가 1 이라 반드시 존재해야 하는 텐서. */
    val REQUIRED_TENSORS = listOf(
        "fixture_layer_norm.weight.pt",
        "fixture_layer_norm.bias.pt",
        "fixture_linear.weight.pt",
        "fixture_linear.bias.pt",
        "fixture_conv_pointwise_weight.pt",
        "fixture_conv_pointwise_bias.pt"
    )

    /** bias 플래그가 0 이라 네이티브가 읽지 않는 텐서 — 파일이 없어도 정상이어야 한다. */
    const val SKIPPED_BIAS_TENSOR = "fixture_conv_depthwise_bias.pt"
    const val DEPTHWISE_WEIGHT = "fixture_conv_depthwise_weight.pt"

    /**
     * 정상 모델 디렉터리를 만든다.
     * 실제 모델과 마찬가지로 [SKIPPED_BIAS_TENSOR] 는 일부러 만들지 않는다.
     */
    fun createValidModel(dir: File): File {
        if (dir.exists()) dir.deleteRecursively()
        dir.mkdirs()
        writeConfig(dir, CONFIG_TEMPLATE)
        for (name in REQUIRED_TENSORS + DEPTHWISE_WEIGHT) writeTensor(dir, name)
        File(dir, VOCAB_NAME).writeText("<pad>\t0\n<unk>\t0\n")
        File(dir, VOCAB_MODEL).writeBytes(ByteArray(64) { 1 })
        File(dir, PHONE_NAME).writeText("<pad>\nSIL\n")
        File(dir, DICT_NAME).writeText("A\tAA\n")
        return dir
    }

    fun writeConfig(dir: File, json: String) {
        File(dir, ModelConfigReader.CONFIG_NAME).writeText(json)
    }

    fun writeTensor(dir: File, name: String, bytes: Int = 16) {
        File(dir, name).writeBytes(ByteArray(bytes) { it.toByte() })
    }

    /** private 인 AsrController.validateModelFiles 를 그대로 호출한다. */
    fun validate(modelDir: File): Boolean {
        val method = AsrController::class.java
            .getDeclaredMethod("validateModelFiles", File::class.java)
        method.isAccessible = true
        return method.invoke(AsrController.getInstance(), modelDir) as Boolean
    }

    /**
     * 실제 model.json 과 같은 형태의 축소판.
     * - layer_norm  : weight/bias 쌍
     * - linear      : weight/bias 쌍
     * - pointwise   : bias=1 인 합성곱 (bias_path 필수)
     * - depthwise   : bias=0 인 합성곱 (bias_path 무시되어야 함)
     */
    val CONFIG_TEMPLATE = """
        {
          "model_path": {
            "format_version": 1,
            "dtype": "float32",
            "model_type": "conformer_stream",
            "vocab_name": "$VOCAB_NAME",
            "vocab_model": "$VOCAB_MODEL",
            "phone_name": "$PHONE_NAME",
            "dict_name": "$DICT_NAME",
            "vocab_size": 2,
            "phone_size": 2,
            "model": {
              "stream_encoder": {
                "layer_norm": {
                  "weight": "fixture_layer_norm.weight.pt",
                  "bias": "fixture_layer_norm.bias.pt",
                  "shape": [1, 2]
                },
                "linear": {
                  "weight": "fixture_linear.weight.pt",
                  "bias": "fixture_linear.bias.pt",
                  "shape": [2, 2]
                },
                "conv_module": {
                  "pointwise1": {
                    "in_channels": 2, "out_channels": 2, "kernel_size": 1,
                    "stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": 1,
                    "weight_path": "fixture_conv_pointwise_weight.pt",
                    "bias_path": "fixture_conv_pointwise_bias.pt",
                    "shape": [2, 2]
                  },
                  "depthwise": {
                    "in_channels": 2, "out_channels": 2, "kernel_size": 3,
                    "stride": 1, "padding": 0, "dilation": 1, "groups": 2, "bias": 0,
                    "weight_path": "$DEPTHWISE_WEIGHT",
                    "bias_path": "$SKIPPED_BIAS_TENSOR",
                    "shape": [2, 3]
                  }
                }
              }
            }
          }
        }
    """.trimIndent()
}
