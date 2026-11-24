package com.chisnova.mercury.asr

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.SmallTest
import com.chisnova.mercury.support.AsrTestSupport
import com.chisnova.mercury.support.ModelConfigReader
import com.chisnova.mercury.support.ModelFixtures
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * AsrController 의 모델 검증(validateModelFiles) 계약 시험.
 *
 * 이 검증은 네이티브 초기화 직전의 마지막 방어선이다. 모델 파일이 빠졌는데도 통과하면
 * C++ 로더가 없는 파일을 열다 프로세스째 죽고, 반대로 멀쩡한 모델을 거부하면 앱이 아예 뜨지 않는다.
 * 두 방향 모두를 조작된 모델 디렉터리로 확인한다.
 */
@RunWith(AndroidJUnit4::class)
@SmallTest
class ModelValidationTest {

    private lateinit var modelDir: File

    @Before
    fun setUp() {
        modelDir = File(AsrTestSupport.context.cacheDir, "model_validation_fixture")
        ModelFixtures.createValidModel(modelDir)
    }

    @After
    fun tearDown() {
        modelDir.deleteRecursively()
    }

    @Test
    fun acceptsWellFormedModelDirectory() {
        assertTrue("정상 픽스처가 거부되면 앱이 기동할 수 없다", ModelFixtures.validate(modelDir))
    }

    /**
     * 실제 모델의 depthwise 합성곱은 bias=0 이라 bias_path 파일이 아예 존재하지 않는다.
     * 이 항목을 필수로 취급하면 정상 모델이 통째로 거부된다.
     */
    @Test
    fun acceptsModelWhereZeroFlaggedBiasTensorIsAbsent() {
        val skipped = File(modelDir, ModelFixtures.SKIPPED_BIAS_TENSOR)
        assertFalse("픽스처 전제 위반: bias=0 텐서는 없어야 한다", skipped.exists())
        assertTrue(ModelFixtures.validate(modelDir))
    }

    /** 반대로 bias=1 인 텐서가 없으면 반드시 거부해야 한다 — 네이티브가 실제로 읽는 파일이다. */
    @Test
    fun rejectsMissingBiasTensorWhenFlagIsOne() {
        assertTrue(File(modelDir, "fixture_conv_pointwise_bias.pt").delete())
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsMissingConfig() {
        assertTrue(File(modelDir, ModelConfigReader.CONFIG_NAME).delete())
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsEmptyConfig() {
        ModelFixtures.writeConfig(modelDir, "")
        assertFalse(ModelFixtures.validate(modelDir))
    }

    /** 자산 복사가 중간에 끊기면 잘린 JSON 이 남을 수 있다. 예외가 밖으로 새지 않아야 한다. */
    @Test
    fun rejectsTruncatedConfigWithoutThrowing() {
        val full = ModelFixtures.CONFIG_TEMPLATE
        ModelFixtures.writeConfig(modelDir, full.substring(0, full.length / 2))
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsConfigWithoutModelPathObject() {
        ModelFixtures.writeConfig(modelDir, """{"something_else": {"vocab_name": "x"}}""")
        assertFalse(ModelFixtures.validate(modelDir))
    }

    /** model_path 가 객체가 아니라 문자열인 경우 — JSONObject 캐스팅 예외를 삼켜야 한다. */
    @Test
    fun rejectsConfigWhereModelPathIsNotAnObject() {
        ModelFixtures.writeConfig(modelDir, """{"model_path": "conformer_stream"}""")
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsMissingVocabDeclaredByConfig() {
        assertTrue(File(modelDir, ModelFixtures.VOCAB_NAME).delete())
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsMissingSentencePieceModelDeclaredByConfig() {
        assertTrue(File(modelDir, ModelFixtures.VOCAB_MODEL).delete())
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsMissingDictionaryDeclaredByConfig() {
        assertTrue(File(modelDir, ModelFixtures.DICT_NAME).delete())
        assertFalse(ModelFixtures.validate(modelDir))
    }

    /** 디스크가 가득 찬 상태에서 복사되면 0바이트 파일이 남는다 — 존재만으로 통과시키면 안 된다. */
    @Test
    fun rejectsZeroLengthAuxiliaryFile() {
        File(modelDir, ModelFixtures.DICT_NAME).writeBytes(ByteArray(0))
        assertFalse(ModelFixtures.validate(modelDir))
    }

    @Test
    fun rejectsZeroLengthTensor() {
        File(modelDir, ModelFixtures.REQUIRED_TENSORS.first()).writeBytes(ByteArray(0))
        assertFalse(ModelFixtures.validate(modelDir))
    }

    /** config 가 이름을 바꿔도(모델 교체) 코드 수정 없이 새 이름을 따라가야 한다. */
    @Test
    fun followsRenamedVocabDeclaredByConfig() {
        val renamed = "renamed_v2.vocab"
        assertTrue(File(modelDir, ModelFixtures.VOCAB_NAME).renameTo(File(modelDir, renamed)))
        assertFalse("이름만 바꾼 상태에서는 거부돼야 한다", ModelFixtures.validate(modelDir))

        ModelFixtures.writeConfig(
            modelDir,
            ModelFixtures.CONFIG_TEMPLATE.replace(ModelFixtures.VOCAB_NAME, renamed)
        )
        assertTrue("config 가 새 이름을 선언하면 통과해야 한다", ModelFixtures.validate(modelDir))
    }

    /** 보조 파일 키가 아예 없는 config 는 경고만 남기고 통과한다(선택 항목 취급). */
    @Test
    fun toleratesConfigWithoutOptionalAuxKeys() {
        val stripped = ModelFixtures.CONFIG_TEMPLATE
            .replace(Regex("\"dict_name\"\\s*:\\s*\"[^\"]*\",\\s*"), "")
        ModelFixtures.writeConfig(modelDir, stripped)
        assertTrue(ModelFixtures.validate(modelDir))
    }

    /** 텐서를 하나도 선언하지 않은 config 는 모델이 아니다. */
    @Test
    fun rejectsConfigWithoutAnyTensorReference() {
        ModelFixtures.writeConfig(
            modelDir,
            """
            {"model_path": {
              "vocab_name": "${ModelFixtures.VOCAB_NAME}",
              "vocab_model": "${ModelFixtures.VOCAB_MODEL}",
              "phone_name": "${ModelFixtures.PHONE_NAME}",
              "dict_name": "${ModelFixtures.DICT_NAME}",
              "model": {}
            }}
            """.trimIndent()
        )
        assertFalse(ModelFixtures.validate(modelDir))
    }

    /** 존재하지 않는 디렉터리를 넘겨도 예외 없이 false 여야 한다. */
    @Test
    fun rejectsMissingDirectory() {
        assertFalse(ModelFixtures.validate(File(modelDir, "does_not_exist")))
    }

    /**
     * 텐서 수가 표본 개수 이하인 모델은 전수 검사돼야 한다.
     * config 뒤쪽에 선언된 텐서를 지워도 검증이 실패해야 한다.
     */
    @Test
    fun validatesEveryTensorWhenModelIsSmallerThanSampleBudget() {
        val tensors = ModelConfigReader.requiredTensors(
            ModelConfigReader.parse(ModelFixtures.CONFIG_TEMPLATE)
        )
        assertTrue("픽스처에 텐서가 충분하지 않다", tensors.size >= 4)
        assertTrue(File(modelDir, tensors.last()).delete())
        assertFalse(ModelFixtures.validate(modelDir))
    }
}
