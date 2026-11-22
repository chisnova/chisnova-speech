#include "json_loader/json_object.h"
#include "module/conformer_conv_module.h"
#include "module/conformer_subsample.h"
#include "module/conformer_feedforward.h"
#include "module/conformer_head.h"
#include "module/conformer_multihead.h"
#include "module/conformer_attention.h"
#include "module/transformer_feedforward.h"
#include "module/transformer_head.h"
#include "module/transformer_multihead.h"
#include "module/conformer_encoder_layer.h"
#include "module/conformer_encoder.h"
#include "module/conformer_decoder.h"
#include "module/transformer_decoder.h"
#include "module/transformer_encoder.h"
#include "module/lstm_layer.h"
#include "module/transformer_transducer_layer.h"
#include "module/transformer_transducer.h"
#include "module/conformer_stream_head.h"
#include "module/conformer_stream_multihead.h"
#include "module/conformer_stream_attention.h"
#include "module/conformer_stream_conv_module.h"
#include "module/conformer_stream_encoder_layer.h"
#include "module/conformer_stream_encoder.h"
#include "module/conformer_stream_encoderV2.h"
#include "module/conformer_ctc.h"
#include "decoder/asr_stream_conformer_trf_transducer.h"
#include <ctime>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>

using namespace MERCURY;

static constexpr float error_threshold = 0.00001f;

std::shared_ptr<Tensor<float>> makeInput(JsonObject input_json, const std::string& base_path) {
    auto input = std::make_shared<Tensor<float>>();
    std::string input_path = base_path + "/" + input_json.parserString("value");
    std::vector<int> shape_input = input_json.parserIntArray("shape");
    input->loadValue(input_path, shape_input);
    return input;
}

std::shared_ptr<Tensor<int>> makeIntegerInput(JsonObject input_json, const std::string& base_path) {
    auto input = std::make_shared<Tensor<int>>();
    std::string input_path = base_path + "/" + input_json.parserString("value");
    std::vector<int> shape_input = input_json.parserIntArray("shape");
    input->loadValue(input_path, shape_input);
    return input;
}

std::shared_ptr<Tensor<float>> makeTarget(JsonObject output_json, const std::string& base_path) {
    auto target = std::make_shared<Tensor<float>>();
    std::string output_path = base_path + "/" + output_json.parserString("value");
    std::vector<int> shape_output = output_json.parserIntArray("shape");
    target->loadValue(output_path, shape_output);
    return target;
}

// RMS error 계산 및 latency 출력, 통과 여부 반환
bool measureLatencyAndError(const std::shared_ptr<Tensor<float>>& out,
                            const std::shared_ptr<Tensor<float>>& target,
                            timespec begin, timespec end,
                            const std::string& test_name) {
    std::cout << test_name << " infernece time: "
              << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0
              << std::endl;
    if (!out || out->value.rows() != target->value.rows() ||
        out->value.cols() != target->value.cols() || !out->value.size()) {
        std::cerr << test_name << " output shape mismatch" << std::endl;
        return false;
    }
    auto diff = std::make_shared<Tensor<float>>();
    diff->value = out->value - target->value;
    double rms_error = std::sqrt(diff->value.array().square().mean());
    if (!std::isfinite(rms_error) || rms_error > error_threshold) {
        std::cout << "RMS ERROR Value is over " << error_threshold
                  << " ERROR Value: " << rms_error << std::endl;
        return false;
    }
    std::cout << "RMS ERROR IS SPEC IN ERROR: " << rms_error << std::endl;
    return true;
}

// 모듈 이름에 따라 적절한 인스턴스 생성 및 forward 호출 (else if로 불필요한 비교 제거)
std::shared_ptr<Tensor<float>> instance_manager(JsonObject& obj,
                                                 const std::shared_ptr<Tensor<float>>& input,
                                                 const std::string& base_path,
                                                 const std::string& test_name) {
    std::shared_ptr<Tensor<float>> out;

    if (test_name == "conformer_conv_module") {
        ConformerConvModule<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_conv_subsample") {
        ConformerSubSample<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_feedforward") {
        ConformerFeedForward<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_head") {
        ConformerHead<float> model(input->value.cols());
        out = model.forwardProjected(input, input, input, input);
    }
    else if (test_name == "conformer_multihead") {
        ConformerMultiHead<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input, input);
    }
    else if (test_name == "conformer_attention") {
        ConformerAttention<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input, input);
    }
    else if (test_name == "conformer_encoder_layer") {
        ConformerEncoderLayer<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input, input);
    }
    else if (test_name == "transformer_feedforward") {
        TransformerFeedForward<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "transformer_head") {
        TransformerHead<float> model(input->value.cols());
        out = model.forwardProjected(input, input, input, true);
    }
    else if (test_name == "transformer_multihead") {
        TransformerMultiHead<float> model(obj.parseObject("model"), base_path);
        out = model.causalAttention(input, input, input);
    }
    else if (test_name == "transformer_encoder_layer") {
        TransformerEncoderLayer<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "transformer_decoder_layer") {
        TransformerDecoderLayer<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input, input);
    }
    else if (test_name == "transformer_encoder") {
        TransformerEncoder<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "transformer_decoder") {
        TransformerDecoder<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input, input);
    }
    else if (test_name == "conformer_encoder") {
        ConformerEncoder<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "transformer_transducer_layer") {
        TransformerTransducerLayer<float> model(obj.parseObject("model"), base_path);
        out = model.forwardContext(input);
    }
    else if (test_name == "conformer_stream_head") {
        ConformerStreamHead<float> model(input->value.cols());
        out = model.forwardProjected(input, input, input, nullptr, nullptr, -1);
    }
    else if (test_name == "conformer_stream_multihead") {
        ConformerStreamMultiHead<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input, nullptr, nullptr, -1);
    }
    else if (test_name == "conformer_stream_attention") {
        ConformerStreamAttention<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_stream_conv_module") {
        ConformerStreamConvModule<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_stream_encoder_layer") {
        ConformerStreamEncoderLayer<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_stream_encoderV2") {
        ConformerStreamEncoderV2<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }
    else if (test_name == "conformer_ctc") {
        ConformerCTC<float> model(obj.parseObject("model"), base_path);
        out = model.forward(input);
    }

    return out;
}

// float 입력 모듈 테스트 (bool 반환, assert 대신)
bool basic_tester(JsonObject obj, const std::string& base_path, const std::string& test_name) {
    auto input = makeInput(obj.parseObject("input"), base_path);
    auto target = makeTarget(obj.parseObject("output"), base_path);
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    auto out = instance_manager(obj, input, base_path, test_name);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return measureLatencyAndError(out, target, begin, end, test_name);
}

// integer 입력 모듈 테스트 (conformer_decoder, transformer_transducer)
bool integer_basic_tester(JsonObject obj, const std::string& base_path, const std::string& test_name) {
    auto enc_input = makeInput(obj.parseObject("enc_input"), base_path);
    auto dec_input = makeIntegerInput(obj.parseObject("dec_input"), base_path);
    auto target = makeTarget(obj.parseObject("output"), base_path);

    std::shared_ptr<Tensor<float>> out;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    if (test_name == "conformer_decoder") {
        ConformerDecoder<float> model(obj.parseObject("model"), base_path);
        out = model.forward(dec_input, enc_input);
    }
    else if (test_name == "transformer_transducer") {
        TransformerTransducer<float> model(obj.parseObject("model"), base_path);
        out = model.forwardContext(dec_input, enc_input, 0);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return measureLatencyAndError(out, target, begin, end, test_name);
}

// decoder 통합 테스트 (asr_stream_conformer_trf_transducer)
bool decoder_basic_tester(JsonObject obj, const std::string& base_path, const std::string& test_name) {
    auto enc_input = makeInput(obj.parseObject("enc_input"), base_path);
    auto dec_input = makeIntegerInput(obj.parseObject("dec_input"), base_path);
    auto target = makeTarget(obj.parseObject("output"), base_path);

    std::shared_ptr<Tensor<float>> out;
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    if (test_name == "asr_stream_conformer_trf_transducer") {
        AsrStreamConformerTrfTransducer<float> model(obj.parseObject("model"), base_path);
        out = model.forward(dec_input, enc_input);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return measureLatencyAndError(out, target, begin, end, test_name);
}

// 단일 테스트 실행: 테스트 이름에 따라 적절한 tester 호출
bool run_single_test(JsonObject& module_test_json, const std::string& name) {
    if (name == "conformer_decoder" || name == "transformer_transducer") {
        return integer_basic_tester(module_test_json.parseObject(name), "python/binary", name);
    } else if (name == "asr_stream_conformer_trf_transducer") {
        return decoder_basic_tester(module_test_json.parseObject(name), "python/binary", name);
    } else {
        return basic_tester(module_test_json.parseObject(name), "python/binary", name);
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [module_name ...]\n"
              << "  If no module_name given, run all tests.\n"
              << "  Available modules:\n"
              << "    conformer_conv_module, conformer_conv_subsample, transformer_feedforward,\n"
              << "    conformer_feedforward, conformer_head, conformer_multihead, conformer_attention,\n"
              << "    transformer_head, transformer_multihead, transformer_transducer_layer,\n"
              << "    transformer_transducer, conformer_encoder_layer, conformer_encoder,\n"
              << "    transformer_encoder_layer, transformer_decoder_layer, transformer_encoder,\n"
              << "    transformer_decoder, conformer_decoder, conformer_stream_head,\n"
              << "    conformer_stream_multihead, conformer_stream_attention,\n"
              << "    conformer_stream_conv_module, conformer_stream_encoder_layer,\n"
              << "    conformer_stream_encoderV2, conformer_ctc,\n"
              << "    asr_stream_conformer_trf_transducer\n";
}

int main(int argc, char* argv[]) {
    // 전체 모듈 테스트 목록 (실행 순서)
    const std::vector<std::string> all_modules = {
        "conformer_conv_module", "conformer_conv_subsample",
        "transformer_feedforward", "conformer_feedforward",
        "conformer_head", "conformer_multihead", "conformer_attention",
        "transformer_head", "transformer_multihead",
        "transformer_transducer_layer", "transformer_transducer",
        "conformer_encoder_layer", "conformer_encoder",
        "transformer_encoder_layer", "transformer_decoder_layer",
        "transformer_encoder", "transformer_decoder",
        "conformer_decoder",
        "conformer_stream_head", "conformer_stream_multihead",
        "conformer_stream_attention", "conformer_stream_conv_module",
        "conformer_stream_encoder_layer",
        "conformer_stream_encoderV2", "conformer_ctc",
        "asr_stream_conformer_trf_transducer"
    };

    // 명령줄 인자로 선택적 테스트 실행 지원
    std::vector<std::string> modules_to_test;
    if (argc > 1) {
        if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        for (int i = 1; i < argc; ++i) {
            std::string name = argv[i];
            bool found = false;
            for (const auto& m : all_modules) {
                if (m == name) { found = true; break; }
            }
            if (!found) {
                std::cerr << "ERROR: Unknown module '" << name << "'\n";
                print_usage(argv[0]);
                return 2;
            }
            modules_to_test.push_back(name);
        }
    } else {
        modules_to_test = all_modules;
    }

    JsonObject config = JsonObject("python/binary/module_test.json");
    JsonObject module_test_json = config.parseObject("module_test");

    int passed = 0;
    int failed = 0;
    std::vector<std::string> failed_modules;

    for (const auto& module_name : modules_to_test) {
        std::cout << "\n===== Testing: " << module_name << " =====\n";
        bool ok = false;
        try {
            ok = run_single_test(module_test_json, module_name);
        } catch (const std::exception& e) {
            std::cerr << "EXCEPTION in " << module_name << ": " << e.what() << std::endl;
            ok = false;
        }

        if (ok) {
            std::cout << "RESULT:" << module_name << ":PASS" << std::endl;
            ++passed;
        } else {
            std::cout << "RESULT:" << module_name << ":FAIL" << std::endl;
            failed_modules.push_back(module_name);
            ++failed;
        }
    }

    // 요약 출력
    std::cout << "\n===== SUMMARY =====\n";
    std::cout << "Passed: " << passed << " / " << (passed + failed) << std::endl;
    if (!failed_modules.empty()) {
        std::cout << "Failed modules:";
        for (const auto& f : failed_modules) {
            std::cout << " " << f;
        }
        std::cout << std::endl;
    }

    return (failed > 0) ? 1 : 0;
}
