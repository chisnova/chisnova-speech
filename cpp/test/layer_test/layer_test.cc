#include "layer/batchnorm1d.h"
#include "layer/conformer_positional.h"
#include "layer/convolution1d.h"
#include "layer/convolution2d.h"
#include "layer/embedding.h"
#include "layer/glu.h"
#include "layer/layernorm.h"
#include "layer/linear.h"
#include "layer/positional_encoding.h"
#include "layer/relu.h"
#include "layer/silu.h"
#include "layer/softmax.h"
#include "layer/sigmoid.h"
#include "layer/tanh.h"
#include "layer/rotary_embedding.h"
#include "layer/gelu.h"
#include "json_loader/json_object.h"
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>

using namespace MERCURY;
struct timespec begin, end;
float error_threshold=0.00001;

std::shared_ptr<Tensor<float>> makeInput(JsonObject input_json, std::string base_path) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>();
    std::string input_path = base_path + "/" +  input_json.parserString("value");
    std::vector<int> shape_input = input_json.parserIntArray("shape");
    input->loadValue(input_path, shape_input);
    return input;
}

std::shared_ptr<Tensor<int>> makeIntegerInput(JsonObject input_json, std::string base_path) {
    std::shared_ptr<Tensor<int>> input = std::make_shared<Tensor<int>>();
    std::string input_path = base_path + "/" +  input_json.parserString("value");
    std::vector<int> shape_input = input_json.parserIntArray("shape");
    input->loadValue(input_path, shape_input);
    return input;
}

std::shared_ptr<Tensor<float>> makeTarget(JsonObject output_json, std::string base_path) {
    std::shared_ptr<Tensor<float>> target = std::make_shared<Tensor<float>>();
    std::string output_path = base_path + "/" +  output_json.parserString("value");
    std::vector<int> shape_output = output_json.parserIntArray("shape");
    target->loadValue(output_path, shape_output);
    return target;
}

bool measureLatencyAndError(std::shared_ptr<Tensor<float>> out, std::shared_ptr<Tensor<float>> target, timespec begin, timespec end, std::string test_name){
    std::cout << test_name << " infernece time: "  << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 << std::endl;
    std::shared_ptr<Tensor<float>> diff = std::make_shared<Tensor<float>>();
    diff->value = out->value - target->value;
    double rms_error = std::sqrt(diff->value.array().square().mean());
    if (rms_error > error_threshold){
        std::cout << "RMS ERROR Value is over " << error_threshold << " ERROR Value: "<< rms_error << std::endl;
        return false;
    }
    std::cout << "RMS ERROR IS SPEC IN ERROR: " << rms_error << std::endl;
    return true;
}


// 각 레이어의 forward 함수 호출 전후로 시간 측정 및 출력
std::shared_ptr<Tensor<float>> float_instance_manager_with_latency(JsonObject& obj, std::shared_ptr<Tensor<float>> input, std::string base_path, std::string test_name, timespec* begin, timespec* end) {
    std::shared_ptr<Tensor<float>> out;
    if(test_name == "linear") {
        Linear<float> linear(obj.parseObject("model"), base_path);
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = linear.forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "cnn1d") {
        Convolution1D<float> conv1d(obj.parseObject("model"), base_path);
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = conv1d.forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "batchnorm1d") {
        BatchNorm1D<float> batch1d(obj.parseObject("model"), base_path);
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = batch1d.forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "positional") {
        PosEncoding<float> pos_enc(obj.parseObject("model"), base_path);
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = pos_enc.forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "conformer_positional") {
        ConformerPositional<float> con_pos(obj.parseObject("model"), base_path);
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = con_pos.forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "layernorm") {
        LayerNorm<float> layernorm(obj.parseObject("model"), base_path);
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = layernorm.forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "relu") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = Relu<float>().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "silu") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = SILU().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "softmax") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = Softmax().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "glu") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = GLU().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "sigmoid") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = Sigmoid().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "tanh") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = Tanh().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "rotary") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = RotaryEmbedding<float>::apply_rotary_pos_emb(input, input, input, 0, input->shape(0));
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    else if(test_name == "gelu") {
        clock_gettime(CLOCK_MONOTONIC, begin);
        out = GELU<float>().forward(input);
        clock_gettime(CLOCK_MONOTONIC, end);
    }
    return out;
}


std::shared_ptr<Tensor<float>> integer_instance_manager(JsonObject& obj, std::shared_ptr<Tensor<int>> input, std::string base_path, std::string test_name) {
    std::shared_ptr<Tensor<float>> out;
    if(test_name == "embedding") {
        Embedding embdding(obj.parseObject("model"), base_path);
        out = embdding.forward(input);
    }
    return out;
}

// Returns true on pass, false on fail (no assert crash)
bool basic_tester(JsonObject obj, std::string base_path, std::string test_name) {
    std::shared_ptr<Tensor<float>> input = makeInput(obj.parseObject("input"), base_path);
    std::shared_ptr<Tensor<float>> target = makeTarget(obj.parseObject("output"), base_path);

    std::shared_ptr<Tensor<float>> out = float_instance_manager_with_latency(obj, input, base_path, test_name, &begin, &end);
    return measureLatencyAndError(out, target, begin, end, test_name);
}

// Returns true on pass, false on fail (no assert crash)
bool integer_basic_tester(JsonObject obj, std::string base_path, std::string test_name) {
    std::shared_ptr<Tensor<int>> input = makeIntegerInput(obj.parseObject("input"), base_path);
    std::shared_ptr<Tensor<float>> target = makeTarget(obj.parseObject("output"), base_path);

    clock_gettime(CLOCK_MONOTONIC, &begin);
    std::shared_ptr<Tensor<float>> out = integer_instance_manager(obj, input, base_path, test_name);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return measureLatencyAndError(out, target, begin, end, test_name);
}

// Returns true on pass, false on fail (no assert crash)
bool conv2d_test(JsonObject obj, std::string base_path, std::string test_name){
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    JsonObject input_json = obj.parseObject("input");

    int num_in_ch = input_json.parserValue("num_channel");
    for(int i=0;i<num_in_ch;++i){
        std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>();
        std::string name = "input" + std::to_string(i);
        JsonObject each_json = input_json.parseObject(name);
        std::string input_path = base_path + "/" +  each_json.parserString("value");
        std::vector<int> shape_input = each_json.parserIntArray("shape");
        input->loadValue(input_path, shape_input);
        inputs.push_back(std::move(input));
    }

    JsonObject output_json = obj.parseObject("output");
    int num_out_ch = output_json.parserValue("num_channel");
    for(int i=0;i<num_out_ch;++i){
        std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>();
        std::string name = "output" + std::to_string(i);
        JsonObject each_json = output_json.parseObject(name);
        std::string output_path = base_path + "/" +  each_json.parserString("value");
        std::vector<int> shape_input = each_json.parserIntArray("shape");
        output->loadValue(output_path, shape_input);
        outputs.push_back(std::move(output));
    }

    Convolution2D<float> convoultion2d(obj.parseObject("model"), base_path);
    clock_gettime(CLOCK_MONOTONIC, &begin);
    std::vector<std::shared_ptr<Tensor<float>>> out = convoultion2d.forward(inputs);
    clock_gettime(CLOCK_MONOTONIC, &end);

    std::cout <<"cnn2d infernece time: "  << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 <<std::endl;
    double rms_error = 0;
    for(int i=0;i<num_out_ch;++i){
        std::shared_ptr<Tensor<float>> diff = std::make_shared<Tensor<float>>();
        diff->value = out[i]->value - outputs[i]->value;
        rms_error += std::sqrt(diff->value.array().square().mean());
    }
    rms_error = rms_error/float(num_out_ch);
    if (rms_error > 0.001){
        std::cout << "RMS ERROR Value is over 0.001 ERROR Value: "<< rms_error << std::endl;
        return false;
    }
    std::cout << "RMS ERROR IS SPEC IN ERROR: " << rms_error << std::endl;
    return true;
}


// Run a single layer test by name. Returns true on pass, false on fail.
bool run_single_test(JsonObject& layer_test_json, const std::string& name) {
    if (name == "embedding") {
        return integer_basic_tester(layer_test_json.parseObject("embedding"), "python/binary", "embedding");
    } else if (name == "cnn2d") {
        return conv2d_test(layer_test_json.parseObject("cnn2d"), "python/binary", "cnn2d");
    } else {
        // All other layers use basic_tester with float input
        return basic_tester(layer_test_json.parseObject(name), "python/binary", name);
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [layer_name]\n"
              << "  If no layer_name given, run all tests.\n"
              << "  Available layers:\n"
              << "    embedding, batchnorm1d, relu, silu, softmax, glu, layernorm,\n"
              << "    positional, conformer_positional, cnn1d, cnn2d, linear,\n"
              << "    sigmoid, tanh, rotary, gelu\n";
}

int main(int argc, char* argv[]){
    // All supported layer test names in default execution order
    const std::vector<std::string> all_layers = {
        "embedding", "batchnorm1d", "relu", "silu", "softmax", "glu",
        "layernorm", "positional", "conformer_positional", "cnn1d", "cnn2d",
        "linear", "sigmoid", "tanh", "rotary", "gelu"
    };

    // Determine which layers to test
    std::vector<std::string> layers_to_test;

    if (argc > 1) {
        if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        // Filter: only run layers specified as arguments
        for (int i = 1; i < argc; ++i) {
            std::string name = argv[i];
            bool found = false;
            for (const auto& l : all_layers) {
                if (l == name) { found = true; break; }
            }
            if (!found) {
                std::cerr << "ERROR: Unknown layer '" << name << "'\n";
                print_usage(argv[0]);
                return 2;
            }
            layers_to_test.push_back(name);
        }
    } else {
        layers_to_test = all_layers;
    }

    JsonObject config = JsonObject("python/binary/layer_test.json");
    JsonObject layer_test_json = config.parseObject("layer_test");

    int passed = 0;
    int failed = 0;
    std::vector<std::string> failed_layers;

    for (const auto& layer_name : layers_to_test) {
        std::cout << "\n===== Testing: " << layer_name << " =====\n";
        bool ok = false;
        try {
            ok = run_single_test(layer_test_json, layer_name);
        } catch (const std::exception& e) {
            std::cerr << "EXCEPTION in " << layer_name << ": " << e.what() << std::endl;
            ok = false;
        }

        if (ok) {
            std::cout << "RESULT:" << layer_name << ":PASS" << std::endl;
            ++passed;
        } else {
            std::cout << "RESULT:" << layer_name << ":FAIL" << std::endl;
            failed_layers.push_back(layer_name);
            ++failed;
        }
    }

    // Summary
    std::cout << "\n===== SUMMARY =====\n";
    std::cout << "Passed: " << passed << " / " << (passed + failed) << std::endl;
    if (!failed_layers.empty()) {
        std::cout << "Failed layers:";
        for (const auto& f : failed_layers) {
            std::cout << " " << f;
        }
        std::cout << std::endl;
    }

    return (failed > 0) ? 1 : 0;
}
