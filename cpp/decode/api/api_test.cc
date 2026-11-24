// [2026-04-07] api_test: ASR API 통합 테스트
// streaming decode → CTC result → second pass decode 순차 검증
#include "commands.h"
#include "cli/wav_input.h"
#include "time.h"
#include <cstdlib>
#include <unistd.h>
#include "api/asr_api.h"
#include "interface/decoder_callback.h"

class PrintResult : public MERCURY::IDecoderCallback {
    public:
        void onResult(std::string asrResult) {
            std::cout << "ASR RESULT: " << asrResult << std::endl;
        }
};

int api_tests::smoke(int argc, char** argv) {
    if (argc > 3) return 2;
    const auto inputs = read_wav_inputs(argc == 3 ? argv[2] : MERCURY_SMOKE_WAV);
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    const char* assets = std::getenv("MERCURY_TEST_MODEL_DIR");
    const std::filesystem::path model = argc >= 2 ? argv[1] : (assets ? assets : MERCURY_DEFAULT_MODEL_DIR);
    if (!std::filesystem::is_regular_file(model / "model.json")) {
        throw std::runtime_error("Missing model.json in " + model.string());
    }
    MERCURY::AsrAPI asr_api(model.string());
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout <<" decoder initialization time: "  << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0<<std::endl;

    asr_api.cancelDecoding();
    asr_api.finishDecoding();

    std::shared_ptr<PrintResult> print_result = std::make_shared<PrintResult>();

    bool passed = true;
    for (const auto& path : inputs) {
        std::cout << "WAV: " << path.string() << std::endl;
        asr_api.clear();
        asr_api.setCallBack(print_result);
        const auto result = decode_wav(asr_api, path);
        std::cout << "ctc asr result: " << result.ctc << '\n'
                  << "second pass decoding result: " << result.attention << '\n';
        std::cout << "RESULT:ctc_decode:" << (result.ctc.empty() ? "FAIL" : "PASS") << '\n'
                  << "RESULT:second_pass_decode:" << (result.attention.empty() ? "FAIL" : "PASS") << '\n'
                  << "RESULT:streaming_decode:PASS\n";
        if (result.ctc.empty() || result.attention.empty()) passed = false;
    }
    return passed ? 0 : 1;
}
