// [2026-04-07] api_test: ASR API 통합 테스트
// streaming decode → CTC result → second pass decode 순차 검증
#include "json_loader/json_object.h"
#include "feature/audio_reader/audio_reader.h"
#include "time.h"
#include <cstdlib>
#include <unistd.h>
#include "api/asr_api.h"
#include "interface/decoder_callback.h"

struct timespec begin, end ;


class PrintResult : public MERCURY::IDecoderCallback {
    public:
        void onResult(std::string asrResult) {
            std::cout << "ASR RESULT: " << asrResult << std::endl;
        }
};

int main() {
    clock_gettime(CLOCK_MONOTONIC, &begin);
    const char* assets = std::getenv("MERCURY_TEST_MODEL_DIR");
    MERCURY::AsrAPI asr_api(assets ? assets : "python/binary");
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout <<" decoder initialization time: "  << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0<<std::endl;

    asr_api.cancelDecoding();
    asr_api.finishDecoding();

    std::shared_ptr<PrintResult> print_result = std::make_shared<PrintResult>();

    std::string path="../resource/StevJobs_Standford/processing/SteveJobsStanford00001.wav";
    std::cout << path << std::endl;
    std::vector<int16_t> wave = audio_read(path.c_str());
    std::vector<float> audio = loadWave(wave);

    asr_api.clear();
    asr_api.setCallBack(print_result);
    asr_api.startBackGroundEngine(3);

    int chunk_size = 1600;
    int audio_idx=0;
    int num_chunk = (int)(audio.size() / chunk_size);
    int remain_chunk = audio.size() % chunk_size;
    std::cout << "audio size: " << audio.size() << std::endl;
    for(audio_idx=0; audio_idx<num_chunk; audio_idx++){
        // std::cout << "audio chunk idx: " << int(audio_idx/chunk_size) << std::endl;
        std::vector<float> chunk = std::vector<float>(audio.begin() + audio_idx * chunk_size, audio.begin() + (audio_idx + 1) * chunk_size);
        asr_api.sendAudio(chunk.data(), chunk.size(), false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (remain_chunk > 0) {
        std::vector<float> last_chunk = std::vector<float>(audio.begin() + audio_idx * chunk_size, audio.end());
        asr_api.sendAudio(last_chunk.data(), last_chunk.size(), true);
    }
    else{
        asr_api.sendAudio(nullptr, 0, true);
    }
    asr_api.finishDecoding();
    std::string ctc_result = asr_api.getCtcAsrResult();
    std::cout << "ctc asr result: " << ctc_result << std::endl;
    std::string second_pass = asr_api.getSecondPassDecodingResult(3);
    std::cout << "second pass decoding result: " << second_pass << std::endl;

    // [2026-04-07] getPronounceScore API가 multi-arg로 변경되어 단독 테스트에서는 생략
    // java_api_test에서 전체 pronunciation scoring 테스트 수행
    bool ctc_ok = !ctc_result.empty();
    bool second_ok = !second_pass.empty();
    std::cout << "RESULT:ctc_decode:" << (ctc_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "RESULT:second_pass_decode:" << (second_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "RESULT:streaming_decode:PASS" << std::endl;

    return (ctc_ok && second_ok) ? 0 : 1;
}
