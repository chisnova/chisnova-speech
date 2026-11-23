// 발음 점수 진단기: 같은 오디오·같은 라벨에 대해 서로 다른 id 배열(구/신 vocab 기준)로
// getPronounceScore 를 호출해 점수 차이를 비교한다.
//
//   ./build/pronounce_score_diag <model_dir> <wav> <label> <ids.json> [<ids2.json> ...]
//
// ids.json 형식 (앱이 assets 의 .jsonl 에서 읽어 넘기는 것과 같은 배열):
//   {"tag":"...","word_idx":[...],"word_idx_group":[...],"phn_idx":[...],"phn_idx_group":[...]}
#include "commands.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "api/asr_api.h"
#include "feature/audio_reader/audio_reader.h"
#include "interface/decoder_callback.h"
#include "nlohmann/json.hpp"

class SilentCallback : public MERCURY::IDecoderCallback {
public:
    void onResult(std::string asrResult) { last = asrResult; }
    std::string last;
};

static std::vector<int> readIntArray(const nlohmann::json& obj, const std::string& key) {
    std::vector<int> ret;
    for (const auto& v : obj.at(key)) ret.push_back(v.get<int>());
    return ret;
}

int api_tests::pronounce(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <model_dir> <wav> <label> <ids.json> [<ids2.json> ...]" << std::endl;
        return 1;
    }
    const std::string model_dir = argv[1];
    const std::string wav_path = argv[2];
    const std::string label = argv[3];

    MERCURY::AsrAPI asr_api(model_dir);
    std::shared_ptr<SilentCallback> callback = std::make_shared<SilentCallback>();

    std::vector<int16_t> wave = audio_read(wav_path.c_str());
    std::vector<float> audio = loadWave(wave);
    std::cout << "wav samples: " << audio.size() << std::endl;

    asr_api.clear();
    asr_api.setCallBack(callback);
    asr_api.startBackGroundEngine(3);

    // 기기와 같은 조건으로 맞추려면 실시간으로 흘려보내야 한다.
    // 한꺼번에 밀어넣으면 엔진이 전체를 한 덩어리로 처리해 사실상 full-context 가 된다.
    const bool realtime = (std::getenv("DIAG_REALTIME") != nullptr);
    const int chunk_size = 1600;
    size_t sent = 0;
    while (sent + chunk_size < audio.size()) {
        asr_api.sendAudio(audio.data() + sent, chunk_size, false);
        sent += chunk_size;
        if (realtime) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    asr_api.sendAudio(audio.data() + sent, int(audio.size() - sent), true);
    asr_api.finishDecoding();

    std::cout << "ctc2 result: " << asr_api.getCtcAsrResult() << std::endl;
    std::cout << "label      : " << label << std::endl;

    for (int i = 4; i < argc; ++i) {
        std::ifstream ids_file(argv[i]);
        if (!ids_file.is_open()) {
            std::cerr << "cannot open " << argv[i] << std::endl;
            continue;
        }
        nlohmann::json ids_json;
        ids_file >> ids_json;

        std::vector<int> word_idx = readIntArray(ids_json, "word_idx");
        std::vector<int> word_group = readIntArray(ids_json, "word_idx_group");
        std::vector<int> phn_idx = readIntArray(ids_json, "phn_idx");
        std::vector<int> phn_group = readIntArray(ids_json, "phn_idx_group");

        std::string tag = ids_json.value("tag", std::string(argv[i]));
        std::string score_json = asr_api.getPronounceScore(
            label,
            word_idx.data(), word_group.data(), int(word_idx.size()),
            phn_idx.data(), phn_group.data(), int(phn_idx.size()));

        std::cout << "---- " << tag << " ----" << std::endl;
        std::cout << score_json << std::endl;
    }
    return 0;
}
