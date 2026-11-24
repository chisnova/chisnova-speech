// iOS 스모크 테스트: 디코더 엔진이 iOS 런타임에서 실제로 도는지 확인한다.
//
// 안드로이드/macOS 에서 같은 wav 로 얻은 결과와 대조하는 것이 목적이라, UI 없이
// 모델을 올리고 한 발화를 디코딩한 뒤 결과를 표준 출력과 파일에 남기고 끝낸다.
//
//   asr_smoke_test <model_dir> <wav> [out_file]
//
// 시뮬레이터는 macOS 프로세스이므로 호스트 절대 경로를 그대로 읽을 수 있다.
// 덕분에 258MB 자산을 앱 번들에 넣지 않고도 검증할 수 있다.
#import <Foundation/Foundation.h>

#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "api/asr_api.h"
#include "feature/audio_reader/audio_reader.h"
#include "interface/decoder_callback.h"

namespace {

class CollectPartials : public MERCURY::IDecoderCallback {
public:
    void onResult(std::string asrResult) override {
        last = asrResult;
        ++count;
    }
    std::string last;
    int count = 0;
};

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 3) {
            std::cerr << "usage: asr_smoke_test <model_dir> <wav> [out_file]" << std::endl;
            return 2;
        }
        const std::string model_dir = argv[1];
        const std::string wav_path = argv[2];
        const std::string out_path = (argc > 3) ? argv[3] : "";

        std::cout << "[ios] model: " << model_dir << std::endl;
        std::cout << "[ios] wav  : " << wav_path << std::endl;

        MERCURY::AsrAPI asr_api(model_dir);
        std::shared_ptr<CollectPartials> callback = std::make_shared<CollectPartials>();

        std::vector<int16_t> wave = audio_read(wav_path.c_str());
        std::vector<float> audio = loadWave(wave);
        std::cout << "[ios] samples: " << audio.size() << std::endl;
        if (audio.empty()) {
            std::cerr << "[ios] wav 를 읽지 못했다" << std::endl;
            return 3;
        }

        asr_api.clear();
        asr_api.setCallBack(callback);
        asr_api.startBackGroundEngine(3);

        const int chunk = 1600;   // 100ms
        size_t sent = 0;
        while (sent + chunk < audio.size()) {
            asr_api.sendAudio(audio.data() + sent, chunk, false);
            sent += chunk;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        asr_api.sendAudio(audio.data() + sent, int(audio.size() - sent), true);
        asr_api.finishDecoding();

        const std::string ctc = asr_api.getCtcAsrResult();
        const std::string second = asr_api.getSecondPassDecodingResult(3);

        std::cout << "[ios] partials    : " << callback->count << std::endl;
        std::cout << "[ios] ctc2 result : " << ctc << std::endl;
        std::cout << "[ios] second pass : " << second << std::endl;

        if (!out_path.empty()) {
            std::ofstream out(out_path);
            out << "partials=" << callback->count << "\n";
            out << "ctc2=" << ctc << "\n";
            out << "second_pass=" << second << "\n";
        }
        return ctc.empty() ? 4 : 0;
    }
}
