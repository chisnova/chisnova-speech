// 스트리밍 CTC 로 발화 종료를 잡을 수 있는지 재 보는 도구.
//
//   ./ctc_endpoint_diag <model_dir> <wav> [chunk_ms]
//
// 오디오를 앱과 같은 방식으로 흘리면서, 청크마다
//   - 지금까지의 1패스 CTC 인식 결과
//   - getCtcTrailingSilenceMs() — 말이 멎은 뒤 흐른 시간
// 을 찍는다.
//
// 보려는 것은 두 가지다.
//   1. 발화 중에는 이 값이 작게 유지되는가(무성 자음 구간에서 튀지 않는가)
//   2. 말이 끝난 뒤에는 청크마다 꾸준히 늘어나는가
// 2 가 성립해야 hangover 판정을 이 값으로 대신할 수 있다.

#include "commands.h"
#include <charconv>
#include <chrono>
#include <limits>
#include <string_view>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "api/asr_api.h"
#include "feature/audio_reader/audio_reader.h"

int api_tests::endpoint(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model_dir> <wav> [chunk_ms]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string wav_path = argv[2];
    int chunk_ms = 100;
    if (argc > 3) {
        std::string_view value(argv[3]);
        auto result = std::from_chars(value.data(), value.data() + value.size(), chunk_ms);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
            chunk_ms <= 0 || chunk_ms > std::numeric_limits<int>::max() / 16) {
            std::cerr << "chunk_ms must be a positive integer within the supported range\n";
            return 2;
        }
    }
    const int chunk_samples = 16 * chunk_ms;

    std::vector<int16_t> raw = audio_read(wav_path.c_str());
    if (raw.empty()) {
        std::cerr << "오디오를 읽지 못했다: " << wav_path << "\n";
        return 1;
    }
    std::vector<float> audio;
    audio.reserve(raw.size());
    for (int16_t sample : raw) audio.push_back(float(sample) / 32768.0f);

    MERCURY::AsrAPI api(model_dir);
    api.startBackGroundEngine(3);

    std::printf("%8s %8s %10s  %s\n", "시각", "무음(ms)", "판정", "인식 결과");
    std::printf("%s\n", std::string(72, '-').c_str());

    int last_reported = -2;
    for (size_t off = 0; off < audio.size(); off += size_t(chunk_samples)) {
        size_t n = std::min(size_t(chunk_samples), audio.size() - off);
        api.sendAudio(audio.data() + off, int(n), false);
        // 실시간처럼 흘린다. 한꺼번에 밀어넣으면 엔진이 뒤따라오기 전에 값을 읽게 되어
        // 언제 무엇이 갱신됐는지 볼 수 없다 — 앱은 마이크 속도로 들어온다.
        std::this_thread::sleep_for(std::chrono::milliseconds(chunk_ms));

        // 엔진은 자기 임계값만큼 모여야 깨어난다. 값이 바뀐 순간만 찍는다.
        int silence = api.getCtcTrailingSilenceMs();
        if (silence != last_reported) {
            const double t = double(off + n) / 16000.0;
            const char* verdict = (silence < 0) ? "발화없음"
                                : (silence >= 900) ? "종료"
                                : "발화중";
            std::printf("%7.2fs %8d %10s  %s\n", t, silence, verdict,
                        api.getCtcAsrResult().c_str());
            last_reported = silence;
        }
    }
    api.sendAudio(nullptr, 0, true);
    api.finishDecoding();

    std::printf("%s\n", std::string(72, '-').c_str());
    std::printf("최종: %s\n", api.getCtcAsrResult().c_str());
    return 0;
}
