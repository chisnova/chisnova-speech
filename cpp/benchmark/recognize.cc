// Native worker for run.py: one WAV path per line, one JSON record per result.
#include "api/asr_api.h"
#include "feature/audio_reader/wavreader.h"
#include "commands.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;
constexpr int sample_rate = 16000;
constexpr int beam_size = 3;

std::vector<float> load_pcm(const std::string& path) {
    std::unique_ptr<void, decltype(&wav_read_close)> wav(wav_read_open(path.c_str()), wav_read_close);
    int format, channels, rate, bits;
    unsigned int bytes;
    if (!wav || !wav_get_header(wav.get(), &format, &channels, &rate, &bits, &bytes) ||
        format != 1 || channels != 1 || rate != sample_rate || bits != 16 || !bytes ||
        bytes % 2 || bytes > INT32_MAX) {
        throw std::runtime_error("Expected a nonempty 16 kHz mono PCM16 WAV: " + path);
    }
    std::vector<int16_t> pcm(bytes / 2);
    if (wav_read_data(wav.get(), reinterpret_cast<unsigned char*>(pcm.data()), bytes) != static_cast<int>(bytes))
        throw std::runtime_error("Incomplete WAV: " + path);
    std::vector<float> audio(pcm.size());
    std::transform(pcm.begin(), pcm.end(), audio.begin(), [](int16_t s) { return s / 32767.0f; });
    return audio;
}

nlohmann::json recognize(MERCURY::AsrAPI& api, std::vector<float>& audio, bool realtime) {
    auto start = Clock::now();
    api.clear();
    api.startBackGroundEngine(beam_size);
    auto stream_start = Clock::now();
    auto last_input = stream_start;
    const size_t chunk = realtime ? sample_rate / 10 : audio.size();
    for (size_t pos = 0; pos < audio.size(); pos += chunk) {
        const auto n = std::min(chunk, audio.size() - pos);
        if (realtime)
            std::this_thread::sleep_until(stream_start + Seconds(double(pos + n) / sample_rate));
        last_input = Clock::now();
        api.sendAudio(audio.data() + pos, static_cast<int>(n), pos + n == audio.size());
    }
    api.finishDecoding();
    auto ctc = api.getCtcAsrResult();
    auto final = api.getSecondPassDecodingResult(beam_size);
    auto end = Clock::now();
    return {{"ctc", ctc}, {"hypothesis", final}, {"audio_s", double(audio.size()) / sample_rate},
            {"elapsed_s", Seconds(end - start).count()}, {"final_latency_ms", Seconds(end - last_input).count() * 1000}};
}
} // namespace

int api_tests::benchmark(int argc, char** argv) {
    if (argc != 3 || (std::string(argv[2]) != "offline" && std::string(argv[2]) != "realtime")) {
        std::cerr << "Usage: api_tester benchmark MODEL_DIR offline|realtime < wav_paths\n";
        return 2;
    }
    if (!std::filesystem::is_regular_file(std::filesystem::path(argv[1]) / "model.json"))
        throw std::runtime_error("Missing model.json");
    const auto begin = Clock::now();
    MERCURY::AsrAPI api(argv[1]);
    const bool realtime = std::string(argv[2]) == "realtime";
    std::cout << "BENCH\t" << nlohmann::json{{"model_load_s", Seconds(Clock::now() - begin).count()}} << std::endl;
    for (std::string path; std::getline(std::cin, path);) {
        auto audio = load_pcm(path);
        auto result = recognize(api, audio, realtime);
        std::cout << "BENCH\t" << result << std::endl;
    }
    return 0;
}
