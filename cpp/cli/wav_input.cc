#include "wav_input.h"
#include <fstream>
#include <string_view>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>

#include "api/asr_api.h"
#include "feature/audio_reader/wavreader.h"

std::vector<std::filesystem::path> read_wav_inputs(const std::filesystem::path& input) {
    std::ifstream file(input, std::ios::binary);
    if (!file || !std::filesystem::is_regular_file(input)) {
        throw std::runtime_error("Cannot open input: " + input.string());
    }
    char signature[12];
    file.read(signature, sizeof(signature));
    if (file.gcount() == sizeof(signature) && std::string_view(signature, 4) == "RIFF" &&
        std::string_view(signature + 8, 4) == "WAVE") return {input};
    if (input.extension() == ".wav" || input.extension() == ".WAV") return {input};
    file.clear();
    file.seekg(0);
    std::vector<std::filesystem::path> paths;
    std::string line;
    while (std::getline(file, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const auto last = line.find_last_not_of(" \t\r");
        paths.push_back(input.parent_path() / line.substr(first, last - first + 1));
    }
    if (!file.eof() || file.bad()) throw std::runtime_error("Cannot read WAV list: " + input.string());
    if (paths.empty()) throw std::runtime_error("Empty WAV list: " + input.string());
    return paths;
}

Transcript decode_wav(MERCURY::AsrAPI& api, const std::filesystem::path& wav_path) {
    if (!std::filesystem::is_regular_file(wav_path)) {
        throw std::runtime_error("Input WAV must be a regular file: " + wav_path.string());
    }
    std::unique_ptr<void, decltype(&wav_read_close)> wav(wav_read_open(wav_path.c_str()), wav_read_close);
    if (!wav) throw std::runtime_error("Cannot open input WAV: " + wav_path.string());

    int format, channels, rate, bits;
    unsigned int remaining;
    if (!wav_get_header(wav.get(), &format, &channels, &rate, &bits, &remaining) ||
        format != 1 || channels != 1 || rate != 16000 || bits != 16 ||
        remaining == 0 || remaining % sizeof(int16_t) != 0) {
        throw std::runtime_error("Expected a nonempty 16 kHz mono 16-bit PCM WAV: " + wav_path.string());
    }

    constexpr int beam_size = 3;
    api.startBackGroundEngine(beam_size);
    // Feed 100 ms at a time at microphone speed, as in the streaming API example.
    std::array<int16_t, 1600> pcm;
    std::array<float, 1600> audio;
    auto next_chunk = std::chrono::steady_clock::now();
    while (remaining > 0) {
        auto bytes = std::min<unsigned int>(remaining, sizeof(pcm));
        if (wav_read_data(wav.get(), reinterpret_cast<unsigned char*>(pcm.data()), bytes)
                != static_cast<int>(bytes)) {
            throw std::runtime_error("Truncated WAV data: " + wav_path.string());
        }
        int samples = bytes / sizeof(int16_t);
        std::transform(pcm.begin(), pcm.begin() + samples, audio.begin(),
                       [](int16_t value) { return value / 32767.0f; });
        remaining -= bytes;
        std::this_thread::sleep_until(next_chunk);
        api.sendAudio(audio.data(), samples, remaining == 0);
        next_chunk += std::chrono::milliseconds(100);
    }
    api.finishDecoding();
    return {api.getCtcAsrResult(), api.getSecondPassDecodingResult(beam_size)};
}
