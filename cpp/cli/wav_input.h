#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace MERCURY { class AsrAPI; }

struct Transcript {
    std::string ctc;
    std::string attention;
};

std::vector<std::filesystem::path> read_wav_inputs(const std::filesystem::path& input);
Transcript decode_wav(MERCURY::AsrAPI& api, const std::filesystem::path& path);
