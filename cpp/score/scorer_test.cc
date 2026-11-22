#include "scorer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace MERCURY;

static std::vector<std::string> readTokens(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path);
    std::vector<std::string> tokens;
    std::string line, token;
    while (std::getline(file, line)) {
        std::istringstream fields(line);
        if (fields >> token) tokens.push_back(token);
    }
    return tokens;
}

// The original posterior dumps can still be used. A one-hot reference also
// exercises the full label-to-score path on a fresh checkout.
static std::shared_ptr<Tensor<float>> posterior(const std::string& path, bool use_dump,
                                                 const std::vector<int>& ids, int classes) {
    auto tensor = std::make_shared<Tensor<float>>();
    const int rows = static_cast<int>(ids.size());
    if (use_dump) {
        if (std::filesystem::file_size(path) != size_t(rows) * classes * sizeof(float))
            throw std::runtime_error("Unexpected posterior shape in " + path);
        tensor->loadValue(path, {rows, classes});
    } else {
        tensor->value.setZero(rows, classes);
        for (int row = 0; row < rows; ++row) tensor->value(row, ids.at(row)) = 1.f;
    }
    return tensor;
}

int main() {
    const auto vocab = readTokens("mercury5.vocab");
    const auto phones = readTokens("phone.txt");
    Scorer scorer(".", "mercury5.model", "mercury5.vocab", "phone.txt", "full.dict", phones.size());
    const std::string label = "i am honored to be with you today at your commencement from one of the finest universities in the world";
    auto words = scorer.makeWordDec(label);
    auto pronunciation = scorer.makePhoneDec(label);
    words.erase(words.begin());
    pronunciation.erase(pronunciation.begin());

    std::vector<int> word_groups, phone_groups;
    int word_index = -1;
    for (int token : words) {
        if (vocab.at(token).rfind("▁", 0) == 0) ++word_index;
        word_groups.push_back(token == 3 ? -1 : word_index);
    }
    int phone_word = 0;
    for (int token : pronunciation) {
        phone_groups.push_back(token == 3 ? -1 : phone_word);
        if (token == 4) ++phone_word;
    }
    const bool use_dump = std::filesystem::exists("w_out.pt") || std::filesystem::exists("p_out.pt");
    auto w = posterior("w_out.pt", use_dump, words, vocab.size());
    auto p = posterior("p_out.pt", use_dump, pronunciation, phones.size());
    auto result = scorer.getScore(label, w, words, word_groups, p, pronunciation, phone_groups);
    std::cout << "final score: " << result.avg_score << std::endl;
    return !std::isfinite(result.avg_score) || result.final_score.size() != size_t(phone_word + 1)
        || (!use_dump && std::abs(result.avg_score - 100.f) > 1e-4f);
}
