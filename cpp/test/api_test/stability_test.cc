#include "api/asr_api.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input was accepted");
}
class ThrowingCallback : public MERCURY::IDecoderCallback {
    void onResult(std::string) override { throw std::runtime_error("test worker failure"); }
};
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "model path required");
        const std::string assets = argv[1];
        MERCURY::AsrAPI uninitialized;
        uninitialized.clear(); // no null decoder access

        MERCURY::JsonObject config;
        rejects([&] { config.parserValue("required"); });
        rejects([&] { config.parseObject("required"); });
        rejects([&] { MERCURY::JsonObject missing(assets + "/does-not-exist.json"); });
        require(config.parserValue("optional", 7) == 7, "optional config default changed");

        MERCURY::Scorer scorer(assets, "mercury.model", "mercury.vocab", "phone.txt", "full.dict", 47);
        auto posterior = std::make_shared<MERCURY::Tensor<float>>();
        posterior->value = Eigen::MatrixXf::Constant(1, 4, 0.25f);
        rejects([&] { scorer.getWordScore(posterior, {1}, {10}, {"hello"}); });
        rejects([&] { scorer.getWordScore(posterior, {100}, {0}, {"hello"}); });
        rejects([&] { scorer.getWordScore(posterior, {1}, {}, {"hello"}); });
        rejects([&] { scorer.getPhoneScore(posterior, {1}, {1}, {"hello"}); });
        require(scorer.getWordScore(posterior, {1}, {0}, {"hello"})[0].score == 25,
                "valid score changed");

        {
            MERCURY::AsrAPI active(assets);
            active.startBackGroundEngine(1);
        } // must join even without audio/final/cancel

        MERCURY::AsrAPI api(assets);
        for (int n = 0; n < 5; ++n) {
            api.startBackGroundEngine(1);
            api.clear(); // stops worker before resetting caches/state
            api.startBackGroundEngine(1);
            api.finishDecoding(); // no explicit final signal; must not hang
            require(api.getCtcAsrResult().empty(), "empty recording produced a result");
            int ids[] = {2, 3};
            rejects([&] { api.getPronounceScore("hello", ids, ids, 0, ids, ids, 0); });
            int bad[] = {2, 1024, 3};
            int groups[] = {-1, 0, -1};
            rejects([&] { api.getPronounceScore("hello", bad, groups, 3, ids, ids, 2); });
            api.clear();
            api.startBackGroundEngine(1);
            api.cancelDecoding();
            api.clear();
        }

        api.setCallBack(std::make_shared<ThrowingCallback>());
        api.startBackGroundEngine(1);
        std::vector<float> audio(24000);
        for (size_t i = 0; i < audio.size(); ++i) audio[i] = 0.05f * std::sin(i * 0.1f);
        api.sendAudio(audio.data(), audio.size(), true);
        rejects([&] { api.finishDecoding(); }); // worker exception crosses join, not thread boundary
        api.clear();
        api.setCallBack(nullptr);
        api.startBackGroundEngine(1);
        api.finishDecoding(); // failed session must not poison next session
        std::cout << "stability regressions passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "stability regression failed: " << e.what() << '\n';
        return 1;
    }
}
