#import "MercuryAsr.h"

#include <memory>
#include <string>
#include <vector>
#include <type_traits>

#include "api/asr_api.h"
#include "interface/decoder_callback.h"

namespace {
// C++ exceptions cannot cross the Swift/Objective-C boundary.
template<class F>
auto GuardNative(F action) -> decltype(action()) {
    try {
        return action();
    } catch (const std::exception& e) {
        NSLog(@"[MercuryAsr] native operation failed: %s", e.what());
    } catch (...) {
        NSLog(@"[MercuryAsr] native operation failed");
    }
    if constexpr (!std::is_void_v<decltype(action())>) return {};
}

/// 블록을 그대로 부르는 디코더 콜백.
///
/// cpp/interface/objectivec_callback 은 target+selector 방식이라 Swift 클로저와 맞추려면
/// @objc 메서드를 따로 만들어야 한다. 블록은 Swift 클로저와 바로 대응하므로 여기서 직접 만든다.
class BlockDecoderCallback : public MERCURY::IDecoderCallback {
public:
    explicit BlockDecoderCallback(void (^block)(NSString *)) : block_(block) {}

    void onResult(std::string asrResult) override {
        if (block_ == nil) {
            return;
        }
        NSString *result = [[NSString alloc] initWithBytes:asrResult.data()
                                                    length:asrResult.size()
                                                  encoding:NSUTF8StringEncoding];
        // 잘린 UTF-8 이면 nil 이 나온다 — 빈 문자열로 대신해 호출부가 옵셔널을 다루지 않게 한다
        block_(result ?: @"");
    }

private:
    // 블록은 ARC 가 관리한다(멤버로 잡고 있으므로 복사되어 힙에 남는다)
    void (^block_)(NSString *);
};

/// NSArray<NSNumber*> -> std::vector<int>
std::vector<int> ToIntVector(NSArray<NSNumber *> *numbers) {
    std::vector<int> out;
    out.reserve(numbers.count);
    for (NSNumber *n in numbers) {
        out.push_back(n.intValue);
    }
    return out;
}

}  // namespace

@implementation MercuryAsr {
    std::unique_ptr<MERCURY::AsrAPI> _api;
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self == nil) {
        return nil;
    }
    try {
        _api = std::make_unique<MERCURY::AsrAPI>(std::string(modelPath.UTF8String));
    } catch (const std::exception &e) {
        NSLog(@"[MercuryAsr] 엔진 초기화 실패: %s", e.what());
        return nil;
    } catch (...) {
        NSLog(@"[MercuryAsr] 엔진 초기화 실패(알 수 없는 예외)");
        return nil;
    }
    return self;
}

- (void)startWithBeamSize:(int)beamSize {
    if (_api == nullptr) {
        return;
    }
    // 콜백은 세션마다 다시 건다 — 블록이 교체됐을 수 있다
    __weak MercuryAsr *weakSelf = self;
    _api->setCallBack(std::make_shared<BlockDecoderCallback>(^(NSString *result) {
        MercuryAsr *strongSelf = weakSelf;
        if (strongSelf.onPartialResult != nil) {
            strongSelf.onPartialResult(result);
        }
    }));
    GuardNative([&] { _api->startBackGroundEngine(beamSize); });
}

- (void)sendAudio:(const float *)samples count:(int)count isFinal:(BOOL)isFinal {
    if (_api == nullptr || count < 0) {
        return;
    }
    // samples 가 NULL 이어도 그냥 돌아가면 안 된다. 종료 신호는 (NULL, 0, YES) 로 오는데,
    // 그것까지 버리면 엔진 스레드가 마지막 청크를 영원히 기다려 finishDecoding 의 join 이
    // 돌아오지 않는다 — 2패스 결과도 발음 점수도 나오지 않고, 다음 세션에 이전 상태가 섞인다.
    if (samples == nullptr && count > 0) {
        return;
    }
    // 네이티브는 비상수 포인터를 받지만 내용을 바꾸지 않는다
    GuardNative([&] { _api->sendAudio(const_cast<float *>(samples), count, isFinal == YES); });
}

- (void)finishDecoding {
    if (_api != nullptr) {
        GuardNative([&] { _api->finishDecoding(); });
    }
}

- (void)cancelDecoding {
    if (_api != nullptr) {
        GuardNative([&] { _api->cancelDecoding(); });
    }
}

- (void)clear {
    if (_api != nullptr) {
        GuardNative([&] { _api->clear(); });
    }
}

- (NSString *)ctcResult {
    if (_api == nullptr) {
        return @"";
    }
    return @(GuardNative([&] { return _api->getCtcAsrResult(); }).c_str());
}

- (int)ctcTrailingSilenceMs {
    if (_api == nullptr) { return -1; }
    return _api->getCtcTrailingSilenceMs();
}

- (NSString *)secondPassResultWithBeamSize:(int)beamSize {
    if (_api == nullptr) {
        return @"";
    }
    return @(GuardNative([&] { return _api->getSecondPassDecodingResult(beamSize); }).c_str());
}

- (NSString *)pronounceScoreForLabel:(NSString *)label
                             wordIdx:(NSArray<NSNumber *> *)wordIdx
                        wordIdxGroup:(NSArray<NSNumber *> *)wordIdxGroup
                              phnIdx:(NSArray<NSNumber *> *)phnIdx
                         phnIdxGroup:(NSArray<NSNumber *> *)phnIdxGroup {
    if (_api == nullptr) {
        return @"";
    }
    // 인덱스와 그룹의 길이가 다르면 네이티브가 짝이 맞지 않는 첨자를 쓴다
    if (wordIdx.count != wordIdxGroup.count || phnIdx.count != phnIdxGroup.count) {
        NSLog(@"[MercuryAsr] 채점 인덱스와 그룹의 길이가 다르다 — 건너뛴다");
        return @"";
    }
    if (wordIdx.count < 2 || phnIdx.count < 2) {
        return @"";
    }

    std::vector<int> word_idx = ToIntVector(wordIdx);
    std::vector<int> word_group = ToIntVector(wordIdxGroup);
    std::vector<int> phn_idx = ToIntVector(phnIdx);
    std::vector<int> phn_group = ToIntVector(phnIdxGroup);

    std::string json = GuardNative([&] { return _api->getPronounceScore(
        std::string(label.UTF8String),
        word_idx.data(), word_group.data(), int(word_idx.size()),
        phn_idx.data(), phn_group.data(), int(phn_idx.size())); });
    return @(json.c_str());
}

- (void)setSentenceScoreWeightsPronunciation:(float)pronunciation
                                 recognition:(float)recognition {
    if (_api == nullptr) {
        return;
    }
    _api->setSentenceScoreWeights(pronunciation, recognition);
}

@end
