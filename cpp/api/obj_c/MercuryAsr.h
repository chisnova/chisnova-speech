#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// C++ ASR 엔진(MERCURY::AsrAPI)을 Swift 에서 쓸 수 있게 감싼 것.
///
/// Swift 는 C++ 를 직접 부를 수 없으므로 Objective-C++ 로 한 겹 덮는다. 여기서 노출하는
/// 메서드는 안드로이드의 AsrAPI.java 와 1:1 로 대응하며, 이름만 Objective-C 관례를 따른다.
///
/// 상태 머신(Idle -> Init -> Running -> Finishing)은 네이티브가 갖고 있다.
/// 허용되지 않은 순서로 부르면 무시되고 로그만 남는다.
@interface MercuryAsr : NSObject

/// 모델 디렉터리(model.json 이 있는 곳)로 엔진을 만든다. 수백 MB 를 읽으므로 느리다.
/// 실패하면 nil 을 돌려준다.
- (nullable instancetype)initWithModelPath:(NSString *)modelPath;

/// 중간 인식 결과. 네이티브 디코딩 스레드에서 불리므로 UI 갱신은 메인 큐로 넘겨야 한다.
/// 발화가 끝나면 마지막으로 2패스 CTC 결과가 한 번 더 온다.
@property (nonatomic, copy, nullable) void (^onPartialResult)(NSString *result);

/// 디코딩 세션 시작. 이전 세션이 남아 있으면 clear 를 먼저 부른다.
- (void)startWithBeamSize:(int)beamSize;

/// 오디오 한 덩어리를 넘긴다. samples 는 [-1, 1] 로 정규화된 16kHz mono PCM.
/// 더 보낼 것이 없다는 신호(count=0, isFinal=YES)를 줄 때는 samples 에 NULL 을 넘겨도 된다.
- (void)sendAudio:(nullable const float *)samples count:(int)count isFinal:(BOOL)isFinal;

/// 입력을 마감하고 마지막 청크까지 처리될 때까지 기다린다(네이티브 스레드 join).
- (void)finishDecoding;

/// 진행 중인 세션을 버린다.
- (void)cancelDecoding;

/// 엔진을 InitState 로 되돌린다. 다음 세션 전에 부른다.
- (void)clear;

/// 발화 종료 후의 최종 인식 결과(2패스 CTC).
- (NSString *)ctcResult;

/// 스트리밍 CTC 가 본 "말이 멎은 뒤 흐른 시간"(ms). 아직 발화가 없으면 -1.
///
/// 음향 에너지가 아니라 모델이 판단하므로 잡음에 강하고, 무성 자음(/s/, /f/, 파열음 앞
/// 폐쇄)을 무음으로 오해하지 않는다. 다만 값은 엔진이 청크를 처리할 때만 갱신되므로
/// 최대 1초까지 오래된 값일 수 있다 — 에너지 기반 판정을 대신하기보다 그쪽을 확인하는
/// 데 쓴다.
- (int)ctcTrailingSilenceMs;

/// 어텐션 빔 서치 결과.
- (NSString *)secondPassResultWithBeamSize:(int)beamSize;

/// 발음 점수 JSON.
///
/// 인덱스 배열은 콘텐츠의 .jsonl 이 갖고 있는 값을 그대로 넘긴다. 라벨만 넘기는 API 는
/// 없다 — 안드로이드에서 그 오버로드가 네이티브 구현 없이 선언만 있어 SIGSEGV 를 냈다.
/// 같은 .jsonl 에서 나온 라벨과 인덱스를 함께 넘긴다. 잘못된 인덱스는
/// 메모리에 접근하기 전에 거부하며 채점 결과는 빈 문자열로 돌려준다.
- (NSString *)pronounceScoreForLabel:(NSString *)label
                             wordIdx:(NSArray<NSNumber *> *)wordIdx
                        wordIdxGroup:(NSArray<NSNumber *> *)wordIdxGroup
                              phnIdx:(NSArray<NSNumber *> *)phnIdx
                         phnIdxGroup:(NSArray<NSNumber *> *)phnIdxGroup;

/// 문장별 최종 점수 비중(발음/인식, 합 1). 콘텐츠 .jsonl 의 score_weights 를
/// pronounceScore 직전에 넘긴다. 음수(예: -1, -1)면 모델 기본 채점으로 되돌린다.
/// clear 시에도 초기화되므로 가중치가 있는 문장마다 다시 지정해야 한다.
- (void)setSentenceScoreWeightsPronunciation:(float)pronunciation
                                 recognition:(float)recognition;

@end

NS_ASSUME_NONNULL_END
