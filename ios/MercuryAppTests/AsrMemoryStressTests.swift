import XCTest
import Foundation
@testable import MercuryApp

/// 세션을 반복해도 메모리가 계속 늘지 않고, 비정상 호출에도 죽지 않는지 본다.
///
/// 안드로이드 `AsrMemoryStressTest.kt` 와 같은 시나리오를 같은 판정 기준으로 확인한다.
/// 두 플랫폼이 같은 C++ 엔진을 쓰므로, 한쪽에서만 새면 브리지 코드가 원인이라는 뜻이 된다.
///
/// ## 어떻게 누수라고 판정하는가
///
/// "N 회 돌린 뒤 늘었다" 는 기준이 못 된다. 할당자는 해제한 블록을 곧바로 OS 에 돌려주지
/// 않고, 첫 세션은 캐시를 채우느라 원래 늘어난다. 그래서 **구간을 둘로 나눠 증가 속도를
/// 견준다.** 진짜 누수면 뒤 구간도 계속 늘고, 예열이었다면 뒤 구간은 평평해진다.
final class AsrMemoryStressTests: XCTestCase {

    /// 예열 구간. 이 동안의 증가는 누수로 세지 않는다.
    private let warmupRounds = 3
    /// 비교할 두 구간 각각의 반복 횟수.
    private let roundsPerPhase = 8
    /// 뒤 구간이 앞 구간의 이 배수를 넘으면 누수로 본다(측정 잡음을 감안한 여유).
    private let growthRatioLimit = 1.6
    /// 이보다 작게 늘었으면 비율을 따지지 않는다 — 작은 절대값에서 비율이 크게 흔들린다.
    private let noiseFloorBytes: Int64 = 4 * 1024 * 1024

    private var asr: MercuryAsr!
    private var utterance: [Float] = []

    // ──────────────────────────────────────────────────────────────
    // 준비
    // ──────────────────────────────────────────────────────────────

    override func setUpWithError() throws {
        try super.setUpWithError()

        guard let modelPath = Self.modelPath else {
            throw XCTSkip("모델이 번들에 없다 — 앱을 기기용으로 빌드했는지 확인하라")
        }
        guard let engine = MercuryAsr(modelPath: modelPath) else {
            XCTFail("엔진을 만들지 못했다: \(modelPath)")
            return
        }
        asr = engine
        utterance = Self.selftestSamples() ?? Self.syntheticSpeech()
    }

    override func tearDown() {
        asr?.clear()
        asr = nil
        super.tearDown()
    }

    /// 앱 번들에 실린 모델 디렉터리. AsrSession.modelPath 와 같은 규칙이다.
    private static var modelPath: String? {
        for bundle in [Bundle(for: AsrMemoryStressTests.self), Bundle.main] {
            guard let root = bundle.resourceURL?.appendingPathComponent("Assets/model_state") else {
                continue
            }
            if FileManager.default.fileExists(
                atPath: root.appendingPathComponent("model.json").path) {
                return root.path
            }
        }
        return nil
    }

    /// 번들의 진단용 녹음. 없으면 합성음으로 대신한다 — 여기서 보는 것은 인식 품질이 아니다.
    private static func selftestSamples() -> [Float]? {
        for bundle in [Bundle(for: AsrMemoryStressTests.self), Bundle.main] {
            guard let url = bundle.resourceURL?
                .appendingPathComponent("Assets/diag/selftest.wav"),
                  let data = try? Data(contentsOf: url), data.count > 44 else { continue }
            // 16bit PCM WAV 로 가정한다(번들의 selftest.wav 가 그렇다).
            let body = data.dropFirst(44)
            var out = [Float]()
            out.reserveCapacity(body.count / 2)
            body.withUnsafeBytes { raw in
                let ints = raw.bindMemory(to: Int16.self)
                for v in ints { out.append(Float(v) / 32768.0) }
            }
            return out.isEmpty ? nil : out
        }
        return nil
    }

    /// 자산이 없을 때 쓰는 대체 신호 — 무음이 아니어야 디코더가 실제로 일한다.
    private static func syntheticSpeech() -> [Float] {
        let count = 16000    // 1초
        return (0..<count).map { i in
            let t = Float(i) / 16000.0
            return 0.2 * sin(2 * .pi * 220 * t) * (1 + 0.5 * sin(2 * .pi * 3 * t))
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 측정 도구
    // ──────────────────────────────────────────────────────────────

    /// 지금 이 프로세스가 쓰는 물리 메모리.
    ///
    /// `phys_footprint` 는 Xcode 메모리 게이지·jetsam 이 보는 값과 같다. resident_size 는
    /// 압축된 페이지를 놓쳐 과소평가되므로 쓰지 않는다.
    private func footprintBytes() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int64(info.phys_footprint)
    }

    /// `work` 를 예열 + 두 구간만큼 돌리고 두 구간의 증가량을 견준다.
    private func assertNoGrowth(_ label: String,
                                file: StaticString = #filePath,
                                line: UInt = #line,
                                work: (Int) -> Void) {
        for i in 0..<warmupRounds { work(i) }

        let start = footprintBytes()
        for i in 0..<roundsPerPhase { work(warmupRounds + i) }
        let mid = footprintBytes()
        for i in 0..<roundsPerPhase { work(warmupRounds + roundsPerPhase + i) }
        let end = footprintBytes()

        let firstPhase = mid - start
        let secondPhase = end - mid
        let detail = "\(label) — 예열 \(warmupRounds)회 뒤 " +
            "1구간(\(roundsPerPhase)회) \(firstPhase / 1024)KB, " +
            "2구간(\(roundsPerPhase)회) \(secondPhase / 1024)KB " +
            "(기준 \(start / 1024 / 1024)MB -> \(end / 1024 / 1024)MB)"

        if secondPhase <= noiseFloorBytes { return }          // 사실상 늘지 않았다

        guard firstPhase > 0 else {
            XCTFail("\(detail) — 1구간은 늘지 않았는데 2구간만 늘었다", file: file, line: line)
            return
        }
        let ratio = Double(secondPhase) / Double(firstPhase)
        XCTAssertLessThanOrEqual(
            ratio, growthRatioLimit,
            "\(detail), 증가비 \(String(format: "%.2f", ratio)) — 반복해도 줄지 않으면 누수다",
            file: file, line: line
        )
    }

    /// 한 세션을 끝까지 돌린다. 앱이 문장 하나를 연습할 때와 같은 순서다.
    private func runSession(samples: [Float], secondPass: Bool) {
        asr.start(withBeamSize: 3)
        samples.withUnsafeBufferPointer { buffer in
            var offset = 0
            let chunk = 1600                       // 100ms — 앱 녹음 경로와 같은 단위
            while offset < buffer.count {
                let n = min(chunk, buffer.count - offset)
                asr.sendAudio(buffer.baseAddress! + offset, count: Int32(n),
                              isFinal: offset + n >= buffer.count)
                offset += n
            }
        }
        // 보낼 것이 없다는 신호. 이걸 빠뜨리면 엔진 스레드가 마지막 청크를 영원히 기다린다.
        asr.sendAudio(nil, count: 0, isFinal: true)
        asr.finishDecoding()
        _ = asr.ctcResult()
        if secondPass { _ = asr.secondPassResult(withBeamSize: 3) }
        asr.clear()
    }

    // ──────────────────────────────────────────────────────────────
    // 메모리
    // ──────────────────────────────────────────────────────────────

    func testRepeatedDecodeSessionsDoNotLeak() {
        assertNoGrowth("디코딩 세션 반복") { _ in
            runSession(samples: utterance, secondPass: false)
        }
    }

    func testRepeatedSecondPassDoesNotLeak() {
        assertNoGrowth("2패스 포함 세션 반복") { _ in
            runSession(samples: utterance, secondPass: true)
        }
    }

    /// 사용자가 녹음을 중간에 끊는 경우와 끝까지 가는 경우를 번갈아 돌린다.
    func testInterleavedCancelAndFinishDoNotLeak() {
        assertNoGrowth("취소·완료 교대") { round in
            if round % 2 == 0 {
                runSession(samples: utterance, secondPass: false)
            } else {
                asr.start(withBeamSize: 3)
                let half = Array(utterance.prefix(utterance.count / 2))
                half.withUnsafeBufferPointer {
                    asr.sendAudio($0.baseAddress!, count: Int32($0.count), isFinal: false)
                }
                asr.cancelDecoding()
                asr.clear()
            }
        }
    }

    /// EPD 가 무음으로 끊었을 때 실제로 도는 경로.
    func testRepeatedSilenceSessionsDoNotLeak() {
        let silence = [Float](repeating: 0, count: 16000)
        assertNoGrowth("무음 세션 반복") { _ in
            runSession(samples: silence, secondPass: false)
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 크래시
    // ──────────────────────────────────────────────────────────────

    /// 상태 머신을 어긋난 순서로 두드린 뒤에도 정상 세션이 되는지 본다.
    ///
    /// 실제 앱에서 화면을 빠르게 오가면 이런 순서가 만들어진다. 특히 종료 신호를 빠뜨리면
    /// finishDecoding 의 join 이 돌아오지 않아 여기서 멈춘다 — 그 회귀를 잡는 것이 목적이다.
    func testAbnormalCallSequencesDoNotCrash() {
        for round in 0..<30 {
            switch round % 6 {
            case 0: asr.finishDecoding()                        // 세션 없이 종료
            case 1: asr.cancelDecoding()                        // 세션 없이 취소
            case 2: asr.clear()                                 // 세션 없이 초기화
            case 3:                                             // 중복 시작
                asr.start(withBeamSize: 3)
                asr.start(withBeamSize: 3)
                asr.cancelDecoding()
            case 4:                                             // 세션 밖 오디오
                utterance.withUnsafeBufferPointer {
                    asr.sendAudio($0.baseAddress!, count: Int32($0.count), isFinal: true)
                }
            default:                                            // 시작하자마자 종료
                asr.start(withBeamSize: 3)
                asr.sendAudio(nil, count: 0, isFinal: true)
                asr.finishDecoding()
            }
            asr.clear()
        }

        runSession(samples: utterance, secondPass: false)
        // 여기까지 왔다는 것 자체가 통과 조건이다 — 위 순서 중 하나라도 죽거나 멈추면
        // 테스트가 이 줄에 닿지 못한다.
    }

    /// 오디오가 한 번도 안 들어온 세션을 연달아 연다.
    func testRepeatedEmptySessionsDoNotCrash() {
        for _ in 0..<20 {
            asr.start(withBeamSize: 3)
            asr.sendAudio(nil, count: 0, isFinal: true)
            asr.finishDecoding()
            asr.clear()
        }
        runSession(samples: utterance, secondPass: false)
    }

    /// 종료 신호를 보내지 않고 끝내려 하면 멈추는지 — 데드락 회귀 감시.
    ///
    /// 실패하면 테스트가 통째로 매달리므로 시간 제한을 둔다.
    func testFinishWithoutEndSignalDoesNotHang() {
        let done = expectation(description: "finishDecoding 이 돌아온다")
        DispatchQueue.global().async { [self] in
            asr.start(withBeamSize: 3)
            utterance.withUnsafeBufferPointer {
                asr.sendAudio($0.baseAddress!, count: Int32($0.count), isFinal: false)
            }
            asr.sendAudio(nil, count: 0, isFinal: true)
            asr.finishDecoding()
            asr.clear()
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
    }
}
