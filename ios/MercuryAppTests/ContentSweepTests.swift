import XCTest
import Foundation
@testable import MercuryApp

/// 번들에 실린 콘텐츠 음원을 실제 엔진에 흘려 전수검사한다.
///
/// 데스크톱 auditV2(contents/run_audit_v2.sh)가 점수 기준의 원천이고, 이 테스트가 보는 것은
/// "실제 앱에 들어간 자산 + 플랫폼 오디오 디코딩 + 브리지 + 단말 엔진"의 결합이다.
/// 안드로이드 `ContentSweepTest.kt` 와 같은 판정 기준을 쓴다.
///
/// ## 판정
/// - 구조 문제(음원 수·JSONL 줄 수 불일치, 디코딩 실패, 빈 인식 결과, 점수 JSON 오류)는 실패.
/// - 문장 점수가 failBelow(기본 85) 미만이면 실패 — auditV2 반영 후 85점 미만은 0건이 기대치다.
/// - warnBelow(기본 90) 미만은 CSV 와 로그에만 남긴다.
///
/// ## 실행 범위와 회수
/// 전체는 4,476문장(약 4.5시간 분량)이라 레벨 폴더로 샤딩할 수 있다:
///
///   xcodebuild test ... -only-testing:MercuryAppTests/ContentSweepTests \
///     TEST_RUNNER_CONTENT_SWEEP_PREFIX=1.basic
///
/// 결과 CSV 는 xcresult 첨부(content_sweep.csv)로 남는다:
///   xcrun xcresulttool export attachments --path <xcresult> --output-path <dir>
final class ContentSweepTests: XCTestCase {

    private static let chunkSamples = 1_600      // 0.1s — 엔진 내부 청크(1.5s)는 네이티브가 관리한다
    private static let beamSize: Int32 = 1       // AsrSession 과 동일

    private var asr: MercuryAsr!
    private var chunkSleepMs: Double = 0

    /// 한 문장의 검사 결과 한 줄.
    private struct Row {
        let topic: String
        let index: Int          // JSONL 줄 번호(0부터) = 음원 번호
        let file: String
        let status: String      // ok | decode_fail | empty_result | score_fail
        let avgScore: Double?
        let minWordScore: Double?
        let recognised: String
    }

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
    }

    override func tearDown() {
        asr?.clear()
        asr = nil
        super.tearDown()
    }

    // ──────────────────────────────────────────────────────────────
    // 전수검사
    // ──────────────────────────────────────────────────────────────

    func testAllContentAudio() async throws {
        let prefix = env("CONTENT_SWEEP_PREFIX")            // 예: "1.basic" 또는 "2.beginner/culture"
        let failBelow = envDouble("CONTENT_SWEEP_FAIL_BELOW") ?? 85
        let warnBelow = envDouble("CONTENT_SWEEP_WARN_BELOW") ?? 90
        // 청크 간 대기(ms). 0이면 최대 속도. 실제 마이크는 100ms 청크가 100ms 마다 오므로
        // 100이면 실시간과 같다 — 최대 속도에서만 생기는 문제를 가려낼 때 쓴다.
        chunkSleepMs = envDouble("CONTENT_SWEEP_CHUNK_SLEEP_MS") ?? 0

        let topics = try Self.topics(matching: prefix)
        XCTAssertFalse(topics.isEmpty,
                       "검사할 주제가 없다 — 접두사 '\(prefix ?? "")' 와 번들 자산을 확인하라")

        var rows: [Row] = []
        var structural: [String] = []
        let started = Date()

        for topic in topics {
            let sentences = try Self.jsonlLines(topic.jsonl)
            let opusFiles = try Self.opusFiles(topic.dir)
            if sentences.count != opusFiles.count {
                structural.append("\(topic.name): JSONL \(sentences.count)줄 vs 음원 \(opusFiles.count)개")
                continue
            }
            for (index, sentence) in sentences.enumerated() {
                let row = await sweep(topic: topic.name, index: index,
                                      opus: opusFiles[index], sentence: sentence,
                                      attachAudioBelow: warnBelow)
                rows.append(row)
            }
            Diag.log(String(format: "[sweep] %@ %d문장 완료 (누적 %d, %.0f초)",
                            topic.name, sentences.count, rows.count,
                            Date().timeIntervalSince(started)))
        }

        attachCsv(rows)

        let broken = rows.filter { $0.status != "ok" }
        let low = rows.filter { $0.status == "ok" && ($0.avgScore ?? 0) < failBelow }
        let warned = rows.filter {
            $0.status == "ok" && ($0.avgScore ?? 0) >= failBelow && ($0.avgScore ?? 0) < warnBelow
        }
        Diag.log("[sweep] 총 \(rows.count)문장 — 구조 문제 \(structural.count), " +
                 "오류 \(broken.count), \(Int(failBelow))점 미만 \(low.count), " +
                 "\(Int(warnBelow))점 미만(경고) \(warned.count)")
        for warn in warned {
            Diag.log(String(format: "[sweep] 경고 %@:%d %.2f점", warn.topic, warn.index + 1,
                            warn.avgScore ?? 0))
        }

        XCTAssertTrue(structural.isEmpty, "구조 문제: \(structural.joined(separator: " / "))")
        XCTAssertTrue(broken.isEmpty, "검사 오류 \(broken.count)건: " +
                      broken.prefix(10).map { "\($0.topic):\($0.index + 1)=\($0.status)" }
                            .joined(separator: ", "))
        XCTAssertTrue(low.isEmpty, "\(Int(failBelow))점 미만 \(low.count)건: " +
                      low.prefix(10).map {
                          String(format: "%@:%d=%.2f", $0.topic, $0.index + 1, $0.avgScore ?? 0)
                      }.joined(separator: ", "))
    }

    /// 한 문장을 엔진에 흘리고 채점까지 마친다.
    /// 점수가 attachAudioBelow 미만이면 단말이 실제로 디코딩한 PCM 을 wav 첨부로 남긴다 —
    /// 문제 문장의 "엔진에 들어간 소리"를 데스크톱 엔진과 대조하기 위함이다.
    private func sweep(topic: String, index: Int, opus: URL,
                       sentence: SentenceMeta, attachAudioBelow: Double = 0) async -> Row {
        guard let samples = await SamplePlayer.decodeMono16k(url: opus), !samples.isEmpty else {
            return Row(topic: topic, index: index, file: opus.lastPathComponent,
                       status: "decode_fail", avgScore: nil, minWordScore: nil, recognised: "")
        }
        let row = runEngine(topic: topic, index: index, opus: opus,
                            samples: samples, sentence: sentence)
        if row.status != "ok" || (row.avgScore ?? 0) < attachAudioBelow {
            attachWav(samples,
                      name: "\(topic.replacingOccurrences(of: "/", with: "_"))_\(index + 1).wav")
        }
        return row
    }

    private func runEngine(topic: String, index: Int, opus: URL,
                           samples: [Float], sentence: SentenceMeta) -> Row {
        // 앱 마이크 경로(AsrSession)와 같은 순서로 보낸다: 데이터 청크는 전부 isFinal=false,
        // 끝나면 빈 최종 청크. 청크 크기도 AudioRecorder.deliveryChunkSize(100ms) 와 같다.
        // 문장별 점수 비중도 앱과 같이 채점 직전에 지정한다(없으면 기본으로 되돌린다).
        if let weights = sentence.scoreWeights {
            asr.setSentenceScoreWeightsPronunciation(weights.pronunciation,
                                                     recognition: weights.recognition)
        } else {
            asr.setSentenceScoreWeightsPronunciation(-1, recognition: -1)
        }
        asr.start(withBeamSize: Self.beamSize)
        samples.withUnsafeBufferPointer { buffer in
            var offset = 0
            while offset < buffer.count {
                let n = min(Self.chunkSamples, buffer.count - offset)
                asr.sendAudio(buffer.baseAddress! + offset, count: Int32(n), isFinal: false)
                offset += n
                if chunkSleepMs > 0 { Thread.sleep(forTimeInterval: chunkSleepMs / 1000) }
            }
        }
        asr.sendAudio(nil, count: 0, isFinal: true)
        asr.finishDecoding()
        let recognised = asr.ctcResult()
        defer { asr.clear() }

        if recognised.trimmingCharacters(in: .whitespaces).isEmpty {
            return Row(topic: topic, index: index, file: opus.lastPathComponent,
                       status: "empty_result", avgScore: nil, minWordScore: nil, recognised: "")
        }

        let json = asr.pronounceScore(forLabel: sentence.label,
                                      wordIdx: sentence.wordIdx.map(NSNumber.init),
                                      wordIdxGroup: sentence.wordGroup.map(NSNumber.init),
                                      phnIdx: sentence.phoneIdx.map(NSNumber.init),
                                      phnIdxGroup: sentence.phoneGroup.map(NSNumber.init))
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let average = object["avg_score"] as? Double, average.isFinite else {
            return Row(topic: topic, index: index, file: opus.lastPathComponent,
                       status: "score_fail", avgScore: nil, minWordScore: nil,
                       recognised: recognised)
        }

        // 채점 제외(graded=false) 단어는 최저 단어 점수에서 뺀다 — 앱 화면과 같은 기준.
        var minWord: Double?
        if let items = object["final_score"] as? [[String: Any]] {
            let scores = items.compactMap { item -> Double? in
                if (item["graded"] as? Bool) == false { return nil }
                guard let score = item["score"] as? Double, score.isFinite else { return nil }
                return score
            }
            minWord = scores.min()
        }
        return Row(topic: topic, index: index, file: opus.lastPathComponent,
                   status: "ok", avgScore: average, minWordScore: minWord,
                   recognised: recognised)
    }

    // ──────────────────────────────────────────────────────────────
    // 번들 자산 열거
    // ──────────────────────────────────────────────────────────────

    private struct Topic {
        let name: String        // contents 기준 상대 경로 (예: "2.beginner/culture/city")
        let dir: URL
        let jsonl: URL
    }

    /// 문장 하나의 채점 인자. AsrSession 이 콘텐츠에서 읽는 것과 같은 필드다.
    private struct SentenceMeta {
        let label: String
        let wordIdx: [Int]
        let wordGroup: [Int]
        let phoneIdx: [Int]
        let phoneGroup: [Int]
        let scoreWeights: (pronunciation: Float, recognition: Float)?
    }

    /// 앱 번들에 실린 모델 디렉터리. AsrMemoryStressTests 와 같은 규칙이다.
    private static var modelPath: String? {
        for bundle in [Bundle(for: ContentSweepTests.self), Bundle.main] {
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

    private static var contentsRoot: URL? {
        for bundle in [Bundle(for: ContentSweepTests.self), Bundle.main] {
            guard let root = bundle.resourceURL?.appendingPathComponent("Assets/contents") else {
                continue
            }
            if FileManager.default.fileExists(atPath: root.path) { return root }
        }
        return nil
    }

    /// 주제 = <이름>.jsonl 을 가진 폴더. prefix 가 있으면 상대 경로가 그걸로 시작해야 한다.
    private static func topics(matching prefix: String?) throws -> [Topic] {
        guard let root = contentsRoot else {
            throw XCTSkip("번들에 Assets/contents 가 없다 — 앱을 기기용으로 빌드했는지 확인하라")
        }
        guard let walker = FileManager.default.enumerator(at: root,
                                                          includingPropertiesForKeys: nil) else {
            return []
        }
        var found: [Topic] = []
        for case let url as URL in walker where url.pathExtension == "jsonl" {
            let dir = url.deletingLastPathComponent()
            guard dir.lastPathComponent == url.deletingPathExtension().lastPathComponent else {
                continue
            }
            let name = dir.path.replacingOccurrences(of: root.path + "/", with: "")
            if let prefix, !prefix.isEmpty, !name.hasPrefix(prefix) { continue }
            found.append(Topic(name: name, dir: dir, jsonl: url))
        }
        return found.sorted { $0.name < $1.name }
    }

    /// opus/<주제>00000.opus 형식 — 이름순 정렬이 곧 문장 순서다(SamplePlayer 와 같은 규칙).
    private static func opusFiles(_ topicDir: URL) throws -> [URL] {
        let dir = topicDir.appendingPathComponent("opus")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return names.filter { $0.hasSuffix(".opus") }.sorted()
            .map { dir.appendingPathComponent($0) }
    }

    private static func jsonlLines(_ url: URL) throws -> [SentenceMeta] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return try text.split(separator: "\n").filter { !$0.isEmpty }.map { line in
            guard let data = line.data(using: .utf8),
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let org = object["org"] as? [String],
                  let wordIdx = object["word_idx"] as? [Int],
                  let wordGroup = object["word_idx_group"] as? [Int],
                  let phoneIdx = object["phn_idx"] as? [Int],
                  let phoneGroup = object["phn_idx_group"] as? [Int] else {
                throw NSError(domain: "ContentSweep", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "\(url.lastPathComponent) JSONL 줄을 읽지 못했다"])
            }
            var weights: (pronunciation: Float, recognition: Float)?
            if let w = object["score_weights"] as? [String: Any],
               let pronunciation = (w["pronunciation"] as? NSNumber)?.floatValue,
               let recognition = (w["recognition"] as? NSNumber)?.floatValue {
                weights = (pronunciation, recognition)
            }
            return SentenceMeta(label: org.joined(separator: " "),
                                wordIdx: wordIdx, wordGroup: wordGroup,
                                phoneIdx: phoneIdx, phoneGroup: phoneGroup,
                                scoreWeights: weights)
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 결과 기록
    // ──────────────────────────────────────────────────────────────

    private func attachCsv(_ rows: [Row]) {
        var csv = "topic,index,file,status,avg_score,min_word_score,recognised\n"
        for row in rows {
            let avg = row.avgScore.map { String(format: "%.2f", $0) } ?? ""
            let minWord = row.minWordScore.map { String(format: "%.2f", $0) } ?? ""
            let recognised = row.recognised.replacingOccurrences(of: "\"", with: "\"\"")
            csv += "\(row.topic),\(row.index + 1),\(row.file),\(row.status),\(avg),\(minWord),\"\(recognised)\"\n"
        }
        let attachment = XCTAttachment(data: Data(csv.utf8), uniformTypeIdentifier: "public.comma-separated-values-text")
        attachment.name = "content_sweep.csv"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    /// 단말이 실제로 디코딩해 엔진에 넣은 PCM 을 16bit WAV 첨부로 남긴다.
    private func attachWav(_ samples: [Float], name: String) {
        var data = Data()
        func le32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func le16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let byteCount = UInt32(samples.count * 2)
        data.append(contentsOf: Array("RIFF".utf8)); le32(36 + byteCount)
        data.append(contentsOf: Array("WAVE".utf8)); data.append(contentsOf: Array("fmt ".utf8))
        le32(16); le16(1); le16(1); le32(16000); le32(32000); le16(2); le16(16)
        data.append(contentsOf: Array("data".utf8)); le32(byteCount)
        for sample in samples {
            let v = Int16(max(-32768, min(32767, Int(sample * 32767))))
            withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
        }
        let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "com.microsoft.waveform-audio")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func env(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]
        return (value?.isEmpty ?? true) ? nil : value
    }

    private func envDouble(_ key: String) -> Double? {
        env(key).flatMap(Double.init)
    }
}
