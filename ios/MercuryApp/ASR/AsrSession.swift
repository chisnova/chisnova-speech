import AVFoundation
import Foundation
import SwiftUI

/// 녹음 -> 엔드포인트 판정 -> 인식 -> 채점을 한 줄로 엮는다.
///
/// 안드로이드에서는 이 흐름이 VoiceRecognitionActivity 와 AudioController 에 흩어져 있는데,
/// 여기서는 화면과 분리해 한곳에 모은다.
@MainActor
final class AsrSession: ObservableObject {

    enum State: Equatable {
        case idle
        case loadingModel
        case ready
        case recording
        case finishing
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    /// 스트리밍 중간 결과(1패스 CTC) — 발화가 끝나면 2패스 결과로 덮인다
    @Published private(set) var partialText: String = ""
    /// 발화 종료 후의 최종 인식 결과
    @Published private(set) var finalText: String = ""
    /// 마지막 종료 사유 — 왜 끊겼는지 화면에 보여주기 위함
    @Published private(set) var lastEndReason: String = ""
    /// 문장 id -> 평균 발음 점수
    @Published private(set) var scores: [Int: Double] = [:]
    /// 문장 id -> 단어별 상세 점수. 단어를 눌렀을 때 보여 줄 음소 점수까지 들고 있다.
    @Published private(set) var wordScores: [Int: [WordScore]] = [:]
    /// 문장 id -> 최종 인식 결과. 연습이 끝나도 지우지 않아 무엇이라 발음했는지 남는다.
    @Published private(set) var recognisedTexts: [Int: String] = [:]

    /// 채점 JSON 의 final_score 한 항목. 안드로이드 AsrController.WordScore 에 대응한다.
    struct WordScore: Identifiable {
        let index: Int
        let word: String
        let score: Double
        let phones: [PhoneScore]
        /// 엔진의 채점 제외 낱말(ungraded_words.txt)이면 false. 구형 JSON 에는 키가 없어 true 로 본다.
        var graded: Bool = true
        var id: Int { index }
    }

    /// 단어 안의 음소 하나. JSON 에서는 음소도 키가 "word" 다.
    struct PhoneScore: Identifiable {
        let index: Int
        let phone: String
        let score: Double
        var id: Int { index }
    }

    /// 이번에 연습 중인 문장. 채점 인덱스를 여기서 가져온다.
    private var currentSentence: Sentence?
    /// 기록에 남길 레슨 정보. 화면에서 넣어 준다.
    var currentLesson: Lesson?
    private let attemptStore: AttemptStore

    init(attemptStore: AttemptStore? = nil) {
        self.attemptStore = attemptStore ?? .shared
    }

    /// 문장 id -> 녹음 파일. "내 녹음" 재생에 쓴다.
    /// @Published 여야 녹음이 생긴 순간 버튼이 살아난다.
    @Published private var recordings: [Int: URL] = [:]

    /// 그 문장의 녹음이 있으면 경로를 준다.
    func recordingURL(for sentenceId: Int) -> URL? {
        scores[sentenceId] == nil ? nil : recordings[sentenceId]
    }

    private var asr: MercuryAsr?
    private let recorder = AudioRecorder()
    private let detector = EndpointDetector()
    private let beamSize: Int32 = 1

    /// 이번 세션에서 오디오가 한 샘플이라도 들어왔는지.
    /// 하나도 없으면 네이티브가 2패스 인코딩을 만들 수 없어 결과 조회를 건너뛴다.
    private var receivedSamples = 0
    private var chunkCount = 0
    private var sessionGeneration = 0

    /// 녹음 중인지. recorder 의 프로퍼티를 그대로 읽으면 @Published 가 아니라
    /// 화면이 갱신되지 않아 "완료" 버튼이 나타나지 않는다 — 발행되는 state 를 본다.
    var isRecording: Bool { state == .recording }

    // MARK: - 모델 적재

    /// 번들에 실린 모델로 엔진을 준비한다. 이미 준비됐으면 아무 일도 하지 않는다.
    func loadModelIfNeeded() async {
        guard asr == nil, state != .loadingModel else { return }
        await loadModel(at: Self.modelPath)
    }

    /// 실기기에서는 앱 번들의 model_state 를 쓴다.
    /// 시뮬레이터 개발 중에는 번들에 없을 수 있어 호스트 경로로 떨어진다.
    private static var modelPath: String {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("Assets/model_state"),
           FileManager.default.fileExists(atPath: bundled.appendingPathComponent("model.json").path) {
            return bundled.path
        }
        return "/Users/chisnova/mercury/android_mercury/app/src/main/assets/model_state"
    }

    /// 모델 디렉터리를 읽어 엔진을 만든다. 수백 MB 라 백그라운드에서 한다.
    func loadModel(at path: String) async {
        state = .loadingModel
        let engine: MercuryAsr? = await Task.detached(priority: .userInitiated) {
            MercuryAsr(modelPath: path)
        }.value

        guard let engine else {
            state = .failed("모델을 불러오지 못했습니다: \(path)")
            return
        }
        engine.onPartialResult = { [weak self] text in
            // 네이티브 디코딩 스레드에서 온다
            Task { @MainActor in
                self?.partialText = text
            }
        }
        asr = engine
        state = .ready
    }

    // MARK: - 세션

    /// 문장 하나를 연습한다. 녹음이 끝나면 그 문장의 채점 인덱스로 점수를 매긴다.
    func practice(_ sentence: Sentence) async {
        currentSentence = sentence
        await startRecording()
    }

    func startRecording() async {
        guard state == .ready, let asr else { return }
        guard await AudioRecorder.requestPermission() else {
            state = .failed("마이크 권한이 필요합니다.")
            return
        }

        partialText = ""
        finalText = ""
        lastEndReason = ""
        if let id = currentSentence?.id {
            // 다시 연습하면 이전 상세 점수는 의미가 없다
            wordScores[id] = nil
            scores[id] = nil
            recognisedTexts[id] = nil
        }
        receivedSamples = 0
        chunkCount = 0
        detector.reset()
        sessionGeneration += 1
        asr.clear()
        asr.start(withBeamSize: beamSize)

        // 오디오 스레드에서 쓸 것들을 미리 잡아 둔다.
        // 콜백 안에서 MainActor 로 격리된 프로퍼티를 건드리면 그 자리에서 죽는다.
        let engine = asr
        let endpoint = detector
        let recorderRef = recorder
        recorder.onAudioChunk = { [weak self] samples, count in
            // 여기는 AVAudioEngine 의 오디오 스레드다 — 블로킹도 UI 접근도 하지 않는다.
            engine.sendAudio(samples, count: Int32(count), isFinal: false)
            // 이 덩어리에 곱해진 이득을 함께 넘긴다 — 증폭분을 빼야 실제 음량이 나온다
            // 에너지만 보면 무성 자음(/s/, /f/, 파열음 앞 폐쇄)이 무음과 구별되지 않아
            // 어말 자음이 잘린다. 모델이 방금까지 말을 봤는지 함께 넘긴다.
            let reason = endpoint.process(samples, count: count, gain: recorderRef.currentGain,
                                          ctcSilenceMs: Int(engine.ctcTrailingSilenceMs()))

            Task { @MainActor in
                guard let self else { return }
                self.receivedSamples += count
                self.chunkCount += 1
                // 3초에 한 번쯤 판정 상태를 남긴다 — 안 끊기는 이유를 가리기 위함
                if self.chunkCount % 20 == 0 {
                    Diag.log(String(format: "판정: level=%.1fdB floor=%.1fdB SNR=%.1f gain=%.1f speech=%@ ctcSilence=%dms 유예=%dms 경과=%.1fs",
                                    endpoint.lastLevelDb, endpoint.lastNoiseFloorDb,
                                    endpoint.lastLevelDb - endpoint.lastNoiseFloorDb,
                                    Double(recorderRef.currentGain),
                                    endpoint.detectedSpeech ? "yes" : "no",
                                    endpoint.lastCtcSilenceMs, endpoint.lastCtcGraceUsedMs,
                                    Double(self.receivedSamples) / 16000.0))
                }
                // 판정은 오디오 청크마다 오므로 종료 신호도 연달아 들어온다.
                // 이미 멈춘 뒤라면 무시한다 — 아니면 같은 종료가 여러 번 처리된다.
                if endpoint.lastDeferredByCtc {
                    Diag.log("endpoint 유예 — CTC 가 발화를 봤다: " +
                             "ctcSilence=\(endpoint.lastCtcSilenceMs)ms 누적유예=\(endpoint.lastCtcGraceUsedMs)ms")
                }
                if reason != .none, self.recorder.isRecording {
                    self.lastEndReason = Self.describe(reason)
                    // 무엇이 끊었는지 남긴다 — 에너지와 CTC 는 서로 다른 이유로 틀리므로
                    // 사유를 가르지 않으면 튜닝할 때 어느 쪽을 손봐야 할지 알 수 없다.
                    Diag.log("종료 사유: \(self.lastEndReason)" +
                             "(\(endpoint.lastEndedByCtc ? "CTC" : "에너지")) " +
                             "level=\(String(format: "%.1f", endpoint.lastLevelDb))dB " +
                             "floor=\(String(format: "%.1f", endpoint.lastNoiseFloorDb))dB " +
                             "ctcSilence=\(endpoint.lastCtcSilenceMs)ms")
                    self.recorder.stop()
                }
            }
        }
        recorder.onFinished = { [weak self] in
            Task { @MainActor in
                // 디코딩 결과와 무관하게 음원부터 확보한다
                self?.dumpRecording()
                await self?.finishSession()
            }
        }

        state = .recording
        do {
            try recorder.start()
        } catch {
            state = .failed("녹음을 시작하지 못했습니다: \(error.localizedDescription)")
            asr.cancelDecoding()
        }
    }

    /// 마이크를 미리 열어 둔다. 화면에 들어올 때 부른다.
    ///
    /// 녹음 버튼을 누른 뒤에 오디오 세션을 활성화하면 마이크가 실제로 열릴 때까지
    /// 1초 가까이 걸려 첫 음절이 통째로 날아간다. 권한이 없으면 조용히 넘어가고,
    /// 실제 요청은 녹음을 시작할 때 한다.
    func prewarmMicrophone() {
        Task {
            guard await AudioRecorder.requestPermission() else { return }
            do {
                try recorder.prewarm()
            } catch {
                Diag.log("마이크 예열 실패: \(error.localizedDescription)")
            }
        }
    }

    /// 화면을 떠날 때 마이크를 놓아 준다.
    func releaseMicrophone() {
        recorder.teardown()
        sessionGeneration += 1
        asr?.cancelDecoding()
        if state == .recording || state == .finishing { state = .ready }
    }

    /// 화면 진도에 포함된 기록만 복원한다. 주제를 다시 열어도 초기화 상태를 유지한다.
    ///
    /// 주제와 문장을 함께 맞춰 다른 주제의 동일 문장이 복원되지 않도록 한다.
    func restoreScores(for sentences: [Sentence]) {
        guard state != .recording, state != .finishing, let lesson = currentLesson else { return }
        let attempts = attemptStore.progressAttempts.filter { $0.belongs(to: lesson) }
        scores.removeAll()
        wordScores.removeAll()
        recognisedTexts.removeAll()
        var restored = 0
        for sentence in sentences {
            // attempts 는 최근 것이 앞에 있으므로 처음 만나는 것이 가장 최근이다
            guard let hit = attempts.first(where: { $0.sentence == sentence.english }) else { continue }
            scores[sentence.id] = hit.score
            if !hit.transcript.isEmpty { recognisedTexts[sentence.id] = hit.transcript }
            wordScores[sentence.id] = hit.words.enumerated().map { index, word in
                WordScore(index: index, word: word.word, score: word.score,
                          phones: word.phones.enumerated().map { phoneIndex, phone in
                              PhoneScore(index: phoneIndex, phone: phone.phone, score: phone.score)
                          },
                          graded: word.isGraded)
            }
            restored += 1
        }
        Diag.log("불러오기: \(restored)/\(sentences.count) 문장 복원")
    }

    /// 불러오기를 직접 누른 경우에는 초기화 이전 기록까지 다시 보여 준다.
    func loadHistory(for sentences: [Sentence]) throws {
        guard state != .recording, state != .finishing, let lesson = currentLesson else { return }
        try attemptStore.restoreProgress(for: lesson)
        restoreScores(for: sentences)
    }

    /// 저장 기록은 보존하고 화면 점수와 진도만 처음부터 다시 시작한다.
    func clearScores() throws {
        guard state != .recording, state != .finishing, let lesson = currentLesson else { return }
        try attemptStore.resetProgress(for: lesson)
        scores.removeAll()
        wordScores.removeAll()
        recognisedTexts.removeAll()
        currentSentence = nil
        partialText = ""
        finalText = ""
        lastEndReason = ""
    }

    /// 재생이 끝난 뒤 마이크를 다시 연다.
    ///
    /// 예시를 틀 때 오디오 세션이 .playback 으로 바뀌면서(그래야 음량이 제대로 나온다)
    /// 예열해 둔 입력이 끊긴다. 그대로 두면 다음 녹음에서 첫 음절이 다시 잘리므로
    /// 완전히 내렸다가 다시 올린다. 녹음 중이면 건드리지 않는다.
    func restoreMicrophone() {
        guard !recorder.isRecording else { return }
        recorder.teardown()
        prewarmMicrophone()
    }

    /// 사용자가 직접 끝낼 때(완료 버튼). 녹음 중이 아니면 아무 일도 없다.
    func stopRecording() {
        guard recorder.isRecording else { return }
        lastEndReason = "사용자 종료"
        Diag.log("종료 사유: 사용자 종료")
        // 화면이 곧바로 "채점 중"으로 바뀌도록 상태를 먼저 옮긴다.
        // recorder.stop() -> onFinished -> finishSession 은 한 박자 뒤에 온다.
        state = .finishing
        recorder.stop()
    }

    /// 엔진에 넣은 것과 똑같은 오디오를 파일로 남긴다.
    /// 인식이 이상할 때 이 파일을 꺼내 다른 환경에서 그대로 디코딩해 보면
    /// 오디오 문제인지 엔진 문제인지 바로 갈린다.
    private func dumpRecording() {
        let samples = recorder.capturedSamples
        guard !samples.isEmpty else {
            Diag.log("덤프 건너뜀 — 캡처된 샘플이 없다")
            return
        }
        let id = currentSentence?.id ?? 0
        guard let data = Self.makeWav(from: samples, sampleRate: 16000) else { return }
        do {
            let url = try LocalDataStorage.documentsDirectory()
                .appendingPathComponent("capture_\(id).wav")
            try data.write(to: url, options: .atomic)
            recordings[id] = url
            Diag.log(String(format: "덤프 %@ (%.2fs, %d samples)",
                            url.lastPathComponent, Double(samples.count) / 16000.0, samples.count))
        } catch {
            Diag.log("덤프 실패: \(error.localizedDescription)")
        }
    }

    // MARK: - 내부

    private func finishSession() async {
        guard let asr, state == .recording || state == .finishing else { return }
        state = .finishing

        // 남은 것이 없다는 신호. 오디오가 아예 없었어도 이걸 보내야 엔진 스레드가 끝난다.
        let generation = sessionGeneration
        Diag.log("finish: final chunk 전송")
        asr.sendAudio(nil, count: 0, isFinal: true)
        await Task.detached { asr.finishDecoding() }.value
        guard generation == sessionGeneration else { return }
        Diag.log("finish: 디코딩 스레드 종료")

        if receivedSamples > 0 {
            finalText = asr.ctcResult()
            if let id = currentSentence?.id {
                let trimmed = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { recognisedTexts[id] = trimmed }
            }
            Diag.log(String(format: "종료: samples=%d (%.2fs) captured=%d ctc='%@'",
                            receivedSamples, Double(receivedSamples) / 16000.0,
                            recorder.capturedSamples.count, finalText))
            scoreCurrentSentence(using: asr)
        } else {
            // 인코딩된 오디오가 없으면 결과 조회 자체가 의미 없다.
            // 이 상태로 채점을 부르면 네이티브가 널을 역참조한다.
            finalText = ""
            lastEndReason = "오디오 없음"
            Diag.log("종료: 받은 오디오가 없다 (captured=\(recorder.capturedSamples.count))")
        }
        state = .ready
    }

    /// 녹음을 16kHz mono wav 로 남긴다 — AVPlayer 로 그대로 재생할 수 있다.
    private func saveRecording() {
        guard let sentence = currentSentence else { return }
        let samples = recorder.capturedSamples
        guard !samples.isEmpty else {
            Diag.log("녹음 저장 건너뜀 — 캡처된 샘플이 없다")
            return
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("recording_\(sentence.id).wav")
        guard let data = Self.makeWav(from: samples, sampleRate: 16000) else { return }
        do {
            try data.write(to: url, options: .atomic)
            recordings[sentence.id] = url
            Diag.log(String(format: "녹음 저장 %@ (%.2fs)", url.lastPathComponent,
                            Double(samples.count) / 16000.0))
        } catch {
            NSLog("[AsrSession] 녹음 저장 실패: %@", error.localizedDescription)
        }
    }

    /// float PCM 을 16bit wav 로 감싼다.
    private static func makeWav(from samples: [Float], sampleRate: Int) -> Data? {
        var data = Data()
        let byteCount = samples.count * 2

        func append(_ string: String) { data.append(contentsOf: Array(string.utf8)) }
        func append32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func append16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }

        append("RIFF");  append32(UInt32(36 + byteCount)); append("WAVE")
        append("fmt ");  append32(16); append16(1); append16(1)
        append32(UInt32(sampleRate)); append32(UInt32(sampleRate * 2))
        append16(2); append16(16)
        append("data");  append32(UInt32(byteCount))

        for sample in samples {
            let clamped = max(-1, min(1, sample))
            append16(UInt16(bitPattern: Int16(clamped * 32767)))
        }
        return data
    }

    /// 발음 점수. 라벨과 인덱스는 반드시 같은 .jsonl 줄에서 나온 것을 함께 넘긴다.
    private func scoreCurrentSentence(using asr: MercuryAsr) {
        guard let sentence = currentSentence else { return }
        guard let meta = sentence.meta else {
            // .jsonl 이 없거나 줄 수가 문장 수와 맞지 않으면 채점할 인덱스가 없다
            Diag.log("채점 메타 없음 — #\(sentence.id) '\(sentence.english)'")
            return
        }

        // .jsonl 의 문장별 점수 비중을 채점 직전에 넘긴다. 없으면 모델 기본으로 되돌린다 —
        // 이전 문장의 비중이 남지 않게 항상 지정한다.
        if let weights = meta.scoreWeights {
            asr.setSentenceScoreWeightsPronunciation(weights.pronunciation,
                                                     recognition: weights.recognition)
        } else {
            asr.setSentenceScoreWeightsPronunciation(-1, recognition: -1)
        }
        let json = asr.pronounceScore(forLabel: meta.label,
                                      wordIdx: meta.wordIdx.map(NSNumber.init),
                                      wordIdxGroup: meta.wordIdxGroup.map(NSNumber.init),
                                      phnIdx: meta.phnIdx.map(NSNumber.init),
                                      phnIdxGroup: meta.phnIdxGroup.map(NSNumber.init))
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let average = object["avg_score"] as? Double,
              average.isFinite else {
            Diag.log("채점 실패 — words=\(meta.wordIdx.count) phones=\(meta.phnIdx.count) json=\(json.prefix(120))")
            return
        }
        Diag.log(String(format: "점수 %.1f — '%@'", average, meta.label))
        scores[sentence.id] = average
        let words = Self.parseWordScores(object)
        wordScores[sentence.id] = words

        // 히스토리·분석 화면이 읽을 수 있게 남긴다
        attemptStore.add(Attempt(
            id: UUID(),
            date: Date(),
            levelLabel: currentLesson?.level.levelLabel ?? "",
            lessonTitle: currentLesson?.title ?? "",
            sentence: sentence.english,
            transcript: recognisedTexts[sentence.id] ?? "",
            score: average,
            words: words.map { word in
                Attempt.WordRecord(word: word.word, score: word.score,
                                   phones: word.phones.map {
                                       Attempt.PhoneRecord(phone: $0.phone, score: $0.score)
                                   },
                                   graded: word.graded)
            }))
    }

    /// 채점 JSON 의 final_score 배열을 단어 점수로 옮긴다.
    /// 음소 항목도 키가 "word" 인 점에 주의 — 네이티브 FinalScoreToJson 이 그렇게 쓴다.
    private static func parseWordScores(_ object: [String: Any]) -> [WordScore] {
        guard let items = object["final_score"] as? [[String: Any]] else { return [] }
        return items.enumerated().compactMap { index, item in
            guard let word = item["word"] as? String else { return nil }
            let score = (item["score"] as? Double) ?? 0
            let phones = (item["phone_score"] as? [[String: Any]] ?? [])
                .enumerated()
                .compactMap { phoneIndex, phoneItem -> PhoneScore? in
                    guard let phone = phoneItem["word"] as? String else { return nil }
                    let phoneScore = (phoneItem["score"] as? Double) ?? 0
                    return PhoneScore(index: phoneIndex, phone: phone,
                                      score: phoneScore.isFinite ? phoneScore : 0)
                }
            return WordScore(index: index, word: word,
                             score: score.isFinite ? score : 0, phones: phones,
                             graded: (item["graded"] as? Bool) ?? true)
        }
    }

    private static func describe(_ reason: EndpointDetector.Reason) -> String {
        switch reason {
        case .none: return ""
        case .silenceAfterSpeech: return "발화 후 무음"
        case .noSpeechTimeout: return "발화 없음(타임아웃)"
        case .maxDuration: return "최대 길이 도달"
        }
    }
}
