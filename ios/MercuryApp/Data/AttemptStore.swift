import Foundation

/// 연습 기록 한 건. 히스토리·분석 화면이 이것만 보고 그린다.
struct Attempt: Codable, Identifiable {
    let id: UUID
    let date: Date
    /// 어느 레벨의 어떤 주제였는지 — 히스토리 줄에 그대로 보인다
    let levelLabel: String
    let lessonTitle: String
    let sentence: String
    /// 그때 실제로 인식된 문장. 예전 기록에는 없을 수 있어 기본값을 둔다.
    let transcript: String
    let score: Double
    let words: [WordRecord]

    /// 같은 콘텐츠 폴더를 공유하는 별칭 주제도 같은 학습 기록으로 취급한다.
    func belongs(to lesson: Lesson) -> Bool {
        guard levelLabel == lesson.level.levelLabel else { return false }
        if lessonTitle == lesson.title { return true }
        return lesson.level == .basic && ContentCatalog.basicTopics.contains {
            $0.folder == lesson.folder && $0.title == lessonTitle
        }
    }

    /// 예전 형식(transcript 가 없던 기록)도 읽을 수 있게 한다.
    enum CodingKeys: String, CodingKey {
        case id, date, levelLabel, lessonTitle, sentence, transcript, score, words
    }

    init(id: UUID, date: Date, levelLabel: String, lessonTitle: String,
         sentence: String, transcript: String, score: Double, words: [WordRecord]) {
        self.id = id
        self.date = date
        self.levelLabel = levelLabel
        self.lessonTitle = lessonTitle
        self.sentence = sentence
        self.transcript = transcript
        self.score = score
        self.words = words
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        date = try c.decode(Date.self, forKey: .date)
        levelLabel = try c.decode(String.self, forKey: .levelLabel)
        lessonTitle = try c.decode(String.self, forKey: .lessonTitle)
        sentence = try c.decode(String.self, forKey: .sentence)
        transcript = try c.decodeIfPresent(String.self, forKey: .transcript) ?? ""
        score = try c.decode(Double.self, forKey: .score)
        words = try c.decode([WordRecord].self, forKey: .words)
    }

    struct WordRecord: Codable {
        let word: String
        let score: Double
        let phones: [PhoneRecord]
        /// 엔진의 채점 제외 낱말(ungraded_words.txt)이면 false. 예전 기록에는 키가 없어 nil 이며 채점됨으로 본다.
        var graded: Bool? = nil
        var isGraded: Bool { graded != false }
    }

    struct PhoneRecord: Codable {
        let phone: String
        let score: Double
    }
}

/// 음소를 발음 유형으로 묶는다 — 안드로이드 FinalScoreSaver 의 표를 그대로 옮겼다.
enum PhoneCategory {

    /// 분석 화면에 나오는 순서
    static let all = ["모음", "파열음", "마찰음", "비음", "유음", "반모음", "파찰음", "기음"]

    static func of(_ phone: String) -> String? {
        let key = phone.uppercased().filter { !$0.isNumber }
        return table[key]
    }

    private static let table: [String: String] = [
        "AA": "모음",
        "AE": "모음",
        "AH": "모음",
        "AO": "모음",
        "AW": "모음",
        "AY": "모음",
        "B": "파열음",
        "CH": "파찰음",
        "D": "파열음",
        "DH": "마찰음",
        "EH": "모음",
        "ER": "모음",
        "EY": "모음",
        "F": "마찰음",
        "G": "파열음",
        "HH": "기음",
        "IH": "모음",
        "IY": "모음",
        "JH": "파찰음",
        "K": "파열음",
        "L": "유음",
        "M": "비음",
        "N": "비음",
        "NG": "비음",
        "OW": "모음",
        "OY": "모음",
        "P": "파열음",
        "R": "유음",
        "S": "마찰음",
        "SH": "마찰음",
        "T": "파열음",
        "TH": "마찰음",
        "UH": "모음",
        "UW": "모음",
        "V": "마찰음",
        "W": "반모음",
        "Y": "반모음",
        "Z": "마찰음",
        "ZH": "마찰음"
    ]
}

/// 연습 기록을 파일에 담아 둔다.
///
/// 안드로이드는 SQLite 를 쓰지만 여기서 필요한 것은 "쌓고, 전부 읽고, 지우기" 뿐이라
/// JSON 한 덩어리로 충분하다. Documents 에 두므로 앱을 다시 열어도 남는다.
@MainActor
final class AttemptStore: ObservableObject {

    static let shared = AttemptStore()

    @Published private(set) var attempts: [Attempt] = []
    @Published private(set) var hiddenAttemptIDs: Set<UUID> = []

    /// 기록은 모두 보관하고, 현재 화면과 진도에 반영할 기록만 따로 고른다.
    var progressAttempts: [Attempt] { attempts.filter { !hiddenAttemptIDs.contains($0.id) } }

    private let url: URL?
    private var progressURL: URL? { url?.deletingPathExtension().appendingPathExtension("progress.json") }

    private init() {
        url = (try? LocalDataStorage.documentsDirectory())?
            .appendingPathComponent("attempts.json")
        load()
    }

    init(url: URL) {
        self.url = url
        load()
    }

    func add(_ attempt: Attempt) {
        attempts.insert(attempt, at: 0)   // 최근 것이 위로
        save()
    }

    func removeAll() {
        attempts.removeAll()
        save()
    }

    /// 초기화 이전 기록은 숨기기만 한다. 이후 새로 추가되는 기록은 바로 진도에 반영된다.
    func resetProgress(for lesson: Lesson) throws {
        let ids = Set(attempts.filter { $0.belongs(to: lesson) }.map(\.id))
        try saveProgressVisibility(hiddenAttemptIDs.union(ids))
    }

    func restoreProgress(for lesson: Lesson) throws {
        let ids = Set(attempts.filter { $0.belongs(to: lesson) }.map(\.id))
        try saveProgressVisibility(hiddenAttemptIDs.subtracting(ids))
    }

    private func saveProgressVisibility(_ ids: Set<UUID>) throws {
        guard let progressURL else { throw CocoaError(.fileNoSuchFile) }
        try JSONEncoder().encode(ids).write(to: progressURL, options: .atomic)
        hiddenAttemptIDs = ids
    }

    // MARK: - 저장·읽기

    private func load() {
        guard let url, let data = try? Data(contentsOf: url) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        attempts = (try? decoder.decode([Attempt].self, from: data)) ?? []
        if let progressURL, let progressData = try? Data(contentsOf: progressURL) {
            hiddenAttemptIDs = (try? JSONDecoder().decode(Set<UUID>.self, from: progressData)) ?? []
        }
    }

    private func save() {
        try? persist(attempts)
    }

    private func persist(_ values: [Attempt]) throws {
        guard let url else { throw CocoaError(.fileNoSuchFile) }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(values)
        try data.write(to: url, options: .atomic)
    }
}
