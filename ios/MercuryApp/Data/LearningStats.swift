import Foundation

struct LessonProgress {
    let done: Int
    let total: Int

    var ratio: Double { total > 0 ? Double(done) / Double(total) : 0 }
    var isComplete: Bool { total > 0 && done == total }
}

/// 홈과 주제 목록이 보여 주는 학습 현황.
///
/// 기록(AttemptStore)만 보고 계산한다. 같은 문장을 여러 번 연습해도 진도는 한 번으로 센다 —
/// 몇 번 했느냐가 아니라 얼마나 훑었느냐가 진도이기 때문이다.
@MainActor
struct LearningStats {

    let attempts: [Attempt]
    private let progressAttempts: [Attempt]

    init(attempts: [Attempt], progressAttempts: [Attempt]? = nil) {
        self.attempts = attempts
        self.progressAttempts = progressAttempts ?? attempts
    }

    // MARK: - 레벨 진행률

    /// 레벨별로 연습해 본 고유 문장 수.
    ///
    /// 기록에는 레벨 이름(levelLabel)이 남아 있어 그것으로 가른다. 예전 기록처럼
    /// 레벨이 비어 있으면 셈에서 빠진다 — 없는 진도를 지어내지 않는다.
    private var doneByLevel: [String: Set<String>] {
        var result: [String: Set<String>] = [:]
        for attempt in progressAttempts where !attempt.levelLabel.isEmpty {
            if attempt.levelLabel == ContentLevel.speech.levelLabel,
               !ContentCatalog.speeches.contains(where: { $0.title == attempt.lessonTitle }) {
                continue
            }
            result[attempt.levelLabel, default: []].insert(attempt.sentence)
        }
        return result
    }

    func progress(of level: ContentLevel) -> Double {
        let total = level.totalSentences
        guard total > 0 else { return 0 }
        let done = doneByLevel[level.levelLabel]?.count ?? 0
        return min(1, Double(done) / Double(total))
    }

    func doneCount(of level: ContentLevel) -> Int {
        doneByLevel[level.levelLabel]?.count ?? 0
    }

    /// 같은 콘텐츠를 가리키는 주제는 진도를 공유하고, 반복 연습은 한 번만 센다.
    func progress(of lesson: Lesson, sentences: [String]) -> LessonProgress {
        let practiced = Set(progressAttempts.filter { $0.belongs(to: lesson) }.map(\.sentence))
        let content = Set(sentences)
        return LessonProgress(done: content.intersection(practiced).count, total: content.count)
    }

    // MARK: - 이어서 학습하기

    /// 가장 최근에 연습한 자리. 홈 맨 위 카드가 이것을 가리킨다.
    var lastPractice: Attempt? { attempts.first }

    // MARK: - 이번 주

    /// 월요일부터 일요일까지, 그날 연습한 문장 수.
    var thisWeek: [(weekday: String, count: Int, isToday: Bool)] {
        var calendar = Calendar(identifier: .gregorian)
        calendar.firstWeekday = 2   // 월요일 시작
        let today = calendar.startOfDay(for: Date())
        guard let start = calendar.dateInterval(of: .weekOfYear, for: today)?.start else { return [] }

        let names = ["월", "화", "수", "목", "금", "토", "일"]
        return (0..<7).map { offset in
            let day = calendar.date(byAdding: .day, value: offset, to: start) ?? start
            let count = attempts.filter { calendar.isDate($0.date, inSameDayAs: day) }.count
            return (names[offset], count, calendar.isDate(day, inSameDayAs: today))
        }
    }

    var weekTotalSentences: Int { thisWeek.reduce(0) { $0 + $1.count } }

    /// 오늘까지 며칠 내리 연습했는지. 오늘 아직 안 했으면 어제까지로 센다.
    var streakDays: Int {
        let calendar = Calendar.current
        let days = Set(attempts.map { calendar.startOfDay(for: $0.date) })
        guard !days.isEmpty else { return 0 }

        var day = calendar.startOfDay(for: Date())
        if !days.contains(day) {
            guard let yesterday = calendar.date(byAdding: .day, value: -1, to: day),
                  days.contains(yesterday) else { return 0 }
            day = yesterday
        }
        var count = 0
        while days.contains(day) {
            count += 1
            guard let previous = calendar.date(byAdding: .day, value: -1, to: day) else { break }
            day = previous
        }
        return count
    }
}
