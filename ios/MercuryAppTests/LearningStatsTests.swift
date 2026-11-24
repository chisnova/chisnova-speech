import XCTest
@testable import MercuryApp

@MainActor
final class LearningStatsTests: XCTestCase {
    private func lesson(_ title: String = "Art", level: ContentLevel = .article1,
                        folder: String = "art") -> Lesson {
        Lesson(id: title, title: title, category: nil, level: level,
               folder: folder, relativePath: "\(level.folderName)/\(folder)")
    }

    private func attempt(_ sentence: String, title: String = "Art",
                         level: String = "Level 2") -> Attempt {
        Attempt(id: UUID(), date: Date(), levelLabel: level, lessonTitle: title,
                sentence: sentence, transcript: "", score: 80, words: [])
    }

    func testRepeatedAttemptsAndOtherTopicsDoNotInflateProgress() {
        let stats = LearningStats(attempts: [
            attempt("A"), attempt("A"), attempt("B", title: "City"),
            attempt("B", level: "Level 3"), attempt("B", level: ""), attempt("Removed")
        ])
        let progress = stats.progress(of: lesson(), sentences: ["A", "B"])
        XCTAssertEqual(progress.done, 1)
        XCTAssertEqual(progress.total, 2)
        XCTAssertEqual(progress.ratio, 0.5)
        XCTAssertFalse(progress.isComplete)
    }

    func testAliasesOfTheSameBasicContentShareProgress() {
        let aliases = ContentCatalog.basicTopics.filter { $0.folder == "expressing_opinions" }
        let stats = LearningStats(attempts: [attempt("A", title: aliases[0].title, level: "Level 1")])
        for alias in aliases {
            let progress = stats.progress(of: lesson(alias.title, level: .basic, folder: alias.folder),
                                          sentences: ["A"])
            XCTAssertEqual(progress.ratio, 1)
            XCTAssertTrue(progress.isComplete)
        }
    }

    func testEmptyContentAndUnstartedTopicsAreNotComplete() {
        let stats = LearningStats(attempts: [])
        for sentences: [String] in [[], ["A", "B"]] {
            let progress = stats.progress(of: lesson(), sentences: sentences)
            XCTAssertEqual(progress.ratio, 0)
            XCTAssertFalse(progress.isComplete)
        }
    }

    func testRemovedSpeechDoesNotCountTowardProgressButKeepsHistory() {
        let stats = LearningStats(attempts: [
            attempt("Removed speech", title: "J.K. Rowling's Speech", level: "Level 5"),
            attempt("Active speech", title: ContentCatalog.speeches[0].title, level: "Level 5")
        ])
        XCTAssertEqual(stats.doneCount(of: .speech), 1)
        XCTAssertEqual(stats.progress(of: .speech), 1.0 / 972, accuracy: 0.000001)
        XCTAssertEqual(stats.weekTotalSentences, 2)
        XCTAssertEqual(stats.attempts.count, 2)
    }

    func testRemovedJobsSpeechKeepsHistoryButDoesNotCountTowardProgress() {
        let stats = LearningStats(attempts: [
            attempt("Removed Jobs sentence", title: "Steve Jobs' Speech", level: "Level 5"),
            attempt("Active sentence", title: ContentCatalog.speeches[0].title, level: "Level 5")
        ])
        XCTAssertFalse(ContentCatalog.speeches.contains { $0.folder == "stevejobs" })
        XCTAssertEqual(stats.doneCount(of: .speech), 1)
        XCTAssertEqual(stats.weekTotalSentences, 2)
        XCTAssertEqual(stats.attempts.count, 2)
    }

    private func stats(_ store: AttemptStore) -> LearningStats {
        LearningStats(attempts: store.attempts, progressAttempts: store.progressAttempts)
    }

    private func removeTestFiles(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
        try? FileManager.default.removeItem(at: url.deletingPathExtension().appendingPathExtension("progress.json"))
    }

    func testResetPreservesHistoryAndExplicitLoadRestoresProgress() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("attempts-\(UUID()).json")
        defer { removeTestFiles(url) }
        let store = AttemptStore(url: url)
        let topic = lesson()
        let sentences = [Sentence(id: 1, english: "A", korean: "", meta: nil),
                         Sentence(id: 2, english: "B", korean: "", meta: nil)]
        store.add(attempt("A"))
        store.add(attempt("B"))
        store.add(attempt("A", title: "City"))
        store.add(attempt("A", level: "Level 3"))
        let session = AsrSession(attemptStore: store)
        session.currentLesson = topic
        session.restoreScores(for: sentences)
        XCTAssertEqual(session.scores.count, 2)
        XCTAssertEqual(stats(store).progress(of: topic, sentences: ["A", "B"]).ratio, 1)
        let weekCount = stats(store).weekTotalSentences
        let savedHistory = try Data(contentsOf: url)

        try session.clearScores()
        session.restoreScores(for: sentences) // 자동 복원은 초기화를 되돌리지 않는다.
        XCTAssertTrue(session.scores.isEmpty)
        XCTAssertTrue(session.wordScores.isEmpty)
        XCTAssertTrue(session.recognisedTexts.isEmpty)
        XCTAssertEqual(store.attempts.count, 4)
        XCTAssertEqual(try Data(contentsOf: url), savedHistory)
        XCTAssertEqual(stats(store).doneCount(of: .article1), 1)
        XCTAssertEqual(stats(store).weekTotalSentences, weekCount)

        let reloaded = AttemptStore(url: url)
        XCTAssertEqual(reloaded.attempts.count, 4)
        XCTAssertEqual(stats(reloaded).progress(of: topic, sentences: ["A", "B"]).ratio, 0)
        reloaded.add(attempt("A"))
        XCTAssertEqual(stats(reloaded).progress(of: topic, sentences: ["A", "B"]).ratio, 0.5)
        let reopened = AsrSession(attemptStore: reloaded)
        reopened.currentLesson = topic
        reopened.restoreScores(for: sentences)
        XCTAssertEqual(reopened.scores.count, 1)

        try reopened.loadHistory(for: sentences)
        XCTAssertEqual(reopened.scores.count, 2)
        XCTAssertEqual(stats(reloaded).progress(of: topic, sentences: ["A", "B"]).ratio, 1)
        XCTAssertEqual(reloaded.attempts.count, 5)
        XCTAssertEqual(stats(AttemptStore(url: url)).progress(of: topic, sentences: ["A", "B"]).ratio, 1)
    }

    func testResetAndRestoreApplyToAllAliasesButNotOtherTopics() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("attempts-\(UUID()).json")
        defer { removeTestFiles(url) }
        let store = AttemptStore(url: url)
        let aliases = ContentCatalog.basicTopics.filter { $0.folder == "expressing_opinions" }
        for alias in aliases {
            store.add(attempt("A", title: alias.title, level: "Level 1"))
        }
        store.add(attempt("B"))
        try store.resetProgress(for: lesson(aliases[0].title, level: .basic, folder: aliases[0].folder))
        try store.resetProgress(for: lesson())
        XCTAssertEqual(AttemptStore(url: url).attempts.count, 3)
        XCTAssertEqual(stats(store).progress(of: .basic), 0)
        for alias in aliases {
            XCTAssertEqual(stats(store).progress(of: lesson(alias.title, level: .basic, folder: alias.folder),
                                                sentences: ["A"]).ratio, 0)
        }
        try store.restoreProgress(for: lesson(aliases[1].title, level: .basic, folder: aliases[1].folder))
        for alias in aliases {
            XCTAssertEqual(stats(store).progress(of: lesson(alias.title, level: .basic, folder: alias.folder),
                                                sentences: ["A"]).ratio, 1)
        }
        XCTAssertEqual(stats(store).progress(of: lesson(), sentences: ["B"]).ratio, 0)
    }

    func testFailedResetLeavesHistoryAndProgressUntouched() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathComponent("missing/attempts.json")
        let store = AttemptStore(url: url)
        store.add(attempt("A"))
        XCTAssertThrowsError(try store.resetProgress(for: lesson()))
        XCTAssertEqual(store.attempts.count, 1)
        XCTAssertEqual(stats(store).progress(of: lesson(), sentences: ["A"]).ratio, 1)
    }
}
