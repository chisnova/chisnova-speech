import SwiftUI

/// 학습 히스토리 — 안드로이드 HistoryActivity(activity_history.xml) 대응.
///
/// 위에 요약 네 칸(평균 점수·총 도전·학습 단어·개선 필요), 아래에 최근 기록 목록.
struct HistoryView: View {
    @StateObject private var store = AttemptStore.shared
    @State private var showClearConfirm = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("학습 히스토리")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(DS.textPrimary)

                summary

                if store.attempts.isEmpty {
                    emptyState
                } else {
                    Text("최근 기록")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(DS.textPrimary)

                    VStack(spacing: 8) {
                        ForEach(store.attempts) { attempt in
                            HistoryRow(attempt: attempt)
                        }
                    }
                }
            }
            .padding(20)
        }
        .background(DS.bg)
        .navigationBarHidden(true)
        .toolbar {
            if !store.attempts.isEmpty {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("초기화") { showClearConfirm = true }
                        .font(.system(size: 13))
                }
            }
        }
        .confirmationDialog("기록을 모두 지울까요?", isPresented: $showClearConfirm) {
            Button("모두 지우기", role: .destructive) { store.removeAll() }
            Button("취소", role: .cancel) {}
        }
    }

    // MARK: - 요약

    private var averageScore: Double {
        guard !store.attempts.isEmpty else { return 0 }
        return store.attempts.map(\.score).reduce(0, +) / Double(store.attempts.count)
    }

    /// 연습하며 채점된 단어 수(같은 단어를 여러 번 해도 각각 센다 — 안드로이드와 같다)
    private var totalWords: Int {
        store.attempts.reduce(0) { $0 + $1.words.count }
    }

    /// 60점 미만이라 더 연습이 필요한 단어 수
    private var needImprovement: Int {
        store.attempts.reduce(0) { $0 + $1.words.filter { $0.isGraded && $0.score < 60 }.count }
    }

    private var summary: some View {
        LazyVGrid(columns: [GridItem(.flexible(), spacing: 10),
                            GridItem(.flexible(), spacing: 10)], spacing: 10) {
            SummaryCard(title: "평균 점수",
                        value: store.attempts.isEmpty ? "—" : "\(Int(averageScore.rounded()))점",
                        tint: DS.primary)
            SummaryCard(title: "총 도전", value: "\(store.attempts.count)회", tint: DS.accentGreen)
            SummaryCard(title: "학습 단어", value: "\(totalWords)개", tint: DS.accentOrange)
            SummaryCard(title: "개선 필요", value: "\(needImprovement)개", tint: DS.danger)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            VectorIcon(name: "ic_calendar_24", size: 34)
                .foregroundStyle(DS.textTertiary)
            Text("아직 연습 기록이 없습니다")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(DS.textSecondary)
            Text("문장을 발음하면 여기에 쌓입니다")
                .font(.system(size: 12))
                .foregroundStyle(DS.textTertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 44)
    }
}

struct SummaryCard: View {
    let title: String
    let value: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.system(size: 12))
                .foregroundStyle(DS.textSecondary)
            Text(value)
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: 14, radius: 14)
    }
}

struct HistoryRow: View {
    let attempt: Attempt

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm"
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(Self.formatter.string(from: attempt.date))
                    .font(.system(size: 11))
                    .foregroundStyle(DS.textTertiary)
                Spacer()
                Text("\(Int(attempt.score.rounded()))점")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(ScoreTint.color(attempt.score))
            }

            Text(attempt.levelLabel.isEmpty
                 ? attempt.lessonTitle
                 : "\(attempt.levelLabel) · \(attempt.lessonTitle)")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(DS.textSecondary)
                .lineLimit(1)

            Text(attempt.sentence)
                .font(.system(size: 14))
                .foregroundStyle(DS.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: 14, radius: 14)
    }
}
