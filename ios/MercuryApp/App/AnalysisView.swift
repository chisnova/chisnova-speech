import SwiftUI

/// 발음 분석 — 안드로이드 GraphActivity(activity_graph.xml) 대응.
///
/// 기간을 골라 그 안의 기록을 발음 유형(모음·파열음 …)별로 묶어 평균을 낸다.
/// 어떤 유형이 강하고 약한지가 핵심이라, 유형별 막대와 강점·약점을 함께 보여 준다.
struct AnalysisView: View {
    @StateObject private var store = AttemptStore.shared
    @State private var period: Period = .week

    enum Period: String, CaseIterable, Identifiable {
        case week = "최근 1주"
        case month = "최근 1달"
        case all = "전체 기간"
        var id: String { rawValue }

        /// 이 기간의 시작. 전체면 nil.
        var start: Date? {
            switch self {
            case .week:  return Calendar.current.date(byAdding: .day, value: -7, to: Date())
            case .month: return Calendar.current.date(byAdding: .month, value: -1, to: Date())
            case .all:   return nil
            }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("발음 분석")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(DS.textPrimary)

                periodPicker

                if filtered.isEmpty {
                    emptyState
                } else {
                    overallCard
                    radarSection
                    typeSection
                }
            }
            .padding(20)
        }
        .background(DS.bg)
        .navigationBarHidden(true)
    }

    // MARK: - 집계

    private var filtered: [Attempt] {
        guard let start = period.start else { return store.attempts }
        return store.attempts.filter { $0.date >= start }
    }

    /// 유형 -> (점수 합, 개수). 음소 하나하나를 유형에 넣어 평균 낸다.
    private var byType: [(type: String, score: Double, count: Int)] {
        var sum: [String: (Double, Int)] = [:]
        for attempt in filtered {
            for word in attempt.words {
                for phone in word.phones {
                    guard let type = PhoneCategory.of(phone.phone) else { continue }
                    let previous = sum[type] ?? (0, 0)
                    sum[type] = (previous.0 + phone.score, previous.1 + 1)
                }
            }
        }
        return PhoneCategory.all.compactMap { type in
            guard let (total, count) = sum[type], count > 0 else { return nil }
            return (type, total / Double(count), count)
        }
    }

    private var totalPhones: Int { byType.reduce(0) { $0 + $1.count } }

    private var overallScore: Double {
        guard !filtered.isEmpty else { return 0 }
        return filtered.map(\.score).reduce(0, +) / Double(filtered.count)
    }

    // MARK: - 조각

    private var periodPicker: some View {
        HStack(spacing: 6) {
            ForEach(Period.allCases) { item in
                Button {
                    period = item
                } label: {
                    Text(item.rawValue)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(period == item ? .white : DS.textSecondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(period == item ? DS.primary : DS.surface)
                        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var overallCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("종합 발음 점수")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(DS.textSecondary)
                    Text("\(byType.count)개 발음 유형 평균")
                        .font(.system(size: 11))
                        .foregroundStyle(DS.textTertiary)
                }
                Spacer()
                Text("\(Int(overallScore.rounded()))점")
                    .font(.system(size: 30, weight: .bold))
                    .foregroundStyle(ScoreTint.color(overallScore))
            }

            Divider()

            HStack(spacing: 0) {
                statColumn("총 발음소", "\(totalPhones)개", DS.textPrimary)
                statColumn("강점 유형", byType.max { $0.score < $1.score }?.type ?? "—", DS.accentGreen)
                statColumn("약점 유형", byType.min { $0.score < $1.score }?.type ?? "—", DS.danger)
            }
        }
        .dsCard()
    }

    private func statColumn(_ title: String, _ value: String, _ tint: Color) -> some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.system(size: 11))
                .foregroundStyle(DS.textTertiary)
            Text(value)
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity)
    }

    /// 유형별 점수를 한눈에 — 안드로이드의 방사형 그래프.
    /// 축이 셋은 되어야 도형이 되므로 그보다 적으면 막대만 보여 준다.
    @ViewBuilder
    private var radarSection: some View {
        if byType.count >= 3 {
            VStack(alignment: .leading, spacing: 10) {
                Text("발음 유형 분포")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(DS.textPrimary)

                RadarChart(values: byType.map { ($0.type, $0.score) })
                    .frame(maxWidth: .infinity)
                    .frame(height: 300)
                    .padding(.vertical, 4)
                    .dsCard()
            }
        }
    }

    private var typeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("발음 유형별 점수")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(DS.textPrimary)

            VStack(spacing: 12) {
                ForEach(byType, id: \.type) { item in
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text(item.type)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(DS.textPrimary)
                            Text("\(item.count)개")
                                .font(.system(size: 11))
                                .foregroundStyle(DS.textTertiary)
                            Spacer()
                            Text("\(Int(item.score.rounded()))점")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundStyle(ScoreTint.color(item.score))
                        }
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Capsule().fill(DS.surfaceSecondary)
                                Capsule()
                                    .fill(ScoreTint.color(item.score))
                                    .frame(width: max(0, min(1, item.score / 100)) * geometry.size.width)
                            }
                        }
                        .frame(height: 8)
                    }
                }
            }
            .dsCard()
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            VectorIcon(name: "ic_bar_chart_24", size: 34)
                .foregroundStyle(DS.textTertiary)
            Text("분석할 기록이 없습니다")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(DS.textSecondary)
            Text("이 기간에 연습한 문장이 없습니다")
                .font(.system(size: 12))
                .foregroundStyle(DS.textTertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 44)
    }
}
