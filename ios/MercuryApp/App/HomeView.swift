import SwiftUI

/// 홈 — 안드로이드 HomeActivity(activity_home.xml) 대응.
///
/// 위에서부터 이어서 학습하기 → 이번 주 학습 → 오늘의 표현 → 학습 레벨 순으로 쌓는다.
/// 앱을 열자마자 "어디까지 했고 다음에 뭘 하면 되는지"가 보이는 것이 이 화면의 목적이다.
struct HomeView: View {
    @StateObject private var store = AttemptStore.shared

    private var stats: LearningStats {
        LearningStats(attempts: store.attempts, progressAttempts: store.progressAttempts)
    }

    var body: some View {
        VStack(spacing: 0) {
            AppHeader(streak: stats.streakDays)

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    WeeklyCard(stats: stats)
                    PurchaseBanner()

                    HStack(alignment: .firstTextBaseline) {
                        Text("학습 레벨")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundStyle(DS.textPrimary)
                        Spacer()
                    }
                    .padding(.top, 4)

                    VStack(spacing: 10) {
                        ForEach(ContentLevel.allCases) { level in
                            LevelRow(level: level, stats: stats)
                        }
                    }
                }
                .padding(16)
            }
        }
        .background(DS.bg)
        .navigationBarHidden(true)
    }
}

// MARK: - 헤더

/// 상단 헤더 — include_header.xml 대응. 오른쪽에 연속 학습일을 붙인다.
struct AppHeader: View {
    var streak: Int = 0

    var body: some View {
        HStack(spacing: 10) {
            // 런처 아이콘과 같은 그림을 쓴다 — 홈 화면에서 앱을 연 사람이 같은 표시를 본다
            Image("AppLogo")
                .resizable()
                .frame(width: 36, height: 36)
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))

            VStack(alignment: .leading, spacing: 1) {
                Text("쉐도잉코치")
                    .font(.system(size: 17, weight: .bold))
                    .foregroundStyle(DS.primary)
                Text("AI와 함께 마스터")
                    .font(.system(size: 11))
                    .foregroundStyle(DS.textTertiary)
            }
            Spacer()

            if streak > 0 {
                HStack(spacing: 4) {
                    Text("🔥").font(.system(size: 12))
                    Text("\(streak)일")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(DS.accentOrange)
                }
                .padding(.horizontal, 11)
                .padding(.vertical, 6)
                .background(DS.accentOrangeLight)
                .clipShape(Capsule())
            }
            NavigationLink { SettingsView() } label: {
                Image(systemName: "gearshape")
                    .font(.system(size: 21)).foregroundStyle(DS.textSecondary)
                    .frame(width: 44, height: 44).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityLabel("설정 및 안내")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(DS.surface)
    }
}

// MARK: - 이번 주 학습

struct WeeklyCard: View {
    let stats: LearningStats

    var body: some View {
        let week = stats.thisWeek
        let peak = max(1, week.map(\.count).max() ?? 1)

        return VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text("이번 주 학습")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(DS.textPrimary)
                Spacer()
                Text("문장 \(stats.weekTotalSentences)개")
                    .font(.system(size: 12))
                    .foregroundStyle(DS.textTertiary)
            }

            HStack(alignment: .bottom, spacing: 0) {
                ForEach(Array(week.enumerated()), id: \.offset) { _, day in
                    VStack(spacing: 6) {
                        // 오늘은 주황으로 강조하고, 기록이 없는 날은 옅은 점으로 자리만 표시한다
                        if day.count > 0 {
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(day.isToday ? DS.accentOrange : DS.primary)
                                .frame(width: 10,
                                       height: max(8, CGFloat(day.count) / CGFloat(peak) * 52))
                        } else {
                            HStack(spacing: 2) {
                                ForEach(0..<3, id: \.self) { _ in
                                    Circle().fill(DS.border).frame(width: 3, height: 3)
                                }
                            }
                            .frame(height: 8)
                        }
                        Text(day.weekday)
                            .font(.system(size: 11, weight: day.isToday ? .bold : .regular))
                            .foregroundStyle(day.isToday ? DS.accentOrange : DS.textTertiary)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(height: 76, alignment: .bottom)
        }
        .dsCard(padding: 18, radius: 16)
    }
}

// MARK: - 학습 레벨

struct LevelRow: View {
    @StateObject private var purchases = PurchaseStore.shared
    let level: ContentLevel
    let stats: LearningStats

    private var tint: (icon: Color, surface: Color) {
        switch level {
        case .basic:    return (DS.primary, DS.primarySurface)
        case .article1: return (DS.accentGreen, DS.accentGreenSurface)
        case .article2: return (DS.accentOrange, DS.accentOrangeSurface)
        case .article3: return (DS.danger, DS.dangerLight)
        case .speech:   return (DS.accentPurple, DS.accentPurpleSurface)
        }
    }

    private var iconName: String {
        switch level {
        case .basic:    return "ic_bookmark_added_24"
        case .article1: return "ic_message_circle_24"
        case .article2: return "ic_book_open_24"
        case .article3: return "ic_book_24"
        case .speech:   return "ic_mic_24"
        }
    }

    var body: some View {
        let unlocked = level == .basic || purchases.hasFullAccess
        let ratio = stats.progress(of: level)

        NavigationLink {
            LessonListView(level: level)
        } label: {
            HStack(spacing: 13) {
                ZStack {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .fill(unlocked ? tint.surface : DS.surfaceSecondary)
                        .frame(width: 52, height: 52)
                    VectorIcon(name: iconName, size: 26)
                        .foregroundStyle(unlocked ? tint.icon : DS.textTertiary)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(level.levelLabel.uppercased() + (level == .basic && !purchases.hasFullAccess ? " · 일부 무료" : ""))
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(unlocked ? tint.icon : DS.textTertiary)

                    Text(level.title)
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(DS.textPrimary)
                        .lineLimit(1)

                    Text(level == .basic && !purchases.hasFullAccess ? "기초 4개 주제 무료 · 나머지는 전체 이용권" : level.cardCaption)
                        .font(.system(size: 11))
                        .foregroundStyle(DS.textTertiary)
                        .lineLimit(1)

                    HStack(spacing: 7) {
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Capsule().fill(DS.surfaceSecondary)
                                if ratio > 0 {
                                    Capsule().fill(tint.icon)
                                        .frame(width: max(4, ratio * geometry.size.width))
                                }
                            }
                        }
                        .frame(height: 5)

                        if unlocked {
                            Text("\(Int((ratio * 100).rounded()))%")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(DS.textSecondary)
                                .frame(width: 34, alignment: .trailing)
                        } else {
                            Image(systemName: "lock.fill")
                                .font(.system(size: 11))
                                .foregroundStyle(DS.textTertiary)
                                .frame(width: 34, alignment: .trailing)
                        }
                    }
                    .padding(.top, 3)
                }

                Image(systemName: "chevron.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(DS.textTertiary)
            }
            .dsCard(padding: 14, radius: 16)
        }
        .buttonStyle(.plain)
    }
}
