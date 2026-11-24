import SwiftUI

/// 레벨 안의 주제 목록 — 안드로이드 Level1/2/3Activity 에 대응.
///
/// 세 레벨이 화면 구조를 공유한다: 제목 헤더 아래에 분류별로 묶인 카드 목록.
/// 다만 연설문(Level 3)만 카드에 인물 사진이 들어가 모양이 다르다.
struct LessonListView: View {
    @StateObject private var purchases = PurchaseStore.shared
    let level: ContentLevel
    @State private var lessons: [Lesson] = []
    @State private var sentencesByLesson: [String: [String]] = [:]
    @StateObject private var store = AttemptStore.shared

    private var stats: LearningStats {
        LearningStats(attempts: store.attempts, progressAttempts: store.progressAttempts)
    }

    private func progress(of lesson: Lesson) -> LessonProgress {
        stats.progress(of: lesson, sentences: sentencesByLesson[lesson.id] ?? [])
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header
                PurchaseBanner()

                if lessons.isEmpty {
                    Text("콘텐츠가 없습니다.")
                        .font(.system(size: 14))
                        .foregroundStyle(DS.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.top, 40)
                } else if level == .speech {
                    speechList
                } else {
                    groupedList
                }
            }
            .padding(20)
        }
        .background(DS.bg)
        .navigationTitle(level.levelLabel)
        .navigationBarTitleDisplayMode(.inline)
        .task {
            lessons = ContentStore.lessons(in: level)
            sentencesByLesson = Dictionary(uniqueKeysWithValues: lessons.map {
                ($0.id, ContentStore.englishSentences(of: $0))
            })
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(level.title)
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(DS.textPrimary)
            Text(level.subtitle)
                .font(.system(size: 13))
                .foregroundStyle(DS.textSecondary)
        }
    }

    // MARK: - 분류별 목록 (Level 1·2)

    /// 화면에 나올 분류를 순서대로 돌려준다. 카탈로그에 없는 분류는 뒤에 붙인다.
    private var groups: [(category: ContentCatalog.Category, lessons: [Lesson])] {
        let known: [ContentCatalog.Category] = level == .basic
            ? ContentCatalog.basicCategories
            : ContentCatalog.articleCategories
                .sorted { $0.key < $1.key }
                .map(\.value)

        var result: [(ContentCatalog.Category, [Lesson])] = []
        var placed = Set<String>()

        for category in known {
            let items = lessons.filter { $0.category == category.id }
            guard !items.isEmpty else { continue }
            items.forEach { placed.insert($0.id) }
            result.append((category, items))
        }

        // 분류를 모르는 주제도 빠뜨리지 않는다
        let rest = lessons.filter { !placed.contains($0.id) }
        if !rest.isEmpty {
            result.append((ContentCatalog.Category(id: "etc", title: "기타", emoji: "📄",
                                                  icon: "ic_book_open_24", tint: "bg_icon_blue"), rest))
        }
        return result
    }

    private var groupedList: some View {
        ForEach(groups, id: \.category.id) { group in
            let isLocked = !group.lessons.contains { purchases.canAccess($0) }
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    // 안드로이드는 분류 머리에 이모지를 쓴다. 아이콘을 함께 두어
                    // 어떤 분류인지 한눈에 들어오게 한다.
                    ZStack {
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .fill(isLocked ? DS.surfaceSecondary : DS.iconTint(group.category.tint).surface)
                            .frame(width: 30, height: 30)
                        VectorIcon(name: group.category.icon, size: 18)
                            .foregroundStyle(isLocked ? DS.textTertiary : DS.iconTint(group.category.tint).icon)
                    }
                    Text(group.category.emoji)
                        .font(.system(size: 15))
                        .saturation(isLocked ? 0 : 1)
                    Text(group.category.title)
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(DS.textPrimary)
                    Spacer()
                    if level == .basic {
                        Text(group.category.id == "basic" ? "4개 주제 무료" : (purchases.hasFullAccess ? "이용 가능" : "전체 이용권 필요"))
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(group.category.id == "basic" ? DS.primary : DS.textSecondary)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(group.category.id == "basic" ? DS.primarySurface : DS.surfaceSecondary)
                            .clipShape(Capsule())
                    }
                    Text("\(group.lessons.count)")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(DS.textTertiary)
                }

                VStack(spacing: 8) {
                    ForEach(group.lessons) { lesson in
                        NavigationLink {
                            SentenceListView(lesson: lesson)
                        } label: {
                            LessonRow(lesson: lesson, progress: progress(of: lesson))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: - 연설문 (Level 3)

    /// 카탈로그에 적힌 순서를 따른다 — 안드로이드 레이아웃과 같은 차례.
    private var orderedSpeeches: [Lesson] {
        let byFolder = Dictionary(lessons.map { ($0.folder, $0) }, uniquingKeysWith: { first, _ in first })
        let known = ContentCatalog.speeches.compactMap { byFolder[$0.folder] }
        let knownIds = Set(known.map(\.id))
        return known + lessons.filter { !knownIds.contains($0.id) }
    }

    private var speechList: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("유명 연설문")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(DS.textPrimary)

            VStack(spacing: 10) {
                ForEach(orderedSpeeches) { lesson in
                    NavigationLink {
                        SentenceListView(lesson: lesson)
                    } label: {
                        SpeechRow(lesson: lesson, progress: progress(of: lesson))
                    }
                    .buttonStyle(.plain)
                }
            }

            speechHighlights
            DisclosureGroup("새 연설 원문과 제작 안내") {
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(ContentCatalog.speeches, id: \.folder) { speech in
                        if let source = ContentStore.speechSource(folder: speech.folder) {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("\(source.author) · \(source.date)").font(.headline)
                                Text(source.title)
                                Text(source.notice).foregroundStyle(DS.textSecondary)
                                Link(source.sourceName, destination: source.sourceURL)
                            }
                        }
                    }
                }
                .font(.footnote)
                .padding(.top, 12)
            }
            .dsCard(padding: 16, radius: 14)
                .padding(.top, 8)
        }
    }

    /// 목록 아래 안내 — 안드로이드 연설문 화면의 "연설문 학습 특징".
    private var speechHighlights: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("연설문 학습 특징")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(DS.textPrimary)

            ForEach(Array(ContentCatalog.speechHighlights.enumerated()), id: \.offset) { index, item in
                HStack(alignment: .top, spacing: 10) {
                    Text("\(index + 1)")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(DS.accentPurple)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(DS.accentPurpleSurface)
                        .clipShape(Capsule())

                    VStack(alignment: .leading, spacing: 3) {
                        Text(item.title)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(DS.textPrimary)
                        Text(item.detail)
                            .font(.system(size: 12))
                            .foregroundStyle(DS.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: 20, radius: 16)
    }
}

/// 주제 카드 — 안드로이드 Level1/2 의 레슨 행(아이콘 + 제목 + 진행률).
struct LessonRow: View {
    @StateObject private var purchases = PurchaseStore.shared
    let lesson: Lesson
    let progress: LessonProgress

    /// 안드로이드 LessonData 가 이 주제에 지정한 아이콘.
    /// 카탈로그에 없는 폴더는 레벨 기본 아이콘으로 떨어진다.
    private var iconName: String {
        if let name = lesson.icon { return name }
        switch lesson.level {
        case .basic:  return "ic_message_circle_24"
        case .speech: return "ic_mic_24"
        default:      return "ic_book_open_24"
        }
    }

    /// 주제별 세 단계는 홈 카드와 같은 색을 쓴다 — 어느 단계인지 색으로 알 수 있게.
    private var tint: Color {
        switch lesson.level {
        case .basic:    return DS.primary
        case .article1: return DS.accentGreen
        case .article2: return DS.accentOrange
        case .article3: return DS.danger
        case .speech:   return DS.accentPurple
        }
    }

    private var tintSurface: Color {
        switch lesson.level {
        case .basic:    return DS.primarySurface
        case .article1: return DS.accentGreenSurface
        case .article2: return DS.accentOrangeSurface
        case .article3: return DS.dangerLight
        case .speech:   return DS.accentPurpleSurface
        }
    }

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(purchases.canAccess(lesson) ? tintSurface : DS.surfaceSecondary)
                    .frame(width: 46, height: 46)
                VectorIcon(name: iconName, size: 26)
                    .foregroundStyle(purchases.canAccess(lesson) ? tint : DS.textTertiary)
            }

            VStack(alignment: .leading, spacing: 7) {
                Text(lesson.title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(DS.textPrimary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                LessonProgressView(progress: progress, tint: tint)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if !purchases.canAccess(lesson) {
                Image(systemName: "lock.fill").foregroundStyle(DS.textTertiary)
                    .accessibilityLabel("전체 이용권 필요")
            }

        }
        .dsCard(padding: 12, radius: 14)
    }
}

/// 연설문 카드 — 인물 사진 + 제목 + 부제(안드로이드 activity_level3 의 행).
struct SpeechRow: View {
    @StateObject private var purchases = PurchaseStore.shared
    let lesson: Lesson
    let progress: LessonProgress

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let name = lesson.imageName, UIImage(named: name) != nil {
                    Image(name)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .saturation(purchases.canAccess(lesson) ? 1 : 0)
                } else {
                    // 사진이 없으면 자리만 지킨다 — 목록이 무너지지 않게
                    ZStack {
                        purchases.canAccess(lesson) ? DS.accentPurpleSurface : DS.surfaceSecondary
                        Image(systemName: "megaphone.fill")
                            .font(.system(size: 20))
                            .foregroundStyle(purchases.canAccess(lesson) ? DS.accentPurple : DS.textTertiary)
                    }
                }
            }
            .frame(width: 64, height: 64)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

            VStack(alignment: .leading, spacing: 3) {
                Text(lesson.title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(DS.textPrimary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                if let subtitle = lesson.subtitle {
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(DS.textSecondary)
                        .lineLimit(1)
                }
                LessonProgressView(progress: progress, tint: DS.accentPurple)
                    .padding(.top, 4)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 4)
            Image(systemName: purchases.canAccess(lesson) ? "chevron.right" : "lock.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(DS.textTertiary)
        }
        .dsCard(padding: 12, radius: 14)
    }
}

private struct LessonProgressView: View {
    let progress: LessonProgress
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 4) {
                Text("\(progress.done)/\(progress.total)문장")
                    .foregroundStyle(DS.textSecondary)
                Spacer(minLength: 0)
                Text("\(Int((progress.ratio * 100).rounded()))%")
                    .foregroundStyle(tint)
            }
            .font(.system(size: 11, weight: .semibold))
            ProgressView(value: progress.ratio)
                .tint(tint)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("학습 진행률")
        .accessibilityValue("전체 \(progress.total)문장 중 \(progress.done)문장 연습, \(Int((progress.ratio * 100).rounded()))퍼센트")
    }
}
