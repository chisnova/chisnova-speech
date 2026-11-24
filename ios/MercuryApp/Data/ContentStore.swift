import Foundation

/// 번들에 실린 학습 콘텐츠를 읽는다.
///
/// 안드로이드 AssetContentLoader 와 같은 규칙을 쓴다:
///   contents/<레벨>/<주제>/<주제>.eng   영어 문장 (줄 단위)
///                          .kor   한국어 번역 (줄 단위, 같은 순서)
///                          .jsonl 발음 채점용 인덱스 (줄 = 문장)
enum ContentLevel: String, CaseIterable, Identifiable {
    /// 폴더 번호가 곧 레벨 번호다. 난이도별로 폴더가 나뉘어 있어 훑기만 하면 된다.
    case basic     = "1.basic"
    case article1  = "2.beginner"
    case article2  = "3.intermediate"
    case article3  = "4.advanced"
    case speech    = "5.speech"

    var id: String { rawValue }

    /// 실제 콘텐츠가 들어 있는 폴더 이름.
    var folderName: String { rawValue }

    var title: String {
        switch self {
        case .basic:    return "빈출표현 정리"
        case .article1: return "주제별 콘텐츠 · 입문"
        case .article2: return "주제별 콘텐츠 · 중급"
        case .article3: return "주제별 콘텐츠 · 심화"
        case .speech:   return "연설문"
        }
    }

    var levelLabel: String {
        switch self {
        case .basic:    return "Level 1"
        case .article1: return "Level 2"
        case .article2: return "Level 3"
        case .article3: return "Level 4"
        case .speech:   return "Level 5"
        }
    }

    /// 이 레벨의 전체 문장 수. 진행률을 그리는 데 쓴다.
    ///
    /// 매번 파일을 세면 홈 화면이 느려진다(69개 주제 × 파일 읽기). 콘텐츠가 바뀔 때만
    /// 달라지는 값이라 미리 세어 둔다.
    var totalSentences: Int {
        switch self {
        case .basic:    return 130
        case .article1: return 1138
        case .article2: return 1003
        case .article3: return 973
        case .speech:   return 972
        }
    }

    /// 홈 카드에 한 줄로 붙는 설명
    var cardCaption: String {
        switch self {
        case .basic:    return "자주 쓰는 표현을 구 단위로 연습"
        case .article1: return "스포츠·음악·문화 등 친숙한 주제"
        case .article2: return "과학·의학·경제 · 어휘 넓히기"
        case .article3: return "사회·정치·경제 이론 · 추상 표현"
        case .speech:   return "연속 발화 · 억양과 전달력까지"
        }
    }

    var subtitle: String {
        switch self {
        case .basic:    return "Level 1 · 영어 발음 기초"
        case .article1: return "Level 2 · 친숙한 주제로 시작하기"
        case .article2: return "Level 3 · 과학·경제로 넓히기"
        case .article3: return "Level 4 · 추상 개념 다루기"
        case .speech:   return "Level 5 · 연설문 따라 읽기"
        }
    }
}

/// 한 문장. 영어/번역과 채점용 인덱스를 함께 들고 다닌다.
struct Sentence: Identifiable {
    let id: Int
    let english: String
    let korean: String
    let meta: PronounceMeta?
}

/// .jsonl 한 줄. 라벨과 인덱스는 반드시 같은 줄에서 나온 것을 함께 써야 한다 —
/// 라벨 단어 수보다 큰 group 인덱스를 넘기면 네이티브가 배열 밖을 읽는다.
struct PronounceMeta {
    let label: String
    let wordIdx: [Int]
    let wordIdxGroup: [Int]
    let phnIdx: [Int]
    let phnIdxGroup: [Int]
    /// 문장별 최종 점수 비중(발음, 인식). 콘텐츠 .jsonl 의 score_weights — 대부분 없다.
    let scoreWeights: (pronunciation: Float, recognition: Float)?
}

/// 한 주제(레슨).
///
/// 레벨마다 폴더 깊이가 다르다:
///   1.basic/<주제>/<주제>.eng
///   2.beginner/<분류>/<주제>/<주제>.eng   ← 난이도 세 단계는 분류가 한 겹 더 있다
///   5.speech/<주제>/<주제>.eng
/// 그래서 폴더 이름만으로는 경로를 만들 수 없어 상대 경로를 따로 들고 다닌다.
struct Lesson: Identifiable {
    /// 목록에서 고유한 열쇠. 폴더와 다를 수 있다 — 빈출표현 정리에는 한 폴더를
    /// 두 항목이 가리키는 곳이 있어(감정 표현하기/의견 표현) 폴더를 id 로 쓰면 겹친다.
    let id: String
    let title: String
    let category: String?     // 분류 id(culture, economics, basic …)
    let level: ContentLevel
    /// 실제 콘텐츠 폴더 이름. 그 안의 파일 이름이기도 하다.
    let folder: String
    let relativePath: String  // contents 기준, 예: "2.beginner/culture/art"
    /// 안드로이드 drawable 이름. 없으면 화면이 이름으로 짐작한다.
    let icon: String?
    /// 연설문에만 있는 부제·사진(안드로이드 activity_level3 의 카드)
    let subtitle: String?
    let imageName: String?

    var basePath: String { "Assets/contents/\(relativePath)/\(folder)" }

    init(id: String, title: String, category: String?, level: ContentLevel,
         folder: String, relativePath: String, icon: String? = nil,
         subtitle: String? = nil, imageName: String? = nil) {
        self.id = id
        self.title = title
        self.category = category
        self.level = level
        self.folder = folder
        self.relativePath = relativePath
        self.icon = icon
        self.subtitle = subtitle
        self.imageName = imageName
    }
}

struct SpeechSource: Decodable {
    let author: String
    let title: String
    let date: String
    let sourceName: String
    let sourceURL: URL
    let notice: String
}

enum ContentStore {
    static func speechSource(folder: String) -> SpeechSource? {
        let base = "Assets/contents/5.speech/\(folder)/\(folder)"
        guard let sourceURL = url(base, ext: "metadata.json"),
              let data = try? Data(contentsOf: sourceURL) else { return nil }
        return try? JSONDecoder().decode(SpeechSource.self, from: data)
    }


    /// 레벨 아래의 주제 목록. 분류가 한 겹 더 있는 레벨도 함께 훑는다.
    static func lessons(in level: ContentLevel) -> [Lesson] {
        guard let root = Bundle.main.resourceURL?
                .appendingPathComponent("Assets/contents/\(level.folderName)") else {
            return []
        }

        // 빈출표현 정리는 폴더를 훑지 않는다. 안드로이드 화면과 목록·순서·이름을 맞추기
        // 위해 카탈로그에 적힌 대로만 내보낸다(같은 폴더를 두 항목이 가리키는 곳이 있다).
        if level == .basic {
            return ContentCatalog.basicTopics.compactMap { topic in
                let folderURL = root.appendingPathComponent(topic.folder)
                guard hasContent(at: folderURL, name: topic.folder) else { return nil }
                return Lesson(id: topic.id,
                              title: topic.title,
                              category: topic.category,
                              level: level,
                              folder: topic.folder,
                              relativePath: "\(level.folderName)/\(topic.folder)",
                              icon: topic.icon)
            }
        }

        var found: [Lesson] = []
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: root.path))?.sorted() ?? []

        for entry in entries where !entry.hasPrefix(".") {
            let entryURL = root.appendingPathComponent(entry)

            // 1단계: <주제>/<주제>.eng
            if hasContent(at: entryURL, name: entry) {
                let speech = ContentCatalog.speeches.first { $0.folder == entry }
                found.append(Lesson(id: entry,
                                    title: speech?.title ?? prettify(entry),
                                    category: nil,
                                    level: level,
                                    folder: entry,
                                    relativePath: "\(level.folderName)/\(entry)",
                                    icon: ContentCatalog.lessonIcons[entry],
                                    subtitle: speech?.subtitle,
                                    imageName: speech?.image))
                continue
            }

            // 2단계: <분류>/<주제>/<주제>.eng
            let inner = (try? FileManager.default.contentsOfDirectory(atPath: entryURL.path))?.sorted() ?? []
            for sub in inner where !sub.hasPrefix(".")
                && hasContent(at: entryURL.appendingPathComponent(sub), name: sub) {
                found.append(Lesson(id: sub,
                                    title: ContentCatalog.articleLessons[sub] ?? prettify(sub),
                                    category: entry,
                                    level: level,
                                    folder: sub,
                                    relativePath: "\(level.folderName)/\(entry)/\(sub)",
                                    icon: ContentCatalog.lessonIcons[sub]))
            }
        }
        return found
    }

    /// .eng 가 있어야 학습 대상으로 본다(opus 같은 보조 폴더 제외).
    private static func hasContent(at folder: URL, name: String) -> Bool {
        FileManager.default.fileExists(
            atPath: folder.appendingPathComponent("\(name).eng").path)
    }

    /// 레슨의 문장들. .eng/.kor 를 줄 단위로 짝지어 읽고 .jsonl 이 있으면 채점 인덱스를 붙인다.
    static func sentences(of lesson: Lesson) -> [Sentence] {
        let english = englishSentences(of: lesson)
        let korean  = lines(lesson.basePath, ext: "kor")
        let metas   = pronounceMetas(lesson.basePath)

        return english.enumerated().map { index, text in
            Sentence(
                id: index + 1,
                english: text,
                korean: index < korean.count ? korean[index] : "",
                meta: index < metas.count ? metas[index] : nil
            )
        }
    }

    /// 목록의 진도 계산에는 채점 인덱스와 번역을 읽을 필요가 없다.
    static func englishSentences(of lesson: Lesson) -> [String] {
        lines(lesson.basePath, ext: "eng")
    }

    // MARK: - 내부

    private static func prettify(_ folder: String) -> String {
        folder.replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }

    private static func url(_ basePath: String, ext: String) -> URL? {
        Bundle.main.resourceURL?.appendingPathComponent("\(basePath).\(ext)")
    }

    private static func lines(_ basePath: String, ext: String) -> [String] {
        guard let url = url(basePath, ext: ext),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func pronounceMetas(_ basePath: String) -> [PronounceMeta] {
        guard let url = url(basePath, ext: "jsonl"),
              let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }

        return text.split(separator: "\n").compactMap { line in
            guard let data = line.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }

            func ints(_ key: String) -> [Int] { (object[key] as? [Int]) ?? [] }
            let org = (object["org"] as? [String]) ?? []
            guard !org.isEmpty else { return nil }

            var weights: (pronunciation: Float, recognition: Float)?
            if let w = object["score_weights"] as? [String: Any],
               let pronunciation = (w["pronunciation"] as? NSNumber)?.floatValue,
               let recognition = (w["recognition"] as? NSNumber)?.floatValue {
                weights = (pronunciation, recognition)
            }

            return PronounceMeta(
                label: org.joined(separator: " "),
                wordIdx: ints("word_idx"),
                wordIdxGroup: ints("word_idx_group"),
                phnIdx: ints("phn_idx"),
                phnIdxGroup: ints("phn_idx_group"),
                scoreWeights: weights
            )
        }
    }
}
