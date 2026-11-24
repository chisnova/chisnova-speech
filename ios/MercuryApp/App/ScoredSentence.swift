import SwiftUI

/// 문장을 단어별 점수 색으로 칠하고, 단어를 누르면 상세를 띄운다.
///
/// 안드로이드 VoiceRecognitionActivity.colorSentenceByScores 와 같은 규칙을 쓴다.
/// 색 기준: 80점 이상 파랑, 50점 이상 주황, 그 아래 빨강, 점수 없으면 기본색.
/// 구두점은 칠하지 않는다 — 안드로이드도 단어만 span 을 건다.
struct ScoredSentence: View {
    let sentence: String
    let wordScores: [AsrSession.WordScore]
    let onSelect: (AsrSession.WordScore) -> Void

    var body: some View {
        // 조각마다 탭을 받아야 하므로 한 Text 로 뭉치지 않고 흘려 배치한다.
        // 조각 안에서는 단어와 구두점의 색을 AttributedString 으로 나눈다.
        FlowLayout(spacing: 0) {
            ForEach(tokens) { token in
                Text(token.text)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(token.color)
                    .onTapGesture {
                        if let score = token.score { onSelect(score) }
                    }
            }
        }
    }

    // MARK: - 토큰

    private struct Token: Identifiable {
        let id: Int
        let text: String
        let color: Color
        let score: AsrSession.WordScore?
    }

    /// 문장을 공백 단위로 끊어 각 조각에 점수를 붙인다.
    /// 조각은 "Hello," 처럼 구두점을 포함한 채로 두어야 줄 배치가 자연스럽다.
    ///
    /// 짝은 자리(순서)로 맞춘다. 채점 결과의 낱말 차례는 문장의 낱말 차례와 같다.
    /// 이름으로 찾으면 "5%," 나 "$32,000." 처럼 글자가 없는 낱말이 빈 열쇠가 되어
    /// 색이 안 칠해지고, 같은 낱말이 두 번 나오면 앞의 점수만 쓰게 된다.
    /// 다만 자리가 어긋난 경우를 대비해 이름이 맞는지 확인하고, 아니면 이름으로 찾는다.
    private var tokens: [Token] {
        let pieces = sentence.split(separator: " ", omittingEmptySubsequences: false)
        let byName = Dictionary(wordScores.map { (Self.normalize($0.word), $0) },
                                uniquingKeysWith: { first, _ in first })
        var result: [Token] = []

        for (index, piece) in pieces.enumerated() {
            let text = String(piece) + (index < pieces.count - 1 ? " " : "")
            let key = Self.normalize(String(piece))

            var score: AsrSession.WordScore?
            if index < wordScores.count {
                let candidate = wordScores[index]
                // 자리로 잡은 것이 이름까지 맞거나, 애초에 이름으로 찾을 수 없는 낱말이면 그대로 쓴다
                if key.isEmpty || Self.normalize(candidate.word) == key {
                    score = candidate
                }
            }
            if score == nil, !key.isEmpty {
                score = byName[key]
            }

            result.append(Token(id: index, text: text,
                                color: Self.color(for: score?.score), score: score))
        }
        return result
    }

    /// 문자와 어포스트로피만 남기고 소문자로 — "morning." 과 "morning" 이 같아지도록.
    private static func normalize(_ word: String) -> String {
        word.lowercased().filter { $0.isLetter || $0 == "'" }
    }

    private static func color(for score: Double?) -> Color {
        guard let score else { return DS.textPrimary }
        switch score {
        case 80...:   return DS.primary
        case 50..<80: return DS.warning
        default:      return DS.dangerDark
        }
    }
}
