import SwiftUI

/// 점수를 색으로 옮긴다. 카드 배지·단어 칩·음소 막대가 같은 기준을 쓴다.
enum ScoreTint {
    static func color(_ score: Double) -> Color {
        switch score {
        case 80...:   return DS.accentGreen
        case 60..<80: return DS.warning
        default:      return DS.danger
        }
    }

    static func label(_ score: Double) -> String {
        switch score {
        case 90...:   return "훌륭해요"
        case 80..<90: return "좋아요"
        case 60..<80: return "조금 더"
        default:      return "연습이 필요해요"
        }
    }
}

/// 중간 결과가 흐르는 동안 뛰는 점 세 개.
struct StreamingDots: View {
    @State private var phase = 0

    private let timer = Timer.publish(every: 0.35, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .frame(width: 3, height: 3)
                    .opacity(phase == index ? 1 : 0.25)
            }
        }
        .onReceive(timer) { _ in
            phase = (phase + 1) % 3
        }
    }
}

/// 단어별 점수 띠 — 안드로이드는 문장의 단어를 눌러 팝업을 띄운다.
/// 여기서는 채점된 단어를 칩으로 늘어놓고, 누르면 음소 점수를 펼친다.
struct WordScoreStrip: View {
    let words: [AsrSession.WordScore]
    let onSelect: (AsrSession.WordScore) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("단어별 점수 · 눌러서 자세히")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(DS.textTertiary)

            FlowLayout(spacing: 6) {
                ForEach(words) { word in
                    Button {
                        onSelect(word)
                    } label: {
                        HStack(spacing: 4) {
                            Text(word.word)
                                .font(.system(size: 13, weight: .semibold))
                            // 채점 제외 낱말은 점수 대신 대시를 보이고 회색으로 둔다
                            Text(word.graded ? "\(Int(word.score.rounded()))" : "—")
                                .font(.system(size: 11, weight: .bold))
                                .opacity(0.85)
                        }
                        .foregroundStyle(word.graded ? ScoreTint.color(word.score) : Color.gray)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background((word.graded ? ScoreTint.color(word.score) : Color.gray).opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

/// 단어 하나의 상세 — 안드로이드 wordInfoPopup 대응.
/// 단어 점수와 음소별 점수, 그리고 낮은 음소의 발음 가이드를 보여 준다.
struct WordScoreSheet: View {
    let word: AsrSession.WordScore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header

                    if word.phones.isEmpty {
                        Text("음소 점수가 없습니다.")
                            .font(.system(size: 13))
                            .foregroundStyle(DS.textTertiary)
                    } else {
                        phoneSection
                        guideSection
                    }
                }
                .padding(20)
            }
            .background(DS.bg)
            .navigationTitle("단어 정보")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(word.word)
                .font(.system(size: 30, weight: .bold))
                .foregroundStyle(DS.textPrimary)

            HStack(spacing: 10) {
                Text("\(Int(word.score.rounded()))점")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(ScoreTint.color(word.score))
                Text(ScoreTint.label(word.score))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(ScoreTint.color(word.score))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(ScoreTint.color(word.score).opacity(0.12))
                    .clipShape(Capsule())
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(DS.surface)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var phoneSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("음소별 점수")
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(DS.textPrimary)

            VStack(spacing: 8) {
                ForEach(word.phones) { phone in
                    HStack(spacing: 10) {
                        Text(phone.phone)
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                            .foregroundStyle(DS.textPrimary)
                            .frame(width: 46, alignment: .leading)

                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Capsule().fill(DS.surfaceSecondary)
                                Capsule()
                                    .fill(ScoreTint.color(phone.score))
                                    .frame(width: max(0, min(1, phone.score / 100)) * geometry.size.width)
                            }
                        }
                        .frame(height: 7)

                        Text("\(Int(phone.score.rounded()))")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(ScoreTint.color(phone.score))
                            .frame(width: 30, alignment: .trailing)
                    }
                }
            }
            .padding(14)
            .background(DS.surface)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
    }

    /// 점수가 낮은 음소부터 발음 가이드를 붙인다 — 안드로이드의 "발음 가이드" 카드.
    private var guideSection: some View {
        let weak = word.phones
            .filter { word.graded && $0.score < 80 }
            .sorted { $0.score < $1.score }
            .prefix(3)

        return Group {
            if weak.isEmpty {
                EmptyView()
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    Text("발음 가이드")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(DS.textPrimary)

                    ForEach(Array(weak)) { phone in
                        if let guide = PhoneGuide.entry(for: phone.phone) {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(spacing: 6) {
                                    Text(phone.phone)
                                        .font(.system(size: 13, weight: .bold, design: .monospaced))
                                    Text("\(Int(phone.score.rounded()))점")
                                        .font(.system(size: 11, weight: .bold))
                                        .foregroundStyle(ScoreTint.color(phone.score))
                                }
                                .foregroundStyle(DS.textPrimary)

                                if !guide.description.isEmpty {
                                    Text(guide.description)
                                        .font(.system(size: 13, weight: .semibold))
                                        .foregroundStyle(DS.textPrimary)
                                }
                                if !guide.points.isEmpty {
                                    guideRow(symbol: "target", text: guide.points)
                                }
                                if !guide.tips.isEmpty {
                                    guideRow(symbol: "lightbulb.fill", text: guide.tips)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(14)
                            .background(DS.surface)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                    }
                }
            }
        }
    }

    private func guideRow(symbol: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 10))
                .foregroundStyle(DS.accentOrange)
                .padding(.top, 3)
            Text(text)
                .font(.system(size: 12))
                .foregroundStyle(DS.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// 칩을 줄 바꿈하며 늘어놓는 배치. HStack 은 넘치면 잘리므로 직접 만든다.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var rowWidth: CGFloat = 0
        var rowHeight: CGFloat = 0
        var totalHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if rowWidth > 0, rowWidth + spacing + size.width > maxWidth {
                totalHeight += rowHeight + spacing
                rowWidth = size.width
                rowHeight = size.height
            } else {
                rowWidth += (rowWidth > 0 ? spacing : 0) + size.width
                rowHeight = max(rowHeight, size.height)
            }
        }
        return CGSize(width: maxWidth == .infinity ? rowWidth : maxWidth,
                      height: totalHeight + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize,
                       subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
