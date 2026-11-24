import SwiftUI

/// 발음 유형별 점수를 방사형으로 보여 준다 — 안드로이드 GraphActivity.RadarChartView 대응.
///
/// 축 개수는 데이터에 따라 정해진다(발음 유형 8가지면 팔각형). 안드로이드와 같은 규칙을 따른다:
/// 12시 방향에서 시작해 시계 방향, 배경은 동심원 네 겹(25·50·75·100점).
struct RadarChart: View {
    /// (유형 이름, 0~100 점수) — 순서가 곧 축 순서다
    let values: [(label: String, score: Double)]

    /// 라벨이 들어갈 자리를 남겨 두고 그린다
    private let labelInset: CGFloat = 34

    var body: some View {
        GeometryReader { geometry in
            let side = min(geometry.size.width, geometry.size.height)
            let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
            let radius = max(10, side / 2 - labelInset)

            ZStack {
                grid(center: center, radius: radius)

                if values.count >= 3 {
                    // 채운 영역
                    dataPath(center: center, radius: radius)
                        .fill(DS.primary.opacity(0.22))
                    // 테두리
                    dataPath(center: center, radius: radius)
                        .stroke(DS.primary, style: StrokeStyle(lineWidth: 2, lineJoin: .round))
                    // 꼭짓점
                    ForEach(Array(values.enumerated()), id: \.offset) { index, item in
                        let p = point(index: index, ratio: item.score / 100,
                                      center: center, radius: radius)
                        Circle()
                            .fill(DS.primary)
                            .frame(width: 7, height: 7)
                            .position(p)
                    }
                }

                labels(center: center, radius: radius)
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    // MARK: - 조각

    /// 동심원 네 겹 + 축 선
    private func grid(center: CGPoint, radius: CGFloat) -> some View {
        ZStack {
            ForEach(1...4, id: \.self) { step in
                Circle()
                    .stroke(DS.border, lineWidth: 1)
                    .frame(width: radius * 2 * CGFloat(step) / 4,
                           height: radius * 2 * CGFloat(step) / 4)
                    .position(center)
            }
            Path { path in
                for index in values.indices {
                    path.move(to: center)
                    path.addLine(to: point(index: index, ratio: 1, center: center, radius: radius))
                }
            }
            .stroke(DS.border, lineWidth: 1)
        }
    }

    private func dataPath(center: CGPoint, radius: CGFloat) -> Path {
        Path { path in
            for (index, item) in values.enumerated() {
                // 0점이어도 가운데에 딱 붙지 않게 아주 작은 여유를 준다
                let ratio = max(0.02, min(1, item.score / 100))
                let p = point(index: index, ratio: ratio, center: center, radius: radius)
                if index == 0 { path.move(to: p) } else { path.addLine(to: p) }
            }
            path.closeSubpath()
        }
    }

    private func labels(center: CGPoint, radius: CGFloat) -> some View {
        ForEach(Array(values.enumerated()), id: \.offset) { index, item in
            let p = point(index: index, ratio: 1, center: center,
                          radius: radius + labelInset * 0.62)
            VStack(spacing: 1) {
                Text(item.label)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(DS.textSecondary)
                Text("\(Int(item.score.rounded()))")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(ScoreTint.color(item.score))
            }
            .position(p)
        }
    }

    /// 12시 방향에서 시작해 시계 방향으로 index 번째 축의 좌표.
    private func point(index: Int, ratio: Double, center: CGPoint, radius: CGFloat) -> CGPoint {
        guard !values.isEmpty else { return center }
        let angle = 2 * Double.pi * Double(index) / Double(values.count) - Double.pi / 2
        return CGPoint(x: center.x + radius * ratio * cos(angle),
                       y: center.y + radius * ratio * sin(angle))
    }
}
