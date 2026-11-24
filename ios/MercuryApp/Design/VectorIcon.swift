import SwiftUI

/// 안드로이드 VectorDrawable 의 pathData 를 그대로 그리는 도형.
///
/// 두 앱이 똑같은 아이콘을 쓰게 하려고 안드로이드 drawable 의 경로 데이터를 그대로 옮겼다.
/// VectorDrawable 의 pathData 는 SVG path 문법과 같아서 파싱만 하면 그릴 수 있다.
/// 벡터라 어떤 크기로 키워도 선명하고, 색은 SwiftUI 쪽에서 정한다.
struct VectorIconShape: Shape {
    let pathData: String
    /// 원본 좌표계 크기(viewportWidth/Height). 24 짜리와 960 짜리가 섞여 있다.
    let viewport: CGSize

    func path(in rect: CGRect) -> Path {
        var parser = SVGPathParser(pathData)
        let raw = parser.parse()
        guard viewport.width > 0, viewport.height > 0 else { return raw }

        // 가로세로 비율을 지켜 가운데에 맞춘다
        let scale = min(rect.width / viewport.width, rect.height / viewport.height)
        let dx = rect.minX + (rect.width - viewport.width * scale) / 2
        let dy = rect.minY + (rect.height - viewport.height * scale) / 2
        return raw.applying(CGAffineTransform(scaleX: scale, y: scale)
                                .concatenating(CGAffineTransform(translationX: dx, y: dy)))
    }
}

/// 이름으로 아이콘을 찾아 그린다. 없는 이름이면 아무것도 그리지 않는다.
struct VectorIcon: View {
    let name: String
    var size: CGFloat = 24

    var body: some View {
        if let icon = IconPaths.table[name] {
            VectorIconShape(pathData: icon.path,
                            viewport: CGSize(width: icon.width, height: icon.height))
                .frame(width: size, height: size)
        } else {
            // 매핑이 빠진 아이콘은 자리만 차지하게 둔다 — 레이아웃이 흔들리지 않게
            Color.clear.frame(width: size, height: size)
        }
    }
}

/// SVG path 문법 파서.
///
/// 안드로이드 아이콘이 쓰는 명령을 모두 다룬다: M m L l H h V v C c S s Q q T t A a Z z.
/// 좌표 구분자는 쉼표·공백·부호 어느 쪽이든 올 수 있고, 같은 명령이 연달아 나오면
/// 문자를 생략한다는 규칙(예: "l1,2 3,4")까지 따른다.
private struct SVGPathParser {
    private let scalars: [Character]
    private var index = 0

    private var path = Path()
    private var current = CGPoint.zero
    private var subpathStart = CGPoint.zero
    /// S/T 가 참조하는 직전 제어점
    private var lastCubicControl: CGPoint?
    private var lastQuadControl: CGPoint?

    init(_ data: String) {
        scalars = Array(data)
    }

    mutating func parse() -> Path {
        var command: Character = " "
        while true {
            skipSeparators()
            guard index < scalars.count else { break }

            if scalars[index].isLetter {
                command = scalars[index]
                index += 1
            } else if command == " " {
                break   // 명령 없이 숫자가 먼저 나오면 해석할 수 없다
            } else if command == "M" {
                command = "L"   // M 뒤의 좌표들은 L 로 이어진다(SVG 규칙)
            } else if command == "m" {
                command = "l"
            }

            guard apply(command) else { break }
        }
        return path
    }

    private mutating func apply(_ command: Character) -> Bool {
        let relative = command.isLowercase
        switch Character(command.uppercased()) {
        case "M":
            guard let p = point(relative: relative) else { return false }
            path.move(to: p)
            current = p
            subpathStart = p
            lastCubicControl = nil; lastQuadControl = nil

        case "L":
            guard let p = point(relative: relative) else { return false }
            path.addLine(to: p)
            current = p
            lastCubicControl = nil; lastQuadControl = nil

        case "H":
            guard let x = number() else { return false }
            let p = CGPoint(x: relative ? current.x + x : x, y: current.y)
            path.addLine(to: p)
            current = p
            lastCubicControl = nil; lastQuadControl = nil

        case "V":
            guard let y = number() else { return false }
            let p = CGPoint(x: current.x, y: relative ? current.y + y : y)
            path.addLine(to: p)
            current = p
            lastCubicControl = nil; lastQuadControl = nil

        case "C":
            guard let c1 = point(relative: relative),
                  let c2 = point(relative: relative),
                  let end = point(relative: relative) else { return false }
            path.addCurve(to: end, control1: c1, control2: c2)
            current = end
            lastCubicControl = c2; lastQuadControl = nil

        case "S":
            // 직전 제어점을 현재 점 기준으로 반사한 것이 첫 제어점이 된다
            guard let c2 = point(relative: relative),
                  let end = point(relative: relative) else { return false }
            let c1 = reflect(lastCubicControl)
            path.addCurve(to: end, control1: c1, control2: c2)
            current = end
            lastCubicControl = c2; lastQuadControl = nil

        case "Q":
            guard let c = point(relative: relative),
                  let end = point(relative: relative) else { return false }
            path.addQuadCurve(to: end, control: c)
            current = end
            lastQuadControl = c; lastCubicControl = nil

        case "T":
            guard let end = point(relative: relative) else { return false }
            let c = reflect(lastQuadControl)
            path.addQuadCurve(to: end, control: c)
            current = end
            lastQuadControl = c; lastCubicControl = nil

        case "A":
            guard let rx = number(), let ry = number(), let rotation = number(),
                  let largeArc = number(), let sweep = number(),
                  let end = point(relative: relative) else { return false }
            addArc(rx: rx, ry: ry, rotationDegrees: rotation,
                   largeArc: largeArc != 0, sweep: sweep != 0, to: end)
            current = end
            lastCubicControl = nil; lastQuadControl = nil

        case "Z":
            path.closeSubpath()
            current = subpathStart
            lastCubicControl = nil; lastQuadControl = nil

        default:
            return false
        }
        return true
    }

    /// 직전 제어점을 현재 점에 대해 반사한다. 없으면 현재 점 그대로.
    private func reflect(_ control: CGPoint?) -> CGPoint {
        guard let control else { return current }
        return CGPoint(x: 2 * current.x - control.x, y: 2 * current.y - control.y)
    }

    /// SVG 의 끝점 방식 호(arc)를 중심점 방식으로 바꿔 그린다(W3C 구현 노트).
    private mutating func addArc(rx: CGFloat, ry: CGFloat, rotationDegrees: CGFloat,
                                 largeArc: Bool, sweep: Bool, to end: CGPoint) {
        var rx = abs(rx), ry = abs(ry)
        // 반지름이 0이면 직선이다
        guard rx > 0, ry > 0, current != end else {
            path.addLine(to: end)
            return
        }

        let phi = rotationDegrees * .pi / 180
        let cosPhi = cos(phi), sinPhi = sin(phi)

        let dx2 = (current.x - end.x) / 2, dy2 = (current.y - end.y) / 2
        let x1 =  cosPhi * dx2 + sinPhi * dy2
        let y1 = -sinPhi * dx2 + cosPhi * dy2

        // 반지름이 너무 작으면 키운다
        let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
        if lambda > 1 {
            let scale = lambda.squareRoot()
            rx *= scale; ry *= scale
        }

        let sign: CGFloat = (largeArc != sweep) ? 1 : -1
        let numerator = max(0, rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1)
        let denominator = rx * rx * y1 * y1 + ry * ry * x1 * x1
        let coefficient = denominator == 0 ? 0 : sign * (numerator / denominator).squareRoot()

        let cx1 =  coefficient * rx * y1 / ry
        let cy1 = -coefficient * ry * x1 / rx

        let cx = cosPhi * cx1 - sinPhi * cy1 + (current.x + end.x) / 2
        let cy = sinPhi * cx1 + cosPhi * cy1 + (current.y + end.y) / 2

        func angle(_ ux: CGFloat, _ uy: CGFloat, _ vx: CGFloat, _ vy: CGFloat) -> CGFloat {
            let dot = ux * vx + uy * vy
            let len = (ux * ux + uy * uy).squareRoot() * (vx * vx + vy * vy).squareRoot()
            guard len > 0 else { return 0 }
            let value = max(-1, min(1, dot / len))
            let result = acos(value)
            return (ux * vy - uy * vx < 0) ? -result : result
        }

        let startAngle = angle(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
        var deltaAngle = angle((x1 - cx1) / rx, (y1 - cy1) / ry,
                               (-x1 - cx1) / rx, (-y1 - cy1) / ry)
        if !sweep, deltaAngle > 0 { deltaAngle -= 2 * .pi }
        if sweep, deltaAngle < 0 { deltaAngle += 2 * .pi }

        // 타원 호는 Path 에 직접 없으므로 변환을 씌운 원호로 그린다
        var transform = CGAffineTransform(translationX: cx, y: cy)
            .rotated(by: phi)
            .scaledBy(x: rx, y: ry)
        path.addArc(center: .zero, radius: 1,
                    startAngle: .radians(startAngle),
                    endAngle: .radians(startAngle + deltaAngle),
                    clockwise: deltaAngle < 0,
                    transform: transform)
    }

    // MARK: - 토큰 읽기

    private mutating func point(relative: Bool) -> CGPoint? {
        guard let x = number(), let y = number() else { return nil }
        return relative ? CGPoint(x: current.x + x, y: current.y + y) : CGPoint(x: x, y: y)
    }

    private mutating func skipSeparators() {
        while index < scalars.count, scalars[index] == " " || scalars[index] == ","
                || scalars[index] == "\n" || scalars[index] == "\t" || scalars[index] == "\r" {
            index += 1
        }
    }

    private mutating func number() -> CGFloat? {
        skipSeparators()
        var text = ""
        // 부호는 맨 앞에만
        if index < scalars.count, scalars[index] == "-" || scalars[index] == "+" {
            text.append(scalars[index]); index += 1
        }
        var seenDot = false
        while index < scalars.count {
            let c = scalars[index]
            if c.isNumber {
                text.append(c); index += 1
            } else if c == ".", !seenDot {
                seenDot = true
                text.append(c); index += 1
            } else if c == "e" || c == "E" {
                // 지수 표기 — 뒤따르는 부호까지 함께 먹는다
                text.append(c); index += 1
                if index < scalars.count, scalars[index] == "-" || scalars[index] == "+" {
                    text.append(scalars[index]); index += 1
                }
            } else {
                break
            }
        }
        guard let value = Double(text) else { return nil }
        return CGFloat(value)
    }
}
