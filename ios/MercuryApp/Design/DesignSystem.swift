import SwiftUI

/// 안드로이드 res/values/colors.xml 의 디자인 토큰을 그대로 옮긴 것.
/// 두 앱이 같은 화면으로 보이도록 값을 임의로 바꾸지 않는다.
enum DS {
    // 브랜드
    static let primary        = Color(hex: 0x2563EB)
    static let primaryLight   = Color(hex: 0xDBEAFE)
    static let primarySurface = Color(hex: 0xEFF6FF)
    static let primaryDark    = Color(hex: 0x1E40AF)

    // 강조
    static let accentOrange        = Color(hex: 0xF97316)
    static let accentOrangeLight   = Color(hex: 0xFFF7ED)
    static let accentOrangeSurface = Color(hex: 0xFFEDD5)
    static let accentGreen         = Color(hex: 0x10B981)
    static let accentGreenLight    = Color(hex: 0xECFDF5)
    static let accentGreenSurface  = Color(hex: 0xD1FAE5)
    static let accentPurple        = Color(hex: 0x8B5CF6)
    static let accentPurpleLight   = Color(hex: 0xF5F3FF)
    static let accentPurpleSurface = Color(hex: 0xEDE9FE)

    // 상태
    static let danger      = Color(hex: 0xEF4444)
    static let dangerLight = Color(hex: 0xFEF2F2)
    static let dangerDark  = Color(hex: 0xDC2626)
    static let warning     = Color(hex: 0xF59E0B)
    static let successDark = Color(hex: 0x047857)

    // 글자
    static let textPrimary   = Color(hex: 0x0F172A)
    static let textSecondary = Color(hex: 0x64748B)
    static let textTertiary  = Color(hex: 0x94A3B8)

    // 배경/면
    static let bg               = Color(hex: 0xF8FAFC)
    static let surface          = Color.white
    static let surfaceSecondary = Color(hex: 0xF1F5F9)
    static let border           = Color(hex: 0xE2E8F0)
    static let borderLight      = Color(hex: 0xF1F5F9)

    // 네비게이션
    static let navActive   = Color(hex: 0x2563EB)
    static let navInactive = Color(hex: 0x94A3B8)

    // MARK: - 안드로이드 bg_icon_* 대응
    //
    // LessonData 가 분류마다 배경색 drawable 을 지정한다. 같은 색을 쓰도록 이름으로 잇는다.
    static func iconTint(_ name: String) -> (icon: Color, surface: Color) {
        switch name {
        case "bg_icon_blue":   return (primary, primarySurface)
        case "bg_icon_green":  return (accentGreen, accentGreenSurface)
        case "bg_icon_orange": return (accentOrange, accentOrangeSurface)
        case "bg_icon_red":    return (danger, dangerLight)
        case "bg_icon_purple": return (accentPurple, accentPurpleSurface)
        default:               return (primary, primarySurface)
        }
    }

}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red:   Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue:  Double(hex & 0xFF) / 255,
            opacity: 1
        )
    }
}

/// 안드로이드의 카드(bg_*_card drawable)에 대응하는 표면 스타일.
struct CardBackground: ViewModifier {
    var padding: CGFloat = 16
    var radius: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(DS.surface)
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(DS.border, lineWidth: 1)
            )
    }
}

extension View {
    func dsCard(padding: CGFloat = 16, radius: CGFloat = 16) -> some View {
        modifier(CardBackground(padding: padding, radius: radius))
    }
}
