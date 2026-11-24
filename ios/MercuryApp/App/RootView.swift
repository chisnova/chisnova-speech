import SwiftUI

/// 하단 네비게이션 — 안드로이드 include_bottom.xml(홈·히스토리·분석) 대응.
///
/// 안드로이드는 BaseActivity 가 액티비티마다 같은 바를 끼워 넣는데,
/// 여기서는 탭 하나가 각자의 NavigationStack 을 갖는다. 그래야 주제 목록 안으로
/// 들어갔다가 탭을 옮겨도 자기 자리를 기억한다.
struct RootView: View {
    enum Tab: Hashable {
        case home, history, analysis
    }

    @State private var tab: Tab = .home

    var body: some View {
        VStack(spacing: 0) {
            // 탭을 바꿔도 각 화면의 상태가 살아 있도록 전부 만들어 두고 보이는 것만 띄운다
            ZStack {
                NavigationStack { HomeView() }
                    .opacity(tab == .home ? 1 : 0)
                    .allowsHitTesting(tab == .home)
                NavigationStack { HistoryView() }
                    .opacity(tab == .history ? 1 : 0)
                    .allowsHitTesting(tab == .history)
                NavigationStack { AnalysisView() }
                    .opacity(tab == .analysis ? 1 : 0)
                    .allowsHitTesting(tab == .analysis)
            }

            BottomNavBar(tab: $tab)
        }
        .background(DS.bg)
    }
}

/// 하단 바 — 안드로이드와 같은 세 칸(홈·히스토리·분석).
struct BottomNavBar: View {
    @Binding var tab: RootView.Tab

    var body: some View {
        HStack(spacing: 0) {
            item(.home, icon: "ic_home_24", title: "홈")
            item(.history, icon: "ic_calendar_24", title: "히스토리")
            item(.analysis, icon: "ic_bar_chart_24", title: "분석")
        }
        .padding(.top, 10)
        .padding(.bottom, 2)
        .background(
            DS.surface
                .overlay(Rectangle().frame(height: 1).foregroundStyle(DS.border),
                         alignment: .top)
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private func item(_ target: RootView.Tab, icon: String, title: String) -> some View {
        let active = tab == target
        return Button {
            tab = target
        } label: {
            VStack(spacing: 3) {
                VectorIcon(name: icon, size: 30)
                    .foregroundStyle(active ? DS.navActive : DS.navInactive)
                Text(title)
                    .font(.system(size: 12, weight: active ? .bold : .regular))
                    .foregroundStyle(active ? DS.navActive : DS.navInactive)
            }
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
