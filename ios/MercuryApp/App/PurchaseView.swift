import SwiftUI

struct PurchaseView: View {
    @StateObject private var purchases = PurchaseStore.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 12) {
                    Image(systemName: purchases.hasFullAccess ? "checkmark.seal.fill" : "book.fill")
                        .font(.system(size: 24)).foregroundStyle(DS.primary)
                        .frame(width: 48, height: 48).background(DS.primarySurface)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    Text(purchases.hasFullAccess ? "전체 콘텐츠 이용 중" : "한 번 구매로\n전체 콘텐츠 이용")
                        .font(.system(size: 22, weight: .bold)).foregroundStyle(DS.textPrimary)
                    Text("원하는 주제를 골라 나만의 속도로 연습하세요.")
                        .font(.system(size: 14)).foregroundStyle(DS.textSecondary)
                    Text("1회 구매 · 기간 제한 없음 · 자동 갱신 없음")
                        .font(.system(size: 12, weight: .semibold)).foregroundStyle(DS.primary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .dsCard(padding: 20, radius: 16)

                VStack(alignment: .leading, spacing: 16) {
                    Text("이용 가능한 콘텐츠").font(.system(size: 16, weight: .bold))
                    VStack(alignment: .leading, spacing: 6) {
                        Text("무료 · Level 1 기초").font(.system(size: 13, weight: .semibold)).foregroundStyle(DS.primary)
                        Text("인사말과 기본 표현 · 일상 대화 표현\n감정 표현하기 · 시간과 날짜 표현")
                            .font(.system(size: 14)).foregroundStyle(DS.textSecondary)
                    }
                    Divider()
                    VStack(alignment: .leading, spacing: 6) {
                        Text("전체 이용권 · Level 1~5").font(.system(size: 13, weight: .semibold))
                        Text("Level 1의 실생활·소통·중요 주제\nLevel 2~4의 주제별 콘텐츠\nLevel 5의 연설문")
                            .font(.system(size: 14)).foregroundStyle(DS.textSecondary)
                    }
                }
                .foregroundStyle(DS.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .dsCard(padding: 20, radius: 16)

                VStack(alignment: .leading, spacing: 12) {
                    Text("전체 이용권").font(.system(size: 16, weight: .bold)).foregroundStyle(DS.textPrimary)
                    Text("현재 기본 패키지의 모든 콘텐츠").font(.system(size: 12)).foregroundStyle(DS.textSecondary)
                    if !purchases.hasFullAccess {
                        Button { Task { await purchases.purchase() } } label: {
                            Text(purchases.product.map { "\($0.displayPrice) · 전체 이용권 구매" }
                                 ?? (purchases.isBusy ? "상품 확인 중" : "상품 확인 필요"))
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(.white).frame(maxWidth: .infinity, minHeight: 52)
                                .background(purchases.product == nil || purchases.isBusy ? DS.textTertiary : DS.primary)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }.buttonStyle(.plain)
                            .disabled(purchases.product == nil || purchases.isBusy)
                        if purchases.product == nil && !purchases.isBusy {
                            Button("상품 다시 불러오기") { Task { await purchases.loadProduct() } }
                                .frame(maxWidth: .infinity, minHeight: 44)
                        }
                    }
                    Button("구매 복원") { Task { await purchases.restore() } }
                        .frame(maxWidth: .infinity, minHeight: 44).disabled(purchases.isBusy)
                    if purchases.isBusy { ProgressView().accessibilityLabel("구매 정보 확인 중") }
                    if let message = purchases.message {
                        Text(message).font(.system(size: 13)).foregroundStyle(DS.textSecondary)
                    }
                }
                .tint(DS.primary).frame(maxWidth: .infinity, alignment: .leading)
                .dsCard(padding: 20, radius: 16)
                Text("구매한 Apple 계정으로 복원할 수 있습니다.\n향후 별도 출시되는 확장 상품은 포함되지 않습니다.")
                    .font(.system(size: 12)).foregroundStyle(DS.textSecondary)
                PurchaseLegalLinks()
            }
            .padding(16)
        }
        .background(DS.bg)
        .navigationTitle("전체 이용권")
        .navigationBarTitleDisplayMode(.inline)
        .task { await purchases.loadProduct() }
    }
}

struct PurchaseBanner: View {
    @StateObject private var purchases = PurchaseStore.shared
    var body: some View {
        NavigationLink {
            PurchaseView()
        } label: {
            HStack {
                Image(systemName: purchases.hasFullAccess ? "checkmark.seal.fill" : "lock.open")
                VStack(alignment: .leading, spacing: 3) {
                    Text(purchases.hasFullAccess ? "전체 콘텐츠 이용 중" : "전체 이용권")
                        .font(.system(size: 15, weight: .bold))
                    Text(purchases.hasFullAccess ? "구매 내역 확인 및 복원" : "기초 네 주제 무료 · 한 번 구매로 전체 이용")
                        .font(.system(size: 12))
                }
                Spacer()
                Image(systemName: "chevron.right")
            }
            .foregroundStyle(DS.primary).dsCard(padding: 16, radius: 14)
        }.buttonStyle(.plain)
    }
}
