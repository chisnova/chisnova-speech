import SwiftUI

struct SettingsView: View {
    private let configuration = LegalConfiguration.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("서비스 안내").font(.system(size: 20, weight: .bold)).foregroundStyle(DS.textPrimary)
                    Text("이용 조건과 개인정보 보호 안내를 확인하세요.")
                        .font(.system(size: 14)).foregroundStyle(DS.textSecondary)
                }.padding(.vertical, 8)
                VStack(spacing: 0) {
                    ForEach(["terms", "privacy"], id: \.self) { id in
                        let document = configuration.document(id)
                        NavigationLink { LegalDocumentView(document: document) } label: {
                            LegalRow(title: document.title, subtitle: document.available ? "내용 확인" : "준비 중")
                        }
                        Divider()
                    }
                    NavigationLink { SupportView() } label: {
                        LegalRow(title: "문의하기", subtitle: "서비스 이용 및 구매 문의")
                    }
                }.buttonStyle(.plain).dsCard(padding: 16, radius: 16)
                if !configuration.operatorName.isEmpty {
                    Text("운영자 · \(configuration.operatorName)").font(.system(size: 13)).foregroundStyle(DS.textSecondary)
                }
                Text("쉐도잉 코치 · 버전 \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")")
                    .font(.system(size: 12)).foregroundStyle(DS.textSecondary)
            }.padding(16)
        }
        .background(DS.bg).navigationTitle("설정 및 안내")
        .navigationBarTitleDisplayMode(.inline).toolbar(.visible, for: .navigationBar)
    }
}

private struct LegalRow: View {
    let title: String
    let subtitle: String
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title).font(.system(size: 15, weight: .semibold)).foregroundStyle(DS.textPrimary)
                Text(subtitle).font(.system(size: 12)).foregroundStyle(DS.textSecondary)
            }
            Spacer()
            Image(systemName: "chevron.right").font(.system(size: 12, weight: .semibold)).foregroundStyle(DS.textTertiary)
        }
        .frame(minHeight: 52).padding(.vertical, 8).contentShape(Rectangle())
    }
}

struct LegalDocumentView: View {
    let document: LegalDocument
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if !document.effectiveDate.isEmpty && document.available {
                    Text("시행일 · \(document.effectiveDate)").font(.system(size: 13)).foregroundStyle(DS.textSecondary)
                }
                if document.readableSections.isEmpty {
                    Text(document.webURL == nil
                         ? "안내 내용을 준비하고 있습니다. 준비되는 대로 이 화면에서 확인하실 수 있습니다."
                         : "아래 웹페이지에서 내용을 확인하실 수 있습니다.")
                        .font(.system(size: 15)).foregroundStyle(DS.textSecondary)
                        .frame(maxWidth: .infinity, alignment: .leading).dsCard(padding: 20)
                } else {
                    VStack(alignment: .leading, spacing: 24) {
                        ForEach(Array(document.readableSections.enumerated()), id: \.offset) { _, section in
                            VStack(alignment: .leading, spacing: 10) {
                                if !section.title.isEmpty {
                                    Text(section.title).font(.system(size: 17, weight: .semibold)).accessibilityAddTraits(.isHeader)
                                }
                                Text(section.body).font(.system(size: 15)).lineSpacing(6)
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }.foregroundStyle(DS.textPrimary).textSelection(.enabled).dsCard(padding: 20)
                }
                if let url = document.webURL { LegalExternalButton(title: "웹에서 보기", url: url) }
            }.padding(16)
        }
        .background(DS.bg).navigationTitle(document.title)
        .navigationBarTitleDisplayMode(.inline).toolbar(.visible, for: .navigationBar)
    }
}

private struct SupportView: View {
    private let configuration = LegalConfiguration.shared
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("서비스 이용 및 구매 문의").font(.system(size: 18, weight: .bold)).foregroundStyle(DS.textPrimary)
                if configuration.emailURL == nil && LegalConfiguration.webURL(configuration.supportUrl) == nil {
                    Text("문의 안내를 준비하고 있습니다.").foregroundStyle(DS.textSecondary)
                }
                if let email = configuration.emailURL {
                    Text("문의 메일은 Gmail을 통해 운영자에게 전달됩니다. 앱이 녹음이나 학습 기록을 자동 첨부하지 않습니다. 불필요한 개인정보는 보내지 말아 주세요.")
                        .font(.system(size: 14)).foregroundStyle(DS.textSecondary)
                    Text(configuration.supportEmail).foregroundStyle(DS.textSecondary).textSelection(.enabled)
                    LegalExternalButton(title: "이메일로 문의", url: email)
                }
                if let url = LegalConfiguration.webURL(configuration.supportUrl) {
                    LegalExternalButton(title: "문의 페이지 열기", url: url)
                }
            }.frame(maxWidth: .infinity, alignment: .leading).dsCard(padding: 20).padding(16)
        }
        .background(DS.bg).navigationTitle("문의하기")
        .navigationBarTitleDisplayMode(.inline).toolbar(.visible, for: .navigationBar)
    }
}

private struct LegalExternalButton: View {
    let title: String
    let url: URL
    @Environment(\.openURL) private var openURL
    @State private var openFailed = false
    var body: some View {
        Button { openURL(url) { accepted in openFailed = !accepted } } label: {
            Label(title, systemImage: "arrow.up.right.square")
                .font(.system(size: 14, weight: .semibold)).frame(minHeight: 44)
        }
        .tint(DS.primary)
        .alert("열 수 없습니다", isPresented: $openFailed) {
            Button("확인", role: .cancel) {}
        } message: {
            Text(url.scheme == "mailto" ? "메일 앱을 확인하거나 표시된 이메일 주소를 복사해 이용해 주세요." : "브라우저와 인터넷 연결을 확인해 주세요.")
        }
    }
}

struct PurchaseLegalLinks: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(["terms", "privacy"], id: \.self) { id in
                let document = LegalConfiguration.shared.document(id)
                NavigationLink { LegalDocumentView(document: document) } label: {
                    HStack {
                        Text(document.title)
                        Spacer()
                        Image(systemName: "chevron.right").font(.system(size: 11))
                    }.frame(minHeight: 44).contentShape(Rectangle())
                }
            }
            NavigationLink { SupportView() } label: {
                Text("구매 관련 문의").frame(minHeight: 44)
            }
        }.font(.system(size: 13)).foregroundStyle(DS.primary).buttonStyle(.plain)
    }
}
