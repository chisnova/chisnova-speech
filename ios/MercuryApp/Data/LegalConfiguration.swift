import Foundation

/// Both apps bundle the same repository-level legal/policies.json.
struct LegalConfiguration: Decodable {
    let operatorName: String
    let supportEmail: String
    let supportUrl: String
    let documents: [LegalDocument]

    static let shared: LegalConfiguration = {
        do {
            guard let url = Bundle.main.url(forResource: "policies", withExtension: "json", subdirectory: "legal") else {
                throw CocoaError(.fileNoSuchFile)
            }
            return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        } catch {
            NSLog("[Legal] Could not load policies: %@", error.localizedDescription)
            return Self(operatorName: "", supportEmail: "", supportUrl: "", documents: [])
        }
    }()

    func document(_ id: String) -> LegalDocument {
        documents.first { $0.id == id } ?? LegalDocument(
            id: id, title: id == "privacy" ? "개인정보 처리방침" : "이용약관",
            effectiveDate: "", url: "", sections: [])
    }

    static func webURL(_ value: String) -> URL? {
        guard let url = URL(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
              url.scheme?.lowercased() == "https", let host = url.host, !host.isEmpty,
              url.user == nil, url.password == nil else { return nil }
        return url
    }

    var emailURL: URL? {
        let email = supportEmail.trimmingCharacters(in: .whitespacesAndNewlines)
        guard email.contains("@"), !email.contains(where: { $0.isWhitespace }) else { return nil }
        var parts = URLComponents()
        parts.scheme = "mailto"
        parts.path = email
        parts.queryItems = [URLQueryItem(name: "subject", value: "쉐도잉 코치 문의")]
        return parts.url
    }
}

struct LegalDocument: Decodable, Identifiable {
    let id: String
    let title: String
    let effectiveDate: String
    let url: String
    let sections: [Section]

    struct Section: Decodable {
        let title: String
        let body: String
    }

    var webURL: URL? { LegalConfiguration.webURL(url) }
    var readableSections: [Section] { sections.filter { !$0.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } }
    var available: Bool { webURL != nil || !readableSections.isEmpty }
}
