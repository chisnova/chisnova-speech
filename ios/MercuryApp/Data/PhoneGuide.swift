import Foundation

/// 음소별 발음 가이드 — contents/phone_info.json.
///
/// 안드로이드 VoiceRecognitionActivity 의 phoneInfoMap 에 대응한다. 점수가 낮은 음소를
/// 눌렀을 때 "어떻게 발음해야 하는지" 를 보여주는 데 쓴다. 12KB 라 한 번만 읽어 캐시한다.
enum PhoneGuide {

    struct Entry {
        let description: String
        let points: String
        let tips: String
    }

    private static let table: [String: Entry] = load()

    /// 음소 기호(대소문자 무시)로 가이드를 찾는다. 강세 숫자(AA1 등)는 떼고 본다.
    static func entry(for phone: String) -> Entry? {
        let key = phone.uppercased().trimmingCharacters(in: .whitespaces)
        if let hit = table[key] { return hit }
        // ARPAbet 은 모음에 강세 숫자가 붙는다 — AA1 -> AA
        let stripped = key.filter { !$0.isNumber }
        return table[stripped]
    }

    private static func load() -> [String: Entry] {
        guard let url = Bundle.main.resourceURL?
                .appendingPathComponent("Assets/contents/phone_info.json"),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        var table: [String: Entry] = [:]
        for (key, value) in object {
            guard let item = value as? [String: Any] else { continue }
            table[key.uppercased()] = Entry(
                description: (item["description"] as? String) ?? "",
                points: (item["points"] as? String) ?? "",
                tips: (item["tips"] as? String) ?? ""
            )
        }
        return table
    }
}
