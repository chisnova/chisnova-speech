import Foundation

/// 기기에서 콘솔 로그를 못 뽑는 상황이 많아, 진단 기록을 파일로도 남긴다.
/// Documents/diag.log 는 devicectl 로 꺼낼 수 있다.
enum Diag {
    private static let url: URL? = {
        (try? LocalDataStorage.documentsDirectory())?
            .appendingPathComponent("diag.log")
    }()

    private static let queue = DispatchQueue(label: "diag.log")

    static func log(_ message: String) {
        NSLog("[Mercury] %@", message)
        guard let url else { return }
        let line = "\(Date().formatted(date: .omitted, time: .standard))  \(message)\n"
        queue.async {
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(Data(line.utf8))
                try? handle.close()
            } else {
                try? Data(line.utf8).write(to: url)
            }
        }
    }
}
