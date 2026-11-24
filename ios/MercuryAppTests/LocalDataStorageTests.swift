import XCTest
@testable import MercuryApp

final class LocalDataStorageTests: XCTestCase {
    func testBackupExclusionPreservesExistingDataAndAtomicReplacements() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let history = directory.appendingPathComponent("attempts.json")
        let original = Data("existing history".utf8)
        try original.write(to: history)

        try LocalDataStorage.excludeFromBackup(directory)
        XCTAssertEqual(try Data(contentsOf: history), original)
        XCTAssertEqual(try directory.resourceValues(forKeys: [.isExcludedFromBackupKey])
            .isExcludedFromBackup, true)

        let updated = Data("updated history".utf8)
        try updated.write(to: history, options: .atomic)
        try LocalDataStorage.excludeFromBackup(directory)
        XCTAssertEqual(try Data(contentsOf: history), updated)
        XCTAssertEqual(try directory.resourceValues(forKeys: [.isExcludedFromBackupKey])
            .isExcludedFromBackup, true)
    }
}
