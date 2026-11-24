import Foundation

/// Keep learning history, recordings and diagnostic files out of OS backups.
enum LocalDataStorage {
    static func documentsDirectory() throws -> URL {
        let directory = try FileManager.default.url(for: .documentDirectory,
            in: .userDomainMask, appropriateFor: nil, create: true)
        return try excludeFromBackup(directory)
    }

    @discardableResult
    static func excludeFromBackup(_ directory: URL) throws -> URL {
        var directory = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try directory.setResourceValues(values)
        return directory
    }
}
