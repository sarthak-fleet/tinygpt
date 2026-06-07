import Foundation

/// Lock file written by `tinygpt train` while a run is active.
/// Consumed by the Mac app's Train tab to attach to CLI-spawned runs
/// without requiring the user to remember which terminal they used.
///
/// Path: `~/.cache/tinygpt/runs/active.json`
public struct RunLockFile: Codable, Sendable, Equatable {
    public let pid: Int32
    public let logJsonlPath: String
    public let canonicalOutPath: String
    public let startedAt: String
    public let totalSteps: Int?

    public init(pid: Int32, logJsonlPath: String, canonicalOutPath: String,
                startedAt: String, totalSteps: Int? = nil) {
        self.pid = pid
        self.logJsonlPath = logJsonlPath
        self.canonicalOutPath = canonicalOutPath
        self.startedAt = startedAt
        self.totalSteps = totalSteps
    }

    public static var url: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/tinygpt/runs/active.json", isDirectory: false)
    }

    public static func write(_ lock: RunLockFile) throws {
        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try enc.encode(lock)
        try data.write(to: url, options: .atomic)
    }

    public static func read() -> RunLockFile? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(RunLockFile.self, from: data)
    }

    public static func clear() {
        try? FileManager.default.removeItem(at: url)
    }

    /// Stale when the PID is dead or the lock is older than 7 days.
    public static func isStale(_ lock: RunLockFile) -> Bool {
        if !processAlive(lock.pid) { return true }
        let fmt = ISO8601DateFormatter()
        guard let started = fmt.date(from: lock.startedAt) else { return true }
        return Date().timeIntervalSince(started) > 7 * 24 * 3600
    }

    private static func processAlive(_ pid: Int32) -> Bool {
        kill(pid, 0) == 0
    }
}
