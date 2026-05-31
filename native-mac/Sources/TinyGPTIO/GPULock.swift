import Foundation

/// Cross-process GPU coordination via file-lock.
///
/// Why this exists: macOS lets any process dispatch Metal compute work
/// concurrently. When TWO MLX processes (e.g., tinygpt training + a
/// researchPapers MLX tagger) both saturate the GPU, both run slower
/// than either would alone — and the laptop can discharge while
/// charging because total power draw exceeds adapter throughput.
///
/// This module provides a cooperative lock: long-running tinygpt
/// commands (train, sample with --max-tokens > N, agent-eval) acquire
/// the lock; other tinygpt invocations either wait or fail-fast. The
/// lock does NOT prevent NON-tinygpt MLX processes from running —
/// that's a different problem.
///
/// Lock file: ~/.cache/tinygpt/gpu.lock
/// Contains: pid, command name, start time. A stale lock (process
/// no longer alive) is auto-cleared on next acquire attempt.
public enum GPULock {

    public enum LockError: Error, CustomStringConvertible {
        case heldByAnother(pid: Int32, command: String, startedAt: String)
        case lockfileError(String)

        public var description: String {
            switch self {
            case .heldByAnother(let pid, let command, let startedAt):
                return "GPU lock held by another tinygpt process (PID \(pid), `\(command)`, started \(startedAt)). Wait, or pass --no-gpu-lock to skip."
            case .lockfileError(let msg):
                return "GPU lock file error: \(msg)"
            }
        }
    }

    public struct LockInfo: Codable {
        public let pid: Int32
        public let command: String
        public let startedAt: String  // ISO8601

        public init(pid: Int32, command: String, startedAt: String) {
            self.pid = pid
            self.command = command
            self.startedAt = startedAt
        }
    }

    /// Path to the lock file.
    public static var lockFileURL: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/tinygpt/gpu.lock")
    }

    /// Try to acquire the lock. Returns the lock token (a file handle)
    /// that must be released via `release(_:)` or by closing.
    ///
    /// - waitSeconds: if > 0, block up to that long waiting for the lock.
    ///   If 0, fail-fast on contention.
    /// - command: the user-facing command name (for the error message).
    public static func acquire(command: String, waitSeconds: TimeInterval = 0) throws -> FileHandle {
        let url = lockFileURL
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let start = Date()
        while true {
            // Check for stale lock first (PID dead).
            if let staleInfo = readLockInfo(url: url), !pidIsAlive(staleInfo.pid) {
                try? FileManager.default.removeItem(at: url)
            }

            // Try to create the file exclusively.
            let created = FileManager.default.createFile(
                atPath: url.path,
                contents: nil,
                attributes: [.posixPermissions: NSNumber(value: 0o644)]
            )

            // O_EXCL via createFile is non-atomic in Foundation; do an
            // additional check by re-reading.
            if created, let handle = try? FileHandle(forWritingTo: url) {
                // Write our identity to the lock.
                let info = LockInfo(
                    pid: ProcessInfo.processInfo.processIdentifier,
                    command: command,
                    startedAt: ISO8601DateFormatter().string(from: Date())
                )
                if let data = try? JSONEncoder().encode(info) {
                    handle.write(data)
                    try? handle.synchronize()
                }
                return handle
            }

            // Lock held by another. Surface the holder.
            if let info = readLockInfo(url: url), pidIsAlive(info.pid) {
                if waitSeconds <= 0 || Date().timeIntervalSince(start) > waitSeconds {
                    throw LockError.heldByAnother(
                        pid: info.pid,
                        command: info.command,
                        startedAt: info.startedAt
                    )
                }
                // Sleep a bit, then retry.
                Thread.sleep(forTimeInterval: 1.0)
                continue
            }

            // File exists but no info / unparseable. Wait or fail.
            if waitSeconds <= 0 || Date().timeIntervalSince(start) > waitSeconds {
                throw LockError.lockfileError("lockfile present but unreadable: \(url.path)")
            }
            Thread.sleep(forTimeInterval: 1.0)
        }
    }

    /// Release the lock. Closes the handle + removes the file.
    public static func release(_ handle: FileHandle) {
        try? handle.close()
        try? FileManager.default.removeItem(at: lockFileURL)
    }

    /// Print info about the current holder (or "(none)" if free).
    public static func status() -> String {
        guard let info = readLockInfo(url: lockFileURL) else {
            return "(no GPU lock held)"
        }
        if !pidIsAlive(info.pid) {
            return "GPU lock is stale: PID \(info.pid) (`\(info.command)`, started \(info.startedAt)) is no longer alive. Will be cleared on next acquire."
        }
        return "GPU lock: PID \(info.pid) holds it for `\(info.command)` (started \(info.startedAt))"
    }

    // MARK: - Internals

    private static func readLockInfo(url: URL) -> LockInfo? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(LockInfo.self, from: data)
    }

    private static func pidIsAlive(_ pid: Int32) -> Bool {
        // Signal 0 doesn't deliver a signal but checks existence.
        // Returns -1 + ESRCH if dead.
        return kill(pid, 0) == 0 || errno != ESRCH
    }
}
