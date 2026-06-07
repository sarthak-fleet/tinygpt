import AppKit
import CryptoKit
import Foundation
import TinyGPTScreen

enum AXCapture {
    private static let defaultOutput = "~/.cache/tinygpt/datasets/vlm-ax-mac"
    private static let defaultInterval: TimeInterval = 10
    private static let defaultMaxCaptures = 10_000
    private static let defaultMaxDepth = AccessibilityTree.defaultMaxDepth
    private static let defaultMaxChildren = AccessibilityTree.defaultMaxChildrenPerNode

    static func run(args: [String]) {
        do {
            let opts = try Options.parse(args)
            if opts.help {
                printUsage()
                return
            }
            if opts.stop {
                try stopDaemon(pidPath: opts.pidFile)
                return
            }
            if opts.daemonize {
                try startDaemon(args: args)
                return
            }
            try runLoop(options: opts)
        } catch {
            fputs("ax-capture: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func runLoop(options: Options) throws {
        let outDir = URL(fileURLWithPath: expandTilde(options.outDir))
        try FileManager.default.createDirectory(
            at: outDir,
            withIntermediateDirectories: true
        )

        let excludes = try Excludes.load(path: options.excludesPath)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601

        var captured = 0
        var lastHash: String?

        while captured < options.maxCaptures {
            do {
                guard let metadata = FrontmostWindow.current() else {
                    fputs("ax-capture: no frontmost window; retrying\n", stderr)
                    sleep(options.interval)
                    continue
                }
                if excludes.matches(metadata) {
                    sleep(options.interval)
                    continue
                }

                let image = try ScreenCapture.captureActiveWindow()
                let root = try AccessibilityTree.readFocused(
                    maxDepth: options.maxDepth,
                    maxChildrenPerNode: options.maxChildren
                )
                let elements = flattenAXTree(root)
                let record = CaptureRecord(
                    timestamp: Date(),
                    bundleID: metadata.bundleID,
                    processName: metadata.processName,
                    pid: metadata.pid,
                    windowTitle: metadata.windowTitle,
                    windowFrame: metadata.frame,
                    imageWidth: image.width,
                    imageHeight: image.height,
                    elements: elements
                )
                let json = try encoder.encode(record)

                let tmpPNG = outDir.appendingPathComponent(".ax-capture-\(UUID().uuidString).png")
                _ = try ScreenCapture.writePNG(image, to: tmpPNG.path)
                let pngData = try Data(contentsOf: tmpPNG)
                let hash = captureHash(pngData: pngData, json: canonicalJSON(record))
                guard hash != lastHash else {
                    try? FileManager.default.removeItem(at: tmpPNG)
                    sleep(options.interval)
                    continue
                }

                let stem = "\(timestampSlug(record.timestamp))-\(sanitize(metadata.bundleID ?? metadata.processName))"
                let pngURL = outDir.appendingPathComponent("\(stem).png")
                let jsonURL = outDir.appendingPathComponent("\(stem).json")
                if FileManager.default.fileExists(atPath: pngURL.path) {
                    try FileManager.default.removeItem(at: pngURL)
                }
                try FileManager.default.moveItem(at: tmpPNG, to: pngURL)
                try json.write(to: jsonURL, options: .atomic)
                try appendIndexLine(
                    CaptureIndexLine(
                        timestamp: record.timestamp,
                        bundleID: record.bundleID,
                        processName: record.processName,
                        windowTitle: record.windowTitle,
                        png: pngURL.lastPathComponent,
                        json: jsonURL.lastPathComponent,
                        elementCount: elements.count,
                        hash: hash
                    ),
                    to: outDir.appendingPathComponent("index.jsonl")
                )

                lastHash = hash
                captured += 1
                print("ax-capture: wrote \(pngURL.lastPathComponent) + \(jsonURL.lastPathComponent) (\(elements.count) elements)")
            } catch ScreenError.accessibilityPermissionDenied {
                throw ScreenError.accessibilityPermissionDenied
            } catch ScreenError.screenRecordingPermissionDenied {
                throw ScreenError.screenRecordingPermissionDenied
            } catch {
                fputs("ax-capture: capture skipped: \(error)\n", stderr)
            }
            sleep(options.interval)
        }
    }

    private static func sleep(_ seconds: TimeInterval) {
        Thread.sleep(forTimeInterval: max(0.1, seconds))
    }

    private static func flattenAXTree(_ root: AXNode) -> [AXElementRecord] {
        var elements: [AXElementRecord] = []
        func visit(_ node: AXNode) {
            let label = firstNonEmpty(node.label, node.title, node.value, node.help, node.roleDescription)
            if let frame = node.frame,
               frame.width > 0,
               frame.height > 0,
               node.enabled != false,
               isUsefulRole(node.role) || label != nil {
                elements.append(AXElementRecord(
                    id: elements.count,
                    role: node.role ?? "AXUnknown",
                    label: label,
                    frame: CaptureFrame(frame),
                    enabled: node.enabled
                ))
            }
            for child in node.children {
                visit(child)
            }
        }
        visit(root)
        return elements
    }

    private static func isUsefulRole(_ role: String?) -> Bool {
        guard let role else { return false }
        return [
            "AXButton",
            "AXCheckBox",
            "AXComboBox",
            "AXLink",
            "AXMenuButton",
            "AXPopUpButton",
            "AXRadioButton",
            "AXSearchField",
            "AXSlider",
            "AXTabGroup",
            "AXTextArea",
            "AXTextField"
        ].contains(role)
    }

    private static func firstNonEmpty(_ values: String?...) -> String? {
        for value in values {
            guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !trimmed.isEmpty else { continue }
            return trimmed
        }
        return nil
    }

    private static func canonicalJSON(_ record: CaptureRecord) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return (try? encoder.encode(record)) ?? Data()
    }

    private static func captureHash(pngData: Data, json: Data) -> String {
        var hasher = SHA256()
        hasher.update(data: pngData)
        hasher.update(data: json)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func appendIndexLine(_ line: CaptureIndexLine, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        var data = try encoder.encode(line)
        data.append(0x0a)
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } else {
            try data.write(to: url, options: .atomic)
        }
    }

    private static func startDaemon(args: [String]) throws {
        let filtered = args.filter { $0 != "--daemonize" }
        let opts = try Options.parse(filtered)
        let pidPath = expandTilde(opts.pidFile)
        let logPath = (expandTilde(opts.outDir) as NSString).appendingPathComponent("ax-capture.log")
        try FileManager.default.createDirectory(
            atPath: expandTilde(opts.outDir),
            withIntermediateDirectories: true
        )

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executablePath())
        process.arguments = ["ax-capture"] + filtered
        let logURL = URL(fileURLWithPath: logPath)
        FileManager.default.createFile(atPath: logPath, contents: nil)
        let log = try FileHandle(forWritingTo: logURL)
        process.standardOutput = log
        process.standardError = log
        try process.run()
        try "\(process.processIdentifier)\n".write(
            toFile: pidPath,
            atomically: true,
            encoding: .utf8
        )
        print("ax-capture: started pid \(process.processIdentifier); pid file \(pidPath); log \(logPath)")
    }

    private static func stopDaemon(pidPath: String) throws {
        let path = expandTilde(pidPath)
        guard let text = try? String(contentsOfFile: path, encoding: .utf8),
              let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw UsageError("no pid file at \(path)")
        }
        if kill(pid, SIGTERM) == 0 {
            try? FileManager.default.removeItem(atPath: path)
            print("ax-capture: stopped pid \(pid)")
        } else {
            throw UsageError("failed to stop pid \(pid)")
        }
    }

    private static func executablePath() -> String {
        let arg0 = CommandLine.arguments[0]
        if arg0.hasPrefix("/") { return arg0 }
        return (FileManager.default.currentDirectoryPath as NSString).appendingPathComponent(arg0)
    }

    private static func timestampSlug(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "-")
    }

    private static func sanitize(_ raw: String?) -> String {
        let source = raw ?? "unknown"
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let scalars = source.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
        return String(scalars).trimmingCharacters(in: CharacterSet(charactersIn: "-")).isEmpty
            ? "unknown"
            : String(scalars)
    }

    private static func expandTilde(_ path: String) -> String {
        (path as NSString).expandingTildeInPath
    }

    private static func printUsage() {
        print("""
        usage: tinygpt ax-capture [flags]

        Capture foreground-window PNG + Accessibility-tree JSON pairs for
        Mac-specific VLM training data.

        Flags:
          --out <dir>              Output dataset directory.
                                   Default: \(defaultOutput)
          --interval-sec <sec>     Seconds between attempts. Default: 10
          --max-captures <n>       Stop after n successful non-duplicate captures.
                                   Default: 10000
          --max-depth <n>          AX recursion depth. Default: \(defaultMaxDepth)
          --max-children <n>       AX children expanded per node. Default: \(defaultMaxChildren)
          --excludes <path.json>   Exclude-list JSON. Default:
                                   ~/.config/tinygpt/ax-capture-excludes.json
          --daemonize              Start in background and write a PID file.
          --pid-file <path>        PID file for --daemonize / --stop.
          --stop                   Stop the daemon named by --pid-file.

        Output:
          <timestamp>-<bundle>.png
          <timestamp>-<bundle>.json
          index.jsonl

        Permissions:
          Screen Recording and Accessibility access must be granted to the
          terminal or tinygpt binary in System Settings.
        """)
    }
}

private struct Options {
    var outDir = AXCaptureDefault.output
    var interval: TimeInterval = AXCaptureDefault.interval
    var maxCaptures = AXCaptureDefault.maxCaptures
    var maxDepth = AccessibilityTree.defaultMaxDepth
    var maxChildren = AccessibilityTree.defaultMaxChildrenPerNode
    var excludesPath: String?
    var daemonize = false
    var stop = false
    var help = false
    var pidFile = "~/.cache/tinygpt/ax-capture.pid"

    static func parse(_ args: [String]) throws -> Options {
        var opts = Options()
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":
                opts.outDir = try value(args, &i, flag: "--out")
            case "--interval-sec":
                opts.interval = TimeInterval(try value(args, &i, flag: "--interval-sec")) ?? opts.interval
            case "--max-captures":
                opts.maxCaptures = Int(try value(args, &i, flag: "--max-captures")) ?? opts.maxCaptures
            case "--max-depth":
                opts.maxDepth = Int(try value(args, &i, flag: "--max-depth")) ?? opts.maxDepth
            case "--max-children":
                opts.maxChildren = Int(try value(args, &i, flag: "--max-children")) ?? opts.maxChildren
            case "--excludes":
                opts.excludesPath = try value(args, &i, flag: "--excludes")
            case "--daemonize":
                opts.daemonize = true
                i += 1
            case "--stop":
                opts.stop = true
                i += 1
            case "--pid-file":
                opts.pidFile = try value(args, &i, flag: "--pid-file")
            case "-h", "--help":
                opts.help = true
                i += 1
            default:
                throw UsageError("unknown flag '\(args[i])'")
            }
        }
        opts.interval = max(0.1, opts.interval)
        opts.maxCaptures = max(1, opts.maxCaptures)
        opts.maxDepth = max(0, opts.maxDepth)
        opts.maxChildren = max(1, opts.maxChildren)
        return opts
    }

    private static func value(_ args: [String], _ index: inout Int, flag: String) throws -> String {
        guard index + 1 < args.count else { throw UsageError("\(flag) requires a value") }
        let value = args[index + 1]
        index += 2
        return value
    }
}

private enum AXCaptureDefault {
    static let output = "~/.cache/tinygpt/datasets/vlm-ax-mac"
    static let interval: TimeInterval = 10
    static let maxCaptures = 10_000
}

private struct UsageError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

private struct FrontmostWindow {
    var bundleID: String?
    var processName: String
    var pid: pid_t
    var windowTitle: String?
    var frame: CaptureFrame?

    static func current() -> FrontmostWindow? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let pid = app.processIdentifier
        var title: String?
        var frame: CaptureFrame?

        if let windows = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[String: Any]] {
            for window in windows {
                guard let ownerPID = window[kCGWindowOwnerPID as String] as? pid_t,
                      ownerPID == pid,
                      let bounds = window[kCGWindowBounds as String] as? [String: Any],
                      let width = bounds["Width"] as? Double,
                      let height = bounds["Height"] as? Double,
                      width >= 64,
                      height >= 64 else { continue }
                title = window[kCGWindowName as String] as? String
                frame = CaptureFrame(
                    x: bounds["X"] as? Double ?? 0,
                    y: bounds["Y"] as? Double ?? 0,
                    width: width,
                    height: height
                )
                break
            }
        }

        return FrontmostWindow(
            bundleID: app.bundleIdentifier,
            processName: app.localizedName ?? app.bundleIdentifier ?? "unknown",
            pid: pid,
            windowTitle: title,
            frame: frame
        )
    }
}

private struct Excludes: Decodable {
    var bundleIDs: [String]
    var windowTitlesMatching: [String]
    var processNames: [String]

    enum CodingKeys: String, CodingKey {
        case bundleIDs = "bundle_ids"
        case windowTitlesMatching = "window_titles_matching"
        case processNames = "process_names"
    }

    init(
        bundleIDs: [String] = [],
        windowTitlesMatching: [String] = [],
        processNames: [String] = []
    ) {
        self.bundleIDs = bundleIDs
        self.windowTitlesMatching = windowTitlesMatching
        self.processNames = processNames
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.bundleIDs = try container.decodeIfPresent([String].self, forKey: .bundleIDs) ?? []
        self.windowTitlesMatching = try container.decodeIfPresent(
            [String].self,
            forKey: .windowTitlesMatching
        ) ?? []
        self.processNames = try container.decodeIfPresent([String].self, forKey: .processNames) ?? []
    }

    static func load(path: String?) throws -> Excludes {
        let defaultPath = "~/.config/tinygpt/ax-capture-excludes.json"
        let rawPath = path ?? defaultPath
        let resolved = (rawPath as NSString).expandingTildeInPath
        if FileManager.default.fileExists(atPath: resolved) {
            let data = try Data(contentsOf: URL(fileURLWithPath: resolved))
            return try JSONDecoder().decode(Excludes.self, from: data).merged(with: defaults)
        }
        return defaults
    }

    static let defaults = Excludes(
        bundleIDs: [
            "com.1password.1password",
            "com.agilebits.onepassword7",
            "com.apple.keychainaccess"
        ],
        windowTitlesMatching: [
            "password",
            "passkey",
            "secret",
            "token",
            "auth",
            "login",
            "keychain"
        ],
        processNames: ["1Password", "Keychain Access"]
    )

    func merged(with base: Excludes) -> Excludes {
        Excludes(
            bundleIDs: Array(Set(base.bundleIDs + bundleIDs)),
            windowTitlesMatching: Array(Set(base.windowTitlesMatching + windowTitlesMatching)),
            processNames: Array(Set(base.processNames + processNames))
        )
    }

    func matches(_ window: FrontmostWindow) -> Bool {
        if let bundleID = window.bundleID,
           bundleIDs.contains(where: { $0.caseInsensitiveCompare(bundleID) == .orderedSame }) {
            return true
        }
        if processNames.contains(where: { $0.caseInsensitiveCompare(window.processName) == .orderedSame }) {
            return true
        }
        let title = window.windowTitle?.lowercased() ?? ""
        return windowTitlesMatching.contains { title.contains($0.lowercased()) }
    }
}

private struct CaptureRecord: Encodable {
    var timestamp: Date
    var bundleID: String?
    var processName: String
    var pid: pid_t
    var windowTitle: String?
    var windowFrame: CaptureFrame?
    var imageWidth: Int
    var imageHeight: Int
    var elements: [AXElementRecord]

    enum CodingKeys: String, CodingKey {
        case timestamp
        case bundleID = "bundle_id"
        case processName = "process_name"
        case pid
        case windowTitle = "window_title"
        case windowFrame = "window_frame"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case elements
    }
}

private struct AXElementRecord: Encodable {
    var id: Int
    var role: String
    var label: String?
    var frame: CaptureFrame
    var enabled: Bool?
}

private struct CaptureFrame: Encodable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double

    init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    init(_ frame: AXNode.Frame) {
        self.init(
            x: frame.x,
            y: frame.y,
            width: frame.width,
            height: frame.height
        )
    }
}

private struct CaptureIndexLine: Encodable {
    var timestamp: Date
    var bundleID: String?
    var processName: String
    var windowTitle: String?
    var png: String
    var json: String
    var elementCount: Int
    var hash: String

    enum CodingKeys: String, CodingKey {
        case timestamp
        case bundleID = "bundle_id"
        case processName = "process_name"
        case windowTitle = "window_title"
        case png
        case json
        case elementCount = "element_count"
        case hash
    }
}
