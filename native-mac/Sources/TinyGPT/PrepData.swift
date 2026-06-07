import Foundation

/// `tinygpt prep-data ...` — compatibility wrapper around the Python
/// data-prep shim in scripts/data-prep/prep_data.py.
enum PrepData {
    static func run(args: [String]) {
        guard let script = findScript() else {
            fputs("prep-data: could not find scripts/data-prep/prep_data.py; run from the repo root or native-mac/\n", stderr)
            exit(1)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", script.path] + args
        process.standardInput = FileHandle.standardInput
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError

        do {
            try process.run()
            process.waitUntilExit()
            exit(process.terminationStatus)
        } catch {
            fputs("prep-data: failed to launch python3: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func findScript() -> URL? {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidates = [
            cwd.appendingPathComponent("scripts/data-prep/prep_data.py"),
            cwd.appendingPathComponent("../scripts/data-prep/prep_data.py"),
            cwd.appendingPathComponent("../../scripts/data-prep/prep_data.py"),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }
}

