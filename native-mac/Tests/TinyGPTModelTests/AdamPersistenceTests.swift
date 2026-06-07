import Foundation
import XCTest
import TinyGPTIO

final class AdamPersistenceTests: XCTestCase {
    private var tinygptBinaryURL: URL? {
        if let p = ProcessInfo.processInfo.environment["TINYGPT_BIN"],
           FileManager.default.isExecutableFile(atPath: p) {
            return URL(fileURLWithPath: p)
        }
        let local = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/arm64-apple-macosx/release/tinygpt")
        return FileManager.default.isExecutableFile(atPath: local.path) ? local : nil
    }

    func test_adamStatePersistsAndRestores_viaTinySubprocessSmoke() throws {
        guard ProcessInfo.processInfo.environment["TINYGPT_RUN_ADAM_PERSISTENCE_SMOKE"] == "1" else {
            throw XCTSkip("set TINYGPT_RUN_ADAM_PERSISTENCE_SMOKE=1 to run the GPU/subprocess Adam persistence smoke")
        }
        guard let bin = tinygptBinaryURL else {
            throw XCTSkip("tinygpt binary not found; set TINYGPT_BIN")
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinygpt-adam-persist-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let out = dir.appendingPathComponent("adam.tinygpt")
        _ = try run(
            bin,
            args: [
                "train", "--preset", "tiny", "--steps", "1",
                "--out", out.path, "--save-every", "1",
                "--seed", "123", "--sample-every", "999999"
            ]
        )

        let file = try TinyGPTFileReader.read(out)
        let nonzeroMomentTensors = file.tensors.filter {
            $0.adamM.contains(where: { $0 != 0 }) || $0.adamV.contains(where: { $0 != 0 })
        }.count
        XCTAssertGreaterThan(nonzeroMomentTensors, 0)

        let resumeLog = try run(
            bin,
            args: [
                "train", "--resume", out.path, "--steps", "2",
                "--out", out.path, "--save-every", "1",
                "--sample-every", "999999"
            ]
        )
        XCTAssertTrue(resumeLog.contains("[resume] restored Adam state"), resumeLog)
    }

    private func run(_ bin: URL, args: [String]) throws -> String {
        let process = Process()
        process.executableURL = bin
        process.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["TINYGPT_NO_POWER_PAUSE"] = "1"
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertEqual(process.terminationStatus, 0, text)
        return text
    }
}

