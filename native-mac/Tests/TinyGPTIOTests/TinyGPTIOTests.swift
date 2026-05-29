import Foundation
import XCTest
@testable import TinyGPTIO

final class TinyGPTIOTests: XCTestCase {

    // MARK: - Helpers

    /// Build a synthetic file with two small tensors. Used by the round-trip tests.
    private func makeFixtureFile() -> TinyGPTFile {
        let entries: [TinyGPTHeader.TensorEntry] = [
            .init(name: "token_embedding.weight", shape: [4, 8]),
            .init(name: "output.weight", shape: [8, 4]),
        ]
        let header = TinyGPTHeader(
            config: .init(layers: 1, dModel: 8, ctx: 16, heads: 2, dMlp: 16, batchSize: 2, backend: "wasm"),
            manifest: entries,
            savedAt: "2026-05-28T00:00:00Z",
            finalLoss: .init(step: 100, train: 1.234, val: 1.456),
            sample: "hello world"
        )
        let tensors = entries.enumerated().map { (i, entry) -> TinyGPTTensor in
            // Distinct, easily-checkable byte patterns per tensor.
            let n = entry.elementCount
            let weight = Self.floats(repeatingPattern: Float(i + 1) * 0.5, count: n)
            let m = Self.floats(repeatingPattern: Float(i + 1) * 0.25, count: n)
            let v = Self.floats(repeatingPattern: Float(i + 1) * 0.125, count: n)
            return TinyGPTTensor(entry: entry, weight: weight, adamM: m, adamV: v)
        }
        return TinyGPTFile(header: header, step: 42, tensors: tensors)
    }

    private static func floats(repeatingPattern value: Float, count: Int) -> Data {
        let buf = [Float](repeating: value, count: count)
        return buf.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func tmpURL(_ name: String = "tinygpt-test.tinygpt") -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinygpt-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(name)
    }

    // MARK: - Header decode

    func test_decodesMinimalHeaderJSON() throws {
        let json = """
        {
          "config": { "layers": 2, "dModel": 16, "ctx": 32 },
          "manifest": [
            { "name": "a.weight", "shape": [4, 4] }
          ]
        }
        """
        let header = try JSONDecoder().decode(TinyGPTHeader.self, from: Data(json.utf8))
        XCTAssertEqual(header.config.layers, 2)
        XCTAssertEqual(header.config.dModel, 16)
        XCTAssertEqual(header.config.ctx, 32)
        XCTAssertEqual(header.manifest.count, 1)
        XCTAssertEqual(header.manifest[0].name, "a.weight")
        XCTAssertEqual(header.manifest[0].elementCount, 16)
    }

    func test_ignoresUnknownTopLevelKeys() throws {
        // Browser ships extra fields (savedAt, finalLoss, lossHistory, gpuBytes, etc.)
        // — the reader must not choke on ones we don't model explicitly.
        let json = """
        {
          "config": { "layers": 1 },
          "manifest": [{ "name": "x", "shape": [2] }],
          "gpuBytes": 9999,
          "futureField": { "nested": [1, 2, 3] }
        }
        """
        XCTAssertNoThrow(
            try JSONDecoder().decode(TinyGPTHeader.self, from: Data(json.utf8))
        )
    }

    // MARK: - Round-trip

    func test_roundTripsByteIdenticallyWhenJSONKeysAreSorted() throws {
        let original = makeFixtureFile()
        let encoded = try TinyGPTFileWriter.encode(original)
        let decoded = try TinyGPTFileReader.decode(encoded, source: tmpURL())

        XCTAssertEqual(decoded.version, original.version)
        XCTAssertEqual(decoded.step, original.step)
        XCTAssertEqual(decoded.header, original.header)
        XCTAssertEqual(decoded.tensors.count, original.tensors.count)
        for (a, b) in zip(original.tensors, decoded.tensors) {
            XCTAssertEqual(a.entry, b.entry)
            XCTAssertEqual(a.weight, b.weight)
            XCTAssertEqual(a.adamM, b.adamM)
            XCTAssertEqual(a.adamV, b.adamV)
        }

        // Second encode reproduces the same bytes — the writer is deterministic.
        let reencoded = try TinyGPTFileWriter.encode(decoded)
        XCTAssertEqual(reencoded, encoded)
    }

    func test_writeThenReadFromDisk() throws {
        let url = tmpURL()
        let original = makeFixtureFile()
        try TinyGPTFileWriter.write(original, to: url)
        let loaded = try TinyGPTFileReader.read(url)
        XCTAssertEqual(loaded.header, original.header)
        XCTAssertEqual(loaded.step, original.step)
        XCTAssertEqual(loaded.tensors.count, original.tensors.count)
        for (a, b) in zip(original.tensors, loaded.tensors) {
            XCTAssertEqual(a.weightFloats, b.weightFloats)
            XCTAssertEqual(a.adamMFloats, b.adamMFloats)
            XCTAssertEqual(a.adamVFloats, b.adamVFloats)
        }
    }

    // MARK: - Error paths

    func test_rejectsBadMagic() {
        var data = Data([0x4e, 0x4f, 0x50, 0x45])  // "NOPE"
        data.append(Data(count: 8))
        XCTAssertThrowsError(try TinyGPTFileReader.decode(data, source: tmpURL()))
    }

    func test_rejectsUnsupportedVersion() {
        var data = Data()
        data.append(contentsOf: TinyGPTFormat.magic)
        var version: UInt32 = 99
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        var headerLen: UInt32 = 0
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        XCTAssertThrowsError(try TinyGPTFileReader.decode(data, source: tmpURL()))
    }

    func test_rejectsTruncatedBody() throws {
        // Build a valid file then chop the last few bytes off so the last tensor
        // is incomplete. The reader should fail with a `truncatedBody` error.
        let file = makeFixtureFile()
        var encoded = try TinyGPTFileWriter.encode(file)
        encoded.removeSubrange((encoded.count - 4)..<encoded.count)
        XCTAssertThrowsError(try TinyGPTFileReader.decode(encoded, source: tmpURL()))
    }

    func test_reportsMissingManifest() throws {
        // Header without `manifest` should produce the dedicated v1-detection error.
        var data = Data()
        data.append(contentsOf: TinyGPTFormat.magic)
        var version: UInt32 = 2
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        let json = #"{"config":{"layers":1}}"#
        let headerBytes = Data(json.utf8)
        var headerLen = UInt32(headerBytes.count)
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(headerBytes)
        data.append(Data(count: 4))  // step counter

        var sawMissingManifest = false
        do {
            _ = try TinyGPTFileReader.decode(data, source: tmpURL())
        } catch TinyGPTFileError.missingManifest {
            sawMissingManifest = true
        } catch {
            // Other error types are acceptable as long as the reader rejected the file.
        }
        XCTAssertTrue(sawMissingManifest)
    }

    // MARK: - fp16 inference layout

    /// Build a fp16 inference-layout file: weight buffer is contiguous, indexed
    /// by floatOffset. No AdamW state.
    private func makeFP16Fixture() -> TinyGPTFile {
        let entries: [TinyGPTHeader.TensorEntry] = [
            .init(name: "a", shape: [2, 2], floatOffset: 0),
            .init(name: "b", shape: [2], floatOffset: 4),
            .init(name: "c", shape: [3], floatOffset: 6),
        ]
        let header = TinyGPTHeader(
            config: .init(layers: 1),
            manifest: entries,
            weightDtype: "fp16",
            includesOptimizerState: false,
            stateByteLength: 9 * 2
        )
        // Build a contiguous fp16 buffer with distinct values per tensor.
        let halfA: [UInt16] = [0x3C00, 0x4000, 0x4200, 0x4400]  // 1.0, 2.0, 3.0, 4.0
        let halfB: [UInt16] = [0x4500, 0x4600]                    // 5.0, 6.0
        let halfC: [UInt16] = [0x4700, 0x4800, 0x4880]            // 7.0, 8.0, 9.0
        func bytes(_ halves: [UInt16]) -> Data {
            return halves.withUnsafeBufferPointer { ptr in
                Data(buffer: UnsafeBufferPointer(start: UnsafeRawPointer(ptr.baseAddress)!
                    .bindMemory(to: UInt8.self, capacity: halves.count * 2),
                    count: halves.count * 2))
            }
        }
        let tensors = [
            TinyGPTTensor(entry: entries[0], weight: bytes(halfA), dtype: .fp16),
            TinyGPTTensor(entry: entries[1], weight: bytes(halfB), dtype: .fp16),
            TinyGPTTensor(entry: entries[2], weight: bytes(halfC), dtype: .fp16),
        ]
        return TinyGPTFile(header: header, step: 1000, tensors: tensors)
    }

    func test_detectsFP16BodyLayout() throws {
        let f = makeFP16Fixture()
        XCTAssertEqual(f.header.bodyLayout, .inferenceFP16)
    }

    func test_roundTripsFP16InferenceLayout() throws {
        let original = makeFP16Fixture()
        let encoded = try TinyGPTFileWriter.encode(original)
        let decoded = try TinyGPTFileReader.decode(encoded, source: tmpURL())

        XCTAssertEqual(decoded.header, original.header)
        XCTAssertEqual(decoded.step, original.step)
        XCTAssertEqual(decoded.tensors.count, original.tensors.count)
        for (a, b) in zip(original.tensors, decoded.tensors) {
            XCTAssertEqual(a.entry, b.entry)
            XCTAssertEqual(a.weight, b.weight)
            XCTAssertEqual(b.dtype, .fp16)
            XCTAssertTrue(b.adamM.isEmpty)
            XCTAssertTrue(b.adamV.isEmpty)
        }
    }

    func test_fp16WeightExpandsToFloat32Correctly() throws {
        let original = makeFP16Fixture()
        let encoded = try TinyGPTFileWriter.encode(original)
        let decoded = try TinyGPTFileReader.decode(encoded, source: tmpURL())
        let tensorA = decoded.tensors[0]
        XCTAssertEqual(tensorA.weightFP16AsFloat32(), [1.0, 2.0, 3.0, 4.0])
        let tensorB = decoded.tensors[1]
        XCTAssertEqual(tensorB.weightFP16AsFloat32(), [5.0, 6.0])
        let tensorC = decoded.tensors[2]
        XCTAssertEqual(tensorC.weightFP16AsFloat32(), [7.0, 8.0, 9.0])
    }

    // MARK: - TensorEntry geometry

    func test_tensorEntryByteLengthIsShapeProductTimesFour() {
        let e = TinyGPTHeader.TensorEntry(name: "x", shape: [3, 4, 5])
        XCTAssertEqual(e.elementCount, 60)
        XCTAssertEqual(e.byteLength, 240)
    }

    // MARK: - Manifest schema — current-set field coverage
    //
    // The header gained a small avalanche of optional fields over the
    // last few weeks (BPE / MoE / MoD / DiffAttn / YOCO / grad-ckpt /
    // sliding-window). A regression in any of them would silently corrupt
    // a `--resume` continuation (architecture flag dropped → checkpoint
    // re-loaded with a different topology). Each of these tests pins one
    // field's round-trip behaviour so a future Codable refactor can't
    // accidentally drop it.

    func test_configFields_roundTripBPEMetadata() throws {
        let cfg = TinyGPTHeader.Config(
            layers: 4, dModel: 128, ctx: 256, heads: 4, dMlp: 512,
            batchSize: 8, backend: "mlx-swift",
            vocabSize: 32_000,
            tokenizerSource: "/models/llama-3-tokenizer"
        )
        let encoded = try JSONEncoder().encode(cfg)
        let decoded = try JSONDecoder().decode(TinyGPTHeader.Config.self, from: encoded)
        XCTAssertEqual(decoded.vocabSize, 32_000)
        XCTAssertEqual(decoded.tokenizerSource, "/models/llama-3-tokenizer")
    }

    func test_configFields_roundTripMoEMetadata() throws {
        let cfg = TinyGPTHeader.Config(
            layers: 4, dModel: 128,
            nExperts: 8, moeTopK: 2, loadBalanceWeight: 0.01
        )
        let encoded = try JSONEncoder().encode(cfg)
        let decoded = try JSONDecoder().decode(TinyGPTHeader.Config.self, from: encoded)
        XCTAssertEqual(decoded.nExperts, 8)
        XCTAssertEqual(decoded.moeTopK, 2)
        XCTAssertEqual(decoded.loadBalanceWeight, 0.01)
    }

    func test_configFields_roundTripArchitectureFlags() throws {
        // The architecture-feature bools (MoD, DiffAttn, YOCO, GradCkpt) +
        // sliding-window. Together these reproduce a checkpoint's
        // structural identity; dropping ANY of them on round-trip
        // breaks --resume.
        let cfg = TinyGPTHeader.Config(
            layers: 12, dModel: 256, ctx: 1024,
            slidingWindow: 256,
            useMoD: true,
            useDifferentialAttention: true,
            useYOCO: true,
            useGradCheckpoint: true
        )
        let encoded = try JSONEncoder().encode(cfg)
        let decoded = try JSONDecoder().decode(TinyGPTHeader.Config.self, from: encoded)
        XCTAssertEqual(decoded.slidingWindow, 256)
        XCTAssertEqual(decoded.useMoD, true)
        XCTAssertEqual(decoded.useDifferentialAttention, true)
        XCTAssertEqual(decoded.useYOCO, true)
        XCTAssertEqual(decoded.useGradCheckpoint, true)
    }

    func test_configFields_omitNilFieldsFromJSON() throws {
        // We need backwards-compatibility for the writer: when a flag
        // isn't set, it MUST NOT appear in the encoded JSON. Older
        // readers (pre-YOCO, pre-MoE) wouldn't have its key and the
        // encoder would otherwise produce `"useYOCO": null` for them.
        let cfg = TinyGPTHeader.Config(layers: 4, dModel: 128)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(cfg)
        let str = String(data: data, encoding: .utf8) ?? ""
        XCTAssertFalse(str.contains("useYOCO"),
                       "nil useYOCO leaked into encoded JSON: \(str)")
        XCTAssertFalse(str.contains("useMoD"),
                       "nil useMoD leaked into encoded JSON: \(str)")
        XCTAssertFalse(str.contains("nExperts"),
                       "nil nExperts leaked into encoded JSON: \(str)")
        XCTAssertFalse(str.contains("tokenizerSource"),
                       "nil tokenizerSource leaked into encoded JSON: \(str)")
    }

    func test_headerRoundTrips_withFullSchema() throws {
        // End-to-end: build a header with every current field populated,
        // run it through encode→decode, byte-compare.
        let entries: [TinyGPTHeader.TensorEntry] = [
            .init(name: "token_embedding.weight", shape: [256, 8]),
            .init(name: "output.weight", shape: [8, 256]),
        ]
        let header = TinyGPTHeader(
            config: .init(
                layers: 2, dModel: 8, ctx: 16, heads: 2, dMlp: 16,
                batchSize: 2, backend: "mlx-swift",
                vocabSize: 256,
                tokenizerSource: nil,
                nExperts: 2, moeTopK: 1, loadBalanceWeight: 0.05,
                slidingWindow: 8,
                useMoD: true,
                useDifferentialAttention: false,
                useYOCO: true,
                useGradCheckpoint: true
            ),
            manifest: entries,
            savedAt: "2026-05-30T00:00:00Z",
            finalLoss: .init(step: 100, train: 1.5, val: 1.6),
            sample: "hello"
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(header)
        let decoded = try JSONDecoder().decode(TinyGPTHeader.self, from: data)
        // Equality should walk every field.
        XCTAssertEqual(decoded, header)
    }

    // MARK: - Legacy v1 file detection
    //
    // v1 files predate the `manifest` field — the reader CAN'T decode them
    // (no schema for tensor layout) but MUST surface a clean
    // `missingManifest` error so the CLI can print the upgrade hint
    // instead of a confusing "headerNotJSON" wrapper.

    func test_v1File_withoutManifestReportsMissingManifest() throws {
        // v1 header has no `manifest` array. Build one by hand:
        //   magic | version=1 | header_len | header JSON | step
        var data = Data()
        data.append(contentsOf: TinyGPTFormat.magic)
        var version: UInt32 = 1
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        let json = #"{"config":{"layers":4,"dModel":128,"ctx":128,"heads":4,"dMlp":512}}"#
        let headerBytes = Data(json.utf8)
        var headerLen = UInt32(headerBytes.count)
        withUnsafeBytes(of: &headerLen) { data.append(contentsOf: $0) }
        data.append(headerBytes)
        data.append(Data(count: 4))  // step counter

        var sawMissingManifest = false
        do {
            _ = try TinyGPTFileReader.decode(data, source: tmpURL())
        } catch TinyGPTFileError.missingManifest {
            sawMissingManifest = true
        } catch {
            // Any other failure also flags v1; the dedicated error path
            // is the desired one for a clear CLI message.
        }
        XCTAssertTrue(sawMissingManifest,
                      "v1 file without manifest should produce missingManifest, not a generic decode error")
    }

    func test_v1File_versionIsAcceptedInTheSupportedSet() {
        // Defensive: the format pin says v1 stays in `supportedVersions`.
        // If a future cleanup drops it (and there's a reasonable case
        // for that), this test fires so the corresponding migration
        // story is reviewed first.
        XCTAssertTrue(TinyGPTFormat.supportedVersions.contains(1),
                      "v1 dropped from supportedVersions — was that intentional?")
        XCTAssertTrue(TinyGPTFormat.supportedVersions.contains(2))
    }
}
