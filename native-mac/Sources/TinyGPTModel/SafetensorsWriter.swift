import Foundation
import MLX

/// safetensors writer — Hugging Face's tagged binary tensor container.
///
/// Format (little-endian throughout):
///   8 bytes        — u64 N, length of JSON header
///   N bytes        — UTF-8 JSON: { name: { dtype, shape, data_offsets: [a, b] }, ... }
///   rest of file   — packed tensor bytes, named tensors at their declared offsets
///
/// We only emit `F32` here (we already store everything as fp32 in the
/// `.tinygpt` format and the GGUF dequant returns fp32). Adding F16
/// would be ~10 lines if a future caller needs it.
///
/// Memory: streams the body to disk via a FileHandle, so a 1B-param
/// model (~4 GB fp32) writes without doubling RAM.
public enum SafetensorsWriter {

    /// One tensor to write. `data` is fp32 in row-major / canonical
    /// HF safetensors layout — caller has already transposed any
    /// Linear weights into the conventional `[out_features, in_features]`
    /// shape (HF's convention; our .tinygpt internally stores them
    /// transposed for WASM-layout reasons).
    public struct Entry {
        public let name: String
        public let data: [Float]
        public let shape: [Int]
        public init(name: String, data: [Float], shape: [Int]) {
            precondition(data.count == shape.reduce(1, *),
                         "entry \(name): data count \(data.count) != prod(shape)=\(shape.reduce(1, *))")
            self.name = name
            self.data = data
            self.shape = shape
        }
    }

    /// Write a safetensors file with the given entries to `url`. The
    /// header keys come out sorted (HF convention; matches what
    /// safetensors-rs writes) so two runs over the same inputs produce
    /// byte-identical files.
    public static func write(entries: [Entry], to url: URL) throws {
        // Build the header JSON. data_offsets are computed in declared
        // order; we sort entries by name to match HF's tooling.
        let sorted = entries.sorted { $0.name < $1.name }
        var dataOffset = 0
        var headerObj: [String: Any] = [:]
        for e in sorted {
            let byteCount = e.data.count * MemoryLayout<Float>.size
            headerObj[e.name] = [
                "dtype": "F32",
                "shape": e.shape,
                "data_offsets": [dataOffset, dataOffset + byteCount],
            ]
            dataOffset += byteCount
        }
        let headerData = try JSONSerialization.data(
            withJSONObject: headerObj,
            options: [.sortedKeys])
        // HF tooling sometimes pads the header to a multiple of 8 with
        // spaces; not required by the format spec but harmless to do.
        var headerBytes = [UInt8](headerData)
        while headerBytes.count % 8 != 0 { headerBytes.append(0x20) }   // space = 0x20

        // Open the file and stream out: u64 length, header bytes, body.
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: url) else {
            throw NSError(domain: "safetensors", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "could not open \(url.path) for writing",
            ])
        }
        defer { try? fh.close() }

        var lenLE = UInt64(headerBytes.count).littleEndian
        try fh.write(contentsOf: Data(bytes: &lenLE, count: 8))
        try fh.write(contentsOf: Data(headerBytes))

        for e in sorted {
            let body = e.data.withUnsafeBufferPointer { Data(buffer: $0) }
            try fh.write(contentsOf: body)
        }
    }

    /// Convenience: collect entries from `(name, MLXArray)` pairs and
    /// write. The MLX arrays are eval'd + downloaded to fp32 inline.
    public static func write(named: [(String, MLXArray)], to url: URL) throws {
        var entries: [Entry] = []
        entries.reserveCapacity(named.count)
        for (name, arr) in named {
            MLX.eval(arr)
            let data = arr.asArray(Float.self)
            entries.append(Entry(name: name, data: data, shape: arr.shape))
        }
        try write(entries: entries, to: url)
    }
}
