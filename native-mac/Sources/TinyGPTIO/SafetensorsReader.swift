import Foundation

/// Reader for HuggingFace's `safetensors` weight format. Spec:
/// https://huggingface.co/docs/safetensors/index
///
/// The format:
///
///   bytes 0..7    little-endian u64: header size N
///   bytes 8..8+N  UTF-8 JSON header — { "<tensor_name>": {dtype, shape, data_offsets}, "__metadata__": {...}? }
///   bytes 8+N..   the raw tensor data, packed contiguously in declaration order
///
/// `data_offsets` is [start, end) within the raw data region (not the
/// whole file — you add 8 + N).
///
/// This is the lightest possible reader. Returns the header + a callback
/// that, given a tensor name, returns its raw `Data` slice. The caller is
/// responsible for dtype interpretation (fp32, fp16, bf16, etc.) — we
/// expose the type string but don't decode it here.
public enum SafetensorsReader {
    public struct TensorInfo: Sendable {
        public let dtype: String   // "F32", "F16", "BF16", "I32", ...
        public let shape: [Int]
        public let dataStart: Int  // absolute offset in the file
        public let dataEnd: Int
        public var byteCount: Int { dataEnd - dataStart }
    }

    public struct File {
        public let url: URL
        public let metadata: [String: String]
        public let tensors: [String: TensorInfo]
        /// Backing `Data` for the whole file. Tensor slices are O(1)
        /// substrings; we don't copy bytes until the caller asks for a
        /// specific tensor.
        public let data: Data

        public func tensorData(_ name: String) -> Data? {
            guard let info = tensors[name] else { return nil }
            return data.subdata(in: info.dataStart..<info.dataEnd)
        }
    }

    public enum ReadError: Error, CustomStringConvertible {
        case tooSmall
        case headerNotJSON(Error)
        case malformedTensor(name: String)

        public var description: String {
            switch self {
            case .tooSmall: return "safetensors file too small to be valid"
            case .headerNotJSON(let e): return "header is not valid JSON: \(e)"
            case .malformedTensor(let n): return "tensor \(n) header is malformed"
            }
        }
    }

    public static func read(_ url: URL) throws -> File {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else { throw ReadError.tooSmall }
        let headerSize = Int(data[0..<8].withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        })
        guard data.count >= 8 + headerSize else { throw ReadError.tooSmall }
        let headerBytes = data.subdata(in: 8..<(8 + headerSize))
        let json: [String: Any]
        do {
            json = try JSONSerialization.jsonObject(with: headerBytes) as! [String: Any]
        } catch {
            throw ReadError.headerNotJSON(error)
        }

        var metadata: [String: String] = [:]
        var tensors: [String: TensorInfo] = [:]
        let dataBase = 8 + headerSize
        for (key, value) in json {
            if key == "__metadata__", let dict = value as? [String: String] {
                metadata = dict
                continue
            }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                throw ReadError.malformedTensor(name: key)
            }
            tensors[key] = TensorInfo(
                dtype: dtype, shape: shape,
                dataStart: dataBase + offsets[0],
                dataEnd: dataBase + offsets[1]
            )
        }
        return File(url: url, metadata: metadata, tensors: tensors, data: data)
    }

    /// Quick inspection — print the tensor inventory of a safetensors
    /// file. Useful for "what's actually in this HuggingFace download?"
    public static func summarize(_ file: File) -> String {
        var out = "safetensors file: \(file.url.path)\n"
        out += "metadata: \(file.metadata)\n"
        out += "tensors: \(file.tensors.count)\n"
        var totalBytes = 0
        for (name, info) in file.tensors.sorted(by: { $0.key < $1.key }) {
            totalBytes += info.byteCount
            out += "  \(name)  \(info.dtype)  shape=\(info.shape)  \(info.byteCount) bytes\n"
        }
        out += "total: \(totalBytes) bytes\n"
        return out
    }
}
