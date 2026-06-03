import Foundation
import MLX
import TinyGPTIO

/// GGUF reader — llama.cpp's tagged-binary model format.
///
/// Tested: ~~ smoke-tested against a hand-built fixture (see
/// TinyGPTModelTests/GGUFReaderTests.swift). Real model loads need an
/// arch mapping from GGUF's metadata-keyed dimensions to our
/// ModelConfig — that ships as a follow-up via HFModelLoader. THIS
/// file is the bit-level parser + dequant primitives.
///
/// Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
///
/// Layout (little-endian):
///   u32  magic = 'GGUF' (0x46554747)
///   u32  version (3 at time of writing)
///   u64  tensor_count
///   u64  metadata_kv_count
///   metadata_kv[metadata_kv_count]
///   tensor_info[tensor_count]
///   ALIGN to general.alignment (default 32) from start of file
///   tensor_data[...]
///
/// First-cut tensor types supported:
///   F32  (ggml type 0)
///   F16  (ggml type 1)
///   Q4_0 (ggml type 2) — 32-element blocks: fp16 scale + 16 packed bytes
///   Q8_0 (ggml type 8) — 32-element blocks: fp16 scale + 32 i8 values
///
/// K-quants (Q4_K, Q6_K, etc.) are common in modern GGUF dumps but use
/// a more elaborate block layout (super-blocks of 256, multiple scales
/// per block, packed quants). Adding them is bounded mechanical work
/// — slot into `dequant(_:type:)` below when needed.
public enum GGUFReader {

    public enum GGUFError: Error, LocalizedError {
        case badMagic(UInt32)
        case unsupportedVersion(UInt32)
        case truncated(String)
        case unsupportedTensorType(UInt32, name: String)
        case unsupportedMetaType(UInt32)
        public var errorDescription: String? {
            switch self {
            case .badMagic(let m): return "bad GGUF magic 0x\(String(m, radix: 16))"
            case .unsupportedVersion(let v): return "GGUF version \(v) — only v2/v3 are supported"
            case .truncated(let what): return "GGUF truncated reading \(what)"
            case .unsupportedTensorType(let t, let n): return "GGUF tensor '\(n)' has unsupported type \(t)"
            case .unsupportedMetaType(let t): return "GGUF metadata value-type \(t) not supported in first cut"
            }
        }
    }

    /// GGML scalar types we recognise. Values match the ggml enum in
    /// llama.cpp/ggml.h.
    public enum GGMLType: UInt32 {
        case f32  = 0
        case f16  = 1
        case q4_0 = 2
        case q8_0 = 8
        case q4_K = 12
        case q5_K = 13
        case q6_K = 14
        case q8_K = 15
    }

    public struct TensorInfo {
        public let name: String
        public let shape: [Int]      // GGUF dims are stored in REVERSE (last-to-first); we de-reverse on read
        public let type: UInt32      // raw ggml type; check against GGMLType
        public let offset: UInt64    // from start of tensor-data section
    }

    public struct ParsedFile {
        public let metadata: [String: Any]
        public let tensors: [TensorInfo]
        public let tensorDataBase: Int  // byte offset where tensor data starts
        public let raw: Data
    }

    /// Parse the header + tensor info. Does NOT dequantise — call
    /// `loadTensor(_:)` once you've matched names against an arch.
    public static func parse(url: URL) throws -> ParsedFile {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        var c = Cursor(data: data)

        let magic = try c.readU32()
        guard magic == 0x46554747 else { throw GGUFError.badMagic(magic) }
        let version = try c.readU32()
        guard version == 2 || version == 3 else { throw GGUFError.unsupportedVersion(version) }
        let tensorCount = try c.readU64()
        let metaCount = try c.readU64()

        var metadata: [String: Any] = [:]
        for _ in 0..<metaCount {
            let key = try c.readGGUFString()
            let valueType = try c.readU32()
            let value = try c.readMetaValue(type: valueType)
            metadata[key] = value
        }

        var tensors: [TensorInfo] = []
        tensors.reserveCapacity(Int(tensorCount))
        for _ in 0..<tensorCount {
            let name = try c.readGGUFString()
            let nDim = try c.readU32()
            var dims: [Int] = []
            for _ in 0..<nDim { dims.append(Int(try c.readU64())) }
            // GGUF dims are stored fastest-varying first; transpose to
            // the row-major convention our code uses (slowest first).
            dims.reverse()
            let type = try c.readU32()
            let offset = try c.readU64()
            tensors.append(TensorInfo(name: name, shape: dims, type: type, offset: offset))
        }

        // ALIGN to `general.alignment` (default 32). Anything that
        // would land between header-end and the alignment boundary is
        // padding bytes.
        let alignment = (metadata["general.alignment"] as? Int) ?? 32
        let pos = c.position
        let padded = ((pos + alignment - 1) / alignment) * alignment
        return ParsedFile(metadata: metadata, tensors: tensors,
                           tensorDataBase: padded, raw: data)
    }

    /// Dequantise a single tensor into a flat fp32 MLXArray. Throws on
    /// unsupported types.
    public static func loadTensor(_ info: TensorInfo, from parsed: ParsedFile) throws -> MLXArray {
        let raw = parsed.raw
        let base = parsed.tensorDataBase + Int(info.offset)
        let nElems = info.shape.reduce(1, *)
        guard let kind = GGMLType(rawValue: info.type) else {
            throw GGUFError.unsupportedTensorType(info.type, name: info.name)
        }
        switch kind {
        case .f32:
            let bytes = nElems * 4
            guard raw.count >= base + bytes else { throw GGUFError.truncated("f32 tensor \(info.name)") }
            let buf = raw.subdata(in: base..<(base + bytes))
            let floats: [Float] = buf.withUnsafeBytes { ptr in
                Array(UnsafeBufferPointer<Float>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: nElems))
            }
            return MLXArray(floats, info.shape)
        case .f16:
            let bytes = nElems * 2
            guard raw.count >= base + bytes else { throw GGUFError.truncated("f16 tensor \(info.name)") }
            let buf = raw.subdata(in: base..<(base + bytes))
            var floats = [Float](repeating: 0, count: nElems)
            buf.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let halves = ptr.bindMemory(to: UInt16.self)
                for i in 0..<nElems { floats[i] = halfToFloat(halves[i]) }
            }
            return MLXArray(floats, info.shape)
        case .q4_0:
            // 32-element blocks: 2-byte fp16 scale + 16 packed bytes
            // (each byte = two int4 values, low nibble first, range
            // [-8, 7] after subtracting 8 from the unsigned nibble).
            precondition(nElems % 32 == 0, "Q4_0 expects multiple-of-32 element count (got \(nElems))")
            let blockBytes = 2 + 16
            let nBlocks = nElems / 32
            guard raw.count >= base + nBlocks * blockBytes else {
                throw GGUFError.truncated("q4_0 tensor \(info.name)")
            }
            var out = [Float](repeating: 0, count: nElems)
            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let bytes = ptr.baseAddress!.advanced(by: base)
                for b in 0..<nBlocks {
                    let blockStart = bytes.advanced(by: b * blockBytes)
                    let scaleU16 = blockStart.load(as: UInt16.self)
                    let scale = halfToFloat(scaleU16)
                    let payload = blockStart.advanced(by: 2).assumingMemoryBound(to: UInt8.self)
                    for i in 0..<16 {
                        let byte = payload[i]
                        let lo = Int(byte & 0x0f) - 8
                        let hi = Int(byte >> 4) - 8
                        out[b * 32 + i] = scale * Float(lo)
                        out[b * 32 + i + 16] = scale * Float(hi)
                    }
                }
            }
            return MLXArray(out, info.shape)
        case .q8_0:
            // 32-element blocks: 2-byte fp16 scale + 32 i8 values.
            precondition(nElems % 32 == 0, "Q8_0 expects multiple-of-32 element count (got \(nElems))")
            let blockBytes = 2 + 32
            let nBlocks = nElems / 32
            guard raw.count >= base + nBlocks * blockBytes else {
                throw GGUFError.truncated("q8_0 tensor \(info.name)")
            }
            var out = [Float](repeating: 0, count: nElems)
            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let bytes = ptr.baseAddress!.advanced(by: base)
                for b in 0..<nBlocks {
                    let blockStart = bytes.advanced(by: b * blockBytes)
                    let scaleU16 = blockStart.load(as: UInt16.self)
                    let scale = halfToFloat(scaleU16)
                    let payload = blockStart.advanced(by: 2).assumingMemoryBound(to: Int8.self)
                    for i in 0..<32 {
                        out[b * 32 + i] = scale * Float(payload[i])
                    }
                }
            }
            return MLXArray(out, info.shape)
        case .q4_K:
            // K-quant: 256-element super-blocks, 144 bytes each.
            //   2 bytes  d     — fp16 outer scale for the 8 sub-block scales
            //   2 bytes  dmin  — fp16 outer scale for the 8 sub-block mins
            //  12 bytes  packed 6-bit (8 sub-scales + 8 sub-mins)
            // 128 bytes  qs    — 256 4-bit quantised values (2 per byte)
            //
            // Sub-block layout (8 sub-blocks of 32 elements). The
            // packing for the (scale, min) pair of sub-block j follows
            // llama.cpp/ggml-quants.c::get_scale_min_k4:
            //   j < 4:   sc = scales[j] & 0x3f
            //            m  = scales[j+4] & 0x3f
            //   j ≥ 4:   sc = (scales[j+4] & 0x0f) | ((scales[j-4] >> 6) << 4)
            //            m  = (scales[j+4] >> 4)   | ((scales[j  ] >> 6) << 4)
            //
            // The qs bytes are read in 32-byte chunks, each chunk
            // covering TWO consecutive sub-blocks (low nibble for
            // sub-block 2i, high nibble for sub-block 2i+1).
            // Dequant per element: x = d·sc·(quant_4bit) - dmin·m.
            precondition(nElems % 256 == 0, "Q4_K expects multiple-of-256 element count (got \(nElems))")
            let blockBytes = 2 + 2 + 12 + 128
            let nBlocks = nElems / 256
            guard raw.count >= base + nBlocks * blockBytes else {
                throw GGUFError.truncated("q4_K tensor \(info.name)")
            }
            var out = [Float](repeating: 0, count: nElems)
            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let bytesBase = ptr.baseAddress!.advanced(by: base)
                for b in 0..<nBlocks {
                    let blockStart = bytesBase.advanced(by: b * blockBytes)
                    let d = halfToFloat(blockStart.load(as: UInt16.self))
                    let dmin = halfToFloat(blockStart.advanced(by: 2).load(as: UInt16.self))
                    let scales = blockStart.advanced(by: 4).assumingMemoryBound(to: UInt8.self)
                    let qs = blockStart.advanced(by: 16).assumingMemoryBound(to: UInt8.self)
                    // Decode all 8 (sc, m) pairs upfront — small enough
                    // to keep on stack-like arrays.
                    var scArr = [UInt8](repeating: 0, count: 8)
                    var mArr  = [UInt8](repeating: 0, count: 8)
                    for j in 0..<4 {
                        scArr[j] = scales[j] & 0x3f
                        mArr[j]  = scales[j + 4] & 0x3f
                    }
                    for j in 4..<8 {
                        // llama.cpp/ggml-quants.c::get_scale_min_k4:
                        //   d = (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4)
                        //   m = (q[j+4] >>  4)  | ((q[j  ] >> 6) << 4)
                        // i.e. q[j+4] is the "fancy-half" byte storing
                        // the LOW 4 bits of both sc and m for this
                        // sub-block, while q[j-4] / q[j] contribute the
                        // top 2 bits of sc / m respectively (in their
                        // own top 2 bits).
                        scArr[j] = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4)
                        mArr[j]  = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4)
                    }
                    // Walk qs in 32-byte chunks; each chunk = two
                    // sub-blocks. Out offset advances by 64 per chunk.
                    var qsOff = 0
                    var outBase = b * 256
                    var sbIdx = 0
                    while sbIdx < 8 {
                        let sc1 = Float(scArr[sbIdx])
                        let m1  = Float(mArr[sbIdx])
                        let sc2 = Float(scArr[sbIdx + 1])
                        let m2  = Float(mArr[sbIdx + 1])
                        let d1 = d * sc1
                        let mm1 = dmin * m1
                        let d2 = d * sc2
                        let mm2 = dmin * m2
                        for l in 0..<32 {
                            let byte = qs[qsOff + l]
                            out[outBase + l]      = d1 * Float(byte & 0x0f) - mm1
                            out[outBase + 32 + l] = d2 * Float(byte >> 4)    - mm2
                        }
                        qsOff += 32
                        outBase += 64
                        sbIdx += 2
                    }
                }
            }
            return MLXArray(out, info.shape)
        case .q5_K:
            // K-quant: 256-element super-blocks, 176 bytes each.
            //   2 bytes  d     — fp16 outer scale for the 8 sub-block scales
            //   2 bytes  dmin  — fp16 outer scale for the 8 sub-block mins
            //  12 bytes  packed 6-bit (same get_scale_min_k4 layout as Q4_K)
            //  32 bytes  qh    — 256 high-bits (1 per element, 8 per byte)
            // 128 bytes  qs    — 256 low-4-bits (2 per byte)
            //
            // Per element: 5-bit value = (qh_bit << 4) | low4, then
            // x = d · sc · value - dmin · m  (same form as Q4_K).
            precondition(nElems % 256 == 0, "Q5_K expects multiple-of-256 element count (got \(nElems))")
            let blockBytes = 2 + 2 + 12 + 32 + 128
            let nBlocks = nElems / 256
            guard raw.count >= base + nBlocks * blockBytes else {
                throw GGUFError.truncated("q5_K tensor \(info.name)")
            }
            var out = [Float](repeating: 0, count: nElems)
            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let bytesBase = ptr.baseAddress!.advanced(by: base)
                for b in 0..<nBlocks {
                    let blockStart = bytesBase.advanced(by: b * blockBytes)
                    let d = halfToFloat(blockStart.load(as: UInt16.self))
                    let dmin = halfToFloat(blockStart.advanced(by: 2).load(as: UInt16.self))
                    let scales = blockStart.advanced(by: 4).assumingMemoryBound(to: UInt8.self)
                    let qh = blockStart.advanced(by: 16).assumingMemoryBound(to: UInt8.self)
                    let qs = blockStart.advanced(by: 16 + 32).assumingMemoryBound(to: UInt8.self)
                    var scArr = [UInt8](repeating: 0, count: 8)
                    var mArr  = [UInt8](repeating: 0, count: 8)
                    for j in 0..<4 {
                        scArr[j] = scales[j] & 0x3f
                        mArr[j]  = scales[j + 4] & 0x3f
                    }
                    for j in 4..<8 {
                        scArr[j] = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4)
                        mArr[j]  = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4)
                    }
                    // Walk in 32-byte qs chunks (low-4-bits) + take 1 bit
                    // per element from qh. Same iteration structure as
                    // Q4_K but with the extra high-bit shifted in.
                    var qsOff = 0
                    var outBase = b * 256
                    var sbIdx = 0
                    while sbIdx < 8 {
                        let sc1 = Float(scArr[sbIdx])
                        let m1  = Float(mArr[sbIdx])
                        let sc2 = Float(scArr[sbIdx + 1])
                        let m2  = Float(mArr[sbIdx + 1])
                        let d1 = d * sc1; let mm1 = dmin * m1
                        let d2 = d * sc2; let mm2 = dmin * m2
                        let qhMask: UInt8 = UInt8(1 << (sbIdx))           // bit for sub-block 2i (low half)
                        let qhMask2: UInt8 = UInt8(1 << (sbIdx + 1))      // bit for sub-block 2i+1
                        for l in 0..<32 {
                            let lowByte = qs[qsOff + l]
                            let lo4 = lowByte & 0x0f
                            let hi4 = lowByte >> 4
                            let hbit1: UInt8 = (qh[l] & qhMask)  != 0 ? 1 : 0
                            let hbit2: UInt8 = (qh[l] & qhMask2) != 0 ? 1 : 0
                            let v1 = UInt8(lo4) | (hbit1 << 4)
                            let v2 = UInt8(hi4) | (hbit2 << 4)
                            out[outBase + l]      = d1 * Float(v1) - mm1
                            out[outBase + 32 + l] = d2 * Float(v2) - mm2
                        }
                        qsOff += 32
                        outBase += 64
                        sbIdx += 2
                    }
                }
            }
            return MLXArray(out, info.shape)
        case .q6_K:
            // K-quant: 256-element super-blocks, 210 bytes each.
            // 128 bytes  ql      — 256 low-4-bits (2 per byte)
            //  64 bytes  qh      — 256 high-2-bits (4 per byte)
            //  16 bytes  scales  — int8 per 16-element sub-block (16 sub-blocks)
            //   2 bytes  d       — fp16 super-block scale
            //
            // No min component — Q6_K is centered. Per element:
            //   value_raw = (qh_2b << 4) | low_4_bits        ∈ [0, 63]
            //   value     = value_raw - 32                    ∈ [-32, 31]
            //   x         = d · int8_scale · value
            precondition(nElems % 256 == 0, "Q6_K expects multiple-of-256 element count (got \(nElems))")
            let blockBytes = 128 + 64 + 16 + 2
            let nBlocks = nElems / 256
            guard raw.count >= base + nBlocks * blockBytes else {
                throw GGUFError.truncated("q6_K tensor \(info.name)")
            }
            var out = [Float](repeating: 0, count: nElems)
            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let bytesBase = ptr.baseAddress!.advanced(by: base)
                for b in 0..<nBlocks {
                    let blockStart = bytesBase.advanced(by: b * blockBytes)
                    let qlAll = blockStart.assumingMemoryBound(to: UInt8.self)
                    let qhAll = blockStart.advanced(by: 128).assumingMemoryBound(to: UInt8.self)
                    let scAll = blockStart.advanced(by: 128 + 64).assumingMemoryBound(to: Int8.self)
                    let d = halfToFloat(blockStart.advanced(by: 128 + 64 + 16).load(as: UInt16.self))
                    // Direct translation of dequantize_row_q6_K from
                    // llama.cpp/ggml-quants.c. Two outer chunks of 128
                    // elements; each chunk uses 32 ql bytes (offset by
                    // 32 between the "row 1" and "row 3" halves), 16
                    // qh bytes, and 8 scales.
                    var outBase = b * 256
                    var qlOff = 0
                    var qhOff = 0
                    var scOff = 0
                    for _ in 0..<2 {  // n in {0, 128}
                        for l in 0..<32 {
                            let isub = l / 16
                            let qhB = qhAll[qhOff + l]
                            let qlL  = qlAll[qlOff + l]
                            let qlH  = qlAll[qlOff + l + 32]
                            let v0 = Int(qlL & 0x0f) | ((Int(qhB >> 0) & 0x3) << 4)
                            let v1 = Int(qlH & 0x0f) | ((Int(qhB >> 2) & 0x3) << 4)
                            let v2 = Int(qlL >> 4)   | ((Int(qhB >> 4) & 0x3) << 4)
                            let v3 = Int(qlH >> 4)   | ((Int(qhB >> 6) & 0x3) << 4)
                            out[outBase + l + 0]  = d * Float(scAll[scOff + isub + 0]) * Float(v0 - 32)
                            out[outBase + l + 32] = d * Float(scAll[scOff + isub + 2]) * Float(v1 - 32)
                            out[outBase + l + 64] = d * Float(scAll[scOff + isub + 4]) * Float(v2 - 32)
                            out[outBase + l + 96] = d * Float(scAll[scOff + isub + 6]) * Float(v3 - 32)
                        }
                        outBase += 128
                        qlOff   += 64
                        qhOff   += 32
                        scOff   += 8
                    }
                }
            }
            return MLXArray(out, info.shape)
        case .q8_K:
            // K-quant: 256-element super-blocks, 292 bytes each.
            //   4 bytes  d        — fp32 super-block scale (not fp16!)
            // 256 bytes  qs       — 256 int8 values
            //  32 bytes  bsums    — 16 int16 partial-sums per sub-block
            //                       (matmul accelerator — ignored for
            //                        plain dequant)
            //
            // Trivially: x = d * qs[i]. Q8_K is primarily a matmul
            // intermediate format; rare in distributed dumps.
            precondition(nElems % 256 == 0, "Q8_K expects multiple-of-256 element count (got \(nElems))")
            let blockBytes = 4 + 256 + 32
            let nBlocks = nElems / 256
            guard raw.count >= base + nBlocks * blockBytes else {
                throw GGUFError.truncated("q8_K tensor \(info.name)")
            }
            var out = [Float](repeating: 0, count: nElems)
            raw.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let bytesBase = ptr.baseAddress!.advanced(by: base)
                for b in 0..<nBlocks {
                    let blockStart = bytesBase.advanced(by: b * blockBytes)
                    let d = blockStart.load(as: Float.self)  // fp32, little-endian
                    let qs = blockStart.advanced(by: 4).assumingMemoryBound(to: Int8.self)
                    for i in 0..<256 {
                        out[b * 256 + i] = d * Float(qs[i])
                    }
                }
            }
            return MLXArray(out, info.shape)
        }
    }

    /// IEEE-754 half → float (scalar). MLX has a vectorised half-to-f32
    /// kernel but we read directly into a Float array for the simple
    /// path; the bottleneck is the file I/O, not this scalar loop.
    @inline(__always)
    public static func halfToFloat(_ h: UInt16) -> Float {
        let sign  = UInt32(h & 0x8000) << 16
        let exp16 = (h >> 10) & 0x1f
        let mant  = h & 0x3ff
        if exp16 == 0 {
            if mant == 0 {
                return Float(bitPattern: sign)
            }
            // Subnormal: renormalise.
            var m = UInt32(mant)
            var e: UInt32 = 0
            while (m & 0x400) == 0 { m <<= 1; e += 1 }
            m &= 0x3ff
            let bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13)
            return Float(bitPattern: bits)
        } else if exp16 == 0x1f {
            // Inf / NaN.
            let bits = sign | (0xff << 23) | (UInt32(mant) << 13)
            return Float(bitPattern: bits)
        }
        let e = UInt32(exp16) + (127 - 15)
        let bits = sign | (e << 23) | (UInt32(mant) << 13)
        return Float(bitPattern: bits)
    }
}

// ============================================================================
// Cursor — little-endian byte reader with bounds-checking. Kept private
// to this module; identical pattern to AWQReader / GPTQReader.
// ============================================================================

private struct Cursor {
    let data: Data
    var position: Int = 0

    mutating func readU8() throws -> UInt8 {
        guard position + 1 <= data.count else { throw GGUFReader.GGUFError.truncated("u8") }
        let v = data[data.startIndex + position]
        position += 1
        return v
    }
    mutating func readBool() throws -> Bool { return (try readU8()) != 0 }
    mutating func readI8() throws -> Int8 { return Int8(bitPattern: try readU8()) }
    mutating func readU16() throws -> UInt16 {
        guard position + 2 <= data.count else { throw GGUFReader.GGUFError.truncated("u16") }
        let v = data.withUnsafeBytes { ptr -> UInt16 in
            ptr.loadUnaligned(fromByteOffset: position, as: UInt16.self).littleEndian
        }
        position += 2; return v
    }
    mutating func readI16() throws -> Int16 { return Int16(bitPattern: try readU16()) }
    mutating func readU32() throws -> UInt32 {
        guard position + 4 <= data.count else { throw GGUFReader.GGUFError.truncated("u32") }
        let v = data.withUnsafeBytes { ptr -> UInt32 in
            ptr.loadUnaligned(fromByteOffset: position, as: UInt32.self).littleEndian
        }
        position += 4; return v
    }
    mutating func readI32() throws -> Int32 { return Int32(bitPattern: try readU32()) }
    mutating func readU64() throws -> UInt64 {
        guard position + 8 <= data.count else { throw GGUFReader.GGUFError.truncated("u64") }
        let v = data.withUnsafeBytes { ptr -> UInt64 in
            ptr.loadUnaligned(fromByteOffset: position, as: UInt64.self).littleEndian
        }
        position += 8; return v
    }
    mutating func readI64() throws -> Int64 { return Int64(bitPattern: try readU64()) }
    mutating func readF32() throws -> Float {
        let bits = try readU32()
        return Float(bitPattern: bits)
    }
    mutating func readF64() throws -> Double {
        let bits = try readU64()
        return Double(bitPattern: bits)
    }
    mutating func readGGUFString() throws -> String {
        let length = Int(try readU64())
        guard position + length <= data.count else { throw GGUFReader.GGUFError.truncated("string body \(length)B") }
        let bytes = data.subdata(in: (data.startIndex + position)..<(data.startIndex + position + length))
        position += length
        return String(data: bytes, encoding: .utf8) ?? ""
    }

    mutating func readMetaValue(type: UInt32) throws -> Any {
        switch type {
        case 0:  return try readU8()
        case 1:  return try readI8()
        case 2:  return try readU16()
        case 3:  return try readI16()
        case 4:  return try readU32()
        case 5:  return try readI32()
        case 6:  return try readF32()
        case 7:  return try readBool()
        case 8:  return try readGGUFString()
        case 9:
            // array — element_type (u32), count (u64), then elements.
            let elemType = try readU32()
            let count = Int(try readU64())
            var arr: [Any] = []
            arr.reserveCapacity(count)
            for _ in 0..<count { arr.append(try readMetaValue(type: elemType)) }
            return arr
        case 10: return try readU64()
        case 11: return try readI64()
        case 12: return try readF64()
        default: throw GGUFReader.GGUFError.unsupportedMetaType(type)
        }
    }
}
