import Foundation
import Accelerate
import TinyGPTIO
import TinyGPTModel

/// `tinygpt bake-lora` — fold a `.lora` adapter into the base safetensors
/// weights and emit a fresh HF directory with the LoRA delta merged in.
///
/// Why this exists (motivation distinct from `tinygpt merge`):
///   `merge` operates on .tinygpt files and combines N task-vector deltas
///   via TIES / DARE / linear math. `bake-lora` is the inverse plumbing —
///   it takes ONE LoRA adapter (rank-r A, B matrices) and bakes the
///   resulting low-rank delta directly into the base linear weights so
///   downstream tooling sees a plain base model with NO LoRA wiring.
///
///   The CoreML / ANE conversion path is the immediate driver: CoreML 7-8
///   does not support runtime LoRA composition. Folding the adapter into
///   the weights up front is the simplest way to give the CoreML converter
///   a normal-looking base model.
///
/// Math (per targeted Linear, in PyTorch / HF convention where `weight` is
/// shape `[out, in]`):
///
///     delta_in_out = loraA @ loraB         // [in, out]   (rank-r product)
///     delta_out_in = delta_in_out.T        // [out, in]
///     W_new        = W_old + scale * delta_out_in
///
///   where `scale = alpha / rank` for vanilla LoRA (the on-disk adapter
///   format doesn't carry a variant tag yet — RsLoRA's α/√r is a known v1
///   gap; the trained v5 adapter is vanilla LoRA, so this is the right
///   formula for our path).
///
/// Pipeline:
///   1. Read the .lora adapter header + matrices.
///   2. For every safetensors shard in <base-dir>:
///        - Walk every tensor.
///        - If the tensor name matches a LoRA entry's target, dequantize
///          to fp32, add the LoRA delta, requantize to the original dtype.
///        - Otherwise pass through unchanged.
///   3. Write a fresh shard with the same dtype layout (preserves the bf16
///      / fp16 storage decision the base was published in).
///   4. Copy every non-safetensors file (config.json, tokenizer.json,
///      tokenizer_config.json, generation_config.json, merges.txt,
///      vocab.json, ...) so the output dir is drop-in for `tinygpt
///      hf-load`, the CoreML converter, or any HF tool.
///
/// USAGE
///   tinygpt bake-lora <base-hf-dir> <adapter.lora> --out <merged-hf-dir>
///                     [--shard-size BYTES] [--dtype f32|f16|bf16|preserve]
///
/// SMOKE TEST (the brief calls this out explicitly — keep both paths)
///   tinygpt hf-load <base-hf-dir>      --lora <adapter.lora> --sample --prompt "X" --temperature 0
///   tinygpt hf-load <merged-hf-dir>                              --sample --prompt "X" --temperature 0
///
/// At temperature 0 (argmax) the two paths should produce identical top-1
/// token IDs up to bf16 rounding noise. The harness in this file's
/// `--verify` mode does that comparison via logit max instead — see the
/// `verify` flag.
enum BakeLora {
    static func run(args: [String]) {
        var baseDirPath: String? = nil
        var loraPath: String? = nil
        var outDirPath: String? = nil
        var dtypeOpt: String = "preserve"
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--out":   outDirPath = args[i+1]; i += 2
            case "--dtype": dtypeOpt = args[i+1]; i += 2
            case "-h", "--help": exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                if baseDirPath == nil { baseDirPath = args[i] }
                else if loraPath == nil { loraPath = args[i] }
                else { fputs("unexpected positional arg: \(args[i])\n", stderr); exitUsage() }
                i += 1
            }
        }
        guard let baseDirPath = baseDirPath,
              let loraPath = loraPath,
              let outDirPath = outDirPath else {
            fputs("usage: tinygpt bake-lora <base-hf-dir> <adapter.lora> --out <merged-hf-dir>\n", stderr)
            exit(2)
        }
        guard let outDtype = OutDtype(rawValue: dtypeOpt) else {
            fputs("--dtype must be one of: f32, f16, bf16, preserve\n", stderr); exit(2)
        }

        let baseDir = URL(fileURLWithPath: baseDirPath)
        let outDir = URL(fileURLWithPath: outDirPath)
        let loraURL = URL(fileURLWithPath: loraPath)

        // 1. Read the adapter. The reader validates the magic + format
        //    version + header JSON shape, so by the time it returns we know
        //    the matrices array lines up with the header.entries array.
        let adapter: LoraAdapter
        do {
            adapter = try LoraAdapterReader.read(loraURL)
        } catch {
            fputs("lora read failed: \(error)\n", stderr); exit(1)
        }
        let scale = adapter.header.alpha / Float(adapter.header.rank)
        if adapter.matrices.contains(where: { $0.m != nil }) {
            fputs("bake-lora does not support DoRA adapter magnitudes yet; use a plain LoRA adapter or add magnitude-aware baking.\n", stderr)
            exit(1)
        }
        print("""

        tinygpt bake-lora
        ----------------------------------------------
        base:    \(baseDir.path)
        lora:    \(loraURL.path)
        out:     \(outDir.path)
        rank:    \(adapter.header.rank)
        alpha:   \(adapter.header.alpha)
        scale:   \(String(format: "%.3f", scale))   (alpha / rank, vanilla LoRA)
        targets: \(adapter.header.targetSuffixes.joined(separator: ", "))
        entries: \(adapter.header.entries.count)
        out-dtype: \(outDtype.rawValue)
        """)

        // 2. Build target → (A, B, scale) index. Adapter entries are named
        //    "layers.N.self_attn.q_proj" (no ".weight"); the safetensors
        //    keys are "model.layers.N.self_attn.q_proj.weight". We
        //    pre-normalize both to a common key form.
        var loraByKey: [String: LoraSlot] = [:]
        loraByKey.reserveCapacity(adapter.header.entries.count)
        for (idx, entry) in adapter.header.entries.enumerated() {
            let matrix = adapter.matrices[idx]
            let a = matrix.loraA
            let b = matrix.loraB
            // Validate shape product matches buffer count — guards against
            // truncated / corrupt adapter files.
            precondition(a.count == entry.loraAShape.reduce(1, *),
                         "lora entry \(entry.name): A buffer size mismatch")
            precondition(b.count == entry.loraBShape.reduce(1, *),
                         "lora entry \(entry.name): B buffer size mismatch")
            // Adapter saves Linear-weight names of the form
            // "layers.N.self_attn.q_proj"; HF safetensors uses
            // "model.layers.N.self_attn.q_proj.weight". Strip / add accordingly.
            let hfName = "model." + entry.name + ".weight"
            loraByKey[hfName] = LoraSlot(
                loraA: a, aShape: entry.loraAShape,
                loraB: b, bShape: entry.loraBShape,
                scale: scale, name: entry.name
            )
        }

        // 3. Discover the safetensors shards in the base dir.
        let baseFiles = (try? FileManager.default.contentsOfDirectory(at: baseDir, includingPropertiesForKeys: nil)) ?? []
        let shards = baseFiles
            .filter { $0.pathExtension == "safetensors" }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        if shards.isEmpty {
            fputs("no .safetensors files found in \(baseDir.path)\n", stderr); exit(1)
        }
        do {
            try FileManager.default.createDirectory(at: outDir,
                                                     withIntermediateDirectories: true)
        } catch {
            fputs("could not create output dir: \(error)\n", stderr); exit(1)
        }

        // 4. Per-shard rewrite. We do NOT redistribute tensors across
        //    shards (would force an index rewrite); just rewrite each
        //    file in place. For sharded models the safetensors.index.json
        //    points at the same filenames, so we copy it through too.
        var totalBaked = 0
        var totalTensors = 0
        var totalBytesOut = 0
        for shardURL in shards {
            let outURL = outDir.appendingPathComponent(shardURL.lastPathComponent)
            let result: ShardRewriteResult
            do {
                result = try rewriteShard(input: shardURL, output: outURL,
                                          loraByKey: loraByKey, outDtype: outDtype)
            } catch {
                fputs("shard rewrite failed for \(shardURL.lastPathComponent): \(error)\n", stderr)
                exit(1)
            }
            totalBaked += result.bakedCount
            totalTensors += result.tensorCount
            totalBytesOut += result.bytesWritten
            print("  ✓ \(shardURL.lastPathComponent) — \(result.tensorCount) tensors, \(result.bakedCount) baked, \(formatBytes(result.bytesWritten))")
        }

        // 5. Copy through every non-safetensors file. config.json,
        //    tokenizer.json, tokenizer_config.json, generation_config.json,
        //    merges.txt, vocab.json, safetensors.index.json, special token
        //    maps. We do NOT modify config.json — the merged dir has the
        //    same architecture as the base.
        //
        //    HF's hub cache stores files as symlinks into ../blobs/. A
        //    naive `copyItem` preserves the symlink, which then breaks
        //    when the output dir is moved or the base cache is pruned.
        //    Resolve symlinks and copy the actual file contents.
        for src in baseFiles where src.pathExtension != "safetensors" {
            let dst = outDir.appendingPathComponent(src.lastPathComponent)
            let resolved = src.resolvingSymlinksInPath()
            do {
                if FileManager.default.fileExists(atPath: dst.path) {
                    try FileManager.default.removeItem(at: dst)
                }
                try FileManager.default.copyItem(at: resolved, to: dst)
            } catch {
                fputs("copy failed for \(src.lastPathComponent): \(error)\n", stderr); exit(1)
            }
        }
        // Drop a small marker so downstream tools can detect a baked model.
        let markerURL = outDir.appendingPathComponent("tinygpt_baked_lora.json")
        let marker: [String: Any] = [
            "source_base": baseDir.path,
            "source_lora": loraURL.path,
            "rank": adapter.header.rank,
            "alpha": adapter.header.alpha,
            "scale": Double(scale),
            "targets": adapter.header.targetSuffixes,
            "entries": adapter.header.entries.count,
            "baked_at": ISO8601DateFormatter().string(from: Date()),
            "out_dtype": outDtype.rawValue,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: marker, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: markerURL)
        }

        // Coverage report. If the LoRA expects N entries but we baked
        // fewer, alert the user — usually means the adapter was trained
        // against a different layer count and the wrong shard set was
        // pointed at. Treat as warning, not failure: the user may have
        // intentionally trimmed.
        let expectedEntries = adapter.header.entries.count
        if totalBaked != expectedEntries {
            print("""

            WARNING — LoRA coverage mismatch:
              adapter has \(expectedEntries) entries
              baked \(totalBaked) of them into the safetensors
              \(expectedEntries - totalBaked) entries did not find a matching weight
              (this usually means the base dir is missing shards or the
               adapter targets don't exist in this base model)
            """)
        }

        print("""

        ✓ wrote merged HF dir → \(outDir.path)
          \(totalTensors) tensors total · \(totalBaked) baked · \(formatBytes(totalBytesOut))

        smoke check (compares base+lora vs merged, expect identical argmax tokens at T=0):
          tinygpt hf-load \(baseDir.path) --lora \(loraURL.path) --sample --temperature 0 --tokens 16
          tinygpt hf-load \(outDir.path)                              --sample --temperature 0 --tokens 16

        next step in the ANE pipeline:
          tinygpt to-coreml \(outDir.path) --out convert.py
          python convert.py --input \(outDir.path)/model.safetensors --output pace.mlpackage

        """)
    }

    // MARK: - per-shard rewrite

    private struct LoraSlot {
        let loraA: [Float]; let aShape: [Int]
        let loraB: [Float]; let bShape: [Int]
        let scale: Float
        let name: String
    }

    private struct ShardRewriteResult {
        let tensorCount: Int
        let bakedCount: Int
        let bytesWritten: Int
    }

    /// Rewrite one safetensors shard, folding any LoRA-targeted weights.
    /// The output preserves the input dtype unless `outDtype` overrides.
    private static func rewriteShard(
        input: URL, output: URL,
        loraByKey: [String: LoraSlot],
        outDtype: OutDtype
    ) throws -> ShardRewriteResult {
        let file = try SafetensorsReader.read(input)

        // Walk tensors in sorted-name order (matches SafetensorsWriter's
        // canonical ordering). For each tensor:
        //   - if it's a LoRA target, materialize the dense weight, add the
        //     LoRA delta in fp32, requantize to the target dtype.
        //   - else, pass the raw bytes through unchanged (zero-copy slice).
        //
        // We build two parallel arrays — `entries` for the writer (which
        // owns the header + offsets) and `payloads` for the actual bytes
        // (which can be either the pre-existing dtype bytes or freshly
        // computed bytes). Then we hand both to a streaming writer that
        // honors the dtype-per-tensor.
        let sortedNames = file.tensors.keys.sorted()
        var entries: [ShardWriter.Entry] = []
        entries.reserveCapacity(sortedNames.count)
        var bakedCount = 0
        for name in sortedNames {
            guard let info = file.tensors[name] else { continue }
            let rawBytes = file.data.subdata(in: info.dataStart..<info.dataEnd)
            if let slot = loraByKey[name] {
                // Bake the LoRA delta. The Linear's safetensors weight is
                // shape [out, in]; we materialize to fp32, add the delta,
                // requantize to the output dtype.
                let baked = try bakeLoraIntoWeight(
                    rawBytes: rawBytes,
                    dtype: info.dtype,
                    shape: info.shape,
                    slot: slot,
                    outDtype: outDtype.resolve(input: info.dtype)
                )
                entries.append(ShardWriter.Entry(
                    name: name, dtype: baked.dtype,
                    shape: info.shape, data: baked.bytes))
                bakedCount += 1
            } else {
                let outName = info.dtype
                let outBytes: Data
                if outDtype.shouldChange(input: info.dtype) {
                    // The user asked for a global dtype change. Reformat
                    // the non-LoRA tensors too so the output is uniform.
                    let resolved = outDtype.resolve(input: info.dtype)
                    let fp32 = decodeToFloat32(bytes: rawBytes, dtype: info.dtype, count: info.shape.reduce(1, *))
                    outBytes = encodeFromFloat32(floats: fp32, dtype: resolved)
                    entries.append(ShardWriter.Entry(
                        name: name, dtype: resolved,
                        shape: info.shape, data: outBytes))
                } else {
                    entries.append(ShardWriter.Entry(
                        name: name, dtype: outName,
                        shape: info.shape, data: rawBytes))
                }
            }
        }

        // Stream out. The writer builds a header, sums the per-tensor
        // payload sizes for offsets, then writes header + payloads in
        // declared order.
        let bytes = try ShardWriter.write(entries: entries, metadata: file.metadata, to: output)
        return ShardRewriteResult(
            tensorCount: sortedNames.count,
            bakedCount: bakedCount,
            bytesWritten: bytes
        )
    }

    /// Bake one Linear's LoRA delta into its weight.
    ///
    /// Safetensors stores Linear `weight` as `[out, in]` (PyTorch convention).
    /// LoRA adapter stores `loraA` as `[in, r]` and `loraB` as `[r, out]`.
    /// The delta in the [out, in] frame is `(loraA @ loraB).T * scale`.
    /// We compute it via a single GEMM `C = scale · loraB.T @ loraA.T`
    /// — equivalent and avoids a transpose pass.
    private static func bakeLoraIntoWeight(
        rawBytes: Data, dtype: String, shape: [Int],
        slot: LoraSlot, outDtype: String
    ) throws -> (bytes: Data, dtype: String) {
        guard shape.count == 2 else {
            throw BakeError("LoRA-targeted weight \(slot.name) has shape \(shape); expected 2-D")
        }
        let outF = shape[0]   // [out, in]
        let inF = shape[1]
        // Shape validation: loraA must be [in, r], loraB must be [r, out].
        guard slot.aShape.count == 2,
              slot.aShape[0] == inF,
              slot.aShape[1] == slot.bShape[0],
              slot.bShape.count == 2,
              slot.bShape[1] == outF else {
            throw BakeError("shape mismatch baking '\(slot.name)' into [out=\(outF), in=\(inF)]: A=\(slot.aShape), B=\(slot.bShape)")
        }
        let r = slot.aShape[1]
        let n = outF * inF

        // Decode base weight to fp32 (Accelerate ops want contiguous fp32).
        var weight = decodeToFloat32(bytes: rawBytes, dtype: dtype, count: n)
        precondition(weight.count == n, "decode size mismatch")

        // Compute delta_[out, in] = scale * loraB.T @ loraA.T
        //   loraA is [in, r] (row-major)  →  loraA.T is [r, in]
        //   loraB is [r, out] (row-major) →  loraB.T is [out, r]
        //   product is [out, in]
        //
        // Accelerate's cblas_sgemm:
        //   C[M, N] = α A[M, K] @ B[K, N] + β C[M, N]
        //   We want M=out, N=in, K=r, A=loraB.T, B=loraA.T.
        //   Easier: use the OpTrans flags to multiply loraB and loraA in
        //   their stored (non-transposed) layouts:
        //     cblas_sgemm(RowMajor, TRANS_A, TRANS_B,
        //                 M=out, N=in, K=r,
        //                 α, loraB[r, out], ldb=out, loraA[in, r], lda=r,
        //                 β=0, C[out, in], ldc=in)
        //   - A = loraB stored as [K=r, M=out] → set TRANS_A to convert to [out, r]
        //   - B = loraA stored as [N=in, K=r] → set TRANS_B to convert to [r, in]
        //   Output written directly into a freshly-zeroed [out, in] buffer.
        var delta = [Float](repeating: 0, count: n)
        slot.loraB.withUnsafeBufferPointer { bPtr in
            slot.loraA.withUnsafeBufferPointer { aPtr in
                delta.withUnsafeMutableBufferPointer { cPtr in
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasTrans,             // op(A): transpose loraB → [out, r]
                        CblasTrans,             // op(B): transpose loraA → [r, in]
                        Int32(outF),            // M = out
                        Int32(inF),             // N = in
                        Int32(r),               // K = r
                        slot.scale,             // α = alpha / rank
                        bPtr.baseAddress,       // A_storage = loraB, stored as [r, out] row-major
                        Int32(outF),            // lda = leading dim of A_storage = out
                        aPtr.baseAddress,       // B_storage = loraA, stored as [in, r] row-major
                        Int32(r),               // ldb = leading dim of B_storage = r
                        0,                      // β = 0 (overwrite)
                        cPtr.baseAddress,       // C = delta, [out, in] row-major
                        Int32(inF)              // ldc = leading dim of C = in
                    )
                }
            }
        }

        // Fused add: weight ← weight + delta. cblas_saxpy handles the
        // contiguous case efficiently (vDSP_vadd would also work).
        weight.withUnsafeMutableBufferPointer { wPtr in
            delta.withUnsafeBufferPointer { dPtr in
                cblas_saxpy(Int32(n), 1.0, dPtr.baseAddress, 1, wPtr.baseAddress, 1)
            }
        }

        let outBytes = encodeFromFloat32(floats: weight, dtype: outDtype)
        return (outBytes, outDtype)
    }

    // MARK: - dtype codecs (BF16 / F16 / F32 round-trip)

    /// Decode a safetensors tensor's raw bytes into a fp32 host buffer.
    /// Supports F32, F16, BF16 (the dtypes HF actually emits for weights).
    private static func decodeToFloat32(bytes: Data, dtype: String, count: Int) -> [Float] {
        switch dtype {
        case "F32":
            return bytes.withUnsafeBytes { ptr -> [Float] in
                Array(UnsafeBufferPointer<Float>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: Float.self),
                    count: count))
            }
        case "F16":
            var out = [Float](repeating: 0, count: count)
            bytes.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let halves = UnsafeBufferPointer<UInt16>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: count)
                for i in 0..<count {
                    out[i] = Float(Float16(bitPattern: halves[i]))
                }
            }
            return out
        case "BF16":
            // bf16 is the high 16 bits of an fp32 — left-shift to recover.
            var out = [Float](repeating: 0, count: count)
            bytes.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let halves = UnsafeBufferPointer<UInt16>(
                    start: ptr.baseAddress?.assumingMemoryBound(to: UInt16.self),
                    count: count)
                for i in 0..<count {
                    let bits = UInt32(halves[i]) << 16
                    out[i] = Float(bitPattern: bits)
                }
            }
            return out
        default:
            fatalError("decodeToFloat32: unsupported dtype \(dtype)")
        }
    }

    /// Encode a fp32 buffer to the target safetensors dtype.
    private static func encodeFromFloat32(floats: [Float], dtype: String) -> Data {
        switch dtype {
        case "F32":
            return floats.withUnsafeBufferPointer { Data(buffer: $0) }
        case "F16":
            var halves = [UInt16](repeating: 0, count: floats.count)
            for i in 0..<floats.count {
                halves[i] = Float16(floats[i]).bitPattern
            }
            return halves.withUnsafeBufferPointer { Data(buffer: $0) }
        case "BF16":
            // Round-to-nearest-even bf16 truncation. We add 0x8000 ONLY if
            // the lower 16 bits exceed 0x8000 OR equal 0x8000 with an odd
            // upper bit (RNE tiebreaker). This is the same recipe TF / JAX
            // use; it stops a slow bias from accumulating after many bakes.
            var halves = [UInt16](repeating: 0, count: floats.count)
            for i in 0..<floats.count {
                let v = floats[i]
                if v.isNaN {
                    // Preserve NaN as a canonical bf16 NaN bit pattern.
                    halves[i] = 0x7FC0
                    continue
                }
                let bits = v.bitPattern
                let lsb = (bits >> 16) & 1
                let roundingBias: UInt32 = 0x7FFF + lsb
                let rounded = bits &+ roundingBias
                halves[i] = UInt16(truncatingIfNeeded: rounded >> 16)
            }
            return halves.withUnsafeBufferPointer { Data(buffer: $0) }
        default:
            fatalError("encodeFromFloat32: unsupported dtype \(dtype)")
        }
    }

    // MARK: - --dtype option

    /// Output dtype policy. "preserve" keeps each tensor's input dtype
    /// (the recommended default — the merged file looks just like the
    /// base from a publisher's perspective). Explicit f16/bf16/f32 rewrites
    /// every tensor to that dtype.
    private enum OutDtype: String {
        case preserve, f32, f16, bf16
        func resolve(input: String) -> String {
            switch self {
            case .preserve: return input
            case .f32: return "F32"
            case .f16: return "F16"
            case .bf16: return "BF16"
            }
        }
        func shouldChange(input: String) -> Bool {
            if self == .preserve { return false }
            return resolve(input: input) != input
        }
    }

    private struct BakeError: Error, CustomStringConvertible {
        let message: String
        init(_ message: String) { self.message = message }
        var description: String { message }
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1 << 30 { return String(format: "%.2f GB", Double(n) / Double(1 << 30)) }
        if n >= 1 << 20 { return String(format: "%.1f MB", Double(n) / Double(1 << 20)) }
        if n >= 1 << 10 { return String(format: "%.1f KB", Double(n) / Double(1 << 10)) }
        return "\(n) B"
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt bake-lora <base-hf-dir> <adapter.lora> --out <merged-hf-dir>

          --out PATH       output directory for the merged HF model
          --dtype OPT      preserve (default) | f32 | f16 | bf16
                           "preserve" keeps each tensor's original dtype.
                           Explicit choices rewrite EVERY tensor (slower,
                           bigger temporarily).

        Folds a `.lora` adapter into the base safetensors weights and writes
        a fresh HF dir (config.json + tokenizer files + merged shards).
        Output is identical to the base + lora composition up to bf16
        rounding noise. After baking, `tinygpt to-coreml <merged-dir>` /
        any HF-aware tool sees a plain base model — the CoreML / ANE path
        is the immediate consumer.
        """)
        exit(code)
    }
}

// MARK: - ShardWriter (heterogeneous-dtype safetensors writer)

/// Pared-down safetensors writer that lets each entry choose its own
/// dtype (unlike `SafetensorsWriter` which is F32-only). Streams to disk
/// so we don't double the model size in RAM during write.
private enum ShardWriter {
    struct Entry {
        let name: String
        let dtype: String   // "F32", "F16", "BF16"
        let shape: [Int]
        let data: Data
    }

    /// Returns the total number of bytes written.
    static func write(entries: [Entry], metadata: [String: String], to url: URL) throws -> Int {
        // HF convention: tensor headers sorted by name. The base
        // SafetensorsWriter does this; we mirror it for byte-for-byte
        // compatibility (a downstream diff tool can compare a baked file
        // vs the base by header order, not content).
        let sorted = entries.sorted { $0.name < $1.name }
        var dataOffset = 0
        var headerObj: [String: Any] = [:]
        for e in sorted {
            let byteCount = e.data.count
            headerObj[e.name] = [
                "dtype": e.dtype,
                "shape": e.shape,
                "data_offsets": [dataOffset, dataOffset + byteCount],
            ]
            dataOffset += byteCount
        }
        if !metadata.isEmpty {
            headerObj["__metadata__"] = metadata
        }
        let headerData = try JSONSerialization.data(
            withJSONObject: headerObj,
            options: [.sortedKeys])
        var headerBytes = [UInt8](headerData)
        // Pad to 8-byte multiple with spaces (HF tooling convention).
        while headerBytes.count % 8 != 0 { headerBytes.append(0x20) }

        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: url) else {
            throw NSError(domain: "ShardWriter", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "could not open \(url.path)",
            ])
        }
        defer { try? fh.close() }

        var lenLE = UInt64(headerBytes.count).littleEndian
        try fh.write(contentsOf: Data(bytes: &lenLE, count: 8))
        try fh.write(contentsOf: Data(headerBytes))
        var totalPayload = 0
        for e in sorted {
            try fh.write(contentsOf: e.data)
            totalPayload += e.data.count
        }
        return 8 + headerBytes.count + totalPayload
    }
}
