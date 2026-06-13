import Foundation
import MLX
import MLXNN
import MLXRandom
import TinyGPTIO
import XCTest
@testable import TinyGPTModel

/// Numerics gate for the MLX-native quantized HF load path (#305 phase 2).
///
/// `mlx_lm convert -q` ships packed-uint32 weights + `.scales`/`.biases`
/// peers, self-described by config.json's `quantization` block.
/// `HFModelLoader` must (a) swap each quantized Linear for a real
/// `QuantizedLinear`, (b) dequantise quantized Embeddings to dense fp32,
/// and (c) produce logits that match BOTH the fp32 master (within int8
/// quantization error) and the in-memory `--quantize` serve path
/// (tightly — identical affine math from the same fp32 weights).
///
/// No-quality-regression rule: this is the automated gate for the
/// quantized HFModel path. It runs on a tiny random-weights checkpoint
/// built in-test — no model download, no multi-GB load.
final class HFQuantizedLoadTests: XCTestCase {

    // Tiny Llama-style config. Every quantized axis (Linear input dims,
    // embedding dims) is a multiple of the group size (64).
    private let configJSON: [String: Any] = [
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 96,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
    ]

    /// Full random fp32 tensor set for the config above, HF naming.
    private func randomWeights() -> [String: MLXArray] {
        MLXRandom.seed(0x5305)
        var t: [String: MLXArray] = [:]
        func w(_ name: String, _ shape: [Int]) {
            t[name] = MLXRandom.normal(shape) * 0.08
        }
        w("model.embed_tokens.weight", [96, 64])
        for i in 0..<2 {
            let p = "model.layers.\(i)."
            t[p + "input_layernorm.weight"] = MLXRandom.uniform(low: 0.8, high: 1.2, [64])
            t[p + "post_attention_layernorm.weight"] = MLXRandom.uniform(low: 0.8, high: 1.2, [64])
            w(p + "self_attn.q_proj.weight", [64, 64])
            w(p + "self_attn.k_proj.weight", [32, 64])
            w(p + "self_attn.v_proj.weight", [32, 64])
            w(p + "self_attn.o_proj.weight", [64, 64])
            w(p + "mlp.gate_proj.weight", [128, 64])
            w(p + "mlp.up_proj.weight", [128, 64])
            w(p + "mlp.down_proj.weight", [64, 128])
        }
        t["model.norm.weight"] = MLXRandom.uniform(low: 0.8, high: 1.2, [64])
        w("lm_head.weight", [96, 64])
        return t
    }

    /// Write a config.json + model.safetensors checkpoint dir in tmp.
    private func writeCheckpoint(tensors: [String: MLXArray],
                                 config: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hf-quant-gate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cfgData = try JSONSerialization.data(withJSONObject: config)
        try cfgData.write(to: dir.appendingPathComponent("config.json"))
        try MLX.save(arrays: tensors, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    /// MLX-pack every Linear/Embedding weight the way `mlx_lm convert -q`
    /// does; norms pass through fp32.
    private func quantizedTensors(_ fp: [String: MLXArray],
                                  bits: Int, groupSize: Int) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (name, w) in fp {
            let isTarget = name.hasSuffix(".weight") && w.ndim == 2
            if isTarget {
                let (wq, scales, biases) = MLX.quantized(
                    w, groupSize: groupSize, bits: bits)
                let base = String(name.dropLast(".weight".count))
                out[base + ".weight"] = wq
                out[base + ".scales"] = scales
                if let biases { out[base + ".biases"] = biases }
            } else {
                out[name] = w
            }
        }
        return out
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.flattened().asType(.float32)
        let y = b.flattened().asType(.float32)
        let num = (x * y).sum()
        let den = MLX.sqrt((x * x).sum()) * MLX.sqrt((y * y).sum())
        return (num / den).item(Float.self)
    }

    /// int8 g64 — the shipping default ("free lunch" per the serve
    /// measurements). Gates: structure, fp32 fidelity, serve-path parity.
    func test_quantizedLoad_int8_matchesFp32AndServePath() throws {
        let fp = randomWeights()
        let fpDir = try writeCheckpoint(tensors: fp, config: configJSON)
        defer { try? FileManager.default.removeItem(at: fpDir) }

        var qConfig = configJSON
        qConfig["quantization"] = ["group_size": 64, "bits": 8]
        let qDir = try writeCheckpoint(
            tensors: quantizedTensors(fp, bits: 8, groupSize: 64), config: qConfig)
        defer { try? FileManager.default.removeItem(at: qDir) }

        let fpLoad = try HFModelLoader.load(from: fpDir)
        XCTAssertNil(fpLoad.quantization, "fp checkpoint must not report quantization")
        let qLoad = try HFModelLoader.load(from: qDir)
        XCTAssertEqual(qLoad.quantization?.bits, 8)
        XCTAssertEqual(qLoad.quantization?.groupSize, 64)

        // Structure: Linears swapped for QuantizedLinear; embedding stays
        // a plain (dequantised-dense) Embedding.
        XCTAssertTrue(qLoad.model.blocks[0].attn.qProj is QuantizedLinear)
        XCTAssertTrue(qLoad.model.blocks[1].attn.oProj is QuantizedLinear)
        XCTAssertTrue(qLoad.model.blocks[1].mlp.fcDown is QuantizedLinear)
        XCTAssertTrue(qLoad.model.lmHead is QuantizedLinear)
        XCTAssertFalse(qLoad.model.tokenEmbedding is QuantizedEmbedding)

        let idx = MLXArray([Int32]([1, 5, 9, 42, 7, 88, 3, 64]), [1, 8])
        let fpLogits = fpLoad.model(idx)
        let qLogits = qLoad.model(idx)
        eval(fpLogits, qLogits)

        // Gate 1: int8 g64 vs the fp32 master.
        let cosFp = cosine(fpLogits, qLogits)
        XCTAssertGreaterThanOrEqual(
            cosFp, 0.999, "quantized-load logits drifted from fp32 (cos=\(cosFp))")

        // Gate 2: parity with the serve-path in-memory MLXNN.quantize —
        // same affine int8 math over the same fp32 weights, so the two
        // quantized paths must agree tightly.
        MLXNN.quantize(model: fpLoad.model, groupSize: 64, bits: 8)
        let serveLogits = fpLoad.model(idx)
        eval(serveLogits)
        let cosServe = cosine(serveLogits, qLogits)
        XCTAssertGreaterThanOrEqual(
            cosServe, 0.9999,
            "native packed load disagrees with in-memory quantize (cos=\(cosServe))")
        let maxAbs = MLX.abs(serveLogits - qLogits).max().item(Float.self)
        XCTAssertLessThanOrEqual(
            maxAbs, 1e-3,
            "native packed load disagrees with in-memory quantize (maxAbs=\(maxAbs))")
    }

    /// int4 g64 also loads + runs (it fails quality gates on real planners,
    /// but the loader must still handle the format — smoke only).
    func test_quantizedLoad_int4_loadsAndRuns() throws {
        let fp = randomWeights()
        var qConfig = configJSON
        qConfig["quantization"] = ["group_size": 64, "bits": 4]
        let qDir = try writeCheckpoint(
            tensors: quantizedTensors(fp, bits: 4, groupSize: 64), config: qConfig)
        defer { try? FileManager.default.removeItem(at: qDir) }

        let qLoad = try HFModelLoader.load(from: qDir)
        XCTAssertEqual(qLoad.quantization?.bits, 4)
        XCTAssertTrue(qLoad.model.blocks[0].mlp.fcGate is QuantizedLinear)

        let idx = MLXArray([Int32]([2, 4, 8, 16]), [1, 4])
        let logits = qLoad.model(idx)
        eval(logits)
        let maxAbs = MLX.abs(logits).max().item(Float.self)
        XCTAssertTrue(maxAbs.isFinite, "int4 logits are not finite")
    }
}
