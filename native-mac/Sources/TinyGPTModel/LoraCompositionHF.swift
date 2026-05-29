import Foundation
import MLX
import MLXNN

/// Multi-LoRA composition for `TinyGPTModelHF`. Parallel to
/// `LoraStackInjection` (which targets the from-scratch `TinyGPTModel`'s
/// LayerNorm + plain MLP), but using HF-native `@ModuleInfo` keys
/// — `self_attn`, `mlp`, `gate_proj`, `up_proj`, `down_proj`.
///
/// Sums all adapter deltas at forward time:
///
///     y = base(x) + Σ_k w_k * (x @ A_k @ B_k * scale_k)
///
/// All stacked adapters must share the same target Linears AND base
/// architecture; clear errors otherwise (the from-scratch comment about
/// "defer until someone asks" applies here too).
public enum LoraStackInjectionHF {
    public static func apply(_ adapters: [LoraAdapter], weights: [Float],
                              to model: TinyGPTModelHF) throws {
        precondition(adapters.count == weights.count,
                     "weights array length must match adapters")
        guard !adapters.isEmpty else { return }

        // Verify target consistency across adapters.
        let firstTargets = Set(adapters[0].header.targetSuffixes)
        for (i, a) in adapters.enumerated() {
            if Set(a.header.targetSuffixes) != firstTargets {
                throw NSError(domain: "TinyGPTLoRA", code: 5,
                              userInfo: [NSLocalizedDescriptionKey:
                                "adapter \(i) targets \(a.header.targetSuffixes) which differs from adapter 0's \(adapters[0].header.targetSuffixes)"])
            }
        }

        // Verify base config consistency.
        let cfg = model.config
        for (i, a) in adapters.enumerated() {
            let h = a.header
            guard h.baseLayers == cfg.nLayers,
                  h.baseDModel == cfg.dModel,
                  h.baseCtx == cfg.contextLength,
                  h.baseHeads == cfg.nHeads,
                  h.baseDMlp == cfg.dMlp else {
                throw NSError(domain: "TinyGPTLoRA", code: 4,
                              userInfo: [NSLocalizedDescriptionKey:
                                "adapter \(i) base config doesn't match loaded HF model"])
            }
        }

        // Per-adapter index into its matrix list — incremented every time
        // we install a Linear-target replacement.
        var matIdx: [Int] = Array(repeating: 0, count: adapters.count)
        var layersList: [NestedItem<String, Module>] = []
        let targetSuffixes = firstTargets
        for block in model.blocks {
            var attnEntries: [String: NestedItem<String, Module>] = [:]
            var mlpEntries: [String: NestedItem<String, Module>] = [:]

            // (name, linear, isAttn, isTarget)
            let projs: [(String, Linear, Bool, Bool)] = [
                ("q_proj",    block.attn.qProj,  true,  targetSuffixes.contains("q_proj")),
                ("k_proj",    block.attn.kProj,  true,  targetSuffixes.contains("k_proj")),
                ("v_proj",    block.attn.vProj,  true,  targetSuffixes.contains("v_proj")),
                ("o_proj",    block.attn.oProj,  true,  targetSuffixes.contains("o_proj")),
                ("gate_proj", block.mlp.fcGate,  false, targetSuffixes.contains("gate_proj")),
                ("up_proj",   block.mlp.fcUp,    false, targetSuffixes.contains("up_proj")),
                ("down_proj", block.mlp.fcDown,  false, targetSuffixes.contains("down_proj")),
            ]

            for (name, lin, isAttn, isTarget) in projs where isTarget {
                var slots: [StackedLoraLinear.Slot] = []
                for (k, adapter) in adapters.enumerated() {
                    let mi = matIdx[k]
                    let aShape = adapter.header.entries[mi].loraAShape
                    let bShape = adapter.header.entries[mi].loraBShape
                    let entry = adapter.matrices[mi]
                    let scale = adapter.header.alpha / Float(adapter.header.rank)
                    slots.append(.init(
                        loraA: MLXArray(entry.loraA, aShape),
                        loraB: MLXArray(entry.loraB, bShape),
                        scale: scale,
                        weight: weights[k]
                    ))
                    matIdx[k] += 1
                }
                let stacked = StackedLoraLinear(wrapping: lin, slots: slots)
                if isAttn {
                    attnEntries[name] = .value(stacked)
                } else {
                    mlpEntries[name] = .value(stacked)
                }
            }
            var blockChildren: [String: NestedItem<String, Module>] = [:]
            if !attnEntries.isEmpty { blockChildren["self_attn"] = .dictionary(attnEntries) }
            if !mlpEntries.isEmpty { blockChildren["mlp"] = .dictionary(mlpEntries) }
            layersList.append(.dictionary(blockChildren))
        }
        var root = NestedDictionary<String, Module>()
        root["layers"] = .array(layersList)
        model.update(modules: root)
    }
}
