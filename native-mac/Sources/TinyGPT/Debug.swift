import Foundation
import MLX
import MLXNN
import TinyGPTIO
import TinyGPTModel

/// `tinygpt debug-names` — print the model's parameter names side-by-side
/// with the file's manifest, so any mismatch is visible at a glance.
enum DebugNames {
    /// Inspect dtypes of loaded weights — maybe MLX is interpreting them
    /// differently than expected (e.g., as fp16 when I think they're fp32).
    static func dtypes(args: [String]) {
        guard let path = args.first else {
            fputs("usage: tinygpt debug-dtypes <path.tinygpt>\n", stderr); exit(2)
        }
        let url = URL(fileURLWithPath: path)
        let file = try! TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let model = TinyGPTModel(cfg)
        try! TinyGPTWeightLoader.load(file, into: model)
        print("dtypes after load:")
        print("  token_embedding.weight: \(model.tokenEmbedding.weight.dtype)")
        print("  position_embedding.weight: \(model.positionEmbedding.weight.dtype)")
        print("  ln_final.weight: \(String(describing: model.lnFinal.weight?.dtype))")
        print("  blocks[0].attn.q_proj.weight: \(model.blocks[0].attn.qProj.weight.dtype)")
        print("  blocks[0].attn.q_proj.bias: \(String(describing: model.blocks[0].attn.qProj.bias?.dtype))")
        print("  blocks[0].ln1.weight: \(String(describing: model.blocks[0].ln1.weight?.dtype))")
        print("  blocks[0].mlp.fc_in.weight: \(model.blocks[0].mlp.fcIn.weight.dtype)")

        // Forward pass on one token and inspect signal magnitude per block.
        let idx = MLXArray([Int32(82), Int32(79), Int32(77), Int32(69), Int32(79), Int32(58)], [1, 6])
        print("\nshape diagnostic:")
        let tokE = model.tokenEmbedding(idx)
        eval(tokE)
        print("  tokenEmbedding(idx [1,6]) → shape \(tokE.shape) dtype \(tokE.dtype)")
        let positions = MLXArray((0..<6).map { Int32($0) })
        let posEmbRaw = model.positionEmbedding(positions)
        eval(posEmbRaw)
        print("  positionEmbedding(pos [6]) → shape \(posEmbRaw.shape) dtype \(posEmbRaw.dtype)")
        let posEmb = posEmbRaw.expandedDimensions(axis: 0)
        var x = tokE + posEmb
        eval(x)
        print("  after add → shape \(x.shape)")
        let tokSqSum = (tokE * tokE).sum().item(Float.self)
        let posSqSum = (posEmbRaw * posEmbRaw).sum().item(Float.self)
        let fullTokSqSum = (model.tokenEmbedding.weight * model.tokenEmbedding.weight).sum().item(Float.self)
        print("  sum(tokE^2) for these 6 tokens: \(String(format: "%.3f", tokSqSum)) (expect ~150 if std≈0.3)")
        print("  sum(posEmb^2) for 6 positions: \(String(format: "%.3f", posSqSum))")
        print("  sum(token_emb.weight^2) full: \(String(format: "%.1f", fullTokSqSum)) over 65536 values → mean(x^2)=\(String(format: "%.4f", fullTokSqSum / 65536))")

        // Check: file's row 82 (bytes 82*256*2 to 83*256*2) vs model.weight[82].
        if let tok = file.tensors.first(where: { $0.entry.name == "token_embedding.weight" }) {
            let allFloats = tok.weightFP16AsFloat32()
            let row82FromFile = Array(allFloats[(82 * 256)..<(83 * 256)])
            print("  FILE row 82 first 8:       \(row82FromFile.prefix(8).map { String(format: "%.4f", $0) })")
            let row0FromFile = Array(allFloats[0..<256])
            print("  FILE row 0 first 8:        \(row0FromFile.prefix(8).map { String(format: "%.4f", $0) })")
            let row0SqSum = row0FromFile.reduce(0) { $0 + $1 * $1 }
            let row82SqSum = row82FromFile.reduce(0) { $0 + $1 * $1 }
            print("  FILE row 0 sum^2: \(String(format: "%.3f", row0SqSum))")
            print("  FILE row 82 sum^2: \(String(format: "%.3f", row82SqSum))")
        }

        // Cross-check: tokenEmbedding(MLXArray([82])) should equal weight[82].
        let viaCall = model.tokenEmbedding(MLXArray([Int32(82)]))
        let viaIndex = model.tokenEmbedding.weight[82]
        eval(viaCall)
        eval(viaIndex)
        let cArr = viaCall.asArray(Float.self)
        let iArr = viaIndex.asArray(Float.self)
        print("  weight[82] first 8:        \(iArr.prefix(8).map { String(format: "%.4f", $0) })")
        print("  tokenEmbedding([82]) first 8: \(cArr.prefix(8).map { String(format: "%.4f", $0) })")
        print("  match: \(iArr == cArr)")
        // Also check sum_squared of weight[82] specifically
        let row82sq = (viaIndex * viaIndex).sum().item(Float.self)
        print("  sum(weight[82]^2) = \(String(format: "%.3f", row82sq))")

        let initSum = (x * x).sum().item(Float.self)
        print("\nforward magnitude trace:")
        print("  after embed:   sum(x^2)=\(String(format: "%.3f", initSum))")
        for (i, block) in model.blocks.enumerated() {
            x = block(x)
            eval(x)
            let s = (x * x).sum().item(Float.self)
            print("  after block \(i): sum(x^2)=\(String(format: "%.3f", s))")
        }
        x = model.lnFinal(x)
        eval(x)
        let lnSum = (x * x).sum().item(Float.self)
        print("  after ln_final: sum(x^2)=\(String(format: "%.3f", lnSum))")
    }

    /// Compute cross-entropy loss on a known Shakespeare excerpt. If the
    /// model is properly loaded, loss should be near 1.22 (the file's
    /// recorded training loss). If it's near 5.5 = ln(256), the forward
    /// pass produces random-baseline logits — pinpoints whether the bug
    /// is in the model or in the sampling/generation code.
    static func sanityLoss(args: [String]) {
        guard let path = args.first else {
            fputs("usage: tinygpt debug-loss <path.tinygpt>\n", stderr); exit(2)
        }
        let url = URL(fileURLWithPath: path)
        let file = try! TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let model = TinyGPTModel(cfg)
        try! TinyGPTWeightLoader.load(file, into: model)

        // A real Shakespeare passage matching the training corpus.
        let text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?"
        let bytes = [UInt8](text.utf8)
        let T = min(bytes.count - 1, cfg.contextLength)

        let inputs = MLXArray(bytes.prefix(T).map { Int32($0) }, [1, T])
        let targets = MLXArray(bytes.dropFirst().prefix(T).map { Int32($0) }, [1, T])

        let loss = model.loss(inputs, targets)
        eval(loss)
        let lossValue = loss.item(Float.self)

        print("Loss on Shakespeare excerpt:")
        print("  excerpt length: \(T) tokens")
        print("  loss:           \(String(format: "%.4f", lossValue))")
        print("  expected for a trained Shakespeare model: ~1.2")
        print("  expected for random init: ~\(String(format: "%.2f", log(Float(cfg.vocabSize))))")

        if lossValue < 2.0 {
            print("  ✓ model is properly loaded — forward pass works")
        } else if lossValue > 5.0 {
            print("  ✗ loss equals random baseline — forward pass produces uniform-ish logits")
        } else {
            print("  ⚠ loss is between random and trained — partial functionality")
        }
    }

    static func logits(args: [String]) {
        // After loading, run a forward pass on "ROMEO:" and print the
        // top-5 next-token candidates by logit. Lets us see if the
        // distribution is degenerate (all probability on one token) vs
        // healthy (top-5 looks like plausible next characters).
        guard let path = args.first else {
            fputs("usage: tinygpt debug-logits <path.tinygpt>\n", stderr); exit(2)
        }
        let url = URL(fileURLWithPath: path)
        let file = try! TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let model = TinyGPTModel(cfg)
        print("step 1: model built")
        fflush(stdout)
        try! TinyGPTWeightLoader.load(file, into: model)
        print("step 2: weights loaded")
        fflush(stdout)

        let prompt = "ROMEO:"
        let bytes = [UInt8](prompt.utf8)
        let idx = MLXArray(bytes.map { Int32($0) }, [1, bytes.count])
        print("step 3: idx shape \(idx.shape)")
        fflush(stdout)
        let logits = model(idx)
        print("step 4: forward done, logits shape \(logits.shape)")
        fflush(stdout)
        eval(logits)
        print("step 5: eval done")
        fflush(stdout)
        // Logits shape: [1, T, 256]. Look at the LAST position.
        let last = logits[0, logits.shape[1] - 1, 0...]
        eval(last)
        print("step 6: last shape \(last.shape)")
        fflush(stdout)
        // Extract all 256 floats at once.
        let values: [Float] = last.asArray(Float.self)
        print("step 7: extracted \(values.count) floats")
        print("  first 8: \(values.prefix(8).map { String(format: "%.3f", $0) })")
        let minV = values.min()!
        let maxV = values.max()!
        let sumV = values.reduce(0, +)
        print("  range: \(String(format: "%.3f", minV)) ... \(String(format: "%.3f", maxV))")
        print("  span:  \(String(format: "%.3f", maxV - minV))")
        print("  mean:  \(String(format: "%.3f", sumV / Float(values.count)))")
        // Top 10 by index without sorting (avoid potential MLX issue).
        var topIds: [(Int, Float)] = []
        for i in 0..<256 {
            topIds.append((i, values[i]))
        }
        topIds.sort { $0.1 > $1.1 }
        print("After '\(prompt)' — top 10 logits:")
        for (id, value) in topIds.prefix(10) {
            let printable = (33...126).contains(id)
                ? (UnicodeScalar(id).map { String($0) } ?? "?")
                : "0x\(String(id, radix: 16))"
            print(String(format: "  %3d  %-4s  %+.3f", id, printable, value))
        }
    }

    static func compareLoaded(args: [String]) {
        // Verify the loader actually changed the model weights — print a
        // few token_embedding values before-and-after loading.
        guard let path = args.first else {
            fputs("usage: tinygpt debug-load <path.tinygpt>\n", stderr); exit(2)
        }
        let url = URL(fileURLWithPath: path)
        let file = try! TinyGPTFileReader.read(url)
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let model = TinyGPTModel(cfg)
        // BEFORE: print fresh-init token_embedding row 0 first 8 values.
        let beforeTok = model.tokenEmbedding.weight[0, 0..<8]
        eval(beforeTok)
        print("BEFORE  token_embedding.weight[0, 0..7]:")
        for i in 0..<8 {
            print(String(format: "  %.6f", beforeTok[i].item(Float.self)))
        }
        try! TinyGPTWeightLoader.load(file, into: model)
        let afterTok = model.tokenEmbedding.weight[0, 0..<8]
        eval(afterTok)
        print("AFTER   token_embedding.weight[0, 0..7]:")
        for i in 0..<8 {
            print(String(format: "  %.6f", afterTok[i].item(Float.self)))
        }
        // Also print expected: the file's first 8 fp16 values, decoded.
        if let tok0 = file.tensors.first(where: { $0.entry.name == "token_embedding.weight" }) {
            let floats = tok0.weightFP16AsFloat32()
            print("EXPECTED  token_embedding.weight first 8 from file:")
            for i in 0..<8 {
                print(String(format: "  %.6f", floats[i]))
            }
        }

        // Same for a Linear weight — these are the "did transposition go wrong?" check.
        print("")
        if let qProj = file.tensors.first(where: { $0.entry.name == "blocks.0.attn.q_proj.weight" }) {
            let floats = qProj.weightFP16AsFloat32()
            print("EXPECTED  blocks.0.attn.q_proj.weight first 8 from file:")
            for i in 0..<8 {
                print(String(format: "  %.6f", floats[i]))
            }
        }
        let afterQ = model.blocks[0].attn.qProj.weight[0, 0..<8]
        eval(afterQ)
        print("AFTER   blocks.0.attn.q_proj.weight[0, 0..7]:")
        for i in 0..<8 {
            print(String(format: "  %.6f", afterQ[i].item(Float.self)))
        }

        // Verify bias loading too (Linear.bias is Optional<MLXArray>, the
        // update logic might be silently skipping it).
        print("")
        if let qBias = file.tensors.first(where: { $0.entry.name == "blocks.0.attn.q_proj.bias" }) {
            let floats = qBias.weightFP16AsFloat32()
            print("EXPECTED  blocks.0.attn.q_proj.bias first 8 from file:")
            for i in 0..<8 {
                print(String(format: "  %.6f", floats[i]))
            }
        }
        if let bias = model.blocks[0].attn.qProj.bias {
            let afterB = bias[0..<8]
            eval(afterB)
            print("AFTER   blocks.0.attn.q_proj.bias[0..7]:")
            for i in 0..<8 {
                print(String(format: "  %.6f", afterB[i].item(Float.self)))
            }
        } else {
            print("⚠ AFTER   blocks.0.attn.q_proj.bias is NIL (loader skipped Optional)")
        }

        // ln1 weight is the most-suspicious-of-not-loading slot.
        print("")
        if let lnW = file.tensors.first(where: { $0.entry.name == "blocks.0.ln1.weight" }) {
            let floats = lnW.weightFP16AsFloat32()
            print("EXPECTED  blocks.0.ln1.weight first 8 from file:")
            for i in 0..<8 {
                print(String(format: "  %.6f", floats[i]))
            }
        }
        if let lnW = model.blocks[0].ln1.weight {
            let afterLN = lnW[0..<8]
            eval(afterLN)
            print("AFTER   blocks.0.ln1.weight[0..7]:")
            for i in 0..<8 {
                print(String(format: "  %.6f", afterLN[i].item(Float.self)))
            }
        }
        // Verify a LATE block's weights too — guard against any order-related bug.
        print("")
        if let lnW = file.tensors.first(where: { $0.entry.name == "blocks.11.attn.o_proj.weight" }) {
            let floats = lnW.weightFP16AsFloat32()
            print("EXPECTED  blocks.11.attn.o_proj.weight first 8 from file:")
            for i in 0..<8 {
                print(String(format: "  %.6f", floats[i]))
            }
        }
        let block11oProj = model.blocks[11].attn.oProj.weight[0, 0..<8]
        eval(block11oProj)
        print("AFTER   blocks.11.attn.o_proj.weight[0, 0..7]:")
        for i in 0..<8 {
            print(String(format: "  %.6f", block11oProj[i].item(Float.self)))
        }
    }

    static func run(args: [String]) {
        guard let path = args.first else {
            fputs("usage: tinygpt debug-names <path.tinygpt>\n", stderr)
            exit(2)
        }
        let url = URL(fileURLWithPath: path)
        let file: TinyGPTFile
        do {
            file = try TinyGPTFileReader.read(url)
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
        let h = file.header.config
        let cfg = ModelConfig(
            vocabSize: 256,
            contextLength: h.ctx ?? 256,
            nLayers: h.layers ?? 12,
            nHeads: h.heads ?? 8,
            dModel: h.dModel ?? 256,
            dMlp: h.dMlp ?? 1024
        )
        let model = TinyGPTModel(cfg)
        let modelKeys = Set(model.parameters().flattened().map { $0.0 })
        let fileKeys = Set(file.tensors.map { $0.entry.name })

        let onlyInModel = modelKeys.subtracting(fileKeys).sorted()
        let onlyInFile = fileKeys.subtracting(modelKeys).sorted()
        let common = modelKeys.intersection(fileKeys).sorted()

        print("model parameters: \(modelKeys.count)")
        print("file tensors:     \(fileKeys.count)")
        print("matched:          \(common.count)")
        print("")
        if !onlyInModel.isEmpty {
            print("⚠ ONLY IN MODEL (file is missing these — model uses random init):")
            for k in onlyInModel { print("    \(k)") }
            print("")
        }
        if !onlyInFile.isEmpty {
            print("⚠ ONLY IN FILE (model can't accept these — unused):")
            for k in onlyInFile { print("    \(k)") }
            print("")
        }

        if onlyInModel.isEmpty && onlyInFile.isEmpty {
            print("✓ all parameter names match perfectly.")
            print("  Shape comparison (model vs file) follows:")
            var anyShapeMismatch = false
            for key in common {
                let modelShape = model.parameters().flattened()
                    .first { $0.0 == key }?.1.shape
                let fileShape = file.tensors.first { $0.entry.name == key }?.entry.shape
                if let m = modelShape, let f = fileShape, m != f {
                    print("    ⚠ shape mismatch on \(key): model=\(m) file=\(f)")
                    anyShapeMismatch = true
                }
            }
            if !anyShapeMismatch {
                print("    ✓ all shapes match.")
            }
        }
    }
}
