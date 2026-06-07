import Foundation
import MLX
import TinyGPTIO
import TinyGPTModel

/// `tinygpt vlm-smoke <hf-vision-dir> [image.png]` — Milestone-1 smoke
/// for the vision encoder primitive (see PRD
/// `docs/prds/factory-vision-specialist.md`).
///
/// Loads a CLIP-style ViT from `<hf-vision-dir>` (a HuggingFace snapshot
/// directory containing `config.json` + `model.safetensors`), runs a
/// single forward pass on either a supplied image or a synthetic
/// gradient, and prints the output shape + a couple of statistic
/// signals (mean, std, finite-ness) so we can verify the loaded weights
/// are at least dimensionally sane.
///
/// Two reasons this lives as a CLI subcommand rather than an
/// XCTestCase:
///
///   1. MLX-Swift's metallib doesn't ship into `swift test`'s product
///      bundle (see TinyGPTModelTests.swift's preamble). Running the
///      forward pass via the CLI executable is the only way to exercise
///      real MLX kernels without booting Xcode.
///   2. M1's smoke is "did the weights load and forward run end-to-end?"
///      — that's a structural verification, not a numerics regression
///      gate. A dedicated test will land alongside M3 when we have the
///      full VLM forward to assert against teacher logprobs.
enum VLMSmoke {
    static func run(args: [String]) {
        guard let dirArg = args.first else {
            fputs("""
            usage: tinygpt vlm-smoke <hf-vision-dir> [image.png]

            Loads a CLIP-style ViT vision encoder from the given HF
            snapshot directory and runs a single forward pass. If
            <image.png> is provided we preprocess and use it; otherwise
            a synthetic gradient image is used.

            Example:
              tinygpt vlm-smoke \\
                ~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/<hash>

            """, stderr)
            exit(2)
        }
        if dirArg == "-h" || dirArg == "--help" {
            run(args: [])
        }
        let dir = URL(fileURLWithPath: dirArg)
        let imagePath = args.dropFirst().first

        print("Loading CLIP vision encoder from: \(dir.path)")
        let loaded: CLIPVisionLoader.LoadResult
        do {
            loaded = try CLIPVisionLoader.load(from: dir)
        } catch {
            fputs("load failed: \(error)\n", stderr)
            exit(1)
        }
        let cfg = loaded.config
        print("""

        VisionConfig:
          hidden_size:      \(cfg.hiddenSize)
          num_hidden_layers:\(cfg.numHiddenLayers)
          num_heads:        \(cfg.numAttentionHeads)
          intermediate_size:\(cfg.intermediateSize)
          image_size:       \(cfg.imageSize)
          patch_size:       \(cfg.patchSize)
          hidden_act:       \(cfg.hiddenAct)
          num_positions:    \(cfg.numPositions)   (1 CLS + \(cfg.numPatches) patches)

        """)

        // Preprocess input image.
        let preprocessCfg = ImagePreprocessConfig.loadFromDir(dir)
        let pixels: MLXArray
        if let p = imagePath {
            print("Preprocessing image: \(p)")
            do {
                pixels = try ImagePreprocess.preprocess(path: p, config: preprocessCfg)
            } catch {
                fputs("preprocess failed: \(error)\n", stderr)
                exit(1)
            }
        } else {
            print("No image path given — using synthetic gradient (size \(preprocessCfg.imageSize))")
            pixels = ImagePreprocess.syntheticTestImage(size: preprocessCfg.imageSize)
        }
        print("  pixels NHWC shape: \(pixels.shape)")

        // Optional dump for the parity check: when TINYGPT_VLM_DUMP is set,
        // write pixels + features as NPY to the given prefix so a
        // Python script can load them and compare against HF
        // CLIPVisionModel. Only fp32 NPY is supported (matches the MLX
        // up-cast in the loader).
        let dumpEnv = ProcessInfo.processInfo.environment["TINYGPT_VLM_DUMP"]
        if let dumpPrefix = dumpEnv {
            writeNPY(pixels, path: "\(dumpPrefix)_pixels.npy")
            print("  dumped pixels → \(dumpPrefix)_pixels.npy")
        }

        // Forward pass.
        let model = loaded.model
        // Capture before/after so the autorelease pool flushes after each
        // op — keeps peak memory bounded while we wait for synchronous
        // eval() at the end.
        let features = model(pixels)
        features.eval()
        print("  features shape:    \(features.shape)    (expected [1, \(cfg.numPositions), \(cfg.hiddenSize)])")

        if let dumpPrefix = dumpEnv {
            writeNPY(features, path: "\(dumpPrefix)_features.npy")
            print("  dumped features → \(dumpPrefix)_features.npy")
        }

        // Quick sanity statistics — finite? non-zero? bounded?
        let flat = features.reshaped([-1])
        let meanVal = flat.mean().asArray(Float.self)[0]
        let std = (flat - meanVal).square().mean().sqrt().asArray(Float.self)[0]
        let mx  = flat.max().asArray(Float.self)[0]
        let mn  = flat.min().asArray(Float.self)[0]
        print("""

          mean:  \(meanVal)
          std:   \(std)
          min:   \(mn)
          max:   \(mx)

        """)

        // Pass criteria for M1: shape is [1, numPositions, hidden] AND
        // every entry is finite AND non-degenerate (std > 0).
        let expectedShape = [1, cfg.numPositions, cfg.hiddenSize]
        guard features.shape == expectedShape else {
            fputs("FAIL: expected shape \(expectedShape), got \(features.shape)\n", stderr)
            exit(1)
        }
        guard std > 1e-4 else {
            fputs("FAIL: feature std is degenerate (\(std)) — likely all-zero weights or layer collapse\n", stderr)
            exit(1)
        }
        // NaN/Inf check via min/max — if any element is NaN, both
        // min and max propagate NaN.
        guard mn.isFinite && mx.isFinite else {
            fputs("FAIL: non-finite values in vision features (min=\(mn), max=\(mx))\n", stderr)
            exit(1)
        }

        print("PASS (M1): vision encoder loaded \(loaded.config.numHiddenLayers) layers, forward produced [1, \(cfg.numPositions), \(cfg.hiddenSize)] finite features.")

        // ---------------------------------------------------------------
        // Milestone-2 smoke: instantiate a random-init CrossModalProjection
        // and forward the patch tokens through it. Verifies the projection
        // produces a sane-shaped output and lets us spot init-time issues
        // (NaN, all-zero, etc.) before M3 wires it into a full VLM forward.
        //
        // We use a fixed LLM hidden size of 2048 here — placeholder for
        // Qwen3-1.7B (the target LLM body, see M4 in the PRD). The
        // projection is trained from scratch at M5 so the exact dim only
        // needs to match the chosen LLM body at M4.
        // ---------------------------------------------------------------
        let projCfg = CrossModalProjectionConfig(
            visionHidden: cfg.hiddenSize,
            llmHidden: 2048,
            hiddenAct: "gelu"
        )
        let proj = CrossModalProjection(projCfg)
        // Drop the CLS token — LLaVA convention feeds only patches.
        let patchOnly = features[0..., 1..., 0...]
        let visionTokens = proj(patchOnly)
        visionTokens.eval()
        print("  projection input  : \(patchOnly.shape)  vision_hidden \(projCfg.visionHidden)")
        print("  projection output : \(visionTokens.shape)  llm_hidden \(projCfg.llmHidden)")

        let projFlat = visionTokens.reshaped([-1])
        let projMean = projFlat.mean().asArray(Float.self)[0]
        let projStd = (projFlat - projMean).square().mean().sqrt().asArray(Float.self)[0]
        let projMax = projFlat.max().asArray(Float.self)[0]
        let projMin = projFlat.min().asArray(Float.self)[0]
        let expectedProjShape = [features.shape[0], cfg.numPatches, projCfg.llmHidden]
        guard visionTokens.shape == expectedProjShape else {
            fputs("FAIL (M2): expected \(expectedProjShape), got \(visionTokens.shape)\n", stderr)
            exit(1)
        }
        guard projStd > 0 && projMin.isFinite && projMax.isFinite else {
            fputs("FAIL (M2): degenerate projection output (std=\(projStd), min=\(projMin), max=\(projMax))\n", stderr)
            exit(1)
        }
        print("""

          projection mean: \(projMean)
          projection std:  \(projStd)
          projection min:  \(projMin)
          projection max:  \(projMax)

        PASS (M2): cross-modal projection forwarded \(cfg.numPatches) patch tokens → [B, \(cfg.numPatches), \(projCfg.llmHidden)] LLM-space tokens, random init produces finite values.
        """)

        // ---------------------------------------------------------------
        // Milestone-3 smoke: full VLM forward (vision + projection + LLM).
        // Random-init tiny LLM (2 layers, hidden=128) keeps this under a
        // second; the test only asserts `(image, tokens) → logits` has
        // the right shape and is finite. Real numerics validation
        // happens at M4 when a real Qwen3-VL-2B checkpoint loads.
        // ---------------------------------------------------------------
        let tinyLLMConfig = ModelConfig(
            modelName: "vlm-smoke-tinylm",
            vocabSize: 1000,
            contextLength: 1024,
            nLayers: 2,
            nHeads: 4,
            nKvHeads: 4,
            dModel: 128,
            dMlp: 256,
            dropout: 0.0,
            tieEmbeddings: true,
            dtype: "float32",
            useRoPE: true,
            ropeBase: 10_000,
            useRMSNorm: true,
            useSwiGLU: true,
            attnBias: false
        )
        let vlmConfig = TinyGPTModelVLM.Config(
            vision: cfg,
            llm: tinyLLMConfig,
            projectionAct: "gelu"
        )
        let vlm = TinyGPTModelVLM(vlmConfig)
        // Synthetic text tokens — 8 ids from the toy vocab.
        let textTokens = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7, 8], [1, 8])
        let logits = vlm(pixels, textTokens)
        logits.eval()
        let expectedSeqLen = cfg.numPatches + textTokens.shape[1]
        let expectedLogitsShape = [pixels.shape[0], expectedSeqLen, tinyLLMConfig.vocabSize]
        print("  vlm output shape: \(logits.shape)  (expected \(expectedLogitsShape))")
        guard logits.shape == expectedLogitsShape else {
            fputs("FAIL (M3): expected \(expectedLogitsShape), got \(logits.shape)\n", stderr)
            exit(1)
        }
        let logitsFlat = logits.reshaped([-1])
        let lMean = logitsFlat.mean().asArray(Float.self)[0]
        let lStd = (logitsFlat - lMean).square().mean().sqrt().asArray(Float.self)[0]
        let lMin = logitsFlat.min().asArray(Float.self)[0]
        let lMax = logitsFlat.max().asArray(Float.self)[0]
        guard lStd > 0 && lMin.isFinite && lMax.isFinite else {
            fputs("FAIL (M3): degenerate logits (std=\(lStd), min=\(lMin), max=\(lMax))\n", stderr)
            exit(1)
        }
        print("""

          logits mean: \(lMean)
          logits std:  \(lStd)
          logits min:  \(lMin)
          logits max:  \(lMax)

        PASS (M3): TinyGPTModelVLM(image=[1,224,224,3], tokens=[1,8]) → logits=[1, \(expectedSeqLen), \(tinyLLMConfig.vocabSize)], all finite (random LLM init).
        """)
    }

    /// Write an MLXArray as a NumPy `.npy` file (V1.0 format, float32,
    /// C-order). Used only by the parity-check path so a Python script
    /// can load the bit-identical tensor and forward through HF's
    /// `CLIPVisionModel` for comparison.
    ///
    /// The NPY format is well-documented in
    /// `numpy/lib/format.py`. Layout:
    ///   `\x93NUMPY` (6 bytes) | major=1 | minor=0 | header_len: u16 LE
    ///   | header: ASCII dict literal padded to 64-byte alignment with
    ///   spaces, terminated with '\n'
    ///   | raw float32 data, little-endian, C-order
    private static func writeNPY(_ arr: MLX.MLXArray, path: String) {
        // Materialise as Float in C-order. asArray flattens; the shape is
        // tracked separately in the NPY header.
        let floats = arr.asArray(Float.self)
        let shapeStr = "(" + arr.shape.map { "\($0)," }.joined() + ")"
        let dictLiteral = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeStr), }"
        // Header (excluding magic + version + len = 10 bytes prefix) must be
        // padded so the total prefix-plus-header length is a multiple of 64.
        let prefixLen = 10
        let baseLen = prefixLen + dictLiteral.utf8.count + 1  // +1 for trailing '\n'
        let padding = (64 - (baseLen % 64)) % 64
        let paddedHeader = dictLiteral + String(repeating: " ", count: padding) + "\n"
        var data = Data()
        data.append(contentsOf: [0x93])
        data.append(contentsOf: "NUMPY".utf8)
        data.append(contentsOf: [0x01, 0x00])   // version 1.0
        let headerBytes = paddedHeader.utf8.count
        precondition(headerBytes < 65_536, "NPY V1.0 header too long; use V2.0")
        let hLo = UInt8(headerBytes & 0xFF)
        let hHi = UInt8((headerBytes >> 8) & 0xFF)
        data.append(contentsOf: [hLo, hHi])
        data.append(contentsOf: paddedHeader.utf8)
        // Raw float32 data — NumPy default is host-order, which on Apple
        // Silicon (and all current Macs) is little-endian; matches '<f4'.
        floats.withUnsafeBufferPointer { ptr in
            data.append(UnsafeBufferPointer(start: ptr.baseAddress, count: ptr.count)
                .withMemoryRebound(to: UInt8.self) { buf in
                    Data(buffer: buf)
                })
        }
        do {
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            fputs("NPY write failed for \(path): \(error)\n", stderr)
        }
    }
}
