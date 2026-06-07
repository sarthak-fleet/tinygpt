import CryptoKit
import Foundation
import MLX
import MLXRandom
import TinyGPTIO
import TinyGPTModel

/// `tinygpt train` — train a model from scratch on a UTF-8 text corpus.
///
/// Long-run features (Tier 0 safety nets):
///   --resume <path.tinygpt>     Resume weights + step from a checkpoint
///                               and Adam state when present/supported.
///   --save-every N              Atomic checkpoint every N steps. A crash
///                               leaves the last successful checkpoint
///                               intact (write-to-.tmp then rename).
///   --depth N                   nanochat-style single-knob override:
///                               derives nLayers=N, dModel=64·N, nHeads=N,
///                               dMlp=4·dModel. Takes precedence over the
///                               preset's L/d/h/MLP fields; preset still
///                               supplies ctx, vocab, dtype.
///   --lr-schedule cosine|wsd|constant  Default cosine. `wsd` is
///                               warmup-stable-decay: linear warmup →
///                               constant maxLR → 1−√(t) decay over the
///                               last `--decay-steps` steps.
///   --warmup N                  Warmup steps (default 500 — standard).
///   --max-lr / --min-lr         Schedule endpoints (defaults 3e-4 / 3e-5).
///   --decay-steps N             WSD decay window (default: 10% of steps).
///   --val-split 0.0-0.2         Hold out last fraction of corpus for val.
///   --val-every N               Eval val loss every N steps (default 200).
///   --eval-every N              Spawn lightweight E3 evals at checkpointed steps.
///
/// Ctrl-C is cooperative: the next step finishes, a final checkpoint is
/// flushed, and the process exits cleanly.
enum Train {
    static func run(args: [String]) {
        // CPU QoS bump (item #3 of the CPU-speedup bundle).
        //
        // macOS schedules a binary launched from the terminal at
        // `.default` QoS by default, which the OS is free to migrate to
        // an E-core under load. The training thread is the host-side
        // driver for an MLX-Swift workload — every MLXArray construction,
        // optimiser update, `eval`, and `item()` runs here. Pinning the
        // thread at `.userInteractive` lands it on a P-core for the
        // duration of the run; the OS treats it like a foreground
        // animation thread and won't demote it.
        //
        // The call is best-effort. If the platform doesn't support the
        // call (impossible on macOS 14+ but cheap to guard) we just
        // continue at default QoS.
        //
        // `TINYGPT_DISABLE_QOS=1` skips the bump. Used by the bundle
        // benchmarks to measure the QoS contribution in isolation —
        // not a user-facing knob.
        if ProcessInfo.processInfo.environment["TINYGPT_DISABLE_QOS"] != "1" {
            TrainSupport.bumpQoSToUserInteractive()
        }
        var preset = "tiny"
        // nanochat-style single-knob depth override: when set, derives
        // (nLayers, dModel, nHeads, dMlp) from N via the standard
        // GPT-2-shaped rules below. Preset still supplies ctx, vocab,
        // tokenizer, and dtype.
        var depthOverride: Int? = nil
        var steps = 500
        var corpusPath: String? = nil
        var outPath: String? = nil
        // Curated-recipe default: bf16 — better range than fp16, ½ memory
        // of fp32, matches modern decoder-only training norms (Llama, Qwen,
        // Mistral). Override with `--dtype float32` for strict reproducibility
        // tests or `--dtype float16` for hardware that lacks bf16.
        var dtype = "bfloat16"
        var batchSize: Int? = nil
        var sampleEvery = 100
        // Tier 0 additions:
        var resumePath: String? = nil
        var saveEvery: Int? = nil
        // B13 support: when set, every --save-every tick ALSO writes a
        // step-numbered copy to `<out-stem>.step-N.tinygpt` alongside the
        // overwriting atomic save. Enables `tinygpt sae --checkpoint-dir`
        // (and future interp-on-checkpoints tools) to replay training
        // dynamics. Disk-hungry on long runs; off by default.
        var saveHistory: Bool = false
        var saveOptState: Bool = true
        // Curated-recipe default: cosine + warmup 500. Standard transformer
        // schedule; constant LR is rarely the right choice past a smoke test.
        // `wsd` (warmup-stable-decay) is the MiniCPM/SmolLM alternative — the
        // decay phase doubles as the annealing window.
        var lrSchedule = "cosine"
        var warmupSteps: Int = 500
        var maxLR: Float = 3e-4
        var minLR: Float = 3e-5
        // -1 sentinel = auto-derive to 10% of `steps` at run time.
        var decaySteps: Int = -1
        // Loss-spike detector — observe-only v1. Logs a warning when a
        // step's loss exceeds `spikeFactor × moving-average over
        // spikeWindow steps`. Off-switch is `--no-spike-detect`.
        var spikeDetectEnabled: Bool = true
        var spikeWindow: Int = 50
        var spikeFactor: Float = 3.0
        // JSONL log emitter (C10 training dashboard). One JSON object per
        // step appended to the file; consumed by browser/src/pages/training-dashboard.
        var logJsonlPath: String? = nil
        // E8 train-time eval hook. Slow evals are skipped rather than queued.
        var evalEvery: Int? = nil
        var evalTasks: String = "arc_easy,gsm8k"
        var evalLimit: Int = 50
        // C9 determinism: --seed N seeds MLXRandom early so model init +
        // any GPU-side dropout/noise is reproducible. Batch sampling via
        // Swift's stdlib `Int.random` is NOT covered yet — v2 will replace
        // it with a seeded host RNG. See banner footer + docs/determinism.md.
        var rngSeed: UInt64? = nil
        var valSplit: Double = 0
        var valEvery: Int = 200
        var tokenizerDir: String? = nil
        var ctxOverride: Int? = nil
        var accumSteps: Int = 1
        // Default-on at 1.0 — standard transformer-LM stability lever, almost
        // never a no-op cost on well-behaved runs, saves bf16 blowups.
        // Pass `--grad-clip 0` to disable.
        var gradClipNorm: Float = 1.0
        // Mixture-of-Experts. `nExperts == 1` = standard dense MLP. When
        // > 1, every block's MLP becomes an MoE with a learned router.
        // Top-K is the number of experts each token activates (1 = Switch
        // Transformer, 2 = Mixtral-style). aux weight scales the load-
        // balance loss that keeps the router from collapsing.
        var moeExperts: Int = 1
        var moeTopK: Int = 1
        var moeAuxWeight: Float = 0.01
        // Multi-Token Prediction horizons (Gloeckle et al., 2024;
        // DeepSeek-V3). 1 = standard next-token. 2-4 typical for the
        // regulariser to bite without ballooning per-step compute.
        var mtpHorizons: Int = 1
        // Sliding-window attention (Mistral / GPT-OSS). nil = full causal.
        // When set, each query attends to only the last `slidingWindow`
        // positions — bounds attn memory/compute at long context.
        var slidingWindow: Int? = nil
        // ALiBi position bias (Press et al., 2021). When set, the model
        // drops learned positional embeddings and uses a per-head linear-
        // distance bias instead. Cleaner generalisation to longer contexts.
        var useALiBi: Bool = false
        // Mixture-of-Depths: per-token sigmoid gate on each block's
        // residual contribution (Raposo et al., 2024). Pure architecture
        // change; the dense compute path is unchanged.
        var useMoD: Bool = false
        // Differential attention (Ye et al., 2024). Doubles the Q/K
        // projections per block + adds a learnable λ — used for less-
        // noisy attention. Mutually exclusive with the standard path.
        var useDiffAttn: Bool = false
        // YOCO (Lin et al., 2024). Second half cross-attends to the
        // anchor; halves KV cache memory at long-context decode.
        var useYOCO: Bool = false
        // Gradient (activation) checkpointing. Trades ~30% extra
        // compute for a large reduction in activation memory. Each
        // TransformerBlock's forward is wrapped in a CustomFunction
        // whose VJP recomputes the block forward at backward time.
        var useGradCheckpoint: Bool = false
        // Optimiser choice (Lion, Sophia, Muon, Adafactor; default
        // AdamW preserves backward compat). See `Optimizers.swift`.
        var optimizerKind: OptimizerKind = .adamw
        // GaLore — rank-R projection of 2-D weight gradients
        // (Zhao et al., 2024). `0` disables; 256 is the paper's
        // typical setting for transformer pretraining.
        var galoreRank: Int = 0
        var galoreUpdateEvery: Int = 200
        // Z-loss weight (PaLM / GShard). 1e-4 default is conservative;
        // 0 disables.
        var zLossWeight: Float = 0
        // DeepNorm scaling for very deep transformers (Wang et al., 2022).
        var useDeepNorm: Bool = false
        // Layer-wise LR decay factor. 1.0 = no decay.
        var lrLayerDecay: Float = 1.0
        // Apply RMSNorm right after the token embedding lookup.
        var useEmbeddingRMSNorm: Bool = false
        // BPE-dropout (Provilkov et al., ACL 2020). Per-merge skip
        // probability — the same surface text yields a slightly
        // different token sequence across epochs, giving the model a
        // light tokenization regulariser. 0 = off (default); 0.1 is the
        // Provilkov paper's recommendation for "BPE families with 30k+
        // vocab". Only active when `--tokenizer` is also set (byte-
        // level BPE). The dropout encoder reads `tokenizer.json`
        // directly — see BPEDropout.swift for the rationale.
        var bpeDropout: Float = 0
        // Quantization-Aware Training. `--qat int4` or `--qat int8`
        // injects per-Linear fake-quant on every forward — round to a
        // per-output-row symmetric int grid, dequantise, propagate the
        // gradient via straight-through. Lifts deployment quality at
        // the matching int width by training the optimiser to route
        // around the quantisation noise.
        var qatBits: Int? = nil
        // Async batch pipeline (item #4 of the CPU-speedup bundle). When
        // on, a background thread builds the next batch's MLXArrays
        // while the current step's forward/backward runs on the GPU.
        // Off by default — measured wins are 0-5% on tiny presets, up
        // to ~10% on huge with large micro-batches. See
        // `docs/cpu_speedup_results.md`.
        var prefetchBatches: Bool = false
        // Sustained-load controls. `throttle=1` means no sleep; 0.5 means
        // sleep roughly one step-time after each step, halving average load.
        var throttle: Double = 1.0
        var maxStepRate: Double = 0
        var throttleFilePath: String? = nil
        var userSpecifiedOut = false
        var domainAdapt = false
        var explicitLRSchedule = false
        var explicitWarmup = false
        var explicitMaxLR = false
        var explicitMinLR = false
        var explicitDecaySteps = false
        var explicitLayerDecay = false
        let cliArgsSnapshot = args

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--preset":      preset = args[i+1]; i += 2
            case "--depth":       depthOverride = Int(args[i+1]); i += 2
            case "--steps":       steps = Int(args[i+1]) ?? steps; i += 2
            case "--corpus":      corpusPath = args[i+1]; i += 2
            case "--out":         outPath = args[i+1]; userSpecifiedOut = true; i += 2
            case "--dtype":       dtype = args[i+1]; i += 2
            case "--batch":       batchSize = Int(args[i+1]); i += 2
            case "--sample-every": sampleEvery = Int(args[i+1]) ?? sampleEvery; i += 2
            case "--throttle":    throttle = clampThrottle(Double(args[i+1]) ?? throttle); i += 2
            case "--max-step-rate": maxStepRate = max(0, Double(args[i+1]) ?? maxStepRate); i += 2
            case "--throttle-file": throttleFilePath = args[i+1]; i += 2
            case "--resume", "--base":
                                  resumePath = args[i+1]; i += 2
            case "--domain-adapt": domainAdapt = true; i += 1
            case "--save-every":  saveEvery = Int(args[i+1]); i += 2
            case "--save-history": saveHistory = true; i += 1
            case "--no-save-opt-state": saveOptState = false; i += 1
            case "--lr-schedule": lrSchedule = args[i+1]; explicitLRSchedule = true; i += 2
            case "--warmup":      warmupSteps = Int(args[i+1]) ?? warmupSteps; explicitWarmup = true; i += 2
            case "--max-lr":      maxLR = Float(args[i+1]) ?? maxLR; explicitMaxLR = true; i += 2
            case "--min-lr":      minLR = Float(args[i+1]) ?? minLR; explicitMinLR = true; i += 2
            case "--decay-steps": decaySteps = Int(args[i+1]) ?? decaySteps; explicitDecaySteps = true; i += 2
            case "--no-spike-detect": spikeDetectEnabled = false; i += 1
            case "--spike-window": spikeWindow = max(2, Int(args[i+1]) ?? spikeWindow); i += 2
            case "--spike-factor": spikeFactor = max(1.01, Float(args[i+1]) ?? spikeFactor); i += 2
            case "--log-jsonl":   logJsonlPath = args[i+1]; i += 2
            case "--eval-every":  evalEvery = Int(args[i+1]); i += 2
            case "--eval-tasks":  evalTasks = args[i+1]; i += 2
            case "--eval-limit":  evalLimit = Int(args[i+1]) ?? evalLimit; i += 2
            case "--seed":        rngSeed = UInt64(args[i+1]); i += 2
            case "--val-split":   valSplit = Double(args[i+1]) ?? valSplit; i += 2
            case "--val-every":   valEvery = Int(args[i+1]) ?? valEvery; i += 2
            case "--tokenizer":   tokenizerDir = args[i+1]; i += 2
            case "--ctx":         ctxOverride = Int(args[i+1]); i += 2
            case "--accum":       accumSteps = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--grad-clip":   gradClipNorm = Float(args[i+1]) ?? gradClipNorm; i += 2
            case "--moe-experts": moeExperts = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--moe-topk":    moeTopK = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--moe-aux-weight": moeAuxWeight = Float(args[i+1]) ?? moeAuxWeight; i += 2
            case "--mtp-horizons":   mtpHorizons = max(1, Int(args[i+1]) ?? 1); i += 2
            case "--sliding-window": slidingWindow = Int(args[i+1]); i += 2
            case "--alibi":          useALiBi = true; i += 1
            case "--mod":            useMoD = true; i += 1
            case "--diff-attn":      useDiffAttn = true; i += 1
            case "--yoco":           useYOCO = true; i += 1
            case "--grad-checkpoint": useGradCheckpoint = true; i += 1
            case "--optimizer":
                guard let k = parseOptimizerKind(args[i+1]) else {
                    fputs("unknown --optimizer '\(args[i+1])'. Pick adamw|lion|sophia|muon|adafactor.\n", stderr); exit(2)
                }
                optimizerKind = k; i += 2
            case "--galore-rank":        galoreRank = max(0, Int(args[i+1]) ?? 0); i += 2
            case "--galore-update-every": galoreUpdateEvery = max(1, Int(args[i+1]) ?? 200); i += 2
            case "--z-loss-weight":      zLossWeight = max(0, Float(args[i+1]) ?? 0); i += 2
            case "--deep-norm":          useDeepNorm = true; i += 1
            case "--lr-layer-decay":     lrLayerDecay = Float(args[i+1]) ?? 1.0; explicitLayerDecay = true; i += 2
            case "--embedding-rmsnorm":  useEmbeddingRMSNorm = true; i += 1
            case "--bpe-dropout":    bpeDropout = Float(args[i+1]) ?? bpeDropout; i += 2
            case "--prefetch":
                let v = args[i+1].lowercased()
                switch v {
                case "on", "true", "1":  prefetchBatches = true
                case "off", "false", "0": prefetchBatches = false
                default:
                    fputs("--prefetch must be on or off (got \(v))\n", stderr); exit(2)
                }
                i += 2
            case "--qat":
                let v = args[i+1].lowercased()
                switch v {
                case "int4", "4":  qatBits = 4
                case "int8", "8":  qatBits = 8
                case "off", "none", "0": qatBits = nil
                default:
                    fputs("--qat must be int4 or int8 (got \(v))\n", stderr); exit(2)
                }
                i += 2
            case "-h", "--help":  exitUsage(0)
            default:
                fputs("unknown flag: \(args[i])\n", stderr); exitUsage()
            }
        }
        if let n = evalEvery, n <= 0 {
            fputs("--eval-every must be a positive step interval\n", stderr); exitUsage()
        }
        if domainAdapt {
            guard resumePath != nil else {
                fputs("--domain-adapt requires --base <checkpoint.tinygpt> (alias of --resume)\n", stderr)
                exit(2)
            }
            if !explicitLRSchedule { lrSchedule = "wsd" }
            if !explicitWarmup { warmupSteps = 100 }
            if !explicitMaxLR { maxLR = 1e-4 }
            if !explicitMinLR { minLR = 1e-5 }
            if !explicitDecaySteps { decaySteps = max(1, steps / 20) }
            if !explicitLayerDecay { lrLayerDecay = 0.85 }
        }
        resolveOutputPaths(
            preset: preset,
            resumePath: resumePath,
            userSpecifiedOut: userSpecifiedOut,
            outPath: &outPath,
            logJsonlPath: &logJsonlPath
        )
        if userSpecifiedOut, let out = outPath {
            warnIfVolatileOutputPath(out)
        }

        // C9: seed MLXRandom BEFORE any model construction or weight init.
        // Model parameter initialization (e.g., He/Xavier init) draws from
        // MLXRandom, so seeding here makes init reproducible across runs.
        // Documented limitation: batch sampling still uses Swift stdlib
        // `Int.random`, which is non-deterministic — full determinism is a
        // v2 follow-up (replace stdlib RNG in sampleBatchRaw with a seeded
        // host generator).
        if let s = rngSeed {
            MLXRandom.seed(s)
        }

        // Model + config — either fresh from preset, or resumed from .tinygpt.
        // If --tokenizer <dir> is set OR the resumed checkpoint carries one,
        // override vocabSize from the HF tokenizer/config and switch to BPE.
        var cfg: ModelConfig
        let model: TinyGPTModel
        var startStep: Int = 0
        var resumeFile: TinyGPTFile? = nil
        if let r = resumePath {
            let url = URL(fileURLWithPath: r)
            let file: TinyGPTFile
            do { file = try TinyGPTFileReader.read(url) }
            catch { fputs("error reading resume file: \(error)\n", stderr); exit(1) }
            resumeFile = file
            let h = file.header.config
            // Resume restores the tokenizer source the model was trained with;
            // ignore --tokenizer if the resumed checkpoint already pins one,
            // because changing tokenizers mid-training corrupts learned weights.
            let resumedTokenizer = h.tokenizerSource ?? tokenizerDir
            cfg = ModelConfig(
                vocabSize: h.vocabSize ?? 256,
                contextLength: h.ctx ?? 256,
                nLayers: h.layers ?? 12,
                nHeads: h.heads ?? 8,
                dModel: h.dModel ?? 256,
                dMlp: h.dMlp ?? 1024,
                tokenizerSource: resumedTokenizer,
                // MoE: if the resumed file carries MoE metadata, restore
                // the same router/expert layout. CLI MoE flags are ignored
                // on resume — changing architecture mid-run corrupts state.
                nExperts: h.nExperts ?? 1,
                moeTopK: h.moeTopK ?? 1,
                loadBalanceWeight: h.loadBalanceWeight ?? 0.01,
                slidingWindow: h.slidingWindow,
                useMoD: h.useMoD ?? false,
                useDifferentialAttention: h.useDifferentialAttention ?? false,
                useYOCO: h.useYOCO ?? false,
                // Grad-checkpoint travels with the checkpoint so a
                // resumed long run keeps the same memory profile. CLI
                // --grad-checkpoint can ALSO promote a non-checkpointed
                // resume into a checkpointed continuation.
                useGradCheckpoint: (h.useGradCheckpoint ?? false) || useGradCheckpoint,
                // GaLore + stability bells. The architectural flag
                // (`useEmbeddingRMSNorm`, `useDeepNorm`) MUST come from the
                // saved manifest because the model layout depends on it.
                // The training-only knobs (`galoreRank`, `zLossWeight`,
                // `lrLayerDecay`) take the CLI value if set, else the
                // saved value — so a resumed run can switch GaLore on /
                // off mid-training without corrupting weights.
                galoreRank: galoreRank > 0 ? galoreRank : h.galoreRank,
                galoreUpdateEvery: h.galoreUpdateEvery ?? galoreUpdateEvery,
                zLossWeight: zLossWeight > 0 ? zLossWeight : (h.zLossWeight ?? 0),
                useDeepNorm: h.useDeepNorm ?? false,
                lrLayerDecay: h.lrLayerDecay ?? lrLayerDecay,
                useEmbeddingRMSNorm: h.useEmbeddingRMSNorm ?? false
            )
            cfg.dtype = dtype
            // QAT bits travel with the model as a config-level toggle
            // — CLI `--qat int4` can also flip a resume into QAT mode,
            // which is useful when fine-tuning a previously-fp32 model
            // into a deployment-int4 one.
            if let q = qatBits { cfg.qatBits = q }
            model = TinyGPTModel(cfg)
            do { try TinyGPTWeightLoader.load(file, into: model) }
            catch { fputs("error loading weights: \(error)\n", stderr); exit(1) }
            startStep = Int(file.step)
            print("resuming from \(r) at step \(startStep)")
        } else {
            cfg = configFor(preset)
            // nanochat-style single-knob: --depth N derives the GPT-2-shaped
            // (nLayers=N, dModel=64N, nHeads=N, dMlp=4·dModel) tuple,
            // overriding whatever the preset specified. Preset still
            // supplies ctx, vocab, dtype. Heads/layers must divide d_model;
            // the rule guarantees this by construction (dModel = 64·N,
            // nHeads = N → headDim = 64, the GPT-2 / Llama / nanochat
            // standard).
            //
            // Also force `nKvHeads = nHeads`: the preset may have set
            // nKvHeads independently for GQA, but nanochat's shape is
            // full multi-head, no GQA. Without this the
            // `nHeads % nKvHeads == 0` precondition fires when the
            // preset's nKvHeads doesn't divide our new nHeads.
            if let d = depthOverride, d >= 1 {
                cfg.nLayers = d
                cfg.dModel = 64 * d
                cfg.nHeads = d
                cfg.nKvHeads = d
                cfg.dMlp = 4 * cfg.dModel
            }
            cfg.dtype = dtype
            // Apply tokenizer override BEFORE building the model — vocabSize
            // determines the token-embedding shape.
            if let tdir = tokenizerDir {
                let hfConfigURL = URL(fileURLWithPath: tdir).appendingPathComponent("config.json")
                if let hfConfig = try? HuggingFaceConfig.read(hfConfigURL) {
                    cfg.vocabSize = hfConfig.vocabSize
                } else {
                    fputs("warning: no config.json in \(tdir) — vocabSize stays at \(cfg.vocabSize)\n", stderr)
                }
                cfg.tokenizerSource = tdir
            }
            // --ctx overrides the preset's context length. Useful when the
            // preset's default is too big for memory or when the user wants
            // longer-range BPE training (Mega default 1024 → 2048 etc).
            if let c = ctxOverride { cfg.contextLength = c }
            // MoE: convert dense MLP blocks into router + expert MLPs. Only
            // honoured on FRESH configs — resumed checkpoints keep whatever
            // structure they were saved with (MoE save/load is a follow-up,
            // see the guard below the model build).
            if moeExperts > 1 {
                cfg.nExperts = moeExperts
                cfg.moeTopK = min(moeTopK, moeExperts)
                cfg.loadBalanceWeight = moeAuxWeight
            }
            // MTP: extra heads materialise inside TinyGPTModel.init when
            // mtpHorizons > 1. They're training-only — see save guard
            // below: manifest entries don't include them, so they're
            // silently dropped on serialise.
            cfg.mtpHorizons = mtpHorizons
            // Sliding window: pure attention-mask change, no extra params.
            // The CausalSelfAttention init reads cfg.slidingWindow.
            if let sw = slidingWindow, sw > 0 { cfg.slidingWindow = sw }
            // ALiBi: when enabled, the model uses NO positional embedding
            // (the position info comes from the attention bias). We still
            // construct the positional embedding table for parameter-name
            // compatibility with the manifest — it's just frozen at init.
            cfg.useALiBi = useALiBi
            // MoD: every block gets a per-token sigmoid gate; manifest
            // gains mod_router.weight/bias per layer.
            cfg.useMoD = useMoD
            // Differential attention: every block gets a diff_attn
            // sibling with 2× Q/K + λ; manifest gains the new
            // q1_proj/k1_proj/q2_proj/k2_proj/v_proj/o_proj/lambda
            // entries per layer (the existing attn entries also stay
            // — see TransformerBlock for the rationale).
            cfg.useDifferentialAttention = useDiffAttn
            // YOCO: half the layers reuse the first half's K, V via
            // cross-attention. Manifest stays identical to the standard
            // dense path; the change is purely in forward orchestration.
            cfg.useYOCO = useYOCO
            // Gradient checkpointing — must be set BEFORE the model is
            // built so each TransformerBlock picks it up at init time.
            //
            // Curated-recipe default: auto-enable for mega/behemoth/titan
            // presets (100M+ params, where activation memory becomes the
            // bottleneck). Tiny/small/huge train fine without it; enabling
            // would just slow them down ~30% for no memory benefit.
            // Override with explicit `--grad-checkpoint` at any preset, or
            // (not supported as a flag) — to disable on mega+, the user
            // would need to pass `--no-grad-checkpoint` if we ever ship one.
            let autoGradCkpt = ["mega", "behemoth", "titan"].contains(preset)
            cfg.useGradCheckpoint = useGradCheckpoint || autoGradCkpt
            // Training-stability bells (Tier 2). Architectural flags
            // (DeepNorm scaling + embedding RMSNorm) MUST be set before
            // the model is built — they change init / layer wiring.
            // Training-only knobs (GaLore, z-loss, layer-LR decay) can
            // be set before or after build; we set them all here for
            // consistency with the manifest round-trip.
            cfg.galoreRank = galoreRank > 0 ? galoreRank : nil
            cfg.galoreUpdateEvery = galoreUpdateEvery
            cfg.zLossWeight = zLossWeight
            cfg.useDeepNorm = useDeepNorm
            cfg.lrLayerDecay = lrLayerDecay
            cfg.useEmbeddingRMSNorm = useEmbeddingRMSNorm
            // QAT bit-width — also must land in cfg BEFORE the model is
            // built so the attention/MLP modules pick it up at init.
            cfg.qatBits = qatBits
            model = TinyGPTModel(cfg)
        }
        // MoE checkpoints now serialise — the manifest gains router +
        // per-expert entries when cfg.isMoE, and the header carries
        // nExperts/moeTopK/loadBalanceWeight so resume + sample can
        // reconstruct the same router/expert layout.
        // bf16 / fp16 training: cast every floating-point parameter to the
        // target dtype. MLX propagates the dtype through all forward / loss
        // / gradient / optimizer ops, so this single cast switches the
        // whole training loop to half precision.
        //
        // bf16 keeps fp32's range (8-bit exponent), so it doesn't need the
        // loss-scaling / master-weights scaffolding fp16 training requires.
        // ~2× memory savings vs fp32 — biggest single lever for fitting
        // larger batches and longer contexts.
        if cfg.mlxDType != .float32 {
            model.apply { $0.dtype.isFloatingPoint ? $0.asType(cfg.mlxDType) : $0 }
            print("model parameters cast to \(cfg.dtype) (memory ~½ of fp32)")
        }

        // Pre-flight memory estimate — runs BEFORE the slow tokenize step so
        // a doomed config can be aborted cheaply. Activations live and die
        // within one micro-batch, so we estimate using the per-micro-batch
        // size (`--batch`), not the effective batch (which is just an
        // accumulator-trick). A >60%-of-RAM projection warns the user.
        let microBatch = batchSize ?? defaultBatch(cfg)
        let memEstimate = OOMGuard.estimate(cfg: cfg, params: model.numParameters(),
                                              batch: microBatch)
        OOMGuard.reportAndWarn(memEstimate)

        // Load the corpus. Two flavours:
        //   - byte-level (vocabSize == 256, no tokenizer): raw bytes →
        //     ByteCorpus. Same shape we've used since day one.
        //   - BPE (vocabSize from HF config): UTF-8 text → HFTokenizer.encode
        //     → TokenizedCorpus. The on-disk size becomes irrelevant; what
        //     matters is the token count.
        //
        // Both expose the same sample-batch closure shape so the training
        // loop below is corpus-flavor-agnostic.
        let sampleTrainBatch: (Int, Int) -> (MLXArray, MLXArray)
        let valSampleBatch: ((Int, Int) -> (MLXArray, MLXArray))?
        let corpusSummary: String
        let trainSummary: String
        let valSummary: String
        if cfg.tokenizerSource != nil {
            let tokDir = URL(fileURLWithPath: cfg.tokenizerSource!)
            print("loading BPE tokenizer from \(tokDir.lastPathComponent)…")
            let tok: HFTokenizer
            do { tok = try HFTokenizer.loadBlocking(from: tokDir) }
            catch { fputs("tokenizer load failed: \(error)\n", stderr); exit(1) }
            guard let p = corpusPath else {
                fputs("--corpus is required when --tokenizer is set\n", stderr); exit(1)
            }
            let corpusURL = URL(fileURLWithPath: p)
            // Persistent token cache — keyed on (corpus, tokenizer, size,
            // mtime, vocab) so any change forces a fresh tokenize. Saves
            // 10-30 min on big corpora across re-runs / --resume cycles.
            let cacheURL = TokenCache.cacheURL(corpus: corpusURL, tokenizerDir: tokDir,
                                                vocabSize: cfg.vocabSize)
            let fileSize = ((try? FileManager.default.attributesOfItem(atPath: p))?[.size]
                            as? NSNumber)?.intValue ?? 0
            // When BPE-dropout is active we'll stream-tokenize anyway, so
            // skip the up-front encode entirely.
            let tokens: [Int32]
            if bpeDropout > 0 {
                tokens = []
            } else if let cu = cacheURL, let cached = TokenCache.read(cu) {
                tokens = cached
                print("loaded \(formatLargeInt(tokens.count)) tokens from cache: \(cu.lastPathComponent)")
            } else {
                let text: String
                do { text = try String(contentsOfFile: p, encoding: .utf8) }
                catch { fputs("error reading corpus: \(error)\n", stderr); exit(1) }
                print("encoding corpus (\(formatBytes(text.utf8.count)))…")
                let ids: [Int]
                do { ids = try tok.encode(text) }
                catch { fputs("tokenize failed: \(error)\n", stderr); exit(1) }
                tokens = ids.map { Int32($0) }
                if let cu = cacheURL {
                    do {
                        try TokenCache.write(tokens, to: cu)
                        print("cached \(formatLargeInt(tokens.count)) tokens → \(cu.lastPathComponent)")
                    } catch {
                        // Non-fatal — next run just re-tokenizes.
                        fputs("warning: cache write failed (\(error))\n", stderr)
                    }
                }
            }
            // BPE-dropout (Provilkov et al., 2020): re-tokenise each batch
            // on the fly with random merge skips. We load merges + vocab
            // directly from tokenizer.json (path b — see BPEDropout.swift)
            // because swift-transformers' BPE is module-internal.
            //
            // We only switch to streaming when bpeDropout > 0 AND the
            // tokenizer.json describes a byte-level BPE model — otherwise
            // we silently keep the cached path (no regularisation, full
            // speed).
            var streamCorpus: StreamingTokenizedCorpus? = nil
            var streamVal: TokenizedCorpus? = nil
            if bpeDropout > 0 {
                let tokenizerJSON = tokDir.appendingPathComponent("tokenizer.json")
                let loadedDropTok: BPEDropoutTokenizer?
                do { loadedDropTok = try BPEDropoutTokenizer.loadFromTokenizerJSON(tokenizerJSON) }
                catch { loadedDropTok = nil }
                if let dropTok = loadedDropTok, dropTok.isByteLevel {
                    let text: String
                    do { text = try String(contentsOfFile: p, encoding: .utf8) }
                    catch { fputs("error reading corpus for streaming: \(error)\n", stderr); exit(1) }
                    print("BPE-dropout: streaming corpus with p_drop=\(bpeDropout) (re-tokenizing per batch)")
                    let stream = StreamingTokenizedCorpus(
                        text: text, tokenizer: dropTok,
                        vocabSize: cfg.vocabSize, pDrop: bpeDropout
                    )
                    let (tr, va) = stream.split(valSplit: valSplit)
                    streamCorpus = tr
                    streamVal = va
                } else {
                    fputs("warning: --bpe-dropout requested but tokenizer isn't byte-level BPE — skipping.\n", stderr)
                }
            }
            if let tr = streamCorpus {
                let va = streamVal
                sampleTrainBatch = { B, T in tr.sampleBatch(batchSize: B, contextLength: T) }
                valSampleBatch = va.map { v in { B, T in v.sampleBatch(batchSize: B, contextLength: T) } }
                corpusSummary = "\(corpusPath ?? "<text>") (\(formatBytes(fileSize)) · streaming · BPE-dropout=\(bpeDropout) · vocab=\(cfg.vocabSize))"
                trainSummary = "\(formatBytes(tr.text.utf8.count)) (streamed)"
                valSummary = va.map { "\(formatLargeInt($0.tokens.count)) tokens (frozen)" } ?? "—"
            } else {
                let full = TokenizedCorpus(tokens: tokens, vocabSize: cfg.vocabSize)
                let (tr, va) = full.split(valSplit: valSplit)
                sampleTrainBatch = { B, T in tr.sampleBatch(batchSize: B, contextLength: T) }
                valSampleBatch = va.map { v in { B, T in v.sampleBatch(batchSize: B, contextLength: T) } }
                corpusSummary = "\(corpusPath ?? "<text>") (\(formatBytes(fileSize)) · \(formatLargeInt(tokens.count)) BPE tokens · vocab=\(cfg.vocabSize))"
                trainSummary = "\(formatLargeInt(tr.tokens.count)) tokens"
                valSummary = va.map { "\(formatLargeInt($0.tokens.count)) tokens" } ?? "—"
            }
        } else {
            let corpusFull: ByteCorpus
            if let p = corpusPath {
                do {
                    corpusFull = try ByteCorpus(contentsOf: URL(fileURLWithPath: p))
                } catch {
                    fputs("error reading corpus: \(error)\n", stderr); exit(1)
                }
            } else {
                print("⚠ no --corpus given, training on random bytes (loss will land at ~ln(256)=5.55)")
                let randomBytes = (0..<1_000_000).map { _ in UInt8.random(in: 0...255) }
                corpusFull = ByteCorpus(Data(randomBytes))
            }
            let (tr, va) = TrainSupport.splitCorpus(corpusFull, valSplit: valSplit)
            sampleTrainBatch = { B, T in tr.sampleBatch(batchSize: B, contextLength: T) }
            valSampleBatch = va.map { v in { B, T in v.sampleBatch(batchSize: B, contextLength: T) } }
            corpusSummary = "\(corpusPath ?? "<random>") (\(formatBytes(corpusFull.bytes.count)) · byte-level)"
            trainSummary = formatBytes(tr.bytes.count)
            valSummary = va.map { formatBytes($0.bytes.count) } ?? "—"
        }

        // Trainer. The compile path is **the** biggest lever — see
        // `docs/cpu_speedup_results.md` — so we lean into it whenever
        // the optimiser kind supports it.
        //
        // The CPU-speedup bundle introduced two new compile sub-paths:
        //   * `useCompiledLR` — bake the LR scalar as part of the
        //     optimiser's `innerState()` MLXArrays. The cosine/warmup
        //     scheduler mutates the array in place each step; the
        //     compiled trace stays valid. Today only `--optimizer adamw`
        //     ships a compile-friendly LR-mutable optimiser.
        //   * `accumMicroBatches` — fold the N-step gradient accumulation
        //     loop INTO the compiled trace. N is fixed-at-compile so the
        //     trace shape is stable. ~30-50% faster than the legacy
        //     host-loop fallback on `--accum >= 2`.
        //
        // Both light up only when the optimiser is AdamW *and* GaLore
        // is off (GaLore mutates projector state out-of-graph). The
        // legacy compile-off-when-scheduled / compile-off-when-accum
        // behaviour is preserved verbatim for non-AdamW optimisers and
        // for the GaLore path, both of which fall through to the
        // existing host-loop code below.
        let useSchedule = (lrSchedule == "cosine" || lrSchedule == "wsd" || warmupSteps > 0)
        // WSD: auto-default to 10% of total steps if --decay-steps wasn't passed.
        let effectiveDecaySteps: Int = decaySteps > 0 ? decaySteps : max(1, steps / 10)
        // Single dispatch point for the LR schedule. Used at init and per-step.
        let lrAtStep: (Int) -> Float = { step in
            switch lrSchedule {
            case "wsd":
                return lrAtWSD(
                    step: step, total: steps, warmup: warmupSteps,
                    decaySteps: effectiveDecaySteps,
                    maxLR: maxLR, minLR: minLR
                )
            case "constant":
                return maxLR
            default:  // cosine
                return TrainSupport.lrAt(
                    step: step, total: steps, warmup: warmupSteps,
                    maxLR: maxLR, minLR: minLR
                )
            }
        }
        let initialLR: Float = useSchedule ? lrAtStep(startStep) : maxLR
        let B = batchSize ?? defaultBatch(cfg)
        let galoreActive = (cfg.galoreRank ?? 0) > 0
        // Pure constant-LR + no-accum: original compile path.
        let legacyCompile = !useSchedule && accumSteps == 1 && !galoreActive
        // New paths — only AdamW + non-GaLore. (`--qat` is OK; it lives
        // inside the model's Linear forward, not the optimiser.)
        let adamwCompileEligible = (optimizerKind == .adamw) && !galoreActive
        // Benchmark instrumentation env vars (not user-facing) — let the
        // bundle benchmark runner toggle items #1/#2 off without rebuilding.
        let disableCompiledLR = ProcessInfo.processInfo.environment["TINYGPT_DISABLE_COMPILED_LR"] == "1"
        let disableFusedAccum = ProcessInfo.processInfo.environment["TINYGPT_DISABLE_FUSED_ACCUM"] == "1"
        let wantCompiledLR = adamwCompileEligible && useSchedule && !disableCompiledLR
        let wantCompiledAccum = adamwCompileEligible && accumSteps > 1 && !disableFusedAccum
        let canCompile = legacyCompile || wantCompiledLR || wantCompiledAccum
        let effectiveClip: Float? = gradClipNorm > 0 ? gradClipNorm : nil
        // Build the GaLore manager iff requested. Pass `nil` when off
        // — the trainer treats it as a no-op.
        let galoreManager: GaLoreManager? = galoreActive
            ? GaLoreManager(rank: cfg.galoreRank!,
                             updateEvery: cfg.galoreUpdateEvery ?? 200)
            : nil
        let trainer = Trainer(model: model, learningRate: initialLR,
                              compileStep: canCompile,
                              gradClipNorm: effectiveClip,
                              optimizer: optimizerKind,
                              galore: galoreManager,
                              lrLayerDecay: cfg.lrLayerDecay,
                              useCompiledLR: wantCompiledLR,
                              accumMicroBatches: wantCompiledAccum ? accumSteps : nil)
        if let resumeFile {
            restoreOptimizerStateIfPossible(from: resumeFile, into: trainer)
        }

        let effB = B * accumSteps
        let throttleURL = URL(fileURLWithPath: throttleFilePath ?? defaultThrottleFilePath(outPath: outPath))
        let trainEval = makeTrainEvalConfig(
            outPath: outPath,
            evalEvery: evalEvery,
            evalTasks: evalTasks,
            evalLimit: evalLimit
        )
        print("""

        TinyGPT — training run
        ---------------------
        recipe:        \(domainAdapt ? "domain-adapt (continued pretrain defaults)" : "pretrain")
        preset:        \(preset) (\(cfg.nLayers)L · d=\(cfg.dModel) · ctx=\(cfg.contextLength))\(cfg.isMoE ? " · MoE(\(cfg.nExperts) experts, top-\(cfg.moeTopK))" : "")\(cfg.mtpHorizons > 1 ? " · MTP(\(cfg.mtpHorizons) horizons)" : "")\(cfg.slidingWindow.map { " · sliding-window=\($0)" } ?? "")\(cfg.useALiBi ? " · ALiBi" : "")
        params:        \(formatLargeInt(model.numParameters()))
        vocab:         \(formatLargeInt(cfg.vocabSize))\(cfg.tokenizerSource != nil ? " (BPE)" : " (byte-level)")
        dtype:         \(cfg.dtype)
        batch size:    \(B)\(accumSteps > 1 ? " × \(accumSteps) accum = \(effB) effective" : "")
        steps:         \(startStep) → \(steps)
        corpus:        \(corpusSummary)
        train/val:     \(trainSummary) / \(valSummary)
        lr schedule:   \(lrSchedule)\(useSchedule ? " (warmup \(warmupSteps), max \(maxLR), min \(minLR)\(lrSchedule == "wsd" ? ", decay \(effectiveDecaySteps)" : ""))" : " @ \(maxLR)")
        seed:          \(rngSeed.map { "\($0) (deterministic init; batch sampling NOT yet covered — see docs/determinism.md)" } ?? "random (non-deterministic)")
        optimizer:     \(optimizerKind.rawValue)
        grad clip:     \(effectiveClip.map { "global L2 ≤ \($0)" } ?? "off")
        grad ckpt:     \(cfg.useGradCheckpoint ? "on (per-block VJP recompute · ~30% slower, ~√L activation mem)" : "off")
        galore:        \(galoreActive ? "rank=\(cfg.galoreRank!) · refresh every \(cfg.galoreUpdateEvery ?? 200) steps" : "off")
        z-loss:        \(cfg.zLossWeight > 0 ? String(format: "weight=%.1e", cfg.zLossWeight) : "off")
        deep-norm:     \(cfg.useDeepNorm ? String(format: "on (α=%.3f, β=%.3f)", cfg.deepNormAlpha, cfg.deepNormBeta) : "off")
        layer-lr decay:\(cfg.lrLayerDecay < 0.9999 ? String(format: " %.3f (deepest layer @ full LR, shallowest @ %.1f%%)", cfg.lrLayerDecay, 100 * pow(cfg.lrLayerDecay, Float(cfg.nLayers - 1))) : " off")
        embed RMSNorm: \(cfg.useEmbeddingRMSNorm ? "on" : "off")
        qat:           \(cfg.qatBits.map { "int\($0) fake-quant + STE on every Linear" } ?? "off")
        save-every:    \(saveEvery.map { "\($0) steps · atomic" } ?? "end only")
        throttle:      \(String(format: "%.0f%%", throttle * 100))\(maxStepRate > 0 ? " · max \(String(format: "%.2f", maxStepRate)) step/s" : "") · control \(throttleURL.path)
        eval:          \(trainEval.map { "every \($0.every) steps · tasks=\($0.tasks) · limit=\($0.limit) · out=\($0.outJsonl.path)" } ?? "off")
        compile:       \(compileLabel(canCompile: canCompile, wantCompiledLR: wantCompiledLR, wantCompiledAccum: wantCompiledAccum, galoreActive: galoreActive, useSchedule: useSchedule, accumSteps: accumSteps, optimizerKind: optimizerKind))
        prefetch:      \(prefetchBatches ? "on (background batch pipeline, capacity=2)" : "off")
        device:        \(Device.defaultDevice())

        """)
        fflush(stdout)

        writeRunReadme(
            runOutPath: outPath,
            cliArgs: cliArgsSnapshot,
            preset: preset,
            cfg: cfg,
            corpusPath: corpusPath,
            steps: steps,
            startStep: startStep
        )

        // Active-run lock — cleared on clean exit (SIGINT included).
        if let out = outPath, let log = logJsonlPath {
            let lock = RunLockFile(
                pid: ProcessInfo.processInfo.processIdentifier,
                logJsonlPath: log,
                canonicalOutPath: out,
                startedAt: ISO8601DateFormatter().string(from: Date()),
                totalSteps: steps
            )
            try? RunLockFile.write(lock)
        }

        // C10 training dashboard log emitter. Append-only JSONL stream
        // consumed by browser/src/pages/training-dashboard. Off unless
        // --log-jsonl <path> is passed. Best-effort — failure to open
        // logs to stderr but does not abort the run.
        let trainLog: TrainLog? = logJsonlPath.flatMap { p in
            if let log = TrainLog(path: p) {
                log.meta(preset: preset, depth: depthOverride,
                         lrSchedule: lrSchedule, warmup: warmupSteps,
                         maxLR: maxLR, minLR: minLR,
                         decaySteps: lrSchedule == "wsd" ? effectiveDecaySteps : nil,
                         totalSteps: steps, params: model.numParameters(),
                         batch: B, ctx: cfg.contextLength,
                         seed: rngSeed)
                return log
            }
            fputs("[log-jsonl] failed to open \(p) — proceeding without log\n", stderr)
            return nil
        }

        // Install SIGINT handler so Ctrl-C flushes a final checkpoint
        // instead of dying mid-step.
        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        // Reset MLX's peak-memory counter at the start of training so
        // the post-run report reflects what training actually consumed
        // (and doesn't include loader/init transients). Always-on; the
        // post-run report uses the same value either way.
        MLX.Memory.peakMemory = 0  // setter triggers mlx_reset_peak_memory

        let t0 = Date()
        var lastLoss: Float = 0
        var lastValLoss: Float? = nil

        // Closure-based sampling — works for both byte and BPE corpora.
        //
        // Item #4 of the CPU-speedup bundle: when `--prefetch on`, spin
        // a background thread that builds the next batch's MLXArrays
        // (random byte/token sampling + Int32 fill + MLXArray buffer
        // copy) while the training thread is busy launching kernels for
        // the current step. The pipeline is bounded — capacity 2 means
        // at most one batch sits ready ahead of the consumer.
        //
        // When off, the previous synchronous sampling path runs
        // unchanged (zero overhead).
        let pipeline: BatchPipeline? = prefetchBatches
            ? BatchPipeline(sampler: sampleTrainBatch, batchSize: B,
                            contextLength: cfg.contextLength, capacity: 2)
            : nil
        // Helper closure that calls into the pipeline when active and
        // falls back to direct sampling otherwise.
        let nextBatch: () -> (MLXArray, MLXArray) = {
            if let p = pipeline { return p.next() }
            return sampleTrainBatch(B, cfg.contextLength)
        }
        var stoppedEarly = false
        var lastStep = startStep
        var activeTrainEval: Process? = nil
        // Pausable-training config — cooperative pause on thermal pressure
        // or battery discharge (despite AC). Polled every 50 steps to
        // avoid IOKit churn. When triggered: same path as SIGINT —
        // atomically save the final checkpoint + exit 0 so the user (or
        // a wrapper script) can `--resume` when conditions clear.
        var pauseCfg = PowerMonitor.PauseConfig()
        // Pause checks are enabled by default. Set TINYGPT_NO_POWER_PAUSE=1
        // to disable (useful for benchmarks where we don't want the run
        // to bail on thermal noise).
        let powerPauseEnabled = ProcessInfo.processInfo.environment["TINYGPT_NO_POWER_PAUSE"] != "1"
        _ = pauseCfg  // suppresses warning until cfg flags wire up
        // Loss-spike detector instance — used only when --no-spike-detect
        // isn't set. Warms up silently for the first `spikeWindow` steps.
        var spikeDetector = LossSpikeDetector(
            window: spikeWindow, factor: spikeFactor
        )
        for step in startStep..<steps {
            let stepStartedAt = Date()
            // LR schedule update. When `useCompiledLR` is on (AdamW +
            // schedule), this still works: `optimizer.learningRate =`
            // routes through `CompiledAdamW.learningRate.set` which
            // `_updateInternal`s the underlying MLXArray in place — the
            // compiled trace keeps using the same scalar object.
            if useSchedule {
                trainer.optimizer.learningRate = lrAtStep(step)
            }

            // Cooperative power/thermal pause check. Every 50 steps to
            // avoid IOKit overhead. Same exit path as SIGINT.
            if powerPauseEnabled && step > startStep && (step - startStep) % 50 == 0 {
                let snapshot = PowerMonitor.sample()
                let (shouldPause, reason) = PowerMonitor.shouldPause(
                    snapshot: snapshot, cfg: pauseCfg
                )
                if shouldPause {
                    fputs("\n[power-pause] \(reason ?? "?") — flushing checkpoint and exiting cleanly\n", stderr)
                    fputs("[power-pause] resume with: tinygpt train --resume \(outPath ?? "?.tinygpt") --steps \(steps)\n", stderr)
                    TrainSupport.stopRequested.set()
                }
            }

            if accumSteps == 1 {
                let (x, y) = nextBatch()
                lastLoss = trainer.step(inputs: x, targets: y)
            } else {
                // Collect N micro-batches before one optimizer update.
                // Effective batch becomes B × accumSteps with the memory
                // cost of just B.
                var micros: [(MLXArray, MLXArray)] = []
                micros.reserveCapacity(accumSteps)
                for _ in 0..<accumSteps {
                    micros.append(nextBatch())
                }
                lastLoss = trainer.accumulatedStep(microBatches: micros)
            }
            lastStep = step + 1

            if step == startStep || (step + 1) % 100 == 0 {
                throttle = readThrottleControlFile(throttleURL) ?? throttle
            }
            applyThrottleSleep(
                stepStartedAt: stepStartedAt,
                throttle: throttle,
                maxStepRate: maxStepRate
            )
            if (step == startStep || (step + 1) % 100 == 0) && (throttle < 0.999 || maxStepRate > 0) {
                let elapsedStep = -stepStartedAt.timeIntervalSinceNow
                if elapsedStep > 0 {
                    fputs(String(format: "[throttle] effective rate %.2f step/s (%.0f%%)\n",
                                 1.0 / elapsedStep, throttle * 100), stderr)
                }
            }

            // Loss-spike detector (observe-only v1). Logs a warning when
            // the new loss exceeds `spikeFactor × moving-average` over the
            // last `spikeWindow` steps. v2 will add rollback; v1 just
            // surfaces the spike so the operator can investigate.
            var lastSpikeMA: Float? = nil
            var lastSpikeWasFiring: Bool? = nil
            if spikeDetectEnabled {
                let (spike, ma) = spikeDetector.observe(loss: lastLoss, step: step)
                lastSpikeMA = ma > 0 ? ma : nil
                lastSpikeWasFiring = spike
                if spike {
                    fputs(String(format:
                        "\n[spike] step %d: loss %.3f > %.1f × moving-avg %.3f over last %d steps. Investigate or --resume from the latest checkpoint with a lower LR.\n",
                        step + 1, lastLoss, spikeFactor, ma, spikeWindow), stderr)
                }
            }

            // Append-only training log (C10). One JSON object per step.
            // The viewer treats `step_per_s` and `peak_rss_mb` as optional
            // — we only attach them on the same 50-step cadence as the
            // stdout status line to avoid IO churn on every step.
            if let log = trainLog {
                let isStatusStep = (step == 0 || (step + 1) % 50 == 0 || step == steps - 1)
                let elapsedNow = -t0.timeIntervalSinceNow
                let doneNow = step - startStep + 1
                let sps: Double? = isStatusStep && elapsedNow > 0
                    ? Double(doneNow) / elapsedNow : nil
                let peakMB: Double? = isStatusStep
                    ? Double(MLX.Memory.peakMemory) / (1024 * 1024) : nil
                log.step(step: step + 1, loss: lastLoss,
                         lr: trainer.optimizer.learningRate,
                         stepPerSec: sps, peakRssMB: peakMB,
                         ma: lastSpikeMA, spike: lastSpikeWasFiring)
            }

            if step == 0 || (step + 1) % 50 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let done = step - startStep + 1
                let stepsPerSec = Double(done) / elapsed
                let eta = Double(steps - step - 1) / max(stepsPerSec, 1e-6)
                let lrTag = useSchedule ?
                    String(format: "  lr=%.2e", trainer.optimizer.learningRate) : ""
                let valTag = lastValLoss.map { String(format: "  val %.3f", $0) } ?? ""
                // QAT diagnostic: relative |W − fakeQuant(W)| / |W| averaged
                // across the first attention block's q_proj weight. The
                // value is BOUNDED by (1/2) · scale / |W_typical| ≈ 1/qMax,
                // so an int4 run should sit around 1/14 ≈ 0.07 and a
                // converged QAT run trends lower as the optimiser learns
                // grid-friendly weights.
                let qatTag: String
                if let bits = cfg.qatBits, let firstBlock = model.blocks.first {
                    let err = QAT.relativeError(firstBlock.attn.qProj.weight, bits: bits)
                    qatTag = String(format: "  qat-err %.3f", err)
                } else { qatTag = "" }
                fputs(String(format: "  step %5d/%5d  loss %.3f%@%@%@  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, lrTag, valTag, qatTag, stepsPerSec, eta), stderr)
            }
            if (step + 1) % sampleEvery == 0 || step == steps - 1 {
                // Inline sample only meaningful for byte-level — BPE prints
                // would need tokenizer decode; use `tinygpt sample` instead.
                if cfg.tokenizerSource == nil {
                    printSample(model: model, cfg: cfg, tag: "step \(step + 1)")
                }
            }
            // Val loss
            if let vsb = valSampleBatch, (step + 1) % valEvery == 0 {
                var total: Float = 0
                let n = 8
                for _ in 0..<n {
                    let (vx, vy) = vsb(B, cfg.contextLength)
                    let loss = model.loss(vx, vy)
                    MLX.eval(loss)
                    total += loss.item(Float.self)
                }
                lastValLoss = total / Float(n)
                fputs(String(format: "    val loss %.3f\n", lastValLoss!), stderr)
                trainLog?.val(step: step + 1, valLoss: lastValLoss!)
            }
            // Atomic checkpoint. `--eval-every` can force checkpoint writes
            // even when --save-every is absent, because E3 evals need a stable
            // on-disk .tinygpt file to serve.
            let completedStep = step + 1
            let saveDue = saveEvery.map { completedStep % $0 == 0 } ?? false
            let evalDue = trainEval.map { completedStep % $0.every == 0 } ?? false
            if let out = outPath, saveDue || evalDue {
                do {
                    try TrainSupport.atomicSave(
                        model: model, cfg: cfg, step: completedStep, finalLoss: lastLoss,
                        weightTranspose: isLinearWeightName,
                        manifestEntries: manifestEntries,
                        optimizerMoments: saveOptState ? optimizerMoments(from: trainer) : [:],
                        to: URL(fileURLWithPath: out)
                    )
                    fputs("    ✓ checkpoint at step \(completedStep) → \(out)\n", stderr)
                    // B13 history mode: also write `<out-stem>.step-N.tinygpt`.
                    // Cheap copy from the just-saved atomic file; we don't
                    // re-serialize the model. A copy failure is logged but
                    // doesn't abort training.
                    var evalModelURL = URL(fileURLWithPath: out)
                    if saveHistory {
                        let outURL = URL(fileURLWithPath: out)
                        let stem = outURL.deletingPathExtension().path
                        let histURL = URL(fileURLWithPath: "\(stem).step-\(completedStep).tinygpt")
                        do {
                            if FileManager.default.fileExists(atPath: histURL.path) {
                                try FileManager.default.removeItem(at: histURL)
                            }
                            try FileManager.default.copyItem(at: outURL, to: histURL)
                            fputs("    ↳ history → \(histURL.lastPathComponent)\n", stderr)
                            evalModelURL = histURL
                        } catch {
                            fputs("    ⚠ history copy failed: \(error)\n", stderr)
                        }
                    }
                    if evalDue, let trainEval {
                        activeTrainEval = launchTrainEvalIfIdle(
                            active: activeTrainEval,
                            config: trainEval,
                            checkpoint: evalModelURL,
                            step: completedStep,
                            tokenizer: cfg.tokenizerSource
                        )
                    }
                } catch {
                    fputs("    ⚠ checkpoint save failed: \(error)\n", stderr)
                }
            }
            // Cooperative cancel
            if TrainSupport.stopRequested.isSet {
                stoppedEarly = true
                fputs("\n[SIGINT] flushing final checkpoint at step \(step + 1)…\n", stderr)
                break
            }
        }
        // Tear down the prefetcher's producer thread (if any). Safe to
        // skip the remaining queued batches — we're done with them.
        pipeline?.stop()
        let elapsed = -t0.timeIntervalSinceNow
        let stepsDone = lastStep - startStep
        let stepsPerSec = elapsed > 0 ? Double(stepsDone) / elapsed : 0
        let summary = stoppedEarly
            ? "interrupted at step \(lastStep) of \(steps) after \(String(format: "%.1f", elapsed))s · loss \(String(format: "%.3f", lastLoss))"
            : "done — \(stepsDone) steps in \(String(format: "%.1f", elapsed))s (\(String(format: "%.1f", stepsPerSec)) step/s) · final loss \(String(format: "%.3f", lastLoss))"
        print("\n\(summary)")
        // Flush + close the JSONL log. Idempotent — safe even if the run
        // was interrupted (the loop's per-step writes already captured
        // everything up to the last completed step).
        trainLog?.done(finalStep: lastStep, finalLoss: lastLoss, totalSeconds: elapsed)

        // Peak GPU-memory report. Always-on at end of training since the
        // counter was reset at start anyway; the line is one of the most
        // useful diagnostics for sizing future runs / verifying that
        // --grad-checkpoint actually reduced activation memory.
        let peak = MLX.Memory.peakMemory
        let snap = MLX.Memory.snapshot()
        print(String(format: "memory:  peak=%@  active=%@  cache=%@%@",
                      formatBytes(peak),
                      formatBytes(snap.activeMemory),
                      formatBytes(snap.cacheMemory),
                      cfg.useGradCheckpoint ? "  · grad-checkpoint=on" : ""))
        // GaLore memory budget — what a fully-GaLore-aware optimiser
        // WOULD use for the 2-D weight params. Compare against the
        // raw AdamW state for the same params.
        if let gm = galoreManager {
            print(gm.summary())
        }

        // Final save (always — covers both completion and Ctrl-C cases).
        if let out = outPath {
            print("saving to \(out)…")
            do {
                try TrainSupport.atomicSave(
                    model: model, cfg: cfg, step: lastStep, finalLoss: lastLoss,
                    weightTranspose: isLinearWeightName,
                    manifestEntries: manifestEntries,
                    optimizerMoments: saveOptState ? optimizerMoments(from: trainer) : [:],
                    to: URL(fileURLWithPath: out)
                )
                print("✓ wrote \(out)")
            } catch {
                fputs("save failed: \(error)\n", stderr); exit(1)
            }
        }
        RunLockFile.clear()
        if stoppedEarly { exit(130) }  // standard "killed by SIGINT" exit code
    }

    private struct TrainEvalConfig {
        let every: Int
        let tasks: String
        let limit: Int
        let outJsonl: URL
        let modelName: String
    }

    private static func makeTrainEvalConfig(
        outPath: String?, evalEvery: Int?, evalTasks: String, evalLimit: Int
    ) -> TrainEvalConfig? {
        guard let outPath, let evalEvery else { return nil }
        let outURL = URL(fileURLWithPath: outPath)
        let stem = outURL.deletingPathExtension()
        let evalURL = URL(fileURLWithPath: "\(stem.path)-evals.jsonl")
        return TrainEvalConfig(
            every: evalEvery,
            tasks: evalTasks,
            limit: max(1, evalLimit),
            outJsonl: evalURL,
            modelName: stem.lastPathComponent
        )
    }

    private static func clampThrottle(_ value: Double) -> Double {
        min(1.0, max(0.01, value))
    }

    private static func defaultThrottleFilePath(outPath: String?) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let dir = home
            .appendingPathComponent(".cache")
            .appendingPathComponent("tinygpt")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let stem: String
        if let outPath {
            stem = URL(fileURLWithPath: outPath).deletingPathExtension().lastPathComponent
        } else {
            stem = "train"
        }
        return dir.appendingPathComponent("\(stem).throttle").path
    }

    private static func readThrottleControlFile(_ url: URL) -> Double? {
        guard let raw = try? String(contentsOf: url, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty,
              let value = Double(raw)
        else { return nil }
        return clampThrottle(value)
    }

    private static func applyThrottleSleep(
        stepStartedAt: Date,
        throttle: Double,
        maxStepRate: Double
    ) {
        let stepTime = -stepStartedAt.timeIntervalSinceNow
        let throttleSleep: Double
        if throttle < 0.999 {
            throttleSleep = stepTime * (1.0 / max(0.01, throttle) - 1.0)
        } else {
            throttleSleep = 0
        }
        let rateSleep: Double
        if maxStepRate > 0 {
            rateSleep = max(0, (1.0 / maxStepRate) - stepTime)
        } else {
            rateSleep = 0
        }
        let sleep = max(throttleSleep, rateSleep)
        if sleep > 0 {
            Thread.sleep(forTimeInterval: sleep)
        }
    }

    private static func launchTrainEvalIfIdle(
        active: Process?,
        config: TrainEvalConfig,
        checkpoint: URL,
        step: Int,
        tokenizer: String?
    ) -> Process? {
        if let active, active.isRunning {
            fputs("    ↳ eval skipped at step \(step): previous eval still running\n", stderr)
            return active
        }
        guard let tokenizer else {
            fputs("    ↳ eval skipped at step \(step): checkpoint has no tokenizerSource\n", stderr)
            return nil
        }

        let fm = FileManager.default
        let parent = config.outJsonl.deletingLastPathComponent()
        try? fm.createDirectory(at: parent, withIntermediateDirectories: true)

        let logURL = URL(fileURLWithPath: "\(config.outJsonl.deletingPathExtension().path).step-\(step).log")
        fm.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try? FileHandle(forWritingTo: logURL)

        let proc = Process()
        let selfPath = CommandLine.arguments.first ?? "tinygpt"
        var args: [String] = []
        if selfPath.contains("/") {
            proc.executableURL = URL(fileURLWithPath: selfPath)
        } else {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            args.append(selfPath)
        }
        let servePort = 8200 + (step % 100)
        args.append(contentsOf: [
            "run-lm-eval",
            "--tinygpt-model", checkpoint.path,
            "--tokenizer", tokenizer,
            "--tasks", config.tasks,
            "--limit", "\(config.limit)",
            "--model-name", config.modelName,
            "--model-step", "\(step)",
            "--out", config.outJsonl.path,
            "--serve-port", "\(servePort)",
            "--work-dir", "/tmp/tinygpt-train-evals"
        ])
        proc.arguments = args
        if let logHandle {
            proc.standardOutput = logHandle
            proc.standardError = logHandle
        }

        do {
            try proc.run()
            fputs("    ↳ eval spawned at step \(step) → \(config.outJsonl.path) (log \(logURL.path))\n", stderr)
            return proc
        } catch {
            fputs("    ⚠ eval spawn failed at step \(step): \(error)\n", stderr)
            return nil
        }
    }

    private static func optimizerMoments(from trainer: Trainer) -> [String: (m: MLXArray, v: MLXArray)] {
        guard let adam = trainer.optimizer as? CompiledAdamW else { return [:] }
        return Dictionary(uniqueKeysWithValues: adam.exportMoments().map { ($0.name, (m: $0.m, v: $0.v)) })
    }

    private static func restoreOptimizerStateIfPossible(from file: TinyGPTFile, into trainer: Trainer) {
        guard let adam = trainer.optimizer as? CompiledAdamW else {
            fputs("[warn] optimizer state restore unsupported for \(trainer.optimizerKind.rawValue); resuming with fresh optimizer state\n", stderr)
            return
        }
        let moments = optimizerMoments(from: file)
        guard !moments.isEmpty else {
            fputs("[warn] no optimizer state found, resuming with fresh Adam — small loss wobble expected\n", stderr)
            return
        }
        if adam.importMoments(moments, matching: trainer.model) {
            fputs("[resume] restored Adam state for \(moments.count) tensors\n", stderr)
        } else {
            fputs("[warn] optimizer state shape mismatch, resuming with fresh Adam — small loss wobble expected\n", stderr)
        }
    }

    private static func optimizerMoments(from file: TinyGPTFile) -> [(name: String, m: MLXArray, v: MLXArray)] {
        var out: [(name: String, m: MLXArray, v: MLXArray)] = []
        out.reserveCapacity(file.tensors.count)
        for tensor in file.tensors {
            guard hasAnyNonZero(tensor.adamM) || hasAnyNonZero(tensor.adamV) else { continue }
            let shape: [Int]
            let m: MLXArray
            let v: MLXArray
            if isLinearWeightName(tensor.entry.name) && tensor.entry.shape.count == 2 {
                shape = [tensor.entry.shape[1], tensor.entry.shape[0]]
                m = MLXArray(tensor.adamM, shape, dtype: .float32).transposed()
                v = MLXArray(tensor.adamV, shape, dtype: .float32).transposed()
            } else {
                shape = tensor.entry.shape
                m = MLXArray(tensor.adamM, shape, dtype: .float32)
                v = MLXArray(tensor.adamV, shape, dtype: .float32)
            }
            out.append((name: tensor.entry.name, m: m, v: v))
        }
        return out
    }

    private static func hasAnyNonZero(_ data: Data) -> Bool {
        data.contains { $0 != 0 }
    }

    private static func printSample(model: TinyGPTModel, cfg: ModelConfig, tag: String) {
        let promptBytes: [UInt8] = [UInt8]("The ".utf8)
        var idx = MLXArray(promptBytes.map { Int32($0) }, [1, promptBytes.count])
        var bytes = promptBytes
        for _ in 0..<60 {
            let T = idx.shape.last!
            let lo = max(0, T - cfg.contextLength)
            let cond = idx[0..., lo..<T]
            let logits = model(cond)
            let last = logits[0..., logits.shape[1] - 1, 0...]
            let next = argMax(last / MLXArray(Float(0.8)), axis: -1).reshaped([1, 1])
            eval(next)
            let id = Int(next.item(Int32.self))
            bytes.append(UInt8(id & 0xff))
            idx = concatenated([idx, next.asType(idx.dtype)], axis: 1)
        }
        let s = String(bytes: bytes, encoding: .utf8) ?? "<non-utf8>"
        let clipped = s.prefix(120).replacingOccurrences(of: "\n", with: "\\n")
        fputs("    [\(tag) sample] \(clipped)\n", stderr)
    }

    /// Param-name manifest order — must match the existing file format so
    /// saves are interoperable with the browser.
    ///
    /// For dense models the layout is fixed (token + position embedding,
    /// ln_final, then per-block ln1/attn/ln2/mlp). For MoE models the
    /// per-block MLP entries are replaced by router + per-expert MLPs;
    /// non-MoE blocks-of-MoE-models don't exist (the choice is uniform
    /// across blocks for now). The browser doesn't load MoE yet, so the
    /// MoE manifest is a Mac-side extension.
    static func manifestEntries(_ cfg: ModelConfig) -> [TinyGPTHeader.TensorEntry] {
        var entries: [TinyGPTHeader.TensorEntry] = []
        var offset = 0
        let push: (String, [Int]) -> Void = { name, shape in
            let size = shape.reduce(1, *)
            entries.append(.init(name: name, shape: shape, floatOffset: offset))
            offset += size
        }
        let C = cfg.dModel, M = cfg.dMlp
        push("token_embedding.weight", [cfg.vocabSize, C])
        push("position_embedding.weight", [cfg.contextLength, C])
        // Embedding-output RMSNorm — only present when the model was
        // built with `useEmbeddingRMSNorm`. Stored RIGHT AFTER the
        // embedding tables (and BEFORE the per-block entries) so the
        // tensor offset layout stays deterministic.
        if cfg.useEmbeddingRMSNorm {
            push("embed_norm.weight", [C])
        }
        push("ln_final.weight", [C])
        push("ln_final.bias", [C])
        for i in 0..<cfg.nLayers {
            push("blocks.\(i).ln1.weight", [C])
            push("blocks.\(i).ln1.bias", [C])
            push("blocks.\(i).attn.q_proj.weight", [C, C])
            push("blocks.\(i).attn.q_proj.bias", [C])
            push("blocks.\(i).attn.k_proj.weight", [C, C])
            push("blocks.\(i).attn.k_proj.bias", [C])
            push("blocks.\(i).attn.v_proj.weight", [C, C])
            push("blocks.\(i).attn.v_proj.bias", [C])
            push("blocks.\(i).attn.o_proj.weight", [C, C])
            push("blocks.\(i).attn.o_proj.bias", [C])
            push("blocks.\(i).ln2.weight", [C])
            push("blocks.\(i).ln2.bias", [C])
            if cfg.isMoE {
                // Router: bias-free Linear(d_model → n_experts).
                push("blocks.\(i).moe.router.weight", [cfg.nExperts, C])
                // Each expert is an MLP — same fc_in/fc_out structure
                // as the dense path, replicated N times per block.
                for e in 0..<cfg.nExperts {
                    push("blocks.\(i).moe.experts.\(e).fc_in.weight", [M, C])
                    push("blocks.\(i).moe.experts.\(e).fc_in.bias", [M])
                    push("blocks.\(i).moe.experts.\(e).fc_out.weight", [C, M])
                    push("blocks.\(i).moe.experts.\(e).fc_out.bias", [C])
                }
            } else {
                push("blocks.\(i).mlp.fc_in.weight", [M, C])
                push("blocks.\(i).mlp.fc_in.bias", [M])
                push("blocks.\(i).mlp.fc_out.weight", [C, M])
                push("blocks.\(i).mlp.fc_out.bias", [C])
            }
            // MoD: per-block sigmoid gate. Linear(d_model → 1) with bias.
            // Tiny — adds C + 1 params per layer.
            if cfg.useMoD {
                push("blocks.\(i).mod_router.weight", [1, C])
                push("blocks.\(i).mod_router.bias", [1])
            }
            // Differential attention extras (Ye et al., 2024). The
            // standard attn entries above are still emitted — the
            // diff_attn sibling adds its own. Bias presence follows
            // cfg.attnBias just like the standard path.
            if cfg.useDifferentialAttention {
                push("blocks.\(i).diff_attn.q1_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.q1_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.k1_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.k1_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.q2_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.q2_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.k2_proj.weight", [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.k2_proj.bias", [C]) }
                push("blocks.\(i).diff_attn.v_proj.weight",  [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.v_proj.bias",  [C]) }
                push("blocks.\(i).diff_attn.o_proj.weight",  [C, C])
                if cfg.attnBias { push("blocks.\(i).diff_attn.o_proj.bias",  [C]) }
                push("blocks.\(i).diff_attn.lambda", [])
            }
        }
        return entries
    }

    /// Predicate for Linear-weight transpose on save (PyTorch [out,in] → WASM [in,out]).
    static func isLinearWeightName(_ name: String) -> Bool {
        guard name.hasSuffix(".weight") else { return false }
        if name == "token_embedding.weight" || name == "position_embedding.weight" {
            return false
        }
        if name.hasSuffix(".ln1.weight") || name.hasSuffix(".ln2.weight")
            || name == "ln_final.weight" || name == "embed_norm.weight" {
            return false
        }
        return true
    }

    private static func configFor(_ preset: String) -> ModelConfig {
        switch preset.lowercased() {
        case "tiny":     return ModelConfig(vocabSize: 256, contextLength: 128, nLayers: 4,
                                             nHeads: 4, dModel: 128, dMlp: 512)
        case "small":    return ModelConfig(vocabSize: 256, contextLength: 256, nLayers: 6,
                                             nHeads: 6, dModel: 192, dMlp: 768)
        case "huge":     return ModelConfig.huge
        case "mega":     return ModelConfig.mega
        case "behemoth": return ModelConfig.behemoth
        case "titan":    return ModelConfig.titan
        default:
            fputs("unknown preset: \(preset). Choose tiny|small|huge|mega|behemoth|titan.\n", stderr)
            exit(2)
        }
    }

    /// Human-readable status line for the `compile:` row of the run
    /// banner. The matrix is small but real: legacy compile vs. new
    /// LR-mutable compile vs. compiled accumulation vs. off-for-reason.
    private static func compileLabel(
        canCompile: Bool, wantCompiledLR: Bool, wantCompiledAccum: Bool,
        galoreActive: Bool, useSchedule: Bool, accumSteps: Int,
        optimizerKind: OptimizerKind
    ) -> String {
        if !canCompile {
            if galoreActive { return "off (GaLore)" }
            if useSchedule {
                return "off (LR scheduling, --optimizer \(optimizerKind.rawValue))"
            }
            if accumSteps > 1 {
                return "off (gradient accumulation, --optimizer \(optimizerKind.rawValue))"
            }
            return "off"
        }
        var bits: [String] = []
        if wantCompiledLR { bits.append("schedule-safe LR") }
        if wantCompiledAccum { bits.append("\(accumSteps)-step fused accum") }
        if bits.isEmpty { return "on" }
        return "on (\(bits.joined(separator: " + ")))"
    }

    // MARK: - Persistent output paths (docs/prds/persistent-training-output.md)

    private static func runsRoot() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/tinygpt/runs", isDirectory: true)
    }

    private static func autoRunName(preset: String) -> String {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone.current
        fmt.dateFormat = "yyyy-MM-dd-HHmmss"
        return "\(preset)-\(fmt.string(from: Date()))"
    }

    /// Default `~/.cache/tinygpt/runs/<name>/<name>.tinygpt` when `--out`
    /// is omitted. Resume-without-`--out` keeps writing to the resume path.
    private static func resolveOutputPaths(
        preset: String,
        resumePath: String?,
        userSpecifiedOut: Bool,
        outPath: inout String?,
        logJsonlPath: inout String?
    ) {
        if outPath == nil {
            if let resume = resumePath {
                outPath = resume
            } else {
                let name = autoRunName(preset: preset)
                let dir = runsRoot().appendingPathComponent(name, isDirectory: true)
                try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                outPath = dir.appendingPathComponent("\(name).tinygpt").path
            }
        }
        guard let out = outPath else { return }
        let outURL = URL(fileURLWithPath: out)
        let runDir = outURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        if logJsonlPath == nil {
            logJsonlPath = outURL.deletingPathExtension().path + ".jsonl"
        }
        _ = userSpecifiedOut  // used by caller for /tmp warning only
    }

    private static func warnIfVolatileOutputPath(_ path: String) {
        guard path.hasPrefix("/tmp/") || path == "/tmp" else { return }
        fputs("""
        [warn] --out points at /tmp — this path is wiped on Mac reboot!
        [warn] If you intend long training, use --out ~/.cache/tinygpt/runs/<name>/<name>.tinygpt
        [warn] Continuing in 3s... (Ctrl-C to abort)

        """, stderr)
        Thread.sleep(forTimeInterval: 3)
    }

    private static func corpusSHA256(path: String?) -> String? {
        guard let path,
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func writeRunReadme(
        runOutPath: String?,
        cliArgs: [String],
        preset: String,
        cfg: ModelConfig,
        corpusPath: String?,
        steps: Int,
        startStep: Int
    ) {
        guard let out = runOutPath else { return }
        let runDir = URL(fileURLWithPath: out).deletingLastPathComponent()
        let readme = runDir.appendingPathComponent("README.md")
        let started = ISO8601DateFormatter().string(from: Date())
        let corpusLine: String
        if let p = corpusPath {
            let hash = corpusSHA256(path: p) ?? "?"
            corpusLine = "- corpus: `\(p)` (sha256: `\(hash.prefix(16))…`)"
        } else {
            corpusLine = "- corpus: (random bytes — no file)"
        }
        let body = """
        # TinyGPT training run

        - started: \(started)
        - preset: `\(preset)`
        - architecture: \(cfg.nLayers)L · d=\(cfg.dModel) · heads=\(cfg.nHeads) · ctx=\(cfg.contextLength)
        - vocab: \(cfg.vocabSize)\(cfg.tokenizerSource.map { " · tokenizer `\($0)`" } ?? "")
        - dtype: \(cfg.dtype)
        - steps: \(startStep) → \(steps)
        \(corpusLine)
        - canonical output: `\(out)`

        ## CLI flags

        ```
        tinygpt train \(cliArgs.joined(separator: " "))
        ```
        """
        try? body.write(to: readme, atomically: true, encoding: .utf8)
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        if cfg.dModel >= 1024 { return 2 }
        if cfg.dModel >= 512 { return 4 }
        if cfg.dModel >= 256 { return 8 }
        return 16
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func formatBytes(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1f MB", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f KB", Double(n) / 1_000) }
        return "\(n) B"
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt train [options]

        The curated default recipe trains a stable, modern transformer with
        sensible choices baked in (bfloat16 + cosine LR + warmup + gradient
        clipping + auto grad-checkpoint on mega+ presets). Most users only
        need to set --preset, --corpus, --steps, and --out. Other flags are
        for fine-grained control or experimentation; see --help-experimental.

        Core:
          --preset tiny|small|huge|mega|behemoth|titan   (default: tiny)
          --depth N                       nanochat-style single-knob override.
                                           Sets nLayers=N, dModel=64·N, nHeads=N,
                                           dMlp=4·dModel. Preset still supplies
                                           ctx, vocab, dtype.
          --steps N                       Training steps (default: 500)
          --corpus path.txt               UTF-8 text file (default: random bytes)
          --out path.tinygpt              Checkpoint path (default:
                                           ~/.cache/tinygpt/runs/<preset>-<ts>/<name>.tinygpt)
          --dtype bfloat16|float32|float16  Training dtype (default: bfloat16)
          --batch N                       Batch size (default: by preset)
          --sample-every N                Print a sample every N steps (default: 100)
          --domain-adapt                  Continued-pretrain recipe for a domain corpus.
                                           Requires --base <checkpoint.tinygpt>. Sets
                                           conservative defaults unless explicitly
                                           overridden: --lr-schedule wsd, --warmup 100,
                                           --max-lr 1e-4, --min-lr 1e-5,
                                           --decay-steps 5% of --steps, and
                                           --lr-layer-decay 0.85.
          --base <path.tinygpt>           Alias for --resume, named for domain-adapt.
          --tokenizer <hf-dir>            Use BPE/SentencePiece from a HF model dir
                                           (vocab size comes from config.json)
          --ctx N                         Override preset's context length
          --accum N                       Gradient accumulation: N micro-batches per
                                           optimizer step (effective batch = batch × N).
                                           Disables compile.
          --grad-clip F                   Global L2 norm cap for gradients (default 1.0
                                           — standard for transformer LM training).
                                           Pass 0 to disable.
          --moe-experts N                 Mixture-of-Experts mode: N experts per block
                                           (default 1 = dense MLP). 8 is Mixtral-class.
          --moe-topk K                    Experts activated per token (default 1; 2 is
                                           Mixtral-style). Capped at --moe-experts.
          --moe-aux-weight F              Load-balance loss scale (default 0.01).
          --mtp-horizons N                Multi-Token Prediction: predict tokens t+1..t+N
                                           at every position via extra heads. Default 1.
                                           2-4 typical. Heads are training-only — saved
                                           checkpoints stay drop-in compatible.
          --sliding-window N              Restrict attention to the last N positions
                                           (Mistral / GPT-OSS recipe). Default: off
                                           (full causal). Cuts O(T²) attn to O(T·N).
          --alibi                         Use ALiBi position bias (Press et al., 2021)
                                           in lieu of positional embeddings/RoPE. Better
                                           extrapolation beyond train context length.
          --optimizer K                   AdamW (default) | lion | sophia | muon | adafactor.
                                           See docs/optimizers.md for memory + tradeoffs.
                                           Drop-in: same --max-lr / --weight-decay etc.
          --bpe-dropout F                 BPE-dropout (Provilkov et al., 2020): per-merge
                                           skip probability. 0 = off (default); 0.1 is
                                           the recommended value. Only applies when
                                           --tokenizer points to a byte-level BPE model
                                           (GPT-2/Llama/Qwen/Gemma/Phi families). The
                                           corpus is re-tokenised per batch (streaming),
                                           which is slower but lets the same text yield
                                           varied token sequences across epochs.
          --grad-checkpoint               Activation (gradient) checkpointing. Wraps each
                                           TransformerBlock's forward in a CustomFunction
                                           whose VJP recomputes the block forward at
                                           backward time. ~30% step-time overhead in
                                           exchange for dramatically lower activation
                                           memory — unlocks bigger models / batches at
                                           the cost of speed.
          --prefetch on|off               Async batch pipeline (item #4 of the CPU
                                           speedup bundle). When on, a background
                                           thread builds the next batch's MLXArrays
                                           while the current step runs. Default off
                                           — gains were 0-5% on tiny presets in
                                           measurements; only flip this on for
                                           large micro-batches where MLXArray
                                           construction starts to bite.
          --qat int4|int8                 Quantization-Aware Training: each Linear's
                                           weight is fake-quantised on every forward
                                           (round-to-nearest in a per-output-row int
                                           grid, then dequantise to fp32). Backward
                                           pass uses the straight-through estimator —
                                           gradients flow through unchanged. Improves
                                           downstream int-deployment quality at the
                                           cost of ~5-10% per step.

        Stability / memory (Tier 2):
          --galore-rank R                 GaLore (Zhao et al., 2024): project each 2-D
                                           weight gradient through a rank-R subspace
                                           before AdamW. Full fine-tuning at LoRA-rank-R
                                           optimiser memory budget. 256 is the paper
                                           default for pretraining. Forces compile off.
          --galore-update-every K         Refresh the GaLore projection basis every K
                                           steps via SVD of the current gradient. 200
                                           default; larger = more stable but slower
                                           adaptation.
          --z-loss-weight F               Add `F · (log Σ exp(logit))²` to the loss
                                           (PaLM / GShard). 1e-4 keeps logit magnitudes
                                           bounded; 0 disables.
          --deep-norm                     DeepNorm scaling for the residual stream
                                           (Wang et al., 2022). α = (2L)^¼ multiplies
                                           the residual; β = (8L)^(-¼) scales v_proj /
                                           o_proj / down_proj init. Stabilises VERY
                                           deep (>100 layer) transformers.
          --lr-layer-decay F              Layer-wise LR decay factor (0 < F ≤ 1). Each
                                           block's gradient is multiplied by
                                           F^(L - 1 - i) so deeper blocks update at
                                           the full LR. 0.85 typical for fine-tuning.
          --embedding-rmsnorm             Apply RMSNorm to the token-embedding output
                                           before positional addition. Lands a new
                                           `embed_norm.weight` tensor in the manifest.

        Long-run safety nets:
          --resume <path.tinygpt>         Continue from a saved checkpoint
                                           and restore Adam state when present
          --throttle F                    Sustained-load fraction 0.01..1.0.
                                           0.5 sleeps one step-time after each step.
          --max-step-rate N               Absolute cap in steps/sec; combines with
                                           --throttle by taking the slower setting.
          --throttle-file PATH            Poll PATH every 100 steps for live throttle
                                           updates (default ~/.cache/tinygpt/<run>.throttle)
          --save-every N                  Atomic checkpoint every N steps
          --save-history                  Also copy each save-every checkpoint to
                                           `<out-stem>.step-N.tinygpt` (B13
                                           interp-on-checkpoints). Disk-hungry.
          --no-save-opt-state             Store zero Adam moments; smaller legacy
                                           checkpoint, resume uses fresh Adam.
          --lr-schedule constant|cosine|wsd
                                          (default: cosine. `wsd` = warmup-stable-decay,
                                           MiniCPM/SmolLM-style; decay phase doubles as
                                           an annealing window.)
          --warmup N                      Warmup steps (default: 500 when cosine/wsd)
          --max-lr / --min-lr             Schedule endpoints (defaults: 3e-4 / 3e-5)
          --decay-steps N                 WSD decay window in steps (default: 10% of --steps).
                                           Ignored unless --lr-schedule wsd.
          --no-spike-detect               Disable loss-spike detector (default: on).
          --spike-window N                Moving-average window for the spike detector
                                           (default: 50 steps; min 2).
          --spike-factor F                Trigger when loss > F × moving-avg (default: 3.0).
          --log-jsonl <path.jsonl>        Append-only JSON-lines log of the training run
                                           (one record per step + val + done event).
                                           Off by default; consumed by the in-house
                                           training dashboard at /training-dashboard.
          --eval-every N                  Spawn a background run-lm-eval at checkpointed
                                           steps divisible by N. If --save-history is on,
                                           evaluates the step-N history checkpoint; otherwise
                                           evaluates the current --out checkpoint. Skips if
                                           the previous eval is still running.
          --eval-tasks <csv>              Tasks for --eval-every (default: arc_easy,gsm8k).
          --eval-limit N                  Example cap per task for --eval-every (default: 50).
                                           Eval rows append to `<out-stem>-evals.jsonl`.
          --seed N                        Seed MLXRandom for reproducible model init +
                                           GPU-side dropout/noise. Two runs with the same
                                           seed produce identical INIT loss. Batch sampling
                                           (Swift stdlib `Int.random`) is NOT seeded in v1
                                           — full bit-exact replay is a v2 follow-up.
          --val-split 0.0-0.2             Hold out last fraction of corpus for val
          --val-every N                   Eval val loss every N steps (default: 200)

        Ctrl-C flushes a final checkpoint then exits cleanly.
        """)
        exit(code)
    }
}
