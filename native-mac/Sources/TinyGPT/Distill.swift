import Foundation
import MLX
import MLXNN
import MLXOptimizers
import TinyGPTIO
import TinyGPTModel

/// `tinygpt distill` — knowledge distillation training.
///
/// Trains a STUDENT model to match a frozen TEACHER's output distribution.
/// Loss is a mix of two terms:
///
///   L = α · T² · KL( softmax(s_logits / T) ‖ softmax(t_logits / T) )
///     + (1 − α) · NLL(s_logits, true_targets)
///
/// Where T (temperature) softens the teacher's distribution so the student
/// learns the FULL "soft probabilities" — not just the argmax. The T²
/// multiplier on the KL term compensates for the gradient scaling that
/// the temperature divide introduces.
///
/// Educational/strategic value: a much smaller model can punch above its
/// weight by absorbing a larger teacher's behaviour. The flagship Phase-4
/// shipping artefact ("a 5M-param model that competes on the leaderboard").
///
/// Reference: Hinton et al., 2015 ("Distilling the Knowledge in a Neural
/// Network"). Modern temperature defaults (T=4-8, α=0.5-0.9) follow the
/// HuggingFace "distil" family recipe.
///
/// USAGE
///   tinygpt distill <student> --teacher <teacher_path> \
///       --corpus <text> --tokenizer <hf-dir> \
///       --steps 5000 --temperature 4 --alpha 0.7 --out distilled.tinygpt
///
/// Both teacher and student MUST share a tokenizer (same vocab size +
/// ids); otherwise the cross-distribution KL is meaningless. The runner
/// asserts vocab equality at startup.
enum Distill {
    static func run(args: [String]) {
        var studentPath: String? = nil
        var teacherPath: String? = nil
        var corpusPath: String? = nil
        var tokenizerDir: String? = nil
        var outPath: String? = nil
        var steps = 1000
        var lr: Float = 3e-4
        var temperature: Float = 4.0
        var alpha: Float = 0.7
        var batchSize: Int? = nil
        var ctxOverride: Int? = nil
        var gradClipNorm: Float = 1.0

        var i = 0
        while i < args.count {
            switch args[i] {
            case "--teacher":     teacherPath = args[i+1]; i += 2
            case "--corpus":      corpusPath = args[i+1]; i += 2
            case "--tokenizer":   tokenizerDir = args[i+1]; i += 2
            case "--out":         outPath = args[i+1]; i += 2
            case "--steps":       steps = Int(args[i+1]) ?? steps; i += 2
            case "--lr":          lr = Float(args[i+1]) ?? lr; i += 2
            case "--temperature", "--temp":
                                  temperature = Float(args[i+1]) ?? temperature; i += 2
            case "--alpha":       alpha = Float(args[i+1]) ?? alpha; i += 2
            case "--batch":       batchSize = Int(args[i+1]); i += 2
            case "--ctx":         ctxOverride = Int(args[i+1]); i += 2
            case "--grad-clip":   gradClipNorm = Float(args[i+1]) ?? gradClipNorm; i += 2
            case "-h", "--help":  exitUsage()
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                studentPath = args[i]; i += 1
            }
        }
        guard let studentPath = studentPath else { fputs("missing <student> path\n", stderr); exitUsage() }
        guard let teacherPath = teacherPath else { fputs("--teacher <path> required\n", stderr); exitUsage() }
        guard let corpusPath = corpusPath else { fputs("--corpus <text> required\n", stderr); exitUsage() }
        guard let outPath = outPath else { fputs("--out <path.tinygpt> required\n", stderr); exitUsage() }

        // Load student (the trainable target) and teacher (frozen).
        print("loading student from \(studentPath)…")
        let studentLoad: ModelLoader.LoadResult
        do { studentLoad = try ModelLoader.load(studentPath) }
        catch { fputs("student load failed: \(error)\n", stderr); exit(1) }
        print("loading teacher from \(teacherPath)…")
        let teacherLoad: ModelLoader.LoadResult
        do { teacherLoad = try ModelLoader.load(teacherPath) }
        catch { fputs("teacher load failed: \(error)\n", stderr); exit(1) }
        let scfg = studentLoad.config
        let tcfg = teacherLoad.config
        guard scfg.vocabSize == tcfg.vocabSize else {
            fputs("vocab mismatch: student=\(scfg.vocabSize) teacher=\(tcfg.vocabSize) — distillation needs a shared tokenizer\n", stderr)
            exit(2)
        }
        if let c = ctxOverride { /* override student ctx noted */ _ = c }

        // Distillation only makes sense for the from-scratch student here:
        // the trainable target carries the gradient, but our `valueAndGrad`
        // is specialised over TinyGPTModel/TinyGPTModelHF leaves. For the
        // first cut we require the student to be from-scratch (the common
        // case — Mega/Huge dense base trained on the same tokenizer).
        guard case .fromScratch(let studentModel) = studentLoad.model else {
            fputs("first-cut distill expects a from-scratch student (HF-student support is follow-up).\n", stderr)
            exit(2)
        }
        let teacherAny = teacherLoad.model

        // Load tokeniser + corpus. Reuses the TokenCache so a 30-minute
        // tokenize doesn't repeat for every distill experiment.
        let tokDir = tokenizerDir.map(URL.init(fileURLWithPath:))
            ?? teacherLoad.hfTokenizerDir
            ?? studentLoad.hfTokenizerDir
        guard let td = tokDir else {
            fputs("no tokenizer found — pass --tokenizer <hf-dir>\n", stderr); exit(2)
        }
        let tok: HFTokenizer
        do { tok = try HFTokenizer.loadBlocking(from: td) }
        catch { fputs("tokenizer load failed: \(error)\n", stderr); exit(1) }
        let corpusURL = URL(fileURLWithPath: corpusPath)
        let cacheURL = TokenCache.cacheURL(corpus: corpusURL,
                                            tokenizerDir: td,
                                            vocabSize: scfg.vocabSize)
        let tokens: [Int32]
        if let cu = cacheURL, let cached = TokenCache.read(cu) {
            tokens = cached
            print("loaded \(cached.count) tokens from cache: \(cu.lastPathComponent)")
        } else {
            let text = (try? String(contentsOfFile: corpusPath, encoding: .utf8)) ?? ""
            print("encoding corpus (\(text.utf8.count) bytes)…")
            let ids: [Int]
            do { ids = try tok.encode(text) }
            catch { fputs("tokenize failed: \(error)\n", stderr); exit(1) }
            tokens = ids.map { Int32($0) }
            if let cu = cacheURL { try? TokenCache.write(tokens, to: cu) }
        }
        let corpus = TokenizedCorpus(tokens: tokens, vocabSize: scfg.vocabSize)

        let B = batchSize ?? defaultBatch(scfg)
        let T = ctxOverride ?? scfg.contextLength

        print("""

        TinyGPT — knowledge distillation
        --------------------------------
        student:        \(studentPath)  (\(scfg.nLayers)L · d=\(scfg.dModel) · \(formatLargeInt(studentLoad.model.numParameters())) params)
        teacher:        \(teacherPath)  (\(tcfg.nLayers)L · d=\(tcfg.dModel) · \(formatLargeInt(teacherLoad.model.numParameters())) params)
        vocab:          \(scfg.vocabSize) (shared)
        loss:           α · T² · KL + (1-α) · NLL    [α=\(alpha) T=\(temperature)]
        steps:          \(steps)
        batch / ctx:    \(B) / \(T)
        lr:             \(lr)
        grad clip:      \(gradClipNorm > 0 ? "global L2 ≤ \(gradClipNorm)" : "off")
        device:         \(Device.defaultDevice())

        """)
        fflush(stdout)

        let stepFn = makeDistillStepFn(student: studentModel, teacher: teacherAny,
                                        lr: lr, temperature: temperature, alpha: alpha,
                                        gradClipNorm: gradClipNorm > 0 ? gradClipNorm : nil)

        TrainSupport.installSigintHandler()
        TrainSupport.stopRequested.reset()

        let t0 = Date()
        var lastLoss: Float = 0
        var stoppedEarly = false
        var lastStep = 0
        for step in 0..<steps {
            let (x, y) = corpus.sampleBatch(batchSize: B, contextLength: T)
            lastLoss = stepFn(x, y)
            lastStep = step + 1
            if step == 0 || (step + 1) % 50 == 0 || step == steps - 1 {
                let elapsed = -t0.timeIntervalSinceNow
                let sps = Double(step + 1) / elapsed
                fputs(String(format: "  step %5d/%5d  loss %.3f  · %.1f step/s · eta %.0fs\n",
                             step + 1, steps, lastLoss, sps,
                             Double(steps - step - 1) / sps), stderr)
            }
            if TrainSupport.stopRequested.isSet {
                stoppedEarly = true
                fputs("\n[SIGINT] flushing checkpoint at step \(lastStep)…\n", stderr)
                break
            }
        }
        let elapsed = -t0.timeIntervalSinceNow
        print(stoppedEarly
            ? String(format: "\ninterrupted at step %d of %d after %.1fs · loss %.3f",
                      lastStep, steps, elapsed, lastLoss)
            : String(format: "\ndone — %d steps in %.1fs (%.1f step/s) · final loss %.3f",
                      steps, elapsed, Double(steps) / elapsed, lastLoss))

        // Save the distilled student. Reuses Train.swift's manifest layout
        // so distilled checkpoints are gallery-compatible.
        do {
            try TrainSupport.atomicSave(
                model: studentModel, cfg: scfg, step: lastStep, finalLoss: lastLoss,
                weightTranspose: Train.isLinearWeightName,
                manifestEntries: Train.manifestEntries,
                to: URL(fileURLWithPath: outPath)
            )
            print("✓ wrote \(outPath)")
        } catch {
            fputs("save failed: \(error)\n", stderr); exit(1)
        }
        if stoppedEarly { exit(130) }
    }

    /// Build a per-step distillation function. Teacher forward is
    /// captured by closure so autograd treats it as a constant — no
    /// gradient flows through the teacher.
    ///
    /// Implementation note: KL(student ‖ teacher) is computed in log-
    /// probability space for numerical stability:
    ///   KL = Σ p_t · (log p_t − log p_s)
    /// We work with the SOFTENED versions (logits / T) of both, then
    /// rescale by T² to keep the gradient magnitude comparable to the
    /// unsoftened NLL term.
    private static func makeDistillStepFn(student: TinyGPTModel,
                                           teacher: AnyModel,
                                           lr: Float,
                                           temperature: Float, alpha: Float,
                                           gradClipNorm: Float?)
        -> (MLXArray, MLXArray) -> Float
    {
        let opt = AdamW(learningRate: lr, weightDecay: 0)
        let clip = gradClipNorm
        let tA = MLXArray(temperature)
        let tSqA = MLXArray(temperature * temperature)
        let alphaA = MLXArray(alpha)
        let oneMinusAlpha = MLXArray(1.0 - alpha)

        let lossFn = { (m: TinyGPTModel, x: MLXArray, y: MLXArray) -> MLXArray in
            let sLogits = m(x)
            // Teacher forward — no grad. The AnyModel-routed call is not
            // a `valueAndGrad` target so MLX treats its outputs as constants.
            let tLogits: MLXArray
            switch teacher {
            case .fromScratch(let tm): tLogits = tm(x)
            case .huggingFace(let tm): tLogits = tm(x)
            }
            let v = sLogits.shape.last!

            // Softened log-probs for both. `logSoftmax` is numerically
            // stable (subtracts the row max under the hood).
            let sLogP = MLXNN.logSoftmax(sLogits / tA, axis: -1)
            let tLogP = MLXNN.logSoftmax(tLogits / tA, axis: -1)
            let tP = MLX.exp(tLogP)
            // KL(t ‖ s) reduced over vocab → per-token scalar; mean over
            // batch × time. Sign matters: we want STUDENT to match TEACHER,
            // so the cross-entropy form is Σ t · (logt - logs).
            let klPerTok = (tP * (tLogP - sLogP)).sum(axis: -1)   // [B, T]
            let klTerm = klPerTok.mean() * tSqA

            // Hard-target NLL on the true next-token. Standard CE — keeps
            // the student grounded in real data even if the teacher is
            // wrong somewhere.
            let nll = crossEntropy(
                logits: sLogits.reshaped([-1, v]),
                targets: y.reshaped([-1]),
                reduction: .mean
            )

            return alphaA * klTerm + oneMinusAlpha * nll
        }
        let gradFn = valueAndGrad(model: student, lossFn)

        return { x, y in
            let (loss, grads) = gradFn(student, x, y)
            let final = clip.map { clipGradNorm(grads, maxNorm: $0) } ?? grads
            opt.update(model: student, gradients: final)
            MLX.eval(loss, student, opt)
            return loss.item(Float.self)
        }
    }

    private static func defaultBatch(_ cfg: ModelConfig) -> Int {
        // Distillation runs ONE forward through the (often-big) teacher
        // per step too, so per-step memory is bigger than train alone.
        // Halve Train's defaults conservatively.
        if cfg.dModel >= 1024 { return 1 }
        if cfg.dModel >= 512 { return 2 }
        if cfg.dModel >= 256 { return 4 }
        return 8
    }

    private static func formatLargeInt(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func exitUsage() -> Never {
        print("""
        usage: tinygpt distill <student> --teacher <path> --corpus <text> [options]

        --teacher <path>         Frozen teacher model (.tinygpt or HF dir) — required
        --corpus <text>          UTF-8 text to distill on — required
        --out <path.tinygpt>     Where to save the distilled student — required
        --tokenizer <hf-dir>     Tokeniser dir (auto-detected from teacher/student if pinned)
        --temperature F          KL temperature (default 4.0; 4-8 typical)
        --alpha F                KL weight in the mix (default 0.7; rest goes to NLL)
        --steps N                Training steps (default 1000)
        --lr F                   Learning rate (default 3e-4)
        --batch N                Batch size (default by student size)
        --ctx N                  Override student context length
        --grad-clip F            Global L2 grad-norm cap (default 1.0; 0 disables)

        Both teacher and student must share a tokenizer (same vocab + ids).
        Ctrl-C flushes a final checkpoint then exits cleanly.
        """)
        exit(2)
    }
}
