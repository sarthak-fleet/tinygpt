import Foundation
import TinyGPTModel

/// `tinygpt quickstart <data>` (B33) — data → trained specialist in one
/// command. Inspect the data, auto-pick a base from the gallery, infer a
/// recipe, train, then drop the user into a side-by-side sample so they
/// can see whether it helped. The CLI sibling of B6's GUI Factory tab.
///
/// The judgement (data-shape → base + recipe) lives in
/// `TinyGPTModel.RecipeResolver` (pure, unit-tested). This command is the
/// orchestration shell: read the file, resolve the plan, print it, emit a
/// `tinygpt.project.json`, and — unless `--dry-run` — run the shipped
/// `sft` / `sample` subcommands. Designed so the model never leaves the
/// device.
enum QuickstartCommand {
    static func run(args: [String]) {
        var dataPath: String?
        var baseOverride: String?
        var galleryPath: String?
        var outPath = "adapter.lora"
        var projectOut = "tinygpt.project.json"
        var assumeYes = false
        var dryRun = false
        var sampleN = 3

        var i = 0
        while i < args.count {
            let a = args[i]
            switch a {
            case "--base": baseOverride = value(args, &i)
            case "--gallery": galleryPath = value(args, &i)
            case "--out": outPath = value(args, &i) ?? outPath
            case "--project-out": projectOut = value(args, &i) ?? projectOut
            case "--samples": sampleN = Int(value(args, &i) ?? "") ?? sampleN
            case "--yes", "-y": assumeYes = true; i += 1
            case "--dry-run": dryRun = true; i += 1
            case "-h", "--help": exitUsage(0)
            default:
                if a.hasPrefix("-") {
                    fputs("unknown flag: \(a)\n", stderr); exitUsage()
                } else if dataPath == nil {
                    dataPath = a; i += 1
                } else {
                    fputs("unexpected argument: \(a)\n", stderr); exitUsage()
                }
            }
        }
        guard let dataPath else { fputs("missing <data> path\n", stderr); exitUsage() }

        // 1. Inspect the data.
        guard let raw = try? String(contentsOfFile: dataPath, encoding: .utf8) else {
            fputs("could not read data file: \(dataPath)\n", stderr); exit(2)
        }
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let inspection = RecipeResolver.inspect(lines: lines)
        if inspection.rowCount == 0 {
            fputs("""
            no usable rows in \(dataPath).
            expected one of: chat JSONL ({"messages":[…]}), instruction JSONL
            ({"instruction","output"} or {"prompt","completion"}), tool-call
            JSONL (messages with tool_calls / a "tools" key), or a raw-text corpus.
            """, stderr)
            exit(2)
        }

        // 2. Resolve a base + recipe.
        let gallery = loadGallery(explicit: galleryPath)
        let plan = RecipeResolver.resolve(inspection: inspection, gallery: gallery, overrideBase: baseOverride)
        let manifestJSON = projectManifestJSON(plan: plan, outPath: outPath)

        // 3. Show the plan.
        printPlan(plan, dataPath: dataPath, outPath: outPath, galleryCount: gallery.count)

        if dryRun {
            print("\n--- \(projectOut) (preview) ---")
            print(manifestJSON)
            print("\n✓ dry run — nothing trained. Re-run without --dry-run to build the specialist.")
            exit(0)
        }

        // 4. Confirm (unless --yes).
        if !assumeYes {
            FileHandle.standardError.write(Data("\nProceed with training? [y/N] ".utf8))
            let answer = (readLine() ?? "").lowercased()
            guard answer.hasPrefix("y") else { print("aborted."); exit(0) }
        }

        // 5. Train → persist project → sample.
        liveRun(plan: plan, dataPath: dataPath, outPath: outPath,
                projectOut: projectOut, manifestJSON: manifestJSON, sampleN: sampleN)
    }

    // MARK: - gallery

    private static func loadGallery(explicit: String?) -> [GalleryModel] {
        let candidates = [explicit, "gallery/manifest.json", "browser/public/gallery/manifest.json"]
            .compactMap { $0 }
        for path in candidates {
            if let m = try? GalleryManifest.load(path: path) { return m.models }
        }
        return []
    }

    // MARK: - project manifest emission

    /// Build the `tinygpt.project.json` body as a dictionary so it decodes
    /// + passes `ProjectManifest.validate()` without needing the (internal)
    /// memberwise initializer from this module.
    private static func projectManifestJSON(plan: RecipeResolver.ResolvedPlan, outPath: String) -> String {
        let adapterId = URL(fileURLWithPath: outPath).deletingPathExtension().lastPathComponent
        var models: [[String: Any]] = []
        if plan.base.fromScratch {
            models.append(["id": adapterId, "role": "base"])
        } else if let baseId = plan.base.galleryId {
            models.append(["id": baseId, "role": "base"])
            models.append(["id": adapterId, "role": "adapter", "applies_to": baseId])
        }
        let dict: [String: Any] = ["name": adapterId, "models": models]
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]),
              let str = String(data: data, encoding: .utf8) else { return "{}" }
        return str
    }

    // MARK: - render

    private static func printPlan(_ plan: RecipeResolver.ResolvedPlan,
                                  dataPath: String, outPath: String, galleryCount: Int) {
        let ins = plan.inspection, r = plan.recipe
        print("")
        print("quickstart plan — \(dataPath)")
        print("  data:   \(ins.rowCount) rows, shape=\(ins.shape.rawValue)"
            + (galleryCount > 0 ? "  (gallery: \(galleryCount) models)" : "  (no gallery found)"))
        if let s = ins.sample { print("  sample: \(s)") }
        print("  base:   \(plan.base.galleryId ?? (plan.base.fromScratch ? "<from scratch>" : "<unresolved>"))")
        print("          \(plan.base.reason)")
        if r.mode == .loraFinetune {
            print("  recipe: LoRA r=\(r.rank) α=\(r.alpha) · steps=\(r.steps) · lr=\(String(format: "%g", r.lr))"
                + " · batch=\(r.batch) · max-seq=\(r.maxSeq) · neftune=\(r.neftuneAlpha)"
                + " · pack=\(r.pack)" + (r.template.map { " · template=\($0)" } ?? ""))
            print("  →       tinygpt sft <base> --data \(dataPath) --out \(outPath) " + r.sftFlags().joined(separator: " "))
        } else {
            print("  recipe: from-scratch pretrain (raw text) — steps=\(r.steps), max-seq=\(r.maxSeq)")
        }
        for w in plan.warnings { print("  ⚠ \(w)") }
    }

    // MARK: - live run

    private static func liveRun(plan: RecipeResolver.ResolvedPlan, dataPath: String,
                                outPath: String, projectOut: String,
                                manifestJSON: String, sampleN: Int) {
        guard !plan.base.fromScratch else {
            fputs("""
            from-scratch (raw-text) training isn't wired into quickstart yet.
            Use `tinygpt train` directly, or provide chat / instruction / tool-call JSONL.
            """, stderr)
            exit(2)
        }
        guard let baseRef = resolveBaseRef(plan: plan) else {
            fputs("""
            couldn't resolve a local base to train from (auto-picked gallery id
            '\(plan.base.galleryId ?? "?")' has no local weights yet).
            Pass --base <local-path-or-hf-id>, or `tinygpt pull` the gallery model first.
            """, stderr)
            exit(2)
        }

        let exe = selfExecutable()
        let sftArgs = ["sft", baseRef, "--data", dataPath, "--out", outPath] + plan.recipe.sftFlags()
        print("\n▶ training: \(exe) \(sftArgs.joined(separator: " "))")
        let status = runProcess(exe, sftArgs)
        if status != 0 { fputs("training failed (exit \(status))\n", stderr); exit(status) }

        // Persist the project file so the result is reproducible + shippable.
        do {
            try manifestJSON.write(toFile: projectOut, atomically: true, encoding: .utf8)
            print("✓ wrote \(projectOut)")
        } catch {
            fputs("warning: could not write \(projectOut): \(error)\n", stderr)
        }

        // "Try it" — N samples from the new specialist vs the base.
        guard sampleN > 0 else { return }
        let prompt = inferTryPrompt(dataPath: dataPath) ?? "Hello!"
        print("\n▶ \(sampleN) sample(s) from the specialist (adapter) — compare against the base with `\(exe) sample \(baseRef) ...`:")
        for n in 1 ... sampleN {
            print("\n  [\(n)/\(sampleN)] prompt: \(prompt)")
            _ = runProcess(exe, ["sample", baseRef, "--lora", outPath, "--prompt", prompt, "--tokens", "128"])
        }
        print("\n✓ done. Specialist adapter: \(outPath) · project: \(projectOut)")
    }

    /// Use the override / a path-like or HF-style id directly; a bare
    /// auto-picked gallery id can't be resolved to local weights in V1.
    private static func resolveBaseRef(plan: RecipeResolver.ResolvedPlan) -> String? {
        guard let id = plan.base.galleryId else { return nil }
        if FileManager.default.fileExists(atPath: id) || id.contains("/") { return id }
        return nil
    }

    /// Best-effort: pull the first user turn from chat/instruction data to
    /// use as a demo prompt. Falls back to a generic greeting.
    private static func inferTryPrompt(dataPath: String) -> String? {
        guard let raw = try? String(contentsOfFile: dataPath, encoding: .utf8),
              let first = raw.split(separator: "\n").first(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }),
              let data = first.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let msgs = obj["messages"] as? [[String: Any]],
           let user = msgs.first(where: { ($0["role"] as? String) == "user" }),
           let content = user["content"] as? String { return content }
        if let instr = obj["instruction"] as? String { return instr }
        if let prompt = obj["prompt"] as? String { return prompt }
        return nil
    }

    private static func selfExecutable() -> String {
        Bundle.main.executablePath ?? CommandLine.arguments.first ?? "tinygpt"
    }

    private static func runProcess(_ exe: String, _ args: [String]) -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        do { try p.run() } catch {
            fputs("failed to launch \(exe): \(error)\n", stderr); return 127
        }
        p.waitUntilExit()
        return p.terminationStatus
    }

    // MARK: - arg helpers

    private static func value(_ args: [String], _ i: inout Int) -> String? {
        guard i + 1 < args.count else { i += 1; return nil }
        let v = args[i + 1]; i += 2; return v
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt quickstart <data> [options]

        Turn a data file into a trained, runnable specialist in one command:
        inspect the data, auto-pick a base from the gallery, infer a LoRA
        recipe, train, and sample the result. Runs entirely on-device.

        <data>  chat JSONL ({"messages":[…]}), instruction JSONL
                ({"instruction","output"} / {"prompt","completion"}),
                tool-call JSONL, or a raw-text corpus.

        Options:
          --base <id|path|hf-id>  override the auto-picked base
          --gallery <path>        gallery manifest.json (default: ./gallery/…)
          --out <path>            adapter output (default: adapter.lora)
          --project-out <path>    project file (default: tinygpt.project.json)
          --samples <N>           demo samples after training (default: 3)
          --yes, -y               skip the train confirmation
          --dry-run               print the resolved plan + project file, don't train

        Exit code: 0 on success / dry-run; non-zero on bad data or a failed step.
        """)
        exit(code)
    }
}
