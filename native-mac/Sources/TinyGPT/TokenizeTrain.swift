import Foundation

/// `tinygpt tokenize-train` — thin wrapper around the repo-local Rust
/// HuggingFace tokenizers helper in `scripts/tokenizer-trainer`.
enum TokenizeTrain {
    static func run(args: [String]) {
        var passthrough = args
        if args.contains("-h") || args.contains("--help") {
            exitUsage(0)
        }
        guard let helper = findHelper() else {
            fputs("""
            tokenize-train: helper binary not found.
            Build it once with:
              cd scripts/tokenizer-trainer && cargo build --release

            Then rerun `tinygpt tokenize-train ...`.
            """, stderr)
            exit(1)
        }
        if !passthrough.contains("--model-type") {
            passthrough.append(contentsOf: ["--model-type", "bpe"])
        }

        let process = Process()
        process.executableURL = helper
        process.arguments = passthrough
        process.standardInput = FileHandle.standardInput
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        do {
            try process.run()
            process.waitUntilExit()
            exit(process.terminationStatus)
        } catch {
            fputs("tokenize-train failed to launch helper: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func findHelper() -> URL? {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let rels = [
            "scripts/tokenizer-trainer/target/release/tinygpt-tokenizer-trainer",
            "../scripts/tokenizer-trainer/target/release/tinygpt-tokenizer-trainer",
            "../../scripts/tokenizer-trainer/target/release/tinygpt-tokenizer-trainer",
        ]
        for rel in rels {
            let url = cwd.appendingPathComponent(rel)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        return nil
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt tokenize-train --corpus corpus.txt --vocab-size 32000 --out tokenizer.json [options]

        Trains a HuggingFace-compatible tokenizer.json via the repo-local
        Rust helper under scripts/tokenizer-trainer.

        --corpus <path>             Text corpus, one document per line or large blob
        --vocab-size N              Target vocabulary size (default helper: 32000)
        --model-type bpe|char       BPE v1; char uses the same byte-level BPE helper
        --special-tokens CSV        Default: <bos>,<eos>,<pad>
        --out <tokenizer.json>      Output tokenizer JSON
        """)
        exit(code)
    }
}
