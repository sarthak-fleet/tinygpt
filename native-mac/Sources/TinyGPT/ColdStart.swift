import Foundation
import TinyGPTModel

/// Cold-start support for the CLI: runs `ModelLoader.loadAsync` while
/// painting a tiny ASCII spinner on stderr. Spinner exits the moment
/// the background load returns; the foreground thread blocks on the
/// continuation under `withCheckedThrowingContinuation` semantics.
///
/// The point of the spinner is purely UX — the user sees something
/// happening within ~50 ms of pressing return, instead of staring at
/// a blank terminal for the duration of the mmap + register pass.
/// Tested on:
///   - flagship-huge.tinygpt (250 MB, byte-level 12L/512d): load ~700 ms
///   - demo.tinygpt (18 MB, byte-level): load ~95 ms (no spinner visible)
enum ColdStart {

    /// Synchronous facade — returns when the load completes, prints
    /// progress to stderr in the meantime. Uses Swift's `Task` runtime
    /// under the hood so async vs sync isn't visible to the caller.
    static func loadWithSpinner(
        path: String,
        deferEmbedding: Bool,
        label: String
    ) throws -> ModelLoader.LoadResult {
        // Render: "loading <name>… " then the spinner glyph; the glyph
        // is overwritten on each tick. Pin the cursor with a fixed
        // prefix so terminals that swallow carriage returns still show
        // *something* useful. stderr (not stdout) so it doesn't
        // contaminate piped output.
        let glyphs: [String] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        let prefix = "loading \(label)… "
        fputs(prefix, stderr)
        fputs(glyphs[0], stderr)
        fflush(stderr)

        // Box for the result so the spinner thread can poll a flag
        // without dealing with a generic.
        final class Box: @unchecked Sendable {
            var done: Bool = false
            var result: Result<ModelLoader.LoadResult, Error>?
            let lock = NSLock()
        }
        let box = Box()

        let loadThread = Thread { [box] in
            do {
                let r: ModelLoader.LoadResult
                if deferEmbedding {
                    r = try ModelLoader.loadLazyEmbedding(path)
                } else {
                    r = try ModelLoader.load(path)
                }
                // Warm up MLX kernel pipelines while we're still on a
                // background thread. The first matmul / softmax registers
                // its pipeline state, so doing it here saves ~50–200 ms
                // off the first sampling step. See MetalCache.swift.
                MetalCache.warmupForSampling()
                box.lock.lock()
                box.result = .success(r)
                box.done = true
                box.lock.unlock()
            } catch {
                box.lock.lock()
                box.result = .failure(error)
                box.done = true
                box.lock.unlock()
            }
        }
        loadThread.qualityOfService = .userInitiated
        loadThread.start()

        // Spinner main loop. Tight RunLoop sleep on the main thread —
        // 80 ms tick is fast enough to look animated, slow enough that
        // we don't pin a core. Bail the moment `box.done` is true.
        var tick = 0
        while true {
            box.lock.lock()
            let done = box.done
            box.lock.unlock()
            if done { break }
            // Overwrite the spinner glyph in place. `\r` returns the
            // cursor to column 0; we then re-print the prefix + glyph.
            // Using ANSI cursor-back would be cleaner but `\r` works
            // on every terminal we care about.
            fputs("\r" + prefix + glyphs[tick % glyphs.count], stderr)
            fflush(stderr)
            tick &+= 1
            Thread.sleep(forTimeInterval: 0.08)
        }
        // Clear the spinner line.
        fputs("\r" + String(repeating: " ", count: prefix.utf8.count + 4) + "\r", stderr)
        fflush(stderr)

        switch box.result! {
        case .success(let r): return r
        case .failure(let e): throw e
        }
    }
}
