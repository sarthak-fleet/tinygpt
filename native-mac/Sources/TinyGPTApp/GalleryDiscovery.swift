import Foundation
import TinyGPTIO

/// One row in the sidebar gallery — a `.tinygpt` checkpoint that the user
/// can load and generate from.
struct GalleryItem: Identifiable, Hashable {
    let id: String              // filename stem ("shakespeare")
    let displayName: String     // "Shakespeare"
    let icon: String            // emoji
    let url: URL                // path to the .tinygpt / .bin file
    let prompt: String          // suggested starting text

    static func == (a: GalleryItem, b: GalleryItem) -> Bool { a.url == b.url }
    func hash(into hasher: inout Hasher) { hasher.combine(url) }
}

/// Look in the standard locations the browser playground uses and return the
/// gallery cards we find. Empty list is fine — the app falls back to "open
/// a checkpoint" via NSOpenPanel.
enum GalleryDiscovery {
    /// Browser ships these four canonical gallery slots. Display order +
    /// pretty names + suggested prompts come from this static map (vs. the
    /// browser's `manifest.json` which duplicates the same info — keeping
    /// the Mac side independent so it doesn't break if the browser changes
    /// the manifest shape).
    private static let slots: [(id: String, name: String, icon: String, prompt: String)] = [
        ("shakespeare", "Shakespeare", "🎭", "MENENIUS:\n"),
        ("tinystories", "TinyStories", "📖", "Once upon a time"),
        ("code",        "Python code", "⌨️", "def "),
        ("chat",        "Q&A chat",    "💬", "User: "),
    ]

    static func discover() -> [GalleryItem] {
        var found: [GalleryItem] = []
        let candidates = candidatePaths()
        for (id, name, icon, prompt) in slots {
            // Try `.bin` first (gallery distribution format), then `.tinygpt`.
            for ext in ["bin", "tinygpt"] {
                for base in candidates {
                    let url = base.appendingPathComponent("\(id).\(ext)")
                    if FileManager.default.fileExists(atPath: url.path) {
                        found.append(GalleryItem(
                            id: id, displayName: name, icon: icon,
                            url: url, prompt: prompt
                        ))
                        break
                    }
                }
                if found.last?.id == id { break }
            }
        }
        // Locally-trained checkpoints — walk ~/.cache/tinygpt/runs/<name>/<name>.tinygpt.
        // These are the user's own training runs (theme-completer, N02, etc.)
        // and deserve a sidebar slot just like the curated gallery models.
        found.append(contentsOf: discoverUserRuns())
        return found
    }

    /// Scan ~/.cache/tinygpt/runs/ for user-trained .tinygpt checkpoints
    /// (one canonical per run directory). LoRA adapters (.lora files) are
    /// NOT surfaced here — they need a base model and live in their own
    /// flow (Sample tab's adapter picker, future feature).
    private static func discoverUserRuns() -> [GalleryItem] {
        let fm = FileManager.default
        let runsDir = URL(fileURLWithPath: NSString("~/.cache/tinygpt/runs").expandingTildeInPath)
        guard fm.fileExists(atPath: runsDir.path) else { return [] }
        guard let entries = try? fm.contentsOfDirectory(at: runsDir,
                                                        includingPropertiesForKeys: nil) else { return [] }
        var out: [GalleryItem] = []
        for runDir in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            where (try? runDir.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            let runName = runDir.lastPathComponent
            // Canonical pattern: <runDir>/<runName>.tinygpt. Step history
            // checkpoints (.step-N.tinygpt) are deliberately excluded —
            // they'd flood the sidebar.
            let canonical = runDir.appendingPathComponent("\(runName).tinygpt")
            if fm.fileExists(atPath: canonical.path) {
                out.append(GalleryItem(
                    id: "run-\(runName)",
                    displayName: runName,
                    icon: "🧪",
                    url: canonical,
                    prompt: ""
                ))
                continue
            }
            // Fall back: first non-step .tinygpt file in the dir.
            if let files = try? fm.contentsOfDirectory(at: runDir, includingPropertiesForKeys: nil),
               let first = files.first(where: { $0.pathExtension == "tinygpt" && !$0.lastPathComponent.contains(".step-") }) {
                out.append(GalleryItem(
                    id: "run-\(runName)",
                    displayName: runName,
                    icon: "🧪",
                    url: first,
                    prompt: ""
                ))
            }
        }
        return out
    }

    /// Where to look. The first matching file wins.
    private static func candidatePaths() -> [URL] {
        var paths: [URL] = []
        let fm = FileManager.default
        // 1. Bundle Resources/gallery (production install location).
        if let resourceURL = Bundle.main.resourceURL {
            paths.append(resourceURL.appendingPathComponent("gallery"))
        }
        // 2. Repo-relative paths — discovered by walking up from the
        //    executable. Cover both the dev layout (`data/gallery/` for
        //    locally-trained native checkpoints) and the browser ship
        //    layout (`browser/public/gallery/` for the in-browser models).
        if let exec = Bundle.main.executableURL {
            var dir = exec.deletingLastPathComponent()
            for _ in 0..<8 {
                for sub in [
                    "data/gallery",
                    "browser/public/gallery",
                    "public/gallery",
                ] {
                    let candidate = dir.appendingPathComponent(sub)
                    if fm.fileExists(atPath: candidate.path) {
                        paths.append(candidate)
                    }
                }
                dir = dir.deletingLastPathComponent()
            }
        }
        // 3. ~/Library/Application Support/TinyGPT/gallery — user cache.
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            paths.append(appSupport.appendingPathComponent("TinyGPT/gallery"))
        }
        return paths
    }
}
