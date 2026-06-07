import SwiftUI
import Foundation

/// "Learn" workspace — left list of in-repo markdown files, right pane
/// renders the selected one. Replaces the prior Education stub.
///
/// Uses AppKit's `NSAttributedString(markdown:)` for rendering. Good
/// enough for headings / code / lists / inline links; not a full
/// CommonMark engine. Click "Open in Finder" for the raw .md.
struct LearnView: View {
    @State private var docs: [LearnDoc] = []
    @State private var selected: LearnDoc? = nil

    var body: some View {
        HStack(spacing: 0) {
            // Left: doc list
            VStack(alignment: .leading, spacing: 0) {
                Text("Learn")
                    .font(.tgDisplay)
                    .foregroundStyle(Theme.fg)
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                Text("docs in this repo — recipes, strategy, sessions, primers")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.muted)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 12)

                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(LearnDoc.Group.allCases, id: \.self) { group in
                            let inGroup = docs.filter { $0.group == group }
                            if !inGroup.isEmpty {
                                Text(group.rawValue.uppercased())
                                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(Theme.faint)
                                    .tracking(1)
                                    .padding(.horizontal, 20)
                                    .padding(.top, 12)
                                    .padding(.bottom, 4)
                                ForEach(inGroup) { doc in
                                    docRow(doc)
                                }
                            }
                        }
                        if docs.isEmpty {
                            Text("no docs found")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.faint)
                                .padding(20)
                        }
                    }
                }
            }
            .frame(width: 320)
            .background(Theme.panel)

            Divider().background(Theme.line)

            // Right: rendered markdown
            if let selected {
                MarkdownView(doc: selected)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 32))
                        .foregroundStyle(Theme.faint)
                    Text("pick a doc to read")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Theme.base)
            }
        }
        .onAppear { docs = LearnDoc.discoverAll() }
    }

    private func docRow(_ doc: LearnDoc) -> some View {
        let active = selected?.id == doc.id
        return Button { selected = doc } label: {
            HStack(spacing: 8) {
                Image(systemName: "doc.text")
                    .font(.system(size: 10))
                    .foregroundStyle(active ? Theme.accent : Theme.faint)
                VStack(alignment: .leading, spacing: 1) {
                    Text(doc.title)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(active ? Theme.fg : Theme.muted)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                    Text(doc.relPath)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(active ? Theme.accent.opacity(0.10) : Color.clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

private struct MarkdownView: View {
    let doc: LearnDoc

    @State private var attributed: AttributedString = AttributedString()

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(doc.title)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text(doc.relPath)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.faint)
                }
                Spacer()
                Button("Open in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([doc.url])
                }
                .controlSize(.small)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 14)
            .background(Theme.panel.opacity(0.5))
            .overlay(Rectangle().fill(Theme.line).frame(height: 1), alignment: .bottom)

            ScrollView {
                Text(attributed)
                    .font(.system(size: 13))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 24)
            }
        }
        .background(Theme.base)
        .onAppear(perform: reload)
        .onChange(of: doc.id) { _, _ in reload() }
    }

    private func reload() {
        guard let raw = try? String(contentsOf: doc.url, encoding: .utf8) else {
            attributed = AttributedString("(could not read file)")
            return
        }
        // Drop YAML frontmatter if present (--- ... ---).
        var body = raw
        if body.hasPrefix("---") {
            if let end = body.range(of: "\n---\n") ?? body.range(of: "\n---\r\n") {
                body = String(body[end.upperBound...])
            }
        }
        // Render via AttributedString. `inlineOnlyPreservingWhitespace`
        // keeps code blocks legible; full block rendering would need
        // an external library.
        do {
            attributed = try AttributedString(markdown: body,
                                              options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace,
                                                             failurePolicy: .returnPartiallyParsedIfPossible))
        } catch {
            attributed = AttributedString(body)
        }
    }
}

// MARK: - Doc discovery

struct LearnDoc: Identifiable, Hashable {
    let id: String      // url path
    let title: String
    let relPath: String
    let url: URL
    let group: Group

    enum Group: String, CaseIterable {
        case primers = "Primers"
        case recipes = "Recipes"
        case sessions = "Sessions"
        case prds = "PRDs"
        case roadmap = "Roadmap"
    }

    static func discoverAll() -> [LearnDoc] {
        guard let repo = RepoLocator.repoRoot() else { return [] }
        var docs: [LearnDoc] = []
        let dirs: [(String, Group)] = [
            ("docs/learn", .primers),
            ("docs/recipes", .recipes),
            ("docs/sessions", .sessions),
            ("docs/prds", .prds),
        ]
        let fm = FileManager.default
        for (rel, group) in dirs {
            let dir = repo.appendingPathComponent(rel)
            guard let entries = try? fm.contentsOfDirectory(at: dir,
                                                            includingPropertiesForKeys: [.contentModificationDateKey])
            else { continue }
            for url in entries where url.pathExtension == "md" {
                docs.append(LearnDoc(
                    id: url.path,
                    title: niceTitle(url),
                    relPath: rel + "/" + url.lastPathComponent,
                    url: url,
                    group: group
                ))
            }
        }
        // Roadmap: top-level docs/PLAN.md + HANDOFF.md
        for top in ["docs/PLAN.md", "HANDOFF.md", "docs/learning_roadmap.md"] {
            let u = repo.appendingPathComponent(top)
            if fm.fileExists(atPath: u.path) {
                docs.append(LearnDoc(id: u.path,
                                     title: niceTitle(u),
                                     relPath: top,
                                     url: u,
                                     group: .roadmap))
            }
        }
        return docs.sorted { $0.title < $1.title }
    }

    private static func niceTitle(_ url: URL) -> String {
        let stem = url.deletingPathExtension().lastPathComponent
        // Try to read the first H1 from the file.
        if let raw = try? String(contentsOf: url, encoding: .utf8) {
            var lines = raw.split(separator: "\n", omittingEmptySubsequences: false)
            if lines.first?.hasPrefix("---") == true {
                if let endIdx = lines.dropFirst().firstIndex(where: { $0.hasPrefix("---") }) {
                    lines = Array(lines[(endIdx + 1)...])
                }
            }
            if let h1 = lines.first(where: { $0.hasPrefix("# ") }) {
                return String(h1.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            }
        }
        return stem.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
    }
}

// MARK: - WorkspaceShelf — shared shell for sub-feature stubs

struct WorkspaceShelf: View {
    let title: String
    let tagline: String
    let items: [Item]

    struct Item: Identifiable {
        let id = UUID()
        let primary: String
        let detail: String
        let status: ItemStatus
        init(_ primary: String, _ detail: String, _ status: ItemStatus) {
            self.primary = primary
            self.detail = detail
            self.status = status
        }
    }
    enum ItemStatus {
        case ok, info, pending
        var color: Color {
            switch self {
            case .ok:      return .green
            case .info:    return .blue
            case .pending: return .orange
            }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(title)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text(tagline)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                }
                ForEach(items) { item in
                    HStack(alignment: .top, spacing: 12) {
                        Circle()
                            .fill(item.status.color)
                            .frame(width: 8, height: 8)
                            .padding(.top, 6)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.primary)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(Theme.fg)
                            Text(item.detail)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.panel)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.line))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.base)
    }
}

// MARK: - RepoLocator (kept here so it survives if WorkspaceStubs was the only consumer)

enum RepoLocator {
    static func repoRoot() -> URL? {
        let fm = FileManager.default
        guard let exec = Bundle.main.executableURL else { return nil }
        var dir = exec.deletingLastPathComponent()
        for _ in 0..<8 {
            let marker = dir.appendingPathComponent("Package.swift")
            let docs = dir.appendingPathComponent("docs")
            if fm.fileExists(atPath: marker.path) || fm.fileExists(atPath: docs.path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }
}
