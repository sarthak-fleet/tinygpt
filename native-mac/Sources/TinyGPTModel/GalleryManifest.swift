import Foundation

/// Swift mirror of `browser/src/gallery-schema.ts`. Single source of
/// truth for the published `gallery/manifest.json`. Browser is the
/// authoring side; this struct lets `tinygpt pull` / `tinygpt
/// validate` / `tinygpt sample <gallery-id>` parse the same file.
///
/// V1 of the schema (per the TS file's `version: 1` field) covered
/// only browser-loadable `.bin` from-scratch models. The B31
/// extension adds the `kind` discriminator + `parent` + `r2_path` +
/// `tags` so Mac-side adapters and GGUF bundles can share the same
/// manifest.
///
/// Edit policy: when this struct changes, change `gallery-schema.ts`
/// in the same PR and bump the manifest `version`. No codegen.
public enum GalleryModelKind: String, Codable, Sendable {
    /// In-browser playground model; weights at
    /// `browser/public/gallery/<file>`. The original V1 shape.
    case browserBin = "browser-bin"
    /// A full-model `.tinygpt` checkpoint loadable by the Mac CLI.
    case macTinygpt = "mac-tinygpt"
    /// A LoRA / DoRA / PEFT adapter loadable atop a base. Requires
    /// `parent` to be set to the base's gallery id.
    case macAdapter = "mac-adapter"
    /// A GGUF bundle (base + tokenizer + config) loadable via
    /// `tinygpt gguf-load`.
    case macGguf = "mac-gguf"
    /// A HuggingFace safetensors bundle (base + tokenizer + config)
    /// loadable via the existing HF loader path.
    case macSafetensorsHf = "mac-safetensors-hf"
}

public struct GallerySubmission: Codable, Sendable, Hashable {
    public let author: String
    public let submittedAt: String
    public let browserTrained: Bool?
    public let featured: Bool?
}

public struct GalleryModel: Codable, Sendable {
    // Required fields (V1 shape, unchanged)
    public let id: String
    public let name: String
    public let file: String   // browser uses .bin name; Mac kinds use a relative path under r2_path
    // Optional fields from V1
    public let icon: String?
    public let blurb: String?
    public let corpus: String?
    public let corpusUrl: String?
    public let fileInt4: String?
    public let fileInt4Bytes: Int?
    public let params: String?
    public let paramCount: Int?
    public let trainLoss: String?
    public let steps: Int?
    public let sample: String?
    public let fileBytes: Int?
    public let gpuBytes: Int?
    public let prompt: String?
    public let trainWallMs: Int?
    public let submission: GallerySubmission?
    public let benchmarks: [String: Double?]?
    // New in B31 (all optional so V1 rows keep parsing)
    /// Defaults to `.browserBin` when absent — that's the V1 implicit
    /// kind. Mac entries set it explicitly.
    public let kind: GalleryModelKind?
    /// For `macAdapter`: the gallery id of the base this adapter
    /// stacks on. Nil for non-adapter kinds.
    public let parent: String?
    /// For Mac kinds whose artifact lives in R2 rather than the
    /// browser's `public/gallery/`. Relative path under the bucket.
    public let r2Path: String?
    /// Loose tags used by the leaderboard's filter UI and by the
    /// Mac app's Factory tab to group offerings.
    public let tags: [String]?

    public var resolvedKind: GalleryModelKind { kind ?? .browserBin }
}

public struct GalleryManifest: Codable, Sendable {
    public let version: Int
    public let note: String?
    public let models: [GalleryModel]

    public static let jsonDecoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    public static let jsonEncoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }()

    public static func decode(from data: Data) throws -> GalleryManifest {
        try jsonDecoder.decode(GalleryManifest.self, from: data)
    }

    public static func load(path: String) throws -> GalleryManifest {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try decode(from: data)
    }

    /// Convenience: look up a model by gallery id. Nil for unknown.
    public func model(id: String) -> GalleryModel? {
        models.first { $0.id == id }
    }
}
