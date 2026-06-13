import Foundation

/// Per-project model + dataset pins. Lives at `./tinygpt.project.json`
/// in the consumer project's root. Read by `tinygpt pull` (fetch
/// every pin from the gallery) and `tinygpt validate` (sanity-check
/// every pin resolves).
///
/// Analogous to `package.json`: visible at the project root, JSON
/// (not TOML — keeps parsers single-language with the browser side),
/// human-editable.
///
/// V1 schema (this file) is exact-id pinning. Versioning + lockfile
/// arrive in V2 if churn warrants — see B31 PRD §"Scope — out".
public struct ProjectManifest: Codable, Sendable {
    /// Display name; not load-bearing.
    public let name: String
    /// Optional minimum tinygpt version (informational; CLI may warn
    /// on mismatch but does not enforce).
    public let tinygptVersion: String?
    /// Model pins. Order is not significant.
    public let models: [ProjectModelPin]
    /// Dataset pins. Order is not significant. Optional —
    /// pure-inference projects don't ship datasets.
    public let datasets: [ProjectDatasetPin]?

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

    public static func decode(from data: Data) throws -> ProjectManifest {
        try jsonDecoder.decode(ProjectManifest.self, from: data)
    }

    public static func load(path: String) throws -> ProjectManifest {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try decode(from: data)
    }
}

public enum ProjectModelRole: String, Codable, Sendable {
    case base
    case adapter
    case draft   // a speculative-decode draft model (B14)
    case judge   // a local model used for E7 judging
    case other
}

public struct ProjectModelPin: Codable, Sendable {
    /// The gallery id (`GalleryModel.id`) being pinned.
    public let id: String
    /// What role this model plays in the project. `base` for the
    /// primary inference model; `adapter` for LoRA/DoRA stacks on
    /// top of a base; `draft` for spec-dec; etc.
    public let role: ProjectModelRole
    /// For `role == .adapter`: the id of the base the adapter
    /// applies to. Must match a gallery model whose kind is one of
    /// the base kinds, AND must match another pin in the same
    /// project file (the cross-check is done by `tinygpt validate`).
    public let appliesTo: String?
    /// Optional opt-out — when present and true, `tinygpt pull`
    /// reports the pin as a no-op (useful for dev-time adapters).
    public let optional: Bool?
}

public struct ProjectDatasetPin: Codable, Sendable {
    /// Dataset id (the same id `tinygpt list-datasets` shows).
    public let id: String
    /// When true, missing dataset doesn't fail `tinygpt validate`.
    public let optional: Bool?
}

/// Lightweight validation: schema-level checks that don't need a
/// gallery to be present. `tinygpt validate` runs both this AND
/// the cross-check against the resolved `GalleryManifest`.
public extension ProjectManifest {
    enum ValidationError: Error, CustomStringConvertible {
        case adapterMissingAppliesTo(id: String)
        case adapterAppliesToNotInManifest(id: String, appliesTo: String)
        case duplicateModelId(id: String)

        public var description: String {
            switch self {
            case .adapterMissingAppliesTo(let id):
                return "adapter '\(id)' must declare applies_to"
            case .adapterAppliesToNotInManifest(let id, let appliesTo):
                return "adapter '\(id)' applies_to '\(appliesTo)' which is not also pinned in this project"
            case .duplicateModelId(let id):
                return "duplicate model id '\(id)' in project pins"
            }
        }
    }

    func validate() throws {
        var seen = Set<String>()
        for pin in models {
            if !seen.insert(pin.id).inserted {
                throw ValidationError.duplicateModelId(id: pin.id)
            }
            if pin.role == .adapter {
                guard let appliesTo = pin.appliesTo else {
                    throw ValidationError.adapterMissingAppliesTo(id: pin.id)
                }
                let basePinned = models.contains { $0.id == appliesTo && $0.role == .base }
                if !basePinned {
                    throw ValidationError.adapterAppliesToNotInManifest(
                        id: pin.id, appliesTo: appliesTo)
                }
            }
        }
    }
}
