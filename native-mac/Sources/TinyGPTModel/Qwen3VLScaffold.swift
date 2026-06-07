import Foundation

/// Build-time Qwen3-VL architecture contracts for the VLM M4 port.
///
/// This file intentionally does not implement a full forward path or a HF
/// weight loader. It captures the three pieces that make Qwen3-VL different
/// from the M3 LLaVA-style smoke path:
/// - multimodal RoPE metadata (`mrope_section`)
/// - `<image>` token embedding replacement
/// - deepstack visual feature injection sites
///
/// The next implementation layer should consume these value types from a
/// Qwen3-VL loader/forward path, then add parity tests against HF PyTorch.

// MARK: - mRoPE

public struct Qwen3VLMRoPESection: Sendable, Equatable {
    public var temporal: Int
    public var height: Int
    public var width: Int

    public var total: Int { temporal + height + width }

    public init(temporal: Int, height: Int, width: Int) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    /// UI-Venus-1.5-2B / Qwen3-VL default observed in HF config:
    /// `[24, 20, 20]` over a 128-wide head dimension.
    public static let uiVenus15_2B = Qwen3VLMRoPESection(
        temporal: 24,
        height: 20,
        width: 20
    )

    public func validate(headDim: Int) throws {
        guard temporal > 0, height > 0, width > 0 else {
            throw Qwen3VLScaffoldError.invalidMRoPESection(
                "mrope_section entries must all be positive: \(self)"
            )
        }
        guard total * 2 == headDim else {
            throw Qwen3VLScaffoldError.invalidMRoPESection(
                "mrope_section sum \(total) must equal head_dim / 2 for head_dim \(headDim)"
            )
        }
    }
}

/// One token's multimodal position. Text tokens advance only `text`;
/// image tokens carry the temporal/height/width coordinates used by mRoPE.
public struct Qwen3VLTokenPosition: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case text
        case image(imageIndex: Int)
    }

    public var kind: Kind
    public var text: Int
    public var temporal: Int
    public var height: Int
    public var width: Int

    public init(kind: Kind, text: Int, temporal: Int, height: Int, width: Int) {
        self.kind = kind
        self.text = text
        self.temporal = temporal
        self.height = height
        self.width = width
    }
}

public struct Qwen3VLImageGrid: Sendable, Equatable {
    public var temporal: Int
    public var height: Int
    public var width: Int

    public var tokenCount: Int { temporal * height * width }

    public init(temporal: Int, height: Int, width: Int) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    public func validate() throws {
        guard temporal > 0, height > 0, width > 0 else {
            throw Qwen3VLScaffoldError.invalidImageGrid(
                "image grid dimensions must be positive: \(self)"
            )
        }
    }
}

/// Per-forward position metadata for a Qwen3-VL prompt after image-token
/// placeholders have been matched to projected vision features.
public struct Qwen3VLMRoPEMetadata: Sendable, Equatable {
    public var section: Qwen3VLMRoPESection
    public var positions: [Qwen3VLTokenPosition]

    public init(section: Qwen3VLMRoPESection, positions: [Qwen3VLTokenPosition]) {
        self.section = section
        self.positions = positions
    }

    public static func build(
        section: Qwen3VLMRoPESection = .uiVenus15_2B,
        replacementPlan: Qwen3VLImageTokenReplacementPlan
    ) throws -> Qwen3VLMRoPEMetadata {
        try section.validate(headDim: replacementPlan.headDim)
        var positions: [Qwen3VLTokenPosition] = []
        positions.reserveCapacity(replacementPlan.tokenIDs.count)

        var textPos = 0
        for tokenIndex in replacementPlan.tokenIDs.indices {
            if let span = replacementPlan.imageSpan(containing: tokenIndex) {
                let offset = tokenIndex - span.tokenRange.lowerBound
                let hw = span.grid.height * span.grid.width
                let t = offset / hw
                let withinFrame = offset % hw
                let h = withinFrame / span.grid.width
                let w = withinFrame % span.grid.width
                positions.append(Qwen3VLTokenPosition(
                    kind: .image(imageIndex: span.imageIndex),
                    text: textPos,
                    temporal: t,
                    height: h,
                    width: w
                ))
            } else {
                positions.append(Qwen3VLTokenPosition(
                    kind: .text,
                    text: textPos,
                    temporal: 0,
                    height: 0,
                    width: 0
                ))
                textPos += 1
            }
        }

        return Qwen3VLMRoPEMetadata(section: section, positions: positions)
    }
}

// MARK: - Image-token replacement

public struct Qwen3VLImageTokenSpan: Sendable, Equatable {
    public var imageIndex: Int
    public var tokenRange: Range<Int>
    public var grid: Qwen3VLImageGrid

    public init(imageIndex: Int, tokenRange: Range<Int>, grid: Qwen3VLImageGrid) {
        self.imageIndex = imageIndex
        self.tokenRange = tokenRange
        self.grid = grid
    }
}

/// Contract for replacing `<image>` placeholder embeddings with projected
/// vision tokens. The actual scatter into an MLX embedding tensor belongs in
/// the Qwen3-VL forward path; this value type validates that the token stream
/// and vision grids agree before that path mutates embeddings.
public struct Qwen3VLImageTokenReplacementPlan: Sendable, Equatable {
    public var imageTokenID: Int
    public var headDim: Int
    public var tokenIDs: [Int]
    public var imageSpans: [Qwen3VLImageTokenSpan]

    public init(
        imageTokenID: Int,
        headDim: Int,
        tokenIDs: [Int],
        imageSpans: [Qwen3VLImageTokenSpan]
    ) {
        self.imageTokenID = imageTokenID
        self.headDim = headDim
        self.tokenIDs = tokenIDs
        self.imageSpans = imageSpans
    }

    public func imageSpan(containing tokenIndex: Int) -> Qwen3VLImageTokenSpan? {
        imageSpans.first { $0.tokenRange.contains(tokenIndex) }
    }

    public static func build(
        tokenIDs: [Int],
        imageGrids: [Qwen3VLImageGrid],
        imageTokenID: Int = Qwen3VLArchitectureDefaults.imageTokenID,
        headDim: Int = Qwen3VLArchitectureDefaults.headDim
    ) throws -> Qwen3VLImageTokenReplacementPlan {
        var imageRuns: [Range<Int>] = []
        var cursor = tokenIDs.startIndex
        while cursor < tokenIDs.endIndex {
            if tokenIDs[cursor] != imageTokenID {
                cursor += 1
                continue
            }
            let start = cursor
            while cursor < tokenIDs.endIndex, tokenIDs[cursor] == imageTokenID {
                cursor += 1
            }
            imageRuns.append(start..<cursor)
        }

        guard imageRuns.count == imageGrids.count else {
            throw Qwen3VLScaffoldError.imageTokenCountMismatch(
                expectedImages: imageGrids.count,
                foundImageRuns: imageRuns.count
            )
        }

        var spans: [Qwen3VLImageTokenSpan] = []
        spans.reserveCapacity(imageGrids.count)
        for (i, grid) in imageGrids.enumerated() {
            try grid.validate()
            let run = imageRuns[i]
            guard run.count == grid.tokenCount else {
                throw Qwen3VLScaffoldError.imageFeatureCountMismatch(
                    imageIndex: i,
                    expected: grid.tokenCount,
                    found: run.count
                )
            }
            spans.append(Qwen3VLImageTokenSpan(
                imageIndex: i,
                tokenRange: run,
                grid: grid
            ))
        }

        return Qwen3VLImageTokenReplacementPlan(
            imageTokenID: imageTokenID,
            headDim: headDim,
            tokenIDs: tokenIDs,
            imageSpans: spans
        )
    }
}

// MARK: - Deepstack

public struct Qwen3VLDeepstackPlan: Sendable, Equatable {
    public var visualIndexes: [Int]
    public var layerCount: Int

    public init(
        visualIndexes: [Int] = Qwen3VLArchitectureDefaults.deepstackVisualIndexes,
        layerCount: Int
    ) throws {
        let sorted = visualIndexes.sorted()
        guard sorted == visualIndexes else {
            throw Qwen3VLScaffoldError.invalidDeepstackIndexes(
                "deepstack_visual_indexes must be sorted: \(visualIndexes)"
            )
        }
        guard Set(visualIndexes).count == visualIndexes.count else {
            throw Qwen3VLScaffoldError.invalidDeepstackIndexes(
                "deepstack_visual_indexes must not contain duplicates: \(visualIndexes)"
            )
        }
        guard visualIndexes.allSatisfy({ $0 >= 0 && $0 < layerCount }) else {
            throw Qwen3VLScaffoldError.invalidDeepstackIndexes(
                "deepstack indexes \(visualIndexes) must be inside layer range 0..<\(layerCount)"
            )
        }
        self.visualIndexes = visualIndexes
        self.layerCount = layerCount
    }

    public func injects(beforeLayer layerIndex: Int) -> Bool {
        visualIndexes.contains(layerIndex)
    }
}

public enum Qwen3VLArchitectureDefaults {
    public static let imageTokenID = 151_655
    public static let headDim = 128
    public static let mropeSection = Qwen3VLMRoPESection.uiVenus15_2B
    public static let spatialMergeSize = 2
    public static let deepstackVisualIndexes = [5, 11, 17]
}

public enum Qwen3VLScaffoldError: Error, CustomStringConvertible, Equatable {
    case invalidMRoPESection(String)
    case invalidImageGrid(String)
    case imageTokenCountMismatch(expectedImages: Int, foundImageRuns: Int)
    case imageFeatureCountMismatch(imageIndex: Int, expected: Int, found: Int)
    case invalidDeepstackIndexes(String)

    public var description: String {
        switch self {
        case .invalidMRoPESection(let msg): return msg
        case .invalidImageGrid(let msg): return msg
        case .imageTokenCountMismatch(let expected, let found):
            return "expected \(expected) image placeholder run(s), found \(found)"
        case .imageFeatureCountMismatch(let imageIndex, let expected, let found):
            return "image \(imageIndex) expected \(expected) placeholder token(s), found \(found)"
        case .invalidDeepstackIndexes(let msg): return msg
        }
    }
}
