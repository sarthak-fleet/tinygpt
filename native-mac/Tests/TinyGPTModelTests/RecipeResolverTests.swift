import Foundation
import XCTest
@testable import TinyGPTModel

/// B33 coverage for the `tinygpt quickstart` decision core: data-shape
/// detection, gallery base selection, and recipe sizing. Pure — no GPU,
/// no filesystem.
final class RecipeResolverTests: XCTestCase {
    // A minimal gallery base; only id / tags / kind / paramCount matter here.
    private func base(_ id: String, tags: [String]? = nil,
                      kind: GalleryModelKind = .macSafetensorsHf,
                      paramCount: Int? = nil) -> GalleryModel {
        GalleryModel(
            id: id, name: id, file: "\(id).safetensors",
            icon: nil, blurb: nil, corpus: nil, corpusUrl: nil,
            fileInt4: nil, fileInt4Bytes: nil, params: nil, paramCount: paramCount,
            trainLoss: nil, steps: nil, sample: nil, fileBytes: nil, gpuBytes: nil,
            prompt: nil, trainWallMs: nil, submission: nil, benchmarks: nil,
            kind: kind, parent: nil, r2Path: nil, tags: tags)
    }

    private func inspect(_ lines: [String]) -> RecipeResolver.DataInspection {
        RecipeResolver.inspect(lines: lines)
    }

    private let chatRow = #"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}"#

    // MARK: - data inspection

    func test_detects_chat_jsonl() {
        let r = inspect([chatRow, chatRow])
        XCTAssertEqual(r.shape, .chat)
        XCTAssertEqual(r.rowCount, 2)
        XCTAssertFalse(r.unparsable)
    }

    func test_detects_tool_call_via_tool_calls() {
        let line = #"{"messages":[{"role":"user","content":"weather?"},{"role":"assistant","tool_calls":[{"function":{"name":"get_weather"}}]}]}"#
        XCTAssertEqual(inspect([line]).shape, .toolCall)
    }

    func test_detects_tool_call_via_tools_key() {
        let line = #"{"tools":[{"name":"x"}],"messages":[{"role":"user","content":"hi"}]}"#
        XCTAssertEqual(inspect([line]).shape, .toolCall)
    }

    func test_detects_instruction_shape() {
        XCTAssertEqual(inspect([#"{"instruction":"Summarize this","output":"ok"}"#]).shape, .instruction)
    }

    func test_detects_prompt_completion_as_instruction() {
        XCTAssertEqual(inspect([#"{"prompt":"2+2=","completion":"4"}"#]).shape, .instruction)
    }

    func test_detects_raw_text() {
        let r = inspect(["The quick brown fox.", "Jumped over the lazy dog.", "Still not JSON."])
        XCTAssertEqual(r.shape, .rawText)
        XCTAssertTrue(r.unparsable)
        XCTAssertEqual(r.rowCount, 3)
    }

    func test_empty_input_is_unknown_zero_rows() {
        let r = inspect(["", "   ", "\n"])
        XCTAssertEqual(r.shape, .unknown)
        XCTAssertEqual(r.rowCount, 0)
    }

    func test_row_count_ignores_blank_lines() {
        XCTAssertEqual(inspect([chatRow, "", chatRow, "   "]).rowCount, 2)
    }

    // MARK: - base selection

    func test_raw_text_resolves_to_from_scratch() {
        let plan = RecipeResolver.resolve(
            inspection: inspect(["corpus line one", "corpus line two"]),
            gallery: [base("qwen3-4b", tags: ["instruct"])])
        XCTAssertTrue(plan.base.fromScratch)
        XCTAssertNil(plan.base.galleryId)
        XCTAssertEqual(plan.recipe.mode, .fromScratch)
    }

    func test_tool_call_prefers_tool_tagged_base() {
        let gallery = [
            base("small-chat", tags: ["chat"], paramCount: 1_000_000_000),
            base("agent-base", tags: ["tool", "agent"], paramCount: 4_000_000_000),
            base("plain", tags: nil, paramCount: 500_000_000),
        ]
        let line = #"{"messages":[{"role":"assistant","tool_calls":[{"x":1}]}]}"#
        let plan = RecipeResolver.resolve(inspection: inspect([line]), gallery: gallery)
        XCTAssertEqual(plan.base.galleryId, "agent-base")
    }

    func test_tiebreak_prefers_smaller_model() {
        let gallery = [
            base("big", tags: ["chat"], paramCount: 14_000_000_000),
            base("small", tags: ["chat"], paramCount: 1_500_000_000),
        ]
        let plan = RecipeResolver.resolve(inspection: inspect([chatRow]), gallery: gallery)
        XCTAssertEqual(plan.base.galleryId, "small")
    }

    func test_skips_adapters_and_browser_bins() {
        let gallery = [
            base("adapter", tags: ["chat"], kind: .macAdapter, paramCount: 100),
            base("browser", tags: ["chat"], kind: .browserBin, paramCount: 100),
            base("real-base", tags: ["chat"], kind: .macSafetensorsHf, paramCount: 2_000_000_000),
        ]
        let plan = RecipeResolver.resolve(inspection: inspect([chatRow]), gallery: gallery)
        XCTAssertEqual(plan.base.galleryId, "real-base")
    }

    func test_no_baseable_gallery_warns() {
        let plan = RecipeResolver.resolve(
            inspection: inspect([chatRow]),
            gallery: [base("adapter", tags: ["chat"], kind: .macAdapter)])
        XCTAssertNil(plan.base.galleryId)
        XCTAssertTrue(plan.warnings.contains { $0.contains("no fine-tunable base") })
    }

    func test_override_base_wins_and_warns_when_unknown() {
        let plan = RecipeResolver.resolve(
            inspection: inspect([chatRow]),
            gallery: [base("gallery-base", tags: ["chat"], paramCount: 1_000_000_000)],
            overrideBase: "my-local/path")
        XCTAssertEqual(plan.base.galleryId, "my-local/path")
        XCTAssertTrue(plan.warnings.contains { $0.contains("not in the gallery") })
    }

    func test_small_dataset_warns() {
        let lines = Array(repeating: chatRow, count: 10)
        let plan = RecipeResolver.resolve(inspection: inspect(lines), gallery: [base("b", tags: ["chat"])])
        XCTAssertTrue(plan.warnings.contains { $0.contains("rows") })
    }

    // MARK: - recipe sizing

    func test_rank_scales_with_data_size() {
        func rank(_ n: Int) -> Int {
            let lines = Array(repeating: chatRow, count: n)
            return RecipeResolver.resolve(inspection: inspect(lines), gallery: [base("b", tags: ["chat"])]).recipe.rank
        }
        XCTAssertEqual(rank(100), 8)
        XCTAssertEqual(rank(1000), 16)
        XCTAssertEqual(rank(6000), 32)
    }

    func test_template_and_maxseq_by_shape() {
        let chat = RecipeResolver.resolve(inspection: inspect([chatRow]), gallery: [base("b", tags: ["chat"])]).recipe
        XCTAssertEqual(chat.template, "chatml")
        XCTAssertEqual(chat.maxSeq, 1024)

        let toolLine = #"{"messages":[{"role":"assistant","tool_calls":[{"x":1}]}]}"#
        let tool = RecipeResolver.resolve(inspection: inspect([toolLine]), gallery: [base("b", tags: ["tool"])]).recipe
        XCTAssertEqual(tool.maxSeq, 2048)

        let instr = RecipeResolver.resolve(
            inspection: inspect([#"{"instruction":"x","output":"y"}"#]),
            gallery: [base("b", tags: ["instruct"])]).recipe
        XCTAssertNil(instr.template)
    }

    func test_sft_flags_render_expected() {
        let recipe = RecipeResolver.resolve(inspection: inspect([chatRow]), gallery: [base("b", tags: ["chat"])]).recipe
        let flags = recipe.sftFlags()
        XCTAssertTrue(flags.contains("--rank"))
        XCTAssertTrue(flags.contains("--pack"))
        XCTAssertTrue(flags.contains("--no-dora"))
        XCTAssertTrue(flags.contains("--template"))
        let lrIdx = flags.firstIndex(of: "--lr")!
        XCTAssertEqual(flags[flags.index(after: lrIdx)], "0.0002")
    }
}
