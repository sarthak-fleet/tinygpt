/**
 * tokenizer.ts — byte-level tokenizer (Phase 4).
 *
 * STATUS: documented stub. No implementation yet.
 *
 * The v0 tokenizer is intentionally trivial: every byte is a token, vocab = 256.
 * This avoids all BPE / merge-table complexity.
 *
 *   encode(text: string): Uint8Array      // UTF-8 bytes
 *   decode(tokens: Uint8Array): string    // bytes -> UTF-8 text
 *
 * Must satisfy the roundtrip test:  decode(encode(text)) === text
 * (see tests/README.md "Tokenizer roundtrip").
 *
 * Guide: docs/model_guide.md ("What you are building")
 *
 * TODO(phase-4): implement encode/decode with TextEncoder/TextDecoder.
 */
export {};
