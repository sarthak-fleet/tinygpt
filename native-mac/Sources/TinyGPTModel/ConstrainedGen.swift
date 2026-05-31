import Foundation

/// Grammar-constrained / JSON-schema-constrained generation.
///
/// At each decode step we run a character-level finite state machine
/// (FSM) driven by the schema, and for every candidate token in the
/// vocabulary we check whether feeding that token's bytes into the FSM
/// would leave it in a valid state. Tokens that would NOT are masked to
/// `-inf` before softmax. The result: every accepted sample is a prefix
/// of some string that matches the schema, and once the FSM reaches a
/// terminal accepting state we can stop.
///
/// This is the same family of algorithm used by `outlines`, `vLLM`'s
/// guided-decoding, and llama.cpp's GBNF — just a character-level FSM
/// instead of a token-level one (token-level requires a heavier
/// vocab-trie build at setup). The trade-off: character-level pays
/// O(token-length × vocab) per step in cycles, but build cost is O(vocab)
/// plus one schema walk. For vocab≤256k that's fine on Apple Silicon.
///
/// SHAPE OF THE FSM
/// The FSM is a stack of `Frame`s. Each frame represents one nesting
/// level (a value, an object, an array, a string, a number). The top
/// of the stack is the "active" position. Transitions are explicit:
/// each per-byte step returns a `Transition` value that tells the
/// driver to mutate, push, or pop.
///
/// WHAT THE FSM ACCEPTS (the grammar)
/// Roughly the JSON grammar, scoped by the schema. Where the schema
/// pins a type, only that type's grammar branch is accepted. Where the
/// schema is `.any`, the full JSON grammar is open. Strings escape via
/// `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, and `\uXXXX`. Numbers
/// follow the JSON number grammar (sign? int frac? exp?), with the
/// `integer` schema variant rejecting `.`/`e`/`E`.

public final class JSONSchemaFSM {

    // MARK: - Frame value types

    fileprivate enum ValuePhase: Equatable { case start, done }

    fileprivate enum ObjectPhase: Equatable {
        /// Just consumed `{`. Either a key string can open (`"`) or
        /// the object can immediately close (`}`) — empty-object case.
        case afterOpen
        /// Just consumed `,`. A key string MUST open next; closing
        /// here would be a trailing-comma error.
        case afterComma
        case afterKey(key: String)
        /// Saw `:`, just pushed a value frame; this key is being
        /// emitted into the value. On value-pop we move to .afterValue
        /// and insert the key into `emitted`.
        case insideValue(key: String)
        case afterValue
        case closed
    }

    fileprivate enum ArrayPhase: Equatable {
        /// Just consumed `[`. Either an item can open (any value byte)
        /// or the array can immediately close (`]`) — empty-array case.
        case afterOpen
        /// Just consumed `,`. An item MUST follow; closing here would
        /// be a trailing-comma error.
        case afterComma
        case afterItem
        case closed
    }

    fileprivate enum StringState: Equatable {
        case body
        case escape
        case unicodeEscape(digitsSeen: Int)
        case closed
    }

    fileprivate enum NumberState: Equatable {
        case sign
        case intZero
        case intDigits
        case afterFracPoint
        case fracDigits
        case afterExp
        case expSign
        case expDigits
    }

    fileprivate struct ObjectFrame {
        var schema: JSONSchemaNode
        var emitted: Set<String>
        var phase: ObjectPhase
    }

    fileprivate struct ArrayFrame {
        var items: JSONSchemaNode
        var phase: ArrayPhase
    }

    fileprivate struct StringFrame {
        var state: StringState
        /// Closed set of permitted contents (nil = unconstrained).
        var enumValues: [String]?
        /// What we've emitted into the body so far (used for enum
        /// prefix filtering AND for stashing object keys on pop).
        var content: String
        /// When true the string frame represents an object key. On pop,
        /// the parent reads `content` as the key name.
        var isKey: Bool
    }

    fileprivate struct NumberFrame {
        var state: NumberState
        var integer: Bool
    }

    fileprivate enum Frame {
        case value(schema: JSONSchemaNode, phase: ValuePhase)
        case object(ObjectFrame)
        case array(ArrayFrame)
        case string(StringFrame)
        case number(NumberFrame)
        case boolean(literal: String, position: Int)
        case null(position: Int)
    }

    /// What a per-frame transition wants the driver to do.
    fileprivate enum Action {
        case reject
        /// Replace top of stack with the new frame; consume the byte.
        case replace(Frame)
        /// Push a child onto the stack; consume the byte.
        case push(Frame)
        /// Replace top with `replacement` and push `child`; consume
        /// the byte. Used when an object sees `:` and pushes a value
        /// frame for the property value.
        case replaceAndPush(replacement: Frame, child: Frame)
        /// Replace top with `replacement`, push `child`, then re-feed
        /// the byte to the new top. Provided for symmetry; not
        /// currently used.
        case replaceAndPushAndRetry(replacement: Frame, child: Frame)
        /// Push `child` onto the stack and re-feed the byte to the new
        /// top. Used by arrays opening their first item.
        case pushAndRetry(Frame)
        /// Pop top of stack; consume the byte. After popping, the
        /// driver advances the parent's phase (key-done / value-done /
        /// item-done / top-done).
        case popConsuming
        /// Pop top of stack; DO NOT consume the byte — re-feed it to
        /// the new top. Used by numbers, which terminate when they see
        /// a structural byte.
        case popAndRetry
    }

    // MARK: - State

    private var stack: [Frame]
    /// Set by the string frame just before it pops, used by the object
    /// parent to read the key text.
    private var lastPoppedKey: String? = nil

    public init(rootSchema: JSONSchemaNode) {
        self.stack = [.value(schema: rootSchema, phase: .start)]
    }

    public func clone() -> JSONSchemaFSM {
        let c = JSONSchemaFSM(rootSchema: .any)
        c.stack = self.stack
        c.lastPoppedKey = self.lastPoppedKey
        return c
    }

    /// True iff a complete JSON value was emitted at the top level.
    public var isComplete: Bool {
        guard stack.count == 1 else { return false }
        if case .value(_, let p) = stack[0], p == .done { return true }
        return false
    }

    public var isDead: Bool { stack.isEmpty }

    // MARK: - Public driver

    @discardableResult
    public func acceptByte(_ b: UInt8) -> Bool {
        return driveByte(b, depth: 0)
    }

    @discardableResult
    public func acceptBytes(_ bytes: [UInt8]) -> Bool {
        let snapshot = stack
        let snapKey = lastPoppedKey
        for b in bytes {
            if !driveByte(b, depth: 0) {
                stack = snapshot
                lastPoppedKey = snapKey
                return false
            }
        }
        return true
    }

    @discardableResult
    public func acceptString(_ s: String) -> Bool {
        return acceptBytes(Array(s.utf8))
    }

    @discardableResult
    public func acceptChar(_ c: Character) -> Bool {
        return acceptString(String(c))
    }

    // MARK: - Driver internals

    /// `depth` guards against ill-formed grammars looping on `popAndRetry`.
    /// In a well-formed driver this should bottom out quickly; we cap at
    /// 64 nested pops to be safe.
    private func driveByte(_ b: UInt8, depth: Int) -> Bool {
        if depth > 64 { return false }
        // Allow insignificant whitespace between structural tokens.
        if isWhitespace(b) && allowsWhitespaceHere() {
            return true
        }
        guard let top = stack.last else { return false }
        let action = transition(top: top, byte: b)
        switch action {
        case .reject:
            return false
        case .replace(let nf):
            stack[stack.count - 1] = nf
            return true
        case .push(let child):
            stack.append(child)
            return true
        case .replaceAndPush(let replacement, let child):
            stack[stack.count - 1] = replacement
            stack.append(child)
            return true
        case .replaceAndPushAndRetry(let replacement, let child):
            let savedTop = stack[stack.count - 1]
            stack[stack.count - 1] = replacement
            stack.append(child)
            let ok = driveByte(b, depth: depth + 1)
            if !ok {
                stack.removeLast()
                stack[stack.count - 1] = savedTop
            }
            return ok
        case .pushAndRetry(let child):
            stack.append(child)
            let ok = driveByte(b, depth: depth + 1)
            if !ok {
                stack.removeLast()
            }
            return ok
        case .popConsuming:
            // Stash the string's content as the key if appropriate
            // BEFORE popping (popped frame is gone after).
            if case .string(let sf) = top, sf.isKey {
                lastPoppedKey = sf.content
            }
            stack.removeLast()
            advanceParentAfterChildPop()
            return true
        case .popAndRetry:
            stack.removeLast()
            advanceParentAfterChildPop()
            return driveByte(b, depth: depth + 1)
        }
    }

    /// Whitespace allowance.
    ///
    /// JSON's grammar permits insignificant whitespace between all
    /// structural tokens, but in constrained generation that's a
    /// liability: if the model's softmax puts even modest mass on
    /// whitespace bytes, the FSM accepts them indefinitely and the
    /// model never advances into the actual structure. Most tiny LLMs
    /// trained on prose (Shakespeare, web text) have a strong
    /// whitespace prior, so this stall is the rule, not the exception.
    ///
    /// Solution: emit MINIFIED JSON. We forbid all insignificant
    /// whitespace and only allow whitespace where it's CONTENT
    /// (inside string bodies). String content includes spaces, tabs,
    /// newlines as legitimate characters; structurally between tokens
    /// they're banned. This matches the behavior of `JSON.stringify()`,
    /// `outlines`, and `vLLM`'s `guided_json`.
    private func allowsWhitespaceHere() -> Bool {
        // Always allow ws inside a string body / escape (it's content).
        // Reject ws everywhere else — the FSM only progresses when the
        // model emits a structurally meaningful byte.
        guard let top = stack.last else { return false }
        if case .string(let sf) = top {
            switch sf.state {
            case .body, .escape, .unicodeEscape: return false  // content path, not ws path
            default: return false
            }
        }
        return false
    }

    private func isWhitespace(_ b: UInt8) -> Bool {
        return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D
    }

    // MARK: - Parent advancement

    private func advanceParentAfterChildPop() {
        // After a child pops, walk UP the stack while we keep finding
        // wrapper frames that just completed. The common case is a
        // single hop, but a value-frame parent (intermediate "value"
        // wrapper for a property/item slot) immediately completes when
        // its concrete child pops, and we need to cascade: pop the
        // value frame and re-trigger the advance on the object/array
        // above it.
        while let parent = stack.last {
            switch parent {
            case .object(var of):
                switch of.phase {
                case .afterOpen, .afterComma:
                    // The popped child was the object's key string.
                    let key = lastPoppedKey ?? ""
                    lastPoppedKey = nil
                    of.phase = .afterKey(key: key)
                case .insideValue(let key):
                    of.emitted.insert(key)
                    of.phase = .afterValue
                case .afterKey, .afterValue, .closed:
                    break
                }
                stack[stack.count - 1] = .object(of)
                return
            case .array(var af):
                af.phase = .afterItem
                stack[stack.count - 1] = .array(af)
                return
            case .value(let s, _):
                // The popped child completed this value slot. If we
                // are the root (only frame on the stack) we stay so
                // `isComplete` can observe it. Otherwise we pop and
                // loop to advance the next layer up (object/array).
                if stack.count == 1 {
                    stack[stack.count - 1] = .value(schema: s, phase: .done)
                    return
                } else {
                    stack.removeLast()
                    continue
                }
            default:
                return
            }
        }
    }

    // MARK: - Per-frame transitions

    private func transition(top: Frame, byte b: UInt8) -> Action {
        switch top {
        case .value(let schema, let phase):
            if phase == .done { return .reject }
            return openValue(schema: schema, byte: b)
        case .object(let of):
            return stepObject(of: of, byte: b)
        case .array(let af):
            return stepArray(af: af, byte: b)
        case .string(let sf):
            return stepString(sf: sf, byte: b)
        case .number(let nf):
            return stepNumber(nf: nf, byte: b)
        case .boolean(let lit, let pos):
            return stepBoolean(literal: lit, position: pos, byte: b)
        case .null(let pos):
            return stepNull(position: pos, byte: b)
        }
    }

    // MARK: - Value (open)

    private func openValue(schema: JSONSchemaNode, byte b: UInt8) -> Action {
        switch schema {
        case .any:
            return openAnyValue(byte: b)
        case .object:
            guard b == UInt8(ascii: "{") else { return .reject }
            // Replace the .value with .object — same level. (The
            // top-level value frame is replaced; subsequent pops will
            // not signal value-done because we lose the .value wrapper.
            // To preserve "top-level completes after the JSON value
            // ends", we PUSH .object on top of .value, and pop both
            // when object closes. But that doesn't match how arrays
            // and other types work... To keep things uniform we adopt
            // the convention: REPLACE the value frame with the
            // concrete container, and consider the root complete when
            // the stack has exactly one frame in a terminal state.
            // Simpler though: PUSH a child object frame and leave the
            // value frame to be popped when the object closes.
            return .push(.object(.init(schema: schema, emitted: [], phase: .afterOpen)))
        case .array(let items):
            guard b == UInt8(ascii: "[") else { return .reject }
            return .push(.array(.init(items: items, phase: .afterOpen)))
        case .string(let enumValues):
            guard b == UInt8(ascii: "\"") else { return .reject }
            return .push(.string(.init(state: .body, enumValues: enumValues, content: "", isKey: false)))
        case .number(let integer):
            if b == UInt8(ascii: "-") {
                return .push(.number(.init(state: .sign, integer: integer)))
            }
            if isDigit(b) {
                let st: NumberState = (b == UInt8(ascii: "0")) ? .intZero : .intDigits
                return .push(.number(.init(state: st, integer: integer)))
            }
            return .reject
        case .boolean:
            if b == UInt8(ascii: "t") { return .push(.boolean(literal: "true", position: 1)) }
            if b == UInt8(ascii: "f") { return .push(.boolean(literal: "false", position: 1)) }
            return .reject
        case .null:
            if b == UInt8(ascii: "n") { return .push(.null(position: 1)) }
            return .reject
        }
    }

    private func openAnyValue(byte b: UInt8) -> Action {
        switch b {
        case UInt8(ascii: "{"):
            return .push(.object(.init(schema: .object(properties: [], required: []),
                                        emitted: [], phase: .afterOpen)))
        case UInt8(ascii: "["):
            return .push(.array(.init(items: .any, phase: .afterOpen)))
        case UInt8(ascii: "\""):
            return .push(.string(.init(state: .body, enumValues: nil, content: "", isKey: false)))
        case UInt8(ascii: "-"):
            return .push(.number(.init(state: .sign, integer: false)))
        case UInt8(ascii: "t"):
            return .push(.boolean(literal: "true", position: 1))
        case UInt8(ascii: "f"):
            return .push(.boolean(literal: "false", position: 1))
        case UInt8(ascii: "n"):
            return .push(.null(position: 1))
        default:
            if isDigit(b) {
                let st: NumberState = (b == UInt8(ascii: "0")) ? .intZero : .intDigits
                return .push(.number(.init(state: st, integer: false)))
            }
            return .reject
        }
    }

    // MARK: - Object

    private func stepObject(of: ObjectFrame, byte b: UInt8) -> Action {
        switch of.phase {
        case .afterOpen:
            // Empty-object close is allowed iff all required keys are
            // already emitted (they will be, since `emitted` is empty
            // at this point — close is allowed iff required is empty).
            if b == UInt8(ascii: "}") {
                if case .object(_, let required) = of.schema {
                    if !required.isSubset(of: of.emitted) { return .reject }
                }
                return .popConsuming
            }
            if b == UInt8(ascii: "\"") {
                let allowedKeys = unemittedKeys(schema: of.schema, emitted: of.emitted)
                if let keys = allowedKeys, keys.isEmpty { return .reject }
                return .push(.string(.init(state: .body, enumValues: allowedKeys,
                                            content: "", isKey: true)))
            }
            return .reject
        case .afterComma:
            // No trailing comma — must open a key string here.
            if b == UInt8(ascii: "\"") {
                let allowedKeys = unemittedKeys(schema: of.schema, emitted: of.emitted)
                if let keys = allowedKeys, keys.isEmpty { return .reject }
                return .push(.string(.init(state: .body, enumValues: allowedKeys,
                                            content: "", isKey: true)))
            }
            return .reject
        case .afterKey(let key):
            if b == UInt8(ascii: ":") {
                var nf = of
                nf.phase = .insideValue(key: key)
                let valSchema = schemaFor(key: key, in: of.schema)
                return .replaceAndPush(
                    replacement: .object(nf),
                    child: .value(schema: valSchema, phase: .start)
                )
            }
            return .reject
        case .insideValue:
            // Child value frame is open; bytes go to the child.
            return .reject
        case .afterValue:
            if b == UInt8(ascii: ",") {
                // Reject if no unemitted property exists — comma would
                // strand us in .afterComma with nothing to open. (`.any`
                // schemas with implicit additionalProperties allow it.)
                let remaining = unemittedKeys(schema: of.schema, emitted: of.emitted)
                if let r = remaining, r.isEmpty { return .reject }
                var nf = of
                nf.phase = .afterComma
                return .replace(.object(nf))
            }
            if b == UInt8(ascii: "}") {
                if case .object(_, let required) = of.schema {
                    if !required.isSubset(of: of.emitted) { return .reject }
                }
                return .popConsuming
            }
            return .reject
        case .closed:
            return .reject
        }
    }

    /// List of property names declared by the schema, minus those already
    /// emitted. Returns `nil` ONLY for `.any`-shaped schemas (no
    /// constraint). For a declared-properties object where every key
    /// has been emitted, returns `[]` — callers should treat that as
    /// "no more keys allowed" and reject further key opens.
    private func unemittedKeys(schema: JSONSchemaNode, emitted: Set<String>) -> [String]? {
        if case .object(let props, _) = schema {
            if props.isEmpty {
                // `.any`-style: no declared properties, additionalProperties
                // implicitly open. Return nil.
                return nil
            }
            return props.map { $0.0 }.filter { !emitted.contains($0) }
        }
        return nil
    }

    private func schemaFor(key: String, in schema: JSONSchemaNode) -> JSONSchemaNode {
        if case .object(let props, _) = schema {
            for (k, v) in props where k == key { return v }
        }
        return .any
    }

    // MARK: - Array

    private func stepArray(af: ArrayFrame, byte b: UInt8) -> Action {
        switch af.phase {
        case .afterOpen:
            if b == UInt8(ascii: "]") { return .popConsuming }
            return .pushAndRetry(.value(schema: af.items, phase: .start))
        case .afterComma:
            // No trailing comma — an item MUST follow.
            return .pushAndRetry(.value(schema: af.items, phase: .start))
        case .afterItem:
            if b == UInt8(ascii: ",") {
                var nf = af; nf.phase = .afterComma
                return .replace(.array(nf))
            }
            if b == UInt8(ascii: "]") { return .popConsuming }
            return .reject
        case .closed: return .reject
        }
    }

    // MARK: - String

    private func stepString(sf: StringFrame, byte b: UInt8) -> Action {
        switch sf.state {
        case .body:
            if b == UInt8(ascii: "\"") {
                if let allowed = sf.enumValues, !allowed.contains(sf.content) {
                    return .reject
                }
                return .popConsuming
            }
            if b == UInt8(ascii: "\\") {
                if sf.enumValues != nil { return .reject }
                var nf = sf; nf.state = .escape
                return .replace(.string(nf))
            }
            if b < 0x20 { return .reject }
            // Constraint: if enumValues set, every value must have
            // `content + char` as a prefix.
            var nf = sf
            let newPrefix = nf.content + String(UnicodeScalar(b))
            if let allowed = sf.enumValues {
                let stillValid = allowed.contains { $0.hasPrefix(newPrefix) }
                if !stillValid { return .reject }
            }
            nf.content = newPrefix
            return .replace(.string(nf))
        case .escape:
            switch b {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"),
                 UInt8(ascii: "b"), UInt8(ascii: "f"), UInt8(ascii: "n"),
                 UInt8(ascii: "r"), UInt8(ascii: "t"):
                var nf = sf
                nf.state = .body
                nf.content.append("\\")
                nf.content.append(Character(UnicodeScalar(b)))
                return .replace(.string(nf))
            case UInt8(ascii: "u"):
                var nf = sf
                nf.state = .unicodeEscape(digitsSeen: 0)
                nf.content.append("\\u")
                return .replace(.string(nf))
            default: return .reject
            }
        case .unicodeEscape(let n):
            if isHex(b) {
                var nf = sf
                nf.content.append(Character(UnicodeScalar(b)))
                nf.state = (n + 1 == 4) ? .body : .unicodeEscape(digitsSeen: n + 1)
                return .replace(.string(nf))
            }
            return .reject
        case .closed: return .reject
        }
    }

    // MARK: - Number

    private func stepNumber(nf: NumberFrame, byte b: UInt8) -> Action {
        // Numbers terminate on structural / whitespace bytes. Those
        // bytes are NOT consumed by the number; we popAndRetry so the
        // parent sees them.
        let isTerm = (b == UInt8(ascii: ",") || b == UInt8(ascii: "}") ||
                      b == UInt8(ascii: "]") ||
                      b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D)
        if isTerm {
            return isNumberAccepting(nf.state) ? .popAndRetry : .reject
        }
        switch nf.state {
        case .sign:
            if b == UInt8(ascii: "0") {
                var x = nf; x.state = .intZero; return .replace(.number(x))
            }
            if isDigit(b) {
                var x = nf; x.state = .intDigits; return .replace(.number(x))
            }
            return .reject
        case .intZero:
            if b == UInt8(ascii: ".") && !nf.integer {
                var x = nf; x.state = .afterFracPoint; return .replace(.number(x))
            }
            if (b == UInt8(ascii: "e") || b == UInt8(ascii: "E")) && !nf.integer {
                var x = nf; x.state = .afterExp; return .replace(.number(x))
            }
            return .reject
        case .intDigits:
            if isDigit(b) { return .replace(.number(nf)) }
            if b == UInt8(ascii: ".") && !nf.integer {
                var x = nf; x.state = .afterFracPoint; return .replace(.number(x))
            }
            if (b == UInt8(ascii: "e") || b == UInt8(ascii: "E")) && !nf.integer {
                var x = nf; x.state = .afterExp; return .replace(.number(x))
            }
            return .reject
        case .afterFracPoint:
            if isDigit(b) {
                var x = nf; x.state = .fracDigits; return .replace(.number(x))
            }
            return .reject
        case .fracDigits:
            if isDigit(b) { return .replace(.number(nf)) }
            if b == UInt8(ascii: "e") || b == UInt8(ascii: "E") {
                var x = nf; x.state = .afterExp; return .replace(.number(x))
            }
            return .reject
        case .afterExp:
            if b == UInt8(ascii: "+") || b == UInt8(ascii: "-") {
                var x = nf; x.state = .expSign; return .replace(.number(x))
            }
            if isDigit(b) {
                var x = nf; x.state = .expDigits; return .replace(.number(x))
            }
            return .reject
        case .expSign:
            if isDigit(b) {
                var x = nf; x.state = .expDigits; return .replace(.number(x))
            }
            return .reject
        case .expDigits:
            if isDigit(b) { return .replace(.number(nf)) }
            return .reject
        }
    }

    private func isNumberAccepting(_ s: NumberState) -> Bool {
        switch s {
        case .intZero, .intDigits, .fracDigits, .expDigits: return true
        default: return false
        }
    }

    // MARK: - Boolean / Null

    private func stepBoolean(literal: String, position: Int, byte b: UInt8) -> Action {
        let chars = Array(literal.utf8)
        if position >= chars.count { return .reject }
        if b == chars[position] {
            let next = position + 1
            if next == chars.count {
                return .popConsuming
            }
            return .replace(.boolean(literal: literal, position: next))
        }
        return .reject
    }

    private func stepNull(position: Int, byte b: UInt8) -> Action {
        let chars = Array("null".utf8)
        if position >= chars.count { return .reject }
        if b == chars[position] {
            let next = position + 1
            if next == chars.count {
                return .popConsuming
            }
            return .replace(.null(position: next))
        }
        return .reject
    }
}

// MARK: - Helpers

private func isDigit(_ b: UInt8) -> Bool { b >= 0x30 && b <= 0x39 }
private func isHex(_ b: UInt8) -> Bool {
    return (b >= 0x30 && b <= 0x39) ||
           (b >= UInt8(ascii: "a") && b <= UInt8(ascii: "f")) ||
           (b >= UInt8(ascii: "A") && b <= UInt8(ascii: "F"))
}

// MARK: - Logits masker
//
// `LogitsMasker` is the bridge between the FSM and the sampler. It
// holds a pre-computed table mapping each token id → its UTF-8 byte
// sequence. At each step the masker:
//
//   1) Snapshots the FSM
//   2) For each token, replays the bytes; if all accepted → token is valid
//   3) Otherwise the token's logit is set to -inf
//
// The FSM is mutated by the chosen token AFTER sampling: the caller
// passes the sampled id back into `commit(tokenId:into:)`, which
// replays the token's bytes into the live FSM. This is cleaner than
// mutating during the mask loop.

public final class LogitsMasker {

    /// Mapping token id → its UTF-8 bytes. Built once at setup.
    public let tokenBytes: [[UInt8]]
    /// Optional EOS token id. If the FSM is complete, this id is
    /// always allowed (the model can finish), and when the model
    /// samples it generation stops.
    public let eosTokenId: Int?

    /// Build the masker. `decodeId(i)` should return the UTF-8 string
    /// produced when token `i` is decoded in isolation. The mask is
    /// approximate when neighbouring tokens change BPE rendering, but
    /// for nearly all JSON-relevant tokens (ASCII-rich) the per-id
    /// rendering matches the contextual rendering.
    public init(vocabSize: Int, eosTokenId: Int?, decodeId: (Int) -> String) {
        var t: [[UInt8]] = []
        t.reserveCapacity(vocabSize)
        for i in 0..<vocabSize {
            let s = decodeId(i)
            t.append(Array(s.utf8))
        }
        self.tokenBytes = t
        self.eosTokenId = eosTokenId
    }

    /// Compute the mask for the current FSM state. Returns a Float
    /// array of length `vocabSize` containing 0 for valid tokens and
    /// -inf for invalid. Caller adds this to the model's logits.
    ///
    /// When `fsm.isComplete`, only EOS is allowed (caller will stop).
    public func mask(forFSM fsm: JSONSchemaFSM) -> [Float] {
        var out = [Float](repeating: -.infinity, count: tokenBytes.count)
        if fsm.isComplete {
            if let eid = eosTokenId, eid >= 0 && eid < out.count {
                out[eid] = 0
            }
            return out
        }
        for id in 0..<tokenBytes.count {
            let bytes = tokenBytes[id]
            if bytes.isEmpty { continue }
            let probe = fsm.clone()
            if probe.acceptBytes(bytes) {
                out[id] = 0
            }
        }
        return out
    }

    /// Replay a sampled token's bytes into the live FSM. Returns true
    /// if accepted (it should always be — the mask guarantees this).
    @discardableResult
    public func commit(tokenId: Int, into fsm: JSONSchemaFSM) -> Bool {
        guard tokenId >= 0 && tokenId < tokenBytes.count else { return false }
        return fsm.acceptBytes(tokenBytes[tokenId])
    }

    /// Count valid tokens at the current state (for debug / stats).
    public func numValid(forFSM fsm: JSONSchemaFSM) -> Int {
        let m = mask(forFSM: fsm)
        return m.reduce(0) { $0 + ($1 == 0 ? 1 : 0) }
    }
}
