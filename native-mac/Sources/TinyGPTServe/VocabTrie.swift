// VocabTrie — shared-prefix vocab walk for grammar mask compute.
// FAILED OPTIMIZATION 2026-06-08 — see docs/prds/serve-fsm-trie-optimization.md.
// Kept for reference / differential testing only. Self.useTrie = false by default.

import Foundation


final class VocabTrieNode {
    var children: [UInt8: VocabTrieNode] = [:]
    var terminalTokenIds: [Int32] = []
    init() {}
}


final class VocabTrie {
    let root: VocabTrieNode
    let vocabSize: Int

    init(tokenBytes: [[UInt8]]) {
        self.vocabSize = tokenBytes.count
        self.root = VocabTrieNode()
        for (id, bytes) in tokenBytes.enumerated() {
            if bytes.isEmpty { continue }
            var node = root
            for b in bytes {
                if let child = node.children[b] {
                    node = child
                } else {
                    let next = VocabTrieNode()
                    node.children[b] = next
                    node = next
                }
            }
            node.terminalTokenIds.append(Int32(id))
        }
    }

    func mask(fsm: ServeByteFSM, into out: inout [Float]) {
        guard !fsm.isComplete else { return }
        descend(node: root, fsm: fsm, out: &out)
    }

    private func descend(node: VocabTrieNode, fsm: ServeByteFSM, out: inout [Float]) {
        for (byte, child) in node.children {
            let probe = fsm.cloneForServe()
            if probe.acceptByte(byte) {
                for tid in child.terminalTokenIds {
                    out[Int(tid)] = 0
                }
                if !child.children.isEmpty {
                    descend(node: child, fsm: probe, out: &out)
                }
            }
        }
    }
}

extension ServeByteFSM {
    func acceptByteDefault(_ b: UInt8) -> Bool { acceptBytes([b]) }
}
