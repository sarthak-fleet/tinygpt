import Foundation
import TinyGPTModel

/// `tinygpt gguf-inspect <path.gguf>` — parse a GGUF file and print its
/// metadata + tensor inventory. Doubles as a smoke test for
/// `GGUFReader.parse` until full HF-style loading lands.
///
/// USAGE
///   tinygpt gguf-inspect <path.gguf> [--dequant <name>]
///
///   --dequant <name>   also dequantise the named tensor and print
///                      its shape + first/last few elements
enum GGUFInspect {
    static func run(args: [String]) {
        var path: String? = nil
        var dequantName: String? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--dequant":     dequantName = args[i+1]; i += 2
            case "-h", "--help":  exitUsage(0)
            default:
                if args[i].hasPrefix("-") { fputs("unknown flag: \(args[i])\n", stderr); exitUsage() }
                path = args[i]; i += 1
            }
        }
        guard let path = path else { fputs("missing <path.gguf>\n", stderr); exitUsage() }

        let parsed: GGUFReader.ParsedFile
        do { parsed = try GGUFReader.parse(url: URL(fileURLWithPath: path)) }
        catch { fputs("parse failed: \(error)\n", stderr); exit(1) }

        print("metadata (\(parsed.metadata.count) entries):")
        for (k, v) in parsed.metadata.sorted(by: { $0.key < $1.key }) {
            let preview: String
            if let s = v as? String { preview = "\"\(s.prefix(60))\"" }
            else if let arr = v as? [Any] { preview = "[\(arr.count) elements]" }
            else { preview = "\(v)" }
            print("  \(k)  =  \(preview)")
        }
        print("\ntensors (\(parsed.tensors.count)):")
        for t in parsed.tensors.prefix(40) {
            let shapeStr = t.shape.map(String.init).joined(separator: ", ")
            let typeName = typeLabel(t.type)
            print("  \(t.name.padding(toLength: 48, withPad: " ", startingAt: 0))  type=\(typeName)  shape=[\(shapeStr)]  off=\(t.offset)")
        }
        if parsed.tensors.count > 40 {
            print("  … +\(parsed.tensors.count - 40) more")
        }

        if let target = dequantName {
            guard let info = parsed.tensors.first(where: { $0.name == target }) else {
                fputs("no tensor named '\(target)' in this file\n", stderr); exit(1)
            }
            do {
                let arr = try GGUFReader.loadTensor(info, from: parsed)
                let flat = arr.asArray(Float.self)
                let head = flat.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", ")
                let tail = flat.suffix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
                print("\ndequantised '\(target)':")
                print("  total elements: \(flat.count)")
                print("  first 8:  [\(head)]")
                print("  last 4:   [\(tail)]")
            } catch {
                fputs("dequant failed: \(error)\n", stderr); exit(1)
            }
        }
    }

    private static func typeLabel(_ t: UInt32) -> String {
        switch GGUFReader.GGMLType(rawValue: t) {
        case .f32:  return "F32"
        case .f16:  return "F16"
        case .q4_0: return "Q4_0"
        case .q8_0: return "Q8_0"
        case nil:   return "unsupported(\(t))"
        }
    }

    private static func exitUsage(_ code: Int32 = 2) -> Never {
        print("""
        usage: tinygpt gguf-inspect <path.gguf> [--dequant <tensor-name>]

        Print metadata + tensor inventory for a GGUF file. First-cut
        recognises F32 / F16 / Q4_0 / Q8_0 tensor types; k-quants
        (Q4_K / Q6_K / etc.) print as 'unsupported(N)'.
        """)
        exit(code)
    }
}
