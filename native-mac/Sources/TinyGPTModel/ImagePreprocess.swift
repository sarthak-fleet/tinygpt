import Foundation
import CoreGraphics
import ImageIO
import MLX

#if canImport(AppKit)
import AppKit
#endif

/// CLIP-style image preprocessing. Takes raw PNG/JPEG bytes (or a
/// path), produces a normalized NHWC `MLXArray` ready to feed a
/// `CLIPVisionModel`.
///
/// Pipeline mirrors HF's `CLIPImageProcessor`:
///
///     bytes
///       ↓ CGImageSource (PNG/JPEG decode)
///       ↓ resize (shorter side → target, bicubic)
///       ↓ center crop to (size, size)
///       ↓ scale to [0, 1] (divide by 255)
///       ↓ normalize per-channel (x - mean) / std
///       ↓ reshape to NHWC [1, H, W, 3]
///
/// Channel order is RGB (after `CGImage` extraction we drop alpha).
/// CLIP uses image_mean=[0.48145466,0.4578275,0.40821073],
/// image_std=[0.26862954,0.26130258,0.27577711]. The defaults below
/// match those exactly; pass different values to support newer ViT
/// preprocessors that ship with different stats (e.g., `image_mean`
/// of `[0.5,0.5,0.5]` for Qwen3-VL).
public struct ImagePreprocessConfig: Sendable, Equatable {
    public var imageSize: Int
    public var imageMean: [Float]   // RGB order, length 3
    public var imageStd: [Float]    // RGB order, length 3
    public var doCenterCrop: Bool
    public var doNormalize: Bool
    public var doResize: Bool

    public init(
        imageSize: Int = 224,
        imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073],
        imageStd:  [Float] = [0.26862954, 0.26130258, 0.27577711],
        doCenterCrop: Bool = true,
        doNormalize: Bool = true,
        doResize: Bool = true
    ) {
        precondition(imageMean.count == 3 && imageStd.count == 3,
                     "mean/std must be length 3 (RGB)")
        self.imageSize = imageSize
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.doCenterCrop = doCenterCrop
        self.doNormalize = doNormalize
        self.doResize = doResize
    }

    /// Read `preprocessor_config.json` from a HuggingFace snapshot dir.
    /// Falls back to the default CLIP stats if the file is absent.
    public static func loadFromDir(_ dir: URL) -> ImagePreprocessConfig {
        let url = dir.appendingPathComponent("preprocessor_config.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else {
            return ImagePreprocessConfig()
        }
        let size: Int = {
            // HF preprocessor may store `size: 224` (CLIP-old) or
            // `size: {shortest_edge: 224}` (CLIP-new / Qwen3-VL).
            if let i = dict["size"] as? Int { return i }
            if let m = dict["size"] as? [String: Any] {
                if let s = m["shortest_edge"] as? Int { return s }
                if let s = m["height"] as? Int { return s }
            }
            return (dict["crop_size"] as? Int) ?? 224
        }()
        let mean: [Float] = (dict["image_mean"] as? [Double])?.map(Float.init)
            ?? [0.48145466, 0.4578275, 0.40821073]
        let std: [Float] = (dict["image_std"] as? [Double])?.map(Float.init)
            ?? [0.26862954, 0.26130258, 0.27577711]
        return ImagePreprocessConfig(
            imageSize: size,
            imageMean: mean,
            imageStd: std,
            doCenterCrop: (dict["do_center_crop"] as? Bool) ?? true,
            doNormalize: (dict["do_normalize"] as? Bool) ?? true,
            doResize: (dict["do_resize"] as? Bool) ?? true
        )
    }
}

public enum ImagePreprocessError: Error, CustomStringConvertible {
    case decodeFailed(String)
    case bitmapFailed(String)

    public var description: String {
        switch self {
        case .decodeFailed(let s): return "could not decode image: \(s)"
        case .bitmapFailed(let s): return "could not extract bitmap pixels: \(s)"
        }
    }
}

public enum ImagePreprocess {
    /// Preprocess from a file path. Returns NHWC `[1, size, size, 3]`.
    public static func preprocess(path: String,
                                   config: ImagePreprocessConfig = .init()) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        return try preprocess(data: data, config: config)
    }

    /// Preprocess from raw image bytes (PNG, JPEG, HEIC — anything
    /// ImageIO accepts). Returns NHWC `[1, size, size, 3]`.
    public static func preprocess(data: Data,
                                   config: ImagePreprocessConfig = .init()) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw ImagePreprocessError.decodeFailed("ImageIO returned nil for the image source")
        }
        return try preprocess(cgImage: cg, config: config)
    }

    /// Preprocess from a decoded `CGImage`. Handles the resize+crop
    /// using Core Graphics (bicubic interpolation), then extracts RGBA
    /// bytes and converts to a normalized float NHWC `MLXArray`.
    public static func preprocess(cgImage source: CGImage,
                                   config: ImagePreprocessConfig = .init()) throws -> MLXArray {
        let target = config.imageSize
        // 1. Resize the SHORTER side to `target` while keeping aspect.
        //    This mirrors `CLIPImageProcessor.resize` with the default
        //    "shortest edge → size, longer scaled proportionally" rule.
        let (resizedW, resizedH): (Int, Int)
        if config.doResize {
            let w = Double(source.width)
            let h = Double(source.height)
            let scale = Double(target) / min(w, h)
            resizedW = Int(round(w * scale))
            resizedH = Int(round(h * scale))
        } else {
            resizedW = source.width
            resizedH = source.height
        }

        // 2. Render the resize result into an RGBA8 bitmap.
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw ImagePreprocessError.bitmapFailed("could not create sRGB colourspace")
        }
        let bytesPerRow = resizedW * 4
        var pixelBuffer = [UInt8](repeating: 0, count: resizedW * resizedH * 4)
        let pixelPtr = pixelBuffer.withUnsafeMutableBufferPointer { $0.baseAddress }
        guard let ctx = CGContext(
            data: pixelPtr,
            width: resizedW,
            height: resizedH,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw ImagePreprocessError.bitmapFailed("could not create CGContext")
        }
        ctx.interpolationQuality = .high
        ctx.draw(source, in: CGRect(x: 0, y: 0, width: resizedW, height: resizedH))

        // 3. Center crop to (target, target).
        let (cropX, cropY): (Int, Int)
        let cropSize: Int
        if config.doCenterCrop {
            cropSize = target
            cropX = max(0, (resizedW - target) / 2)
            cropY = max(0, (resizedH - target) / 2)
        } else {
            cropSize = min(resizedW, resizedH)
            cropX = 0
            cropY = 0
        }

        // 4. Walk the cropped region row by row, extracting RGB (drop
        //    alpha) and scaling to [0, 1].
        let mean = config.imageMean
        let std  = config.imageStd
        var out = [Float](repeating: 0, count: cropSize * cropSize * 3)
        for y in 0..<cropSize {
            let srcRow = (y + cropY) * bytesPerRow
            let dstRow = y * cropSize * 3
            for x in 0..<cropSize {
                let srcIdx = srcRow + (x + cropX) * 4
                let dstIdx = dstRow + x * 3
                var r = Float(pixelBuffer[srcIdx + 0]) / 255.0
                var g = Float(pixelBuffer[srcIdx + 1]) / 255.0
                var b = Float(pixelBuffer[srcIdx + 2]) / 255.0
                if config.doNormalize {
                    r = (r - mean[0]) / std[0]
                    g = (g - mean[1]) / std[1]
                    b = (b - mean[2]) / std[2]
                }
                out[dstIdx + 0] = r
                out[dstIdx + 1] = g
                out[dstIdx + 2] = b
            }
        }

        // 5. Wrap as NHWC `[1, H, W, 3]`.
        return MLXArray(out, [1, cropSize, cropSize, 3])
    }

    /// Generate a synthetic test image — useful for smoke tests when we
    /// don't want to bundle an actual JPEG.
    public static func syntheticTestImage(size: Int = 224) -> MLXArray {
        // Produce a smooth gradient with three channels so the post-norm
        // values stay in a sane range. NHWC `[1, size, size, 3]`.
        var data = [Float](repeating: 0, count: size * size * 3)
        for y in 0..<size {
            for x in 0..<size {
                let idx = (y * size + x) * 3
                data[idx + 0] = Float(x) / Float(size) - 0.5      // R channel ~ horizontal
                data[idx + 1] = Float(y) / Float(size) - 0.5      // G channel ~ vertical
                data[idx + 2] = Float(x + y) / Float(2 * size) - 0.5  // B channel ~ diagonal
            }
        }
        return MLXArray(data, [1, size, size, 3])
    }
}
