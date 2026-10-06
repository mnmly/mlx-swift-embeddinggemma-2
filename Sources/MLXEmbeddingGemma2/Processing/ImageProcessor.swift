// Copyright © 2026
//
// Port of transformers `Gemma4ImageProcessor` / `EmbeddingGemma2VideoProcessor` pixel handling:
// aspect-ratio-preserving resize to a soft-token budget (antialiased bicubic, PIL/torchvision
// kernel), rescale to [0, 1], patchify to `[P, p·p·3]` with `(x, y)` positions, pad to the budget.

import CoreGraphics
import Foundation

/// An 8-bit RGB raster in row-major `[height, width, 3]` order.
public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3, "expected width·height·3 bytes")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Decodes `image` into RGB bytes without color conversion when it is already RGB
    /// (matching PIL's `convert("RGB")`), otherwise via sRGB. Alpha is dropped (composited on black).
    public init(_ image: CGImage) {
        let width = image.width, height = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBytes { buffer in
            let ctx = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.interpolationQuality = .none
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0 ..< width * height {
            rgb[3 * i] = rgba[4 * i]
            rgb[3 * i + 1] = rgba[4 * i + 1]
            rgb[3 * i + 2] = rgba[4 * i + 2]
        }
        self.init(width: width, height: height, pixels: rgb)
    }
}

/// One image (or video frame) ready for the vision tower.
public struct PatchifiedImage: Sendable {
    /// `[maxPatches · patchSize² · 3]` row-major 8-bit patches, zero-padded. Rescaled to `[0, 1]`
    /// (`× 1/255`, as the reference's `rescale_factor`) when batched, on the GPU.
    public var patches: [UInt8]
    /// `[maxPatches · 2]` `(x, y)` patch positions, -1 for padding.
    public var positions: [Int32]
    /// Soft tokens this image contributes after pooling.
    public var softTokens: Int
}

public struct ImageProcessor: Sendable {
    /// Allowed soft-token budgets per image or frame.
    public static let supportedSoftTokens = [70, 140, 280, 560, 1120]

    public var patchSize = 16
    public var poolingKernelSize = 3
    public var maxSoftTokens = 280

    public init(patchSize: Int = 16, poolingKernelSize: Int = 3, maxSoftTokens: Int = 280) {
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.maxSoftTokens = maxSoftTokens
    }

    var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }
    public var patchDimension: Int { patchSize * patchSize * 3 }

    /// The largest size whose sides are multiples of `pooling·patch` and that fits the patch budget.
    public func targetSize(height: Int, width: Int) throws -> (height: Int, width: Int) {
        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        let side = poolingKernelSize * patchSize
        var h = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var w = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        if h == 0, w == 0 {
            throw EmbeddingGemma2Error.invalidInput("image \(width)x\(height) would resize to 0x0")
        }
        let maxSide = (maxPatches / (poolingKernelSize * poolingKernelSize)) * side
        if h == 0 {
            h = side
            w = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
        } else if w == 0 {
            w = side
            h = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
        }
        return (h, w)
    }

    public func callAsFunction(_ image: RGBImage) throws -> PatchifiedImage {
        guard Self.supportedSoftTokens.contains(maxSoftTokens) else {
            throw EmbeddingGemma2Error.invalidInput("max_soft_tokens must be one of \(Self.supportedSoftTokens)")
        }
        let (h, w) = try targetSize(height: image.height, width: image.width)
        let resized = (h == image.height && w == image.width) ? image : resizeBicubicAA(image, width: w, height: h)

        let p = patchSize
        let (ph, pw) = (h / p, w / p)
        let real = ph * pw
        let dim = patchDimension
        var patches = [UInt8](repeating: 0, count: maxPatches * dim)
        var positions = [Int32](repeating: -1, count: maxPatches * 2)
        resized.pixels.withUnsafeBufferPointer { src in
            patches.withUnsafeMutableBufferPointer { dst in
                for py in 0 ..< ph {
                    for px in 0 ..< pw {
                        let patch = py * pw + px
                        positions[2 * patch] = Int32(px)
                        positions[2 * patch + 1] = Int32(py)
                        var o = patch * dim
                        // Patch layout is (row, col, channel), matching `convert_image_to_patches`.
                        for r in 0 ..< p {
                            let rowStart = ((py * p + r) * w + px * p) * 3
                            for c in 0 ..< (p * 3) {
                                dst[o] = src[rowStart + c]
                                o += 1
                            }
                        }
                    }
                }
            }
        }
        return PatchifiedImage(
            patches: patches, positions: positions,
            softTokens: real / (poolingKernelSize * poolingKernelSize))
    }
}

// MARK: - Antialiased bicubic resampling (PIL `ImagingResample` / torchvision `antialias=True`)

private func bicubicKernel(_ x: Double) -> Double {
    let a = -0.5
    let x = abs(x)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
    return 0
}

/// Per-output-pixel source span and normalized weights for one axis.
private func resampleCoefficients(input: Int, output: Int) -> (starts: [Int], weights: [[Double]]) {
    let scale = Double(input) / Double(output)
    let filterScale = max(scale, 1)
    let support = 2 * filterScale
    var starts: [Int] = []
    var weights: [[Double]] = []
    for i in 0 ..< output {
        let center = (Double(i) + 0.5) * scale
        let lo = max(Int(center - support + 0.5), 0)
        let hi = min(Int(center + support + 0.5), input)
        var w = (lo ..< hi).map { bicubicKernel((Double($0) - center + 0.5) / filterScale) }
        let total = w.reduce(0, +)
        if total != 0 { w = w.map { $0 / total } }
        starts.append(lo)
        weights.append(w)
    }
    return (starts, weights)
}

/// PIL-style 8-bit fixed-point pass rounding (22 fractional bits), clamped to `0...255`.
private let precisionBits = 22

private func fixedPoint(_ weights: [[Double]]) -> [[Int64]] {
    weights.map { $0.map { Int64(($0 * Double(1 << precisionBits)).rounded(.toNearestOrAwayFromZero)) } }
}

private func clampByte(_ acc: Int64) -> UInt8 {
    let v = (acc + (1 << (precisionBits - 1))) >> precisionBits
    return UInt8(clamping: min(max(v, 0), 255))
}

/// Horizontal pass then vertical pass, each rounded to 8 bits, like PIL and torchvision's uint8 path.
func resizeBicubicAA(_ image: RGBImage, width: Int, height: Int) -> RGBImage {
    var current = image
    if width != image.width {
        current = resamplePass(current, outputCount: width, horizontal: true)
    }
    if height != current.height {
        current = resamplePass(current, outputCount: height, horizontal: false)
    }
    return current
}

/// One separable pass along x (`horizontal`) or y, with flattened fixed-point taps.
private func resamplePass(_ image: RGBImage, outputCount: Int, horizontal: Bool) -> RGBImage {
    let inputCount = horizontal ? image.width : image.height
    let (starts, w) = resampleCoefficients(input: inputCount, output: outputCount)
    let taps = w.map(\.count).max() ?? 0
    var kernel = [Int64](repeating: 0, count: outputCount * taps)
    for (i, row) in fixedPoint(w).enumerated() {
        for (j, k) in row.enumerated() { kernel[i * taps + j] = k }
    }
    let counts = w.map(\.count)
    let (outW, outH) = horizontal ? (outputCount, image.height) : (image.width, outputCount)
    // Element stride between consecutive taps: one pixel along x, one row along y.
    let tapStride = horizontal ? 3 : image.width * 3
    var out = [UInt8](repeating: 0, count: outW * outH * 3)
    let srcWidth = image.width
    image.pixels.withUnsafeBufferPointer { srcBuffer in
        kernel.withUnsafeBufferPointer { kBuffer in
            out.withUnsafeMutableBufferPointer { dstBuffer in
                nonisolated(unsafe) let (src, k, dst) = (srcBuffer, kBuffer, dstBuffer)
                // Output rows are independent; each writes a disjoint slice of `dst`.
                DispatchQueue.concurrentPerform(iterations: outH) { y in
                    for x in 0 ..< outW {
                        let i = horizontal ? x : y
                        var p = horizontal ? (y * srcWidth + starts[i]) * 3 : (starts[i] * srcWidth + x) * 3
                        var r: Int64 = 0, g: Int64 = 0, b: Int64 = 0
                        for j in 0 ..< counts[i] {
                            let kw = k[i * taps + j]
                            r += Int64(src[p]) * kw
                            g += Int64(src[p + 1]) * kw
                            b += Int64(src[p + 2]) * kw
                            p += tapStride
                        }
                        let o = (y * outW + x) * 3
                        dst[o] = clampByte(r)
                        dst[o + 1] = clampByte(g)
                        dst[o + 2] = clampByte(b)
                    }
                }
            }
        }
    }
    return RGBImage(width: outW, height: outH, pixels: out)
}
