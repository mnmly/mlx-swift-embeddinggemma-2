import CoreGraphics
import XCTest

@testable import MLXEmbeddingGemma2

/// `RGBImage(CGImage)` on colour spaces that can't back an 8-bit bitmap context. Used to trap on a
/// force-unwrapped `CGContext` (seen on archive scans with an input-only scanner ICC profile).
final class RGBImageTests: XCTestCase {
    /// A 2×1 half-float image (red, blue) tagged with `name`.
    func image(in name: CFString) throws -> CGImage {
        let space = try XCTUnwrap(CGColorSpace(name: name))
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: 2, height: 1, bitsPerComponent: 16, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.floatComponents.rawValue
                | CGBitmapInfo.byteOrder16Little.rawValue))
        ctx.setFillColor(CGColor(colorSpace: space, components: [1, 0, 0, 1])!)
        ctx.fill(CGRect(x: 0, y: 0, width: 1, height: 1))
        ctx.setFillColor(CGColor(colorSpace: space, components: [0, 0, 1, 1])!)
        ctx.fill(CGRect(x: 1, y: 0, width: 1, height: 1))
        return try XCTUnwrap(ctx.makeImage())
    }

    func testExtendedSRGBFallsBackToSRGB() throws {
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.extendedSRGB))
        // The precondition for this test: no 8-bit context in this space.
        XCTAssertNil(CGContext(data: nil, width: 2, height: 1, bitsPerComponent: 8, bytesPerRow: 8, space: space,
                               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))

        let rgb = RGBImage(try image(in: CGColorSpace.extendedSRGB))
        XCTAssertEqual(rgb.width, 2)
        XCTAssertEqual(rgb.height, 1)
        XCTAssertEqual(Array(rgb.pixels[0 ..< 3]), [255, 0, 0])
        XCTAssertEqual(Array(rgb.pixels[3 ..< 6]), [0, 0, 255])
    }
}
