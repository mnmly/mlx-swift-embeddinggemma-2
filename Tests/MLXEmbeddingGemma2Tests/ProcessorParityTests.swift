import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import MLXEmbeddingGemma2

/// Swift preprocessing vs transformers' `EmbeddingGemma2Processor` on the same raw media, then the
/// full Swift pipeline (processor + model) vs the reference embeddings.
final class ProcessorParityTests: XCTestCase {
    nonisolated(unsafe) static var cached: (EmbeddingGemma2Model, EmbeddingGemma2Processor)?

    func load() async throws -> (EmbeddingGemma2Model, EmbeddingGemma2Processor) {
        if let cached = Self.cached { return cached }
        let dir = try await resolveModelDirectory()
        let model = try EmbeddingGemma2Model.load(directory: dir)
        let processor = try await EmbeddingGemma2Processor.load(directory: dir, configuration: model.configuration)
        Self.cached = (model, processor)
        return (model, processor)
    }

    struct Index: Decodable { let texts: [String] }

    static func index() throws -> Index {
        let url = try fixtureURL("index", "json")
        return try JSONDecoder().decode(Index.self, from: Data(contentsOf: url))
    }

    static func png(_ name: String) throws -> RGBImage {
        let url = try fixtureURL(name, "png")
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        return RGBImage(try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil)))
    }

    /// Unpadded token rows of a reference batch.
    static func rows(_ f: Fixture) -> [[Int32]] {
        let ids = f["input_ids"].asArray(Int32.self).chunked(f["input_ids"].dim(1))
        let mask = f["attention_mask"].asArray(Int32.self).chunked(f["input_ids"].dim(1))
        return zip(ids, mask).map { row, m in zip(row, m).filter { $0.1 != 0 }.map(\.0) }
    }

    func check(_ name: String, _ inputs: [EmbeddingInput], pixelTolerance: Float = 2.0 / 255) async throws {
        let f = try Fixture.load(name)
        let (model, processor) = try await load()
        let got = try processor(inputs)
        XCTAssertEqual(Self.rows(Fixture(arrays: ["input_ids": got.inputIds, "attention_mask": got.attentionMask])),
                       Self.rows(f), "input_ids")
        for (key, value) in [("pixel_values", got.pixelValues), ("pixel_values_videos", got.pixelValuesVideos)] {
            guard let want = f.arrays[key] else { continue }
            let value = try XCTUnwrap(value, key)
            XCTAssertEqual(value.shape, want.shape, key)
            let diff = MLX.abs(value - want)
            print("[\(name)] \(key) max|Δ| \(diff.max().item(Float.self) * 255)/255, mean \(diff.mean().item(Float.self) * 255)/255")
            XCTAssertLessThanOrEqual(diff.max().item(Float.self), pixelTolerance, key)
        }
        for (key, value) in [("image_position_ids", got.imagePositionIds), ("video_position_ids", got.videoPositionIds)] {
            guard let want = f.arrays[key] else { continue }
            XCTAssertEqual(try XCTUnwrap(value, key).asArray(Int32.self), want.asArray(Int32.self), key)
        }
        if let want = f.arrays["input_features"] {
            let features = try XCTUnwrap(got.inputFeatures)
            XCTAssertEqual(features.shape, want.shape)
            let diff = MLX.abs(features - want).max().item(Float.self)
            print("[\(name)] input_features max|Δ| \(diff)")
            // float32 FFT (MLX) vs numpy's float64, amplified by log() near the 1e-3 mel floor.
            XCTAssertLessThan(diff, 5e-3)
            XCTAssertEqual(try XCTUnwrap(got.inputFeaturesMask).asArray(Bool.self),
                           f["input_features_mask"].asType(.bool).asArray(Bool.self))
        }
        let (_, embeddings) = try model(got)
        assertClose(embeddings, f["text_embeds"], minCosine: 0.999)
    }

    func testText() async throws {
        try await check("text", try Self.index().texts.map { .text($0) })
    }

    func testImage() async throws {
        try await check("image", [EmbeddingInput([.image(try Self.png("image_landscape"))])])
    }

    func testTextImage() async throws {
        try await check("text_image", [
            EmbeddingInput(prompt: EmbeddingPrompt.document, [.text("A red circle"), .image(try Self.png("image_portrait"))])
        ])
    }

    func testAudio() async throws {
        let wave = try Fixture.load("audio")["waveform"].asArray(Float.self)
        try await check("audio", [EmbeddingInput([.audio(wave)])])
    }

    func testVideo() async throws {
        let raw = try Fixture.load("video")["frames"]
        let (n, h, w) = (raw.dim(0), raw.dim(1), raw.dim(2))
        let bytes = raw.asArray(UInt8.self)
        let frames = (0 ..< n).map {
            RGBImage(width: w, height: h, pixels: Array(bytes[($0 * h * w * 3) ..< (($0 + 1) * h * w * 3)]))
        }
        XCTAssertEqual(VideoProcessor().sampleIndices(frameCount: 6, sourceFPS: 2, duration: 3), [0, 2, 4])
        try await check("video", [EmbeddingInput([.video(VideoClip(frames: frames, fps: 2, duration: 3))])])
    }

    /// A real H.264 file decoded by `MediaLoader.video` (AVFoundation) vs PyAV + the reference processor.
    func testVideoFile() async throws {
        let url = try fixtureURL("video_clip", "mp4")
        let f = try Fixture.load("video_file")
        let clip = try await MediaLoader.video(url)
        XCTAssertEqual(clip.frames.count, 4, "1 fps sampling of a 4 s, 5 fps clip")

        // Decoder difference alone (YUV→RGB conversion), frame by frame.
        let want = f["sampled_frames"].asType(.float32)
        for (i, frame) in clip.frames.enumerated() {
            XCTAssertEqual([frame.height, frame.width], [want.dim(1), want.dim(2)])
            let got = MLXArray(frame.pixels, [frame.height, frame.width, 3]).asType(.float32)
            let diff = MLX.abs(got - want[i])
            print("[video_file] frame \(i) decode max|Δ| \(diff.max().item(Float.self)) mean \(diff.mean().item(Float.self))")
        }

        let (model, processor) = try await load()
        let got = try processor([EmbeddingInput([.video(clip)])])
        XCTAssertEqual(Self.rows(Fixture(arrays: ["input_ids": got.inputIds, "attention_mask": got.attentionMask])),
                       Self.rows(f), "input_ids")
        let (_, embedding) = try model(got)
        let cos = dot(embedding.asArray(Float.self), f["text_embeds"].asArray(Float.self))
        print("[video_file] end-to-end cosine \(cos)")
        XCTAssertGreaterThan(cos, 0.995)
    }

    func testHeterogeneousBatch() async throws {
        let wave = try Fixture.load("audio")["waveform"].asArray(Float.self)
        try await check("hetero_batch", [
            .text("a short sentence", prompt: EmbeddingPrompt.document),
            EmbeddingInput([.image(try Self.png("image_landscape"))]),
            EmbeddingInput([.audio(wave)]),
        ])
    }

    func testVideoSamplingCapsUniformly() {
        var v = VideoProcessor()
        v.maxFrames = 4
        // 10 s at 30 fps → 10 frames at 1 fps → uniform 4 of them (np.linspace(0, 9, 4, dtype=int) = 0, 3, 6, 9).
        XCTAssertEqual(v.sampleIndices(frameCount: 300, sourceFPS: 30, duration: 10), [0, 90, 180, 270])
    }
}
