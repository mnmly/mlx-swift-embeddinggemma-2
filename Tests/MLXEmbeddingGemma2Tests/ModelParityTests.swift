import Foundation
import MLX
import XCTest

@testable import MLXEmbeddingGemma2

/// Model-level parity against `Tools/gen_fixtures.py` (mlx-vlm reference, BF16): the Swift model is
/// fed the reference processor's tensors, so these tests isolate the network from preprocessing.
/// Needs the checkpoint in the HF hub cache (downloaded on first run).
final class ModelParityTests: XCTestCase {
    nonisolated(unsafe) static var model: EmbeddingGemma2Model?

    func loadModel() async throws -> EmbeddingGemma2Model {
        if let model = Self.model { return model }
        let model = try EmbeddingGemma2Model.load(directory: try await resolveModelDirectory())
        Self.model = model
        return model
    }

    func testText() async throws {
        let f = try Fixture.load("text")
        let model = try await loadModel()
        let (_, got) = try model(f.inputs)
        assertClose(got, f["text_embeds"], minCosine: 0.9999)

        let e = got.asArray(Float.self).chunked(768)
        let scores = [dot(e[0], e[1]), dot(e[0], e[2])]
        XCTAssertEqual(scores[0], 0.6771, accuracy: 0.005, "venus")
        XCTAssertEqual(scores[1], 0.8532, accuracy: 0.005, "mars")
    }

    func testImageFeatures() async throws {
        let f = try Fixture.load("image")
        let model = try await loadModel()
        let features = try model.imageFeatures(f["pixel_values"], positions: f["image_position_ids"])
        XCTAssertEqual(features.shape, f["image_features"].shape)
        // Per-token features are looser than pooled embeddings: the SwiftPM-built Metal kernels and the
        // Python wheel's differ by an ulp in a handful of transcendentals (rsqrt/pow; ~10 of 1.8M
        // elements after the first q_norm), which 16 bf16 layers amplify. GEMMs match bit-for-bit.
        assertRowsClose(features, f["image_features"], minCosine: 0.995)
    }

    func testImage() async throws { try await checkCase("image") }
    func testTextImage() async throws { try await checkCase("text_image") }

    func testAudioTower() async throws {
        let f = try Fixture.load("audio")
        let model = try await loadModel()
        let features = try model.audioFeatures(f["input_features"], mask: f["input_features_mask"])
        XCTAssertEqual(features.shape, f["audio_features"].shape)
        assertRowsClose(features, f["audio_features"], minCosine: 0.999)
    }

    func testAudio() async throws { try await checkCase("audio") }
    func testVideo() async throws { try await checkCase("video") }
    func testHeterogeneousBatch() async throws { try await checkCase("hetero_batch") }

    private func checkCase(_ name: String) async throws {
        let f = try Fixture.load(name)
        let model = try await loadModel()
        let (_, got) = try model(f.inputs)
        assertClose(got, f["text_embeds"], minCosine: 0.999)
    }
}

// MARK: - Helpers

/// Fixtures are generated, not checked in: `make fixtures` (Tools/gen_fixtures.py) writes them to
/// `Tests/Fixtures`. Tests skip when they are missing.
let fixturesDirectory = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .appending(path: "Fixtures")

func fixtureURL(_ name: String, _ ext: String) throws -> URL {
    let url = fixturesDirectory.appending(path: "\(name).\(ext)")
    guard FileManager.default.fileExists(atPath: url.path()) else {
        throw XCTSkip("Missing fixture \(name).\(ext): run `make fixtures` to generate it.")
    }
    return url
}

struct Fixture {
    let arrays: [String: MLXArray]

    static func load(_ name: String) throws -> Fixture {
        Fixture(arrays: try MLX.loadArrays(url: try fixtureURL(name, "safetensors")))
    }

    subscript(_ key: String) -> MLXArray { arrays[key]! }

    var inputs: EmbeddingGemma2Inputs {
        EmbeddingGemma2Inputs(
            inputIds: arrays["input_ids"]!, attentionMask: arrays["attention_mask"]!,
            pixelValues: arrays["pixel_values"], imagePositionIds: arrays["image_position_ids"],
            pixelValuesVideos: arrays["pixel_values_videos"], videoPositionIds: arrays["video_position_ids"],
            inputFeatures: arrays["input_features"], inputFeaturesMask: arrays["input_features_mask"])
    }
}

func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

/// Row-wise cosine between unit-normalized embeddings.
func assertClose(_ got: MLXArray, _ want: MLXArray, minCosine: Float, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(got.shape, want.shape, file: file, line: line)
    let d = want.dim(-1)
    for (i, (g, w)) in zip(got.asType(.float32).asArray(Float.self).chunked(d), want.asArray(Float.self).chunked(d)).enumerated() {
        let cos = dot(g, w)
        XCTAssertGreaterThan(cos, minCosine, "row \(i) cosine \(cos)", file: file, line: line)
    }
}

/// Row-wise cosine between unnormalized feature rows.
func assertRowsClose(_ got: MLXArray, _ want: MLXArray, minCosine: Float, file: StaticString = #filePath, line: UInt = #line) {
    let d = want.dim(-1)
    var worst: Float = 1
    for (g, w) in zip(got.asType(.float32).asArray(Float.self).chunked(d), want.asArray(Float.self).chunked(d)) {
        let n = (dot(g, g) * dot(w, w)).squareRoot()
        worst = min(worst, n > 0 ? dot(g, w) / n : 1)
    }
    XCTAssertGreaterThan(worst, minCosine, "worst row cosine \(worst)", file: file, line: line)
}
