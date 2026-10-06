import Foundation
import MLX
import XCTest

@testable import MLXEmbeddingGemma2

/// The large-job path: lazy loading, budget packing across lookahead windows, per-item failures.
final class StreamingTests: XCTestCase {
    nonisolated(unsafe) static var session: EmbeddingSession?

    /// Tiny budgets so a dozen inputs span several windows and batches.
    func load() async throws -> EmbeddingSession {
        if let session = Self.session { return session }
        var config = EmbeddingSessionConfiguration()
        config.batchSize = 3
        config.maxBatchTokens = 600
        config.maxBatchVisionFrames = 2
        config.lookahead = 5
        let session = try await EmbeddingSession.load(config)
        Self.session = session
        return session
    }

    func testMixedStreamMatchesSingleInputsAndIsolatesFailures() async throws {
        let session = try await load()
        let image = try ProcessorParityTests.png("image_landscape")
        let wave = try Fixture.load("audio")["waveform"].asArray(Float.self)
        let frames = (0 ..< 3).map { i in
            RGBImage(width: 96, height: 64, pixels: (0 ..< 96 * 64 * 3).map { UInt8(($0 * (i + 3)) % 251) })
        }
        let sentence = "Embeddings place similar meanings close together. "
        var inputs: [EmbeddingInput] = (0 ..< 8).map {
            .text(String(repeating: sentence, count: 1 + 5 * $0), prompt: EmbeddingPrompt.document)
        }
        inputs.insert(EmbeddingInput([.image(image)]), at: 2)
        inputs.insert(EmbeddingInput([.audio(wave)]), at: 5)
        inputs.insert(EmbeddingInput(prompt: EmbeddingPrompt.document, [.text("frames"), .video(VideoClip(frames: frames, fps: nil))]), at: 7)
        let bad = [4, 9]
        inputs.insert(EmbeddingInput([.audio([0, 0, 0])]), at: bad[0])  // too short to yield a token
        inputs.insert(EmbeddingInput([.video(VideoClip(frames: [], fps: nil))]), at: bad[1])  // no frames

        var results = [Int: Result<[Float], any Error>]()
        var duplicates = 0
        try await session.embed(count: inputs.count, load: { [inputs] in inputs[$0] }) { index, result in
            if results[index] != nil { duplicates += 1 }
            results[index] = result
        }
        XCTAssertEqual(duplicates, 0)
        XCTAssertEqual(results.count, inputs.count, "every index reported once")

        for (i, input) in inputs.enumerated() {
            let result = try XCTUnwrap(results[i])
            if bad.contains(i) {
                if case .success = result { XCTFail("input \(i) should fail") }
                continue
            }
            let streamed = try result.get()
            let single = try await session.embed(input)
            XCTAssertGreaterThan(dot(streamed, single), 0.999, "input \(i): batched vs single")
        }
    }

    func testThrowingLoaderOnlyFailsThatIndex() async throws {
        let session = try await load()
        struct Unreadable: Error {}
        var failures: [Int] = [], successes = 0
        try await session.embed(count: 7, load: { i in
            if i == 3 { throw Unreadable() }
            return .text("item \(i)")
        }) { index, result in
            switch result {
            case .success: successes += 1
            case .failure: failures.append(index)
            }
        }
        XCTAssertEqual(failures, [3])
        XCTAssertEqual(successes, 6)
    }

    func testPackingRespectsBudgets() async throws {
        let session = try await load()
        let c = session.configuration
        let items = try (0 ..< 20).map { i in
            (i, try session.processor.prepare(.text(String(repeating: "word ", count: 3 + i * 7))))
        }
        let batches = session.pack(items)
        XCTAssertEqual(batches.flatMap { $0.map(\.0) }.sorted(), Array(0 ..< 20))
        for batch in batches where batch.count > 1 {
            XCTAssertLessThanOrEqual(batch.count, c.batchSize)
            XCTAssertLessThanOrEqual(batch.count * batch.map(\.1.tokenCount).max()!, c.maxBatchTokens)
        }
    }

    /// The per-clip formula equals `replace_audio_token` applied to the padded batch mask.
    func testAudioSoftTokensMatchBatchMask() {
        let fx = AudioFeatureExtractor()
        let lengths = [161, 320, 321, 480, 481, 1000, 16_000, 37_920, 160_000, 480_000]
        let (_, mask) = fx(lengths.map { [Float](repeating: 0.1, count: $0) })
        let rows = mask.asArray(Bool.self).chunked(mask.dim(1))
        for (n, row) in zip(lengths, rows) {
            var m = row
            for _ in 0 ..< 2 {
                let tOut = (m.count + 2 - 3) / 2 + 1
                m = Array(stride(from: 0, to: m.count, by: 2).map { m[$0] }.prefix(tOut))
            }
            XCTAssertEqual(fx.softTokens(samples: n), m.filter { $0 }.count, "\(n) samples")
        }
    }
}
