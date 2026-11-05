import Foundation
import MLX
import XCTest

@testable import MLXEmbeddingGemma2

/// `tokenStates` must reproduce `embed` and split a cosine exactly over tokens.
final class TokenStatesTests: XCTestCase {
    nonisolated(unsafe) static var session: EmbeddingSession?

    func load() async throws -> EmbeddingSession {
        if let session = Self.session { return session }
        let session = try await EmbeddingSession.load()
        Self.session = session
        return session
    }

    func testEmbeddingMatchesAndContributionsSumToCosine() async throws {
        let session = try await load()
        let image = try ProcessorParityTests.png("image_landscape")
        let a = EmbeddingInput(prompt: EmbeddingPrompt.document, [.image(image)])
        let b = EmbeddingInput.text("a wide landscape photograph", prompt: EmbeddingPrompt.document)

        let sa = try session.tokenStates(a)
        let embedded = try await session.embed(a)
        XCTAssertGreaterThan(dot(sa.embedding, embedded), 0.999, "tokenStates embedding vs embed")

        // Image layout: one contiguous run, grid matches the run length.
        XCTAssertEqual(sa.images.count, 1)
        let img = try XCTUnwrap(sa.images.first)
        XCTAssertEqual(img.gridWidth * img.gridHeight, img.tokens.count)
        XCTAssertTrue(img.tokens.allSatisfy { sa.tokenIds[$0] == session.model.configuration.imageTokenId })

        // Exact decomposition: per-token shares sum to the cosine.
        let eb = try await session.embed(b)
        let shares = sa.contributions(toward: eb)
        XCTAssertEqual(shares.count, sa.tokenCount)
        XCTAssertEqual(shares.reduce(0, +), dot(sa.embedding, eb), accuracy: 1e-3)
    }

    private func dot(_ x: [Float], _ y: [Float]) -> Float { zip(x, y).reduce(0) { $0 + $1.0 * $1.1 } }
}
