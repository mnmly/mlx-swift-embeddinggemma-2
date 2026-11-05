// Copyright © 2026

import Foundation
import MLX

/// The per-token output states of one input, for attributing a similarity to the tokens that carry it.
///
/// An embedding is the mask-aware mean of the encoder's final token states, L2-normalised. Because the
/// mean is linear, the cosine between two inputs A and B splits exactly over their tokens:
/// `cos(A, B) = Σᵢ aᵢ · ê_B / (L_A · ‖ā‖)`, where `aᵢ` are A's token states, `ā` their mean, `L_A` the
/// token count and `ê_B` B's embedding. ``contributions(toward:)`` returns those per-token terms, and the
/// image soft tokens among them map to cells of the resized image via ``images``.
public struct TokenStates: Sendable {
    /// Row-major `[tokenCount × dimension]` final token states (after the output projection, before
    /// pooling), float32.
    public let states: [Float]
    /// Width of each state row (the full embedding width, 768).
    public let dimension: Int
    /// Token ids, including `<bos>`, `<eos>`, the prompt and the expanded media placeholders.
    public let tokenIds: [Int]
    /// The input's images, in order, with the rows of ``states`` that are their soft tokens.
    public let images: [ImageTokens]
    /// The input's embedding — the normalised mean of ``states``, as ``EmbeddingSession/embed(_:dimension:)``
    /// returns it at full width.
    public let embedding: [Float]
    /// Norm of the un-normalised mean of ``states`` (the divisor that turns a token's dot product into its
    /// share of a cosine).
    public let meanNorm: Float

    /// Number of tokens (rows of ``states``).
    public var tokenCount: Int { tokenIds.count }

    /// One image's soft tokens: a contiguous run of the sequence, row-major over its pooled grid.
    public struct ImageTokens: Sendable {
        /// Rows of ``TokenStates/states`` holding this image's soft tokens.
        public let tokens: Range<Int>
        /// Pooled grid width in soft tokens.
        public let gridWidth: Int
        /// Pooled grid height in soft tokens (`gridWidth · gridHeight == tokens.count`).
        public let gridHeight: Int
        /// Side, in pixels of the resized image, of the square cell each soft token pools
        /// (`patchSize · poolingKernelSize`).
        public let cellSize: Int
    }

    /// Each token's share of the cosine between this input and `other`.
    ///
    /// `cᵢ = sᵢ · ê / (L · ‖s̄‖)`; the shares sum (over all tokens, prompt and template tokens included) to
    /// `cos(self, other)`.
    ///
    /// - Parameter other: The other input's embedding, unit length and ``dimension`` wide.
    /// - Returns: One share per token, in sequence order; empty if the widths differ.
    public func contributions(toward other: [Float]) -> [Float] {
        guard other.count == dimension, meanNorm > 0 else { return [] }
        let scale = 1 / (Float(tokenCount) * meanNorm)
        return (0 ..< tokenCount).map { i in
            var dot: Float = 0
            let row = i * dimension
            for k in 0 ..< dimension { dot += states[row + k] * other[k] }
            return dot * scale
        }
    }
}

extension EmbeddingSession {
    /// Runs one input through the encoder and returns its per-token states alongside its embedding.
    ///
    /// For explaining a similarity rather than computing one: a single unbatched forward pass, so the
    /// tokens are exactly those the mean pool sees (no padding).
    ///
    /// - Parameter input: The input to encode.
    /// - Returns: The token states, the image token layout and the embedding.
    /// - Throws: Preprocessing or model errors for this input.
    public func tokenStates(_ input: EmbeddingInput) throws -> TokenStates {
        let prepared = try processor.prepare(input)
        let output = try model(processor.collate([prepared]))
        let hidden = output.lastHiddenState[0].asType(.float32)          // [L, D]
        let mean = hidden.mean(axis: 0)
        let norm = MLX.sqrt((mean * mean).sum())
        eval(hidden, norm, output.embeddings)

        let ids = prepared.ids
        let imageId = model.configuration.imageTokenId
        var runs: [Range<Int>] = []
        var i = 0
        while i < ids.count {
            guard ids[i] == imageId else { i += 1; continue }
            var j = i
            while j < ids.count, ids[j] == imageId { j += 1 }
            runs.append(i ..< j)
            i = j
        }
        let k = processor.image.poolingKernelSize
        var images: [TokenStates.ImageTokens] = []
        for (run, image) in zip(runs, prepared.images) {
            var maxX: Int32 = -1, maxY: Int32 = -1
            for p in stride(from: 0, to: image.positions.count, by: 2) where image.positions[p] >= 0 {
                maxX = max(maxX, image.positions[p])
                maxY = max(maxY, image.positions[p + 1])
            }
            let w = (Int(maxX) + 1) / k, h = (Int(maxY) + 1) / k
            guard w * h == run.count else { continue }       // layout mismatch: leave the image out
            images.append(.init(tokens: run, gridWidth: w, gridHeight: h,
                                cellSize: processor.image.patchSize * k))
        }
        return TokenStates(
            states: hidden.asArray(Float.self), dimension: hidden.dim(1), tokenIds: ids, images: images,
            embedding: output.embeddings[0].asArray(Float.self), meanNorm: norm.item(Float.self))
    }
}
