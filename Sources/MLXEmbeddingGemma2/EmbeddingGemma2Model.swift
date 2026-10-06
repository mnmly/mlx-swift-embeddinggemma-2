// Copyright © 2026
//
// Port of mlx-vlm `embedding_gemma2/embedding_gemma2.py`.

import Foundation
import MLX
import MLXNN

/// Model-ready tensors for one padded batch, as produced by ``EmbeddingGemma2Processor``.
public struct EmbeddingGemma2Inputs: @unchecked Sendable {
    /// `[B, L]` token ids, media soft tokens expanded.
    public var inputIds: MLXArray
    /// `[B, L]`, 1 for real tokens, 0 for padding.
    public var attentionMask: MLXArray
    /// `[nImages, maxPatches, 3·p²]` patchified images in `[0, 1]`.
    public var pixelValues: MLXArray?
    /// `[nImages, maxPatches, 2]` `(x, y)` patch positions, -1 for padding.
    public var imagePositionIds: MLXArray?
    /// `[nFrames, maxPatches, 3·p²]` patchified video frames (all videos, concatenated).
    public var pixelValuesVideos: MLXArray?
    /// `[nFrames, maxPatches, 2]`.
    public var videoPositionIds: MLXArray?
    /// `[nAudio, T, 128]` log-mel features.
    public var inputFeatures: MLXArray?
    /// `[nAudio, T]`, true for real frames.
    public var inputFeaturesMask: MLXArray?
    /// Soft tokens per image / video frame / audio clip, when known on the host (the processor sets
    /// them). With them the forward pass builds one lazy graph with no GPU→CPU synchronization;
    /// without them the counts are read back from the towers' validity masks.
    public var imageTokenCounts: [Int]?
    public var videoTokenCounts: [Int]?
    public var audioTokenCounts: [Int]?

    public init(
        inputIds: MLXArray, attentionMask: MLXArray,
        pixelValues: MLXArray? = nil, imagePositionIds: MLXArray? = nil,
        pixelValuesVideos: MLXArray? = nil, videoPositionIds: MLXArray? = nil,
        inputFeatures: MLXArray? = nil, inputFeaturesMask: MLXArray? = nil
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.pixelValues = pixelValues
        self.imagePositionIds = imagePositionIds
        self.pixelValuesVideos = pixelValuesVideos
        self.videoPositionIds = videoPositionIds
        self.inputFeatures = inputFeatures
        self.inputFeaturesMask = inputFeaturesMask
    }
}

public enum EmbeddingGemma2Error: LocalizedError {
    case missingTower(String)
    case mediaTokenMismatch(kind: String, tokens: Int, features: Int)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .missingTower(let name): "\(name) inputs require the \(name) tower, which this model was loaded without."
        case .mediaTokenMismatch(let kind, let tokens, let features):
            "\(kind) token count (\(tokens)) does not match feature count (\(features))."
        case .invalidInput(let message): message
        }
    }
}

/// The EmbeddingGemma 2 network: text encoder plus optional vision and audio towers.
public final class EmbeddingGemma2Model: Module {
    public let configuration: EmbeddingGemma2Configuration

    @ModuleInfo(key: "language_model") var languageModel: TextEncoder
    @ModuleInfo(key: "vision_tower") var visionTower: VisionTower?
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder?
    @ModuleInfo(key: "audio_tower") var audioTower: AudioTower?
    @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder?

    public init(_ configuration: EmbeddingGemma2Configuration) {
        self.configuration = configuration
        let text = configuration.textConfig
        _languageModel.wrappedValue = TextEncoder(text)
        if let vision = configuration.visionConfig {
            _visionTower.wrappedValue = VisionTower(vision)
            _embedVision.wrappedValue = MultimodalEmbedder(
                embeddingDim: vision.hiddenSize, textHiddenSize: text.hiddenSize, eps: vision.rmsNormEps)
        }
        if let audio = configuration.audioConfig {
            _audioTower.wrappedValue = AudioTower(audio)
            _embedAudio.wrappedValue = MultimodalEmbedder(
                embeddingDim: audio.outputProjDims ?? audio.hiddenSize, textHiddenSize: text.hiddenSize,
                eps: audio.rmsNormEps)
        }
    }

    /// Projected soft tokens of every valid patch neighbourhood, images concatenated: `[Σ tokens, hidden]`.
    func imageFeatures(_ patches: MLXArray, positions: MLXArray, counts: [Int]? = nil) throws -> MLXArray {
        guard let visionTower, let embedVision else { throw EmbeddingGemma2Error.missingTower("vision") }
        let (pooled, valid) = visionTower(patches, positions: positions)
        let counts = counts ?? valid.asType(.int32).sum(axis: 1).asArray(Int32.self).map(Int.init)
        let rows = counts.enumerated().map { pooled[$0.offset, ..<$0.element] }
        return embedVision(concatenated(rows, axis: 0))
    }

    /// Projected audio soft tokens of every valid frame, clips concatenated: `[Σ tokens, hidden]`.
    func audioFeatures(_ features: MLXArray, mask: MLXArray?, counts: [Int]? = nil) throws -> MLXArray {
        guard let audioTower, let embedAudio else { throw EmbeddingGemma2Error.missingTower("audio") }
        let invalid = mask.map { .!$0.asType(.bool) } ?? MLXArray.zeros([features.dim(0), features.dim(1)], type: Bool.self)
        let (encoded, invalidOut) = audioTower(features, invalid: invalid)
        let projected = embedAudio(encoded)
        let counts = counts ?? (.!invalidOut).asType(.int32).sum(axis: 1).asArray(Int32.self).map(Int.init)
        return concatenated(counts.enumerated().map { projected[$0.offset, ..<$0.element] }, axis: 0)
    }

    /// Replaces the embeddings at `tokenId` positions (row-major) with `features` in order.
    /// With `hostCount` (the processor's consistent counts) no token count is read back from the GPU.
    private func scatter(
        _ embeddings: MLXArray, inputIds: MLXArray, tokenId: Int, features: MLXArray, kind: String, hostCount: Int?
    ) throws -> MLXArray {
        let mask = inputIds .== Int32(tokenId)
        let count = hostCount ?? mask.asType(.int32).sum().item(Int.self)
        guard count == features.dim(0) else {
            throw EmbeddingGemma2Error.mediaTokenMismatch(kind: kind, tokens: count, features: features.dim(0))
        }
        guard count > 0 else { return embeddings }
        let indices = MLX.maximum(cumsum(mask.reshaped(-1).asType(.int32)) - 1, 0)
        let aligned = features[indices].reshaped(embeddings.shape)
        return MLX.where(mask.expandedDimensions(axis: -1), aligned.asType(embeddings.dtype), embeddings)
    }

    /// Runs the encoder.
    ///
    /// - Returns: `lastHiddenState` `[B, L, embeddingDim]` and `embeddings` `[B, embeddingDim]`
    ///   (mask-aware mean pool, L2-normalized, float32).
    public func callAsFunction(_ inputs: EmbeddingGemma2Inputs, positionIds: MLXArray? = nil) throws
        -> (lastHiddenState: MLXArray, embeddings: MLXArray)
    {
        let c = configuration
        let ids = inputs.inputIds
        guard inputs.attentionMask.shape == ids.shape else {
            throw EmbeddingGemma2Error.invalidInput("attention_mask must have the same shape as input_ids")
        }
        let media = (ids .== Int32(c.imageTokenId)) .|| (ids .== Int32(c.videoTokenId))
            .|| (ids .== Int32(c.audioTokenId))
        let textIds = MLX.where(media, Int32(c.textConfig.padTokenId), ids)
        var embeddings = languageModel.embedTokens(textIds)
        embeddings = embeddings
            * MLXArray(Float(c.textConfig.hiddenSize).squareRoot(), dtype: embeddings.dtype)

        if let pixels = inputs.pixelValues, let positions = inputs.imagePositionIds {
            embeddings = try scatter(
                embeddings, inputIds: ids, tokenId: c.imageTokenId,
                features: imageFeatures(pixels, positions: positions, counts: inputs.imageTokenCounts), kind: "Image",
                hostCount: inputs.imageTokenCounts?.reduce(0, +))
        }
        if let pixels = inputs.pixelValuesVideos, let positions = inputs.videoPositionIds {
            embeddings = try scatter(
                embeddings, inputIds: ids, tokenId: c.videoTokenId,
                features: imageFeatures(pixels, positions: positions, counts: inputs.videoTokenCounts), kind: "Video",
                hostCount: inputs.videoTokenCounts?.reduce(0, +))
        }
        if let features = inputs.inputFeatures {
            embeddings = try scatter(
                embeddings, inputIds: ids, tokenId: c.audioTokenId,
                features: audioFeatures(features, mask: inputs.inputFeaturesMask, counts: inputs.audioTokenCounts),
                kind: "Audio", hostCount: inputs.audioTokenCounts?.reduce(0, +))
        }

        let hidden = languageModel(embeddings, attentionMask: inputs.attentionMask, positionIds: positionIds)
        return (hidden, Self.meanPoolNormalize(hidden, mask: inputs.attentionMask))
    }

    /// Mask-aware mean pooling followed by L2 normalization, in float32.
    static func meanPoolNormalize(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        let weights = mask.asType(.float32).expandedDimensions(axis: -1)
        let pooled = (hidden * weights).sum(axis: 1) / MLX.maximum(weights.sum(axis: 1), 1e-9)
        return pooled / MLX.maximum(MLX.sqrt((pooled * pooled).sum(axis: -1, keepDims: true)), 1e-9)
    }

    /// Mirrors the reference `sanitize`: strips a `model.` prefix, drops weights of absent towers and
    /// rotary buffers, and converts PyTorch-layout conv weights to MLX's channels-last layout.
    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (rawKey, rawValue) in weights {
            let key = rawKey.hasPrefix("model.") ? String(rawKey.dropFirst("model.".count)) : rawKey
            var value = rawValue
            if visionTower == nil, key.hasPrefix("vision_tower.") || key.hasPrefix("embed_vision.") { continue }
            if audioTower == nil, key.hasPrefix("audio_tower.") || key.hasPrefix("embed_audio.") { continue }
            if key.contains("rotary_emb.inv_freq") { continue }
            if key.contains("subsample_conv_projection"), key.hasSuffix("conv.weight"),
                let audio = configuration.audioConfig
            {
                let channels = key.contains(".layer0.") ? 1 : audio.subsamplingConvChannels[0]
                if value.dim(-1) != channels { value = value.transposed(0, 2, 3, 1) }
            }
            if key.hasSuffix("depthwise_conv1d.weight"), value.dim(-1) != 1 {
                value = value.transposed(0, 2, 1)
            }
            out[key] = value
        }
        return out
    }
}
