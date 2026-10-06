// Copyright © 2026
//
// Mirrors mlx-vlm `embedding_gemma2/config.py` and `gemma4/config.py` (AudioConfig, VisionConfig).

import Foundation

/// Top-level `config.json` of an EmbeddingGemma 2 checkpoint.
public struct EmbeddingGemma2Configuration: Codable, Sendable {
    public var textConfig: TextConfiguration
    /// `nil` when the checkpoint (or caller) drops the image/video tower.
    public var visionConfig: VisionConfiguration?
    /// `nil` when the checkpoint (or caller) drops the audio tower.
    public var audioConfig: AudioConfiguration?
    public var imageTokenId: Int = 258_880
    public var audioTokenId: Int = 258_881
    public var videoTokenId: Int = 258_884
    public var boiTokenId: Int = 255_999
    public var eoiTokenId: Int = 258_882
    public var boaTokenId: Int = 256_000
    public var eoaTokenId: Int = 258_883

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_index"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(TextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfig)
        audioConfig = try c.decodeIfPresent(AudioConfiguration.self, forKey: .audioConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? imageTokenId
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? audioTokenId
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? videoTokenId
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? boiTokenId
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? eoiTokenId
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: .boaTokenId) ?? boaTokenId
        eoaTokenId = try c.decodeIfPresent(Int.self, forKey: .eoaTokenId) ?? eoaTokenId
    }
}

/// The bidirectional text encoder (`text_config`).
public struct TextConfiguration: Codable, Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    /// Width of the per-layer input embeddings (0 disables them).
    public var hiddenSizePerLayerInput: Int
    /// Output width after `embedding_projection` (768).
    public var embeddingDim: Int
    public var rmsNormEps: Float
    /// Local window: a token attends to keys with `|i - j| <= slidingWindow`.
    public var slidingWindow: Int
    public var attentionBias: Bool
    public var padTokenId: Int
    /// `"sliding_attention"` or `"full_attention"` per layer; the last layer is always full.
    public var layerTypes: [String]
    /// Per-layer overrides keyed by zero-padded layer index.
    public var perLayerConfig: [String: LayerOverride]
    /// RoPE base per layer type.
    public var ropeTheta: [String: Float]

    public struct LayerOverride: Codable, Sendable {
        public var headDim: Int?
        public var numAttentionHeads: Int?
        public var numKeyValueHeads: Int?
        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
        }
    }

    struct RopeParameters: Codable { var rope_theta: Float }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case embeddingDim = "embedding_dim"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case attentionBias = "attention_bias"
        case padTokenId = "pad_token_id"
        case layerTypes = "layer_types"
        case perLayerConfig = "per_layer_config"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262_144
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 4
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        hiddenSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 512
        embeddingDim = try c.decodeIfPresent(Int.self, forKey: .embeddingDim) ?? 768
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0

        var types = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< numHiddenLayers).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
        guard types.count == numHiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: c, debugDescription: "layer_types must have one entry per text layer")
        }
        types[types.count - 1] = "full_attention"
        layerTypes = types

        if let raw = try c.decodeIfPresent([String: LayerOverride].self, forKey: .perLayerConfig) {
            // Normalize keys to two digits, as the reference does.
            perLayerConfig = Dictionary(
                uniqueKeysWithValues: raw.map { (String(format: "%02d", Int($0.key) ?? 0), $0.value) })
        } else {
            perLayerConfig = Dictionary(
                uniqueKeysWithValues: types.enumerated().filter { $0.element == "full_attention" }.map {
                    (String(format: "%02d", $0.offset), LayerOverride(headDim: 512, numKeyValueHeads: 1))
                })
        }

        if let rope = try c.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters) {
            ropeTheta = rope.mapValues(\.rope_theta)
        } else {
            ropeTheta = ["full_attention": 1_000_000, "sliding_attention": 10_000]
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(hiddenSizePerLayerInput, forKey: .hiddenSizePerLayerInput)
        try c.encode(embeddingDim, forKey: .embeddingDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(padTokenId, forKey: .padTokenId)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(perLayerConfig, forKey: .perLayerConfig)
        try c.encode(ropeTheta.mapValues { RopeParameters(rope_theta: $0) }, forKey: .ropeParameters)
    }

    /// Effective `(heads, kvHeads, headDim)` of layer `index` after `per_layer_config` overrides.
    func attentionShape(layer index: Int) -> (heads: Int, kvHeads: Int, headDim: Int) {
        let o = perLayerConfig[String(format: "%02d", index)]
        return (
            o?.numAttentionHeads ?? numAttentionHeads,
            o?.numKeyValueHeads ?? numKeyValueHeads,
            o?.headDim ?? headDim
        )
    }
}

extension TextConfiguration.LayerOverride {
    init(headDim: Int?, numKeyValueHeads: Int?) {
        self.headDim = headDim
        self.numAttentionHeads = nil
        self.numKeyValueHeads = numKeyValueHeads
    }
}

/// The Gemma 4 vision tower (`vision_config`).
public struct VisionConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 768
    public var intermediateSize: Int = 3072
    public var numHiddenLayers: Int = 16
    public var numAttentionHeads: Int = 12
    public var numKeyValueHeads: Int = 12
    public var headDim: Int = 64
    public var rmsNormEps: Float = 1e-6
    public var defaultOutputLength: Int = 280
    public var patchSize: Int = 16
    public var positionEmbeddingSize: Int = 10_240
    public var poolingKernelSize: Int = 3
    public var standardize: Bool = false
    public var useClippedLinears: Bool = false
    public var ropeTheta: Float = 100

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case standardize
        case useClippedLinears = "use_clipped_linears"
        case ropeParameters = "rope_parameters"
    }

    struct RopeParameters: Codable { var rope_theta: Float? }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        hiddenSize = try d(.hiddenSize, hiddenSize)
        intermediateSize = try d(.intermediateSize, intermediateSize)
        numHiddenLayers = try d(.numHiddenLayers, numHiddenLayers)
        numAttentionHeads = try d(.numAttentionHeads, numAttentionHeads)
        numKeyValueHeads = try d(.numKeyValueHeads, numKeyValueHeads)
        headDim = try d(.headDim, headDim)
        rmsNormEps = try d(.rmsNormEps, rmsNormEps)
        defaultOutputLength = try d(.defaultOutputLength, defaultOutputLength)
        patchSize = try d(.patchSize, patchSize)
        positionEmbeddingSize = try d(.positionEmbeddingSize, positionEmbeddingSize)
        poolingKernelSize = try d(.poolingKernelSize, poolingKernelSize)
        standardize = try d(.standardize, standardize)
        useClippedLinears = try d(.useClippedLinears, useClippedLinears)
        ropeTheta = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)?.rope_theta ?? ropeTheta
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(defaultOutputLength, forKey: .defaultOutputLength)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(positionEmbeddingSize, forKey: .positionEmbeddingSize)
        try c.encode(poolingKernelSize, forKey: .poolingKernelSize)
        try c.encode(standardize, forKey: .standardize)
        try c.encode(useClippedLinears, forKey: .useClippedLinears)
        try c.encode(RopeParameters(rope_theta: ropeTheta), forKey: .ropeParameters)
    }
}

/// The Gemma 4 Conformer audio tower (`audio_config`).
public struct AudioConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 12
    public var numAttentionHeads: Int = 8
    public var subsamplingConvChannels: [Int] = [128, 32]
    public var convKernelSize: Int = 5
    public var residualWeight: Float = 0.5
    public var attentionChunkSize: Int = 12
    public var attentionContextLeft: Int = 13
    public var attentionContextRight: Int = 0
    public var attentionLogitCap: Float = 50
    public var attentionInvalidLogitsValue: Float = -1e9
    public var useClippedLinears: Bool = true
    public var rmsNormEps: Float = 1e-6
    public var gradientClipping: Float = 1e10
    /// Width of `output_proj`; `nil` means no output projection.
    public var outputProjDims: Int? = 1536

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case residualWeight = "residual_weight"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case useClippedLinears = "use_clipped_linears"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case outputProjDims = "output_proj_dims"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? fallback
        }
        hiddenSize = try d(.hiddenSize, hiddenSize)
        numHiddenLayers = try d(.numHiddenLayers, numHiddenLayers)
        numAttentionHeads = try d(.numAttentionHeads, numAttentionHeads)
        subsamplingConvChannels = try d(.subsamplingConvChannels, subsamplingConvChannels)
        convKernelSize = try d(.convKernelSize, convKernelSize)
        residualWeight = try d(.residualWeight, residualWeight)
        attentionChunkSize = try d(.attentionChunkSize, attentionChunkSize)
        attentionContextLeft = try d(.attentionContextLeft, attentionContextLeft)
        attentionContextRight = try d(.attentionContextRight, attentionContextRight)
        attentionLogitCap = try d(.attentionLogitCap, attentionLogitCap)
        attentionInvalidLogitsValue = try d(.attentionInvalidLogitsValue, attentionInvalidLogitsValue)
        useClippedLinears = try d(.useClippedLinears, useClippedLinears)
        rmsNormEps = try d(.rmsNormEps, rmsNormEps)
        gradientClipping = try d(.gradientClipping, gradientClipping)
        outputProjDims = try c.decodeIfPresent(Int.self, forKey: .outputProjDims) ?? outputProjDims
    }
}
