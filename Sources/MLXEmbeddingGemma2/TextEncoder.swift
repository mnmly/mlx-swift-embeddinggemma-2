// Copyright © 2026
//
// Port of mlx-vlm `embedding_gemma2/language.py`: a bidirectional Gemma4-style encoder with
// alternating sliding/full attention, per-layer inputs (PLE) and a final projection to the
// embedding width.

import Foundation
import MLX
import MLXFast
import MLXNN

/// RMS normalization without a learned scale (`RMSNormNoScale`).
func rmsNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps)
}

/// Standard rotary embedding that keeps FP32 angles and rounds cos/sin to the activation dtype,
/// matching the reference (and PyTorch) rather than `MLXFast.RoPE`'s internal precision.
struct TextRotaryEmbedding {
    let frequencies: MLXArray

    init(dimensions: Int, base: Float) {
        let exponents = MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) / Float(dimensions) })
        frequencies = 1 / MLX.pow(MLXArray(base), exponents)
    }

    /// - Parameters:
    ///   - x: `[B, H, L, D]`.
    ///   - positions: `[B or 1, L]` integer positions.
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        var angles = positions.asType(.float32).expandedDimensions(axis: -1) * frequencies
        angles = concatenated([angles, angles], axis: -1).expandedDimensions(axis: 1)
        let cos = MLX.cos(angles).asType(x.dtype)
        let sin = MLX.sin(angles).asType(x.dtype)
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos + rotated * sin
    }
}

final class TextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let eps: Float
    let rope: TextRotaryEmbedding

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: TextConfiguration, layer: Int) {
        (heads, kvHeads, headDim) = config.attentionShape(layer: layer)
        eps = config.rmsNormEps
        let hidden = config.hiddenSize
        _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, hidden, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        let theta = config.ropeTheta[config.layerTypes[layer]] ?? 10_000
        rope = TextRotaryEmbedding(dimensions: headDim, base: theta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = qNorm(qProj(x).reshaped(B, L, heads, -1))
        let k = kNorm(kProj(x).reshaped(B, L, kvHeads, -1))
        let v = rmsNormNoScale(vProj(x).reshaped(B, L, kvHeads, -1), eps: eps)
        let out = MLXFast.scaledDotProductAttention(
            queries: rope(q.transposed(0, 2, 1, 3), positions: positions),
            keys: rope(k.transposed(0, 2, 1, 3), positions: positions),
            values: v.transposed(0, 2, 1, 3),
            scale: 1.0, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hidden: Int, intermediate: Int) {
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// Projects the input embeddings into one small input per layer (`ple`).
final class PerLayerInputs: Module {
    let layers: Int
    let width: Int
    let scale: Float

    @ModuleInfo(key: "per_layer_model_projection") var projection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var norm: RMSNorm

    init(_ config: TextConfiguration) {
        layers = config.numHiddenLayers
        width = config.hiddenSizePerLayerInput
        scale = pow(Float(config.hiddenSize), -0.5)
        _projection.wrappedValue = Linear(config.hiddenSize, layers * width, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: width, eps: config.rmsNormEps)
    }

    /// `[B, L, hidden]` → `[B, L, layers, width]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = (projection(x) * scale).reshaped(x.dim(0), x.dim(1), layers, width)
        return norm(projected)
    }
}

/// Gates a layer's per-layer input into the residual stream (`ple_block`).
final class PerLayerBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var gate: Linear
    @ModuleInfo(key: "per_layer_projection") var projection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var norm: RMSNorm

    init(_ config: TextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
        _projection.wrappedValue = Linear(config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray) -> MLXArray {
        x + norm(projection(geluApproximate(gate(x)) * perLayerInput))
    }
}

final class TextEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TextAttention
    @ModuleInfo var mlp: TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "ple_block") var pleBlock: PerLayerBlock?
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: TextConfiguration, layer: Int) {
        let h = config.hiddenSize
        _selfAttn.wrappedValue = TextAttention(config, layer: layer)
        _mlp.wrappedValue = TextMLP(hidden: h, intermediate: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _pleBlock.wrappedValue = config.hiddenSizePerLayerInput > 0 ? PerLayerBlock(config) : nil
        _layerScalar.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, positions: MLXArray, perLayerInput: MLXArray?
    ) -> MLXArray {
        var h = x + postAttentionLayerNorm(selfAttn(inputLayerNorm(x), mask: mask, positions: positions))
        h = h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        if let pleBlock, let perLayerInput {
            h = pleBlock(h, perLayerInput: perLayerInput)
        }
        return h * layerScalar
    }
}

/// The bidirectional text encoder (`language_model`).
final class TextEncoder: Module {
    let config: TextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var ple: PerLayerInputs?
    @ModuleInfo var layers: [TextEncoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    init(_ config: TextConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _ple.wrappedValue = config.hiddenSizePerLayerInput > 0 ? PerLayerInputs(config) : nil
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { TextEncoderLayer(config, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _embeddingProjection.wrappedValue = Linear(config.hiddenSize, config.embeddingDim, bias: false)
    }

    /// - Parameters:
    ///   - inputsEmbeds: `[B, L, hidden]`, already scaled by `sqrt(hidden)`.
    ///   - attentionMask: `[B, L]`, nonzero for real tokens.
    ///   - positionIds: `[B, L]` or `nil` for `0 ..< L`.
    /// - Returns: `[B, L, embeddingDim]` projected token states.
    func callAsFunction(_ inputsEmbeds: MLXArray, attentionMask: MLXArray, positionIds: MLXArray? = nil)
        -> MLXArray
    {
        let L = inputsEmbeds.dim(1)
        let fullMask = attentionMask.asType(.bool)[0..., .newAxis, .newAxis, 0...]
        let positions = MLXArray(Int32(0) ..< Int32(L))
        let distance = MLX.abs(positions[0..., .newAxis] - positions[.newAxis, 0...])
        let slidingMask = fullMask .&& (distance .<= Int32(config.slidingWindow))
        let positionIds = positionIds ?? positions[.newAxis]

        let perLayerInputs = ple?(inputsEmbeds)
        var h = inputsEmbeds
        for (i, layer) in layers.enumerated() {
            let mask = config.layerTypes[i] == "sliding_attention" ? slidingMask : fullMask
            h = layer(h, mask: mask, positions: positionIds, perLayerInput: perLayerInputs?[0..., 0..., i])
        }
        return embeddingProjection(norm(h))
    }
}
