// Copyright © 2026
//
// Port of mlx-vlm `gemma4/vision.py` (the processor-supplied patches + position ids path), with
// structure borrowed from mlx-swift-lm `MLXVLM/Models/Gemma4.swift`. Images arrive already
// patchified (`[N, maxPatches, 3·p²]`) with `(x, y)` patch positions padded with -1.

import Foundation
import MLX
import MLXFast
import MLXNN

/// `nn.Linear` with optional input/output clamping; bounds are scalar checkpoint buffers.
final class ClippableLinear: Module, UnaryLayer {
    @ModuleInfo var linear: Linear
    @ParameterInfo(key: "input_min") var inputMin: MLXArray?
    @ParameterInfo(key: "input_max") var inputMax: MLXArray?
    @ParameterInfo(key: "output_min") var outputMin: MLXArray?
    @ParameterInfo(key: "output_max") var outputMax: MLXArray?

    init(_ inputs: Int, _ outputs: Int, bias: Bool = false, clipping: Bool) {
        _linear.wrappedValue = Linear(inputs, outputs, bias: bias)
        if clipping {
            _inputMin.wrappedValue = MLXArray(-Float.infinity)
            _inputMax.wrappedValue = MLXArray(Float.infinity)
            _outputMin.wrappedValue = MLXArray(-Float.infinity)
            _outputMax.wrappedValue = MLXArray(Float.infinity)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let inputMin, let inputMax { x = MLX.clip(x, min: inputMin, max: inputMax) }
        x = linear(x)
        if let outputMin, let outputMax { x = MLX.clip(x, min: outputMin, max: outputMax) }
        return x
    }
}

/// Gemma4 vision RMSNorm: computed entirely in float32, then cast back.
final class VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        visionRMSNormalize(x, eps: eps, weight: weight)
    }
}

func visionRMSNormalize(_ x: MLXArray, eps: Float, weight: MLXArray? = nil) -> MLXArray {
    let xf = x.asType(.float32)
    var normed = xf * MLX.rsqrt(MLX.mean(xf * xf, axis: -1, keepDims: true) + eps)
    if let weight { normed = normed * weight.asType(.float32) }
    return normed.asType(x.dtype)
}

/// 2-D RoPE: the head is split into one half per spatial axis, each rotated independently.
/// - Parameters:
///   - x: `[B, L, N, H]`.
///   - positions: `[B, L, 2]` integer `(x, y)` positions.
func applyMultidimensionalRoPE(_ x: MLXArray, positions: MLXArray, base: Float) -> MLXArray {
    let headDim = x.dim(-1)
    let ndim = positions.dim(-1)
    let channels = 2 * (headDim / (2 * ndim))
    let half = channels / 2
    let exponents = MLXArray((0 ..< half).map { 2 * Float($0) / Float(channels) })
    let timescale = MLX.pow(MLXArray(base), exponents)

    var parts: [MLXArray] = []
    for d in 0 ..< ndim {
        let part = x[.ellipsis, (d * channels) ..< ((d + 1) * channels)]
        let sinusoid = positions[.ellipsis, d ..< (d + 1)].asType(.float32) / timescale
        let cos = concatenated([MLX.cos(sinusoid), MLX.cos(sinusoid)], axis: -1).asType(x.dtype)
            .expandedDimensions(axis: 2)
        let sin = concatenated([MLX.sin(sinusoid), MLX.sin(sinusoid)], axis: -1).asType(x.dtype)
            .expandedDimensions(axis: 2)
        let rotated = concatenated([-part[.ellipsis, half...], part[.ellipsis, ..<half]], axis: -1)
        parts.append(part * cos + rotated * sin)
    }
    return concatenated(parts, axis: -1)
}

final class VisionAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeBase: Float

    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm

    init(_ config: VisionConfiguration) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        ropeBase = config.ropeTheta
        let h = config.hiddenSize
        let clip = config.useClippedLinears
        _qProj.wrappedValue = ClippableLinear(h, heads * headDim, clipping: clip)
        _kProj.wrappedValue = ClippableLinear(h, kvHeads * headDim, clipping: clip)
        _vProj.wrappedValue = ClippableLinear(h, kvHeads * headDim, clipping: clip)
        _oProj.wrappedValue = ClippableLinear(heads * headDim, h, clipping: clip)
        _qNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        _kNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(B, L, heads, headDim))
        var k = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim))
        let v = visionRMSNormalize(vProj(x).reshaped(B, L, kvHeads, headDim), eps: 1e-6)
        q = applyMultidimensionalRoPE(q, positions: positions, base: ropeBase)
        k = applyMultidimensionalRoPE(k, positions: positions, base: ropeBase)
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
            values: v.transposed(0, 2, 1, 3), scale: 1.0, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: ClippableLinear

    init(_ config: VisionConfiguration) {
        let clip = config.useClippedLinears
        _gateProj.wrappedValue = ClippableLinear(config.hiddenSize, config.intermediateSize, clipping: clip)
        _upProj.wrappedValue = ClippableLinear(config.hiddenSize, config.intermediateSize, clipping: clip)
        _downProj.wrappedValue = ClippableLinear(config.intermediateSize, config.hiddenSize, clipping: clip)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
    @ModuleInfo var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    init(_ config: VisionConfiguration) {
        let h = config.hiddenSize
        _selfAttn.wrappedValue = VisionAttention(config)
        _mlp.wrappedValue = VisionMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + postAttentionLayerNorm(selfAttn(inputLayerNorm(x), positions: positions, mask: mask))
        return h + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
    }
}

final class VisionPatchEmbedder: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ParameterInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(_ config: VisionConfiguration) {
        _inputProj.wrappedValue = Linear(3 * config.patchSize * config.patchSize, config.hiddenSize, bias: false)
        _positionEmbeddingTable.wrappedValue = MLXArray.ones([2, config.positionEmbeddingSize, config.hiddenSize])
    }

    /// - Parameters:
    ///   - patches: `[B, P, 3·p²]` in `[0, 1]`.
    ///   - positions: `[B, P, 2]`; `padding` is `[B, P]`, true for padded patches.
    func callAsFunction(_ patches: MLXArray, positions: MLXArray, padding: MLXArray) -> MLXArray {
        let hidden = inputProj((2 * (patches - 0.5)).asType(inputProj.weight.dtype))
        // Equivalent to the reference's one-hot @ table: each axis contributes exactly one row.
        let clamped = MLX.maximum(positions, 0)
        let px = positionEmbeddingTable[0].take(clamped[.ellipsis, 0], axis: 0)
        let py = positionEmbeddingTable[1].take(clamped[.ellipsis, 1], axis: 0)
        let positional = MLX.where(padding.expandedDimensions(axis: -1), MLXArray(0, dtype: px.dtype), px + py)
        return hidden + positional
    }
}

final class VisionEncoder: Module {
    @ModuleInfo var layers: [VisionBlock]

    init(_ config: VisionConfiguration) {
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in VisionBlock(config) }
    }
}

/// The Gemma 4 vision tower (`vision_tower`).
final class VisionTower: Module {
    let config: VisionConfiguration

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: VisionPatchEmbedder
    @ModuleInfo var encoder: VisionEncoder
    @ParameterInfo(key: "std_bias") var stdBias: MLXArray?
    @ParameterInfo(key: "std_scale") var stdScale: MLXArray?

    init(_ config: VisionConfiguration) {
        self.config = config
        _patchEmbedder.wrappedValue = VisionPatchEmbedder(config)
        _encoder.wrappedValue = VisionEncoder(config)
        if config.standardize {
            _stdBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            _stdScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
    }

    /// Encodes and pools a batch of patchified images (or video frames).
    ///
    /// - Parameters:
    ///   - patches: `[N, P, 3·p²]`.
    ///   - positions: `[N, P, 2]`, -1 for padding.
    /// - Returns: `[N, P / k², hidden]` pooled soft tokens (valid ones first, the rest zero) and
    ///   `[N, P / k²]` validity.
    func callAsFunction(_ patches: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
        let k2 = config.poolingKernelSize * config.poolingKernelSize
        let outputLength = patches.dim(1) / k2
        let padding = MLX.all(positions .== -1, axis: -1)

        var h = patchEmbedder(patches, positions: positions, padding: padding)

        // Bidirectional additive mask over valid patches; -1e4 (not -inf) like the reference.
        let valid = .!padding
        let pairs = valid.expandedDimensions(axis: 1) .&& valid.expandedDimensions(axis: 2)
        let mask = MLX.where(pairs, MLXArray(0, dtype: h.dtype), MLXArray(-1e4, dtype: h.dtype))
            .expandedDimensions(axis: 1)
        for layer in encoder.layers {
            h = layer(h, positions: positions, mask: mask)
        }

        // Pool k×k patch neighbourhoods by their (x, y) positions.
        h = MLX.where(padding.expandedDimensions(axis: -1), MLXArray(0, dtype: h.dtype), h)
        let k = Int(Double(h.dim(1) / outputLength).squareRoot())
        let clamped = MLX.maximum(positions, 0)
        let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
        let kernel = MLX.floorDivide(clamped, Int32(k))
        let index = kernel[.ellipsis, 0] + MLX.floorDivide(maxX, Int32(k)) * kernel[.ellipsis, 1]
        let weights = (index.expandedDimensions(axis: -1) .== MLXArray(Int32(0) ..< Int32(outputLength)))
            .asType(.float32) / Float(k * k)
        var pooled = matmul(weights.transposed(0, 2, 1), h.asType(.float32)).asType(h.dtype)
        let pooledValid = MLX.any(weights .!= 0, axis: 1)
        pooled = pooled * MLXArray(Float(config.hiddenSize).squareRoot(), dtype: pooled.dtype)
        if let stdBias, let stdScale {
            pooled = (pooled - stdBias) * stdScale
        }
        return (pooled, pooledValid)
    }
}

/// Projects tower outputs into the text hidden space (`embed_vision` / `embed_audio`).
final class MultimodalEmbedder: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self.eps = eps
        _embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(rmsNormNoScale(x, eps: eps))
    }
}
