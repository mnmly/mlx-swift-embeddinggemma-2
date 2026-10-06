// Copyright © 2026
//
// Port of mlx-vlm `gemma4/audio.py`: the Gemma 4 Conformer audio encoder (Universal Speech Model).
// SSCP (2× stride-2 Conv2d) → 12 macaron Conformer blocks with chunked local attention, relative
// position logits and softcapping → output projection.
//
// Dtypes intentionally follow the reference op-for-op: the float32 log-mel input and the float32
// attention promote activations to float32 against the bf16 weights.

import Foundation
import MLX
import MLXFast
import MLXNN

final class SSCPConvBlock: Module {
    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: LayerNorm

    init(_ config: AudioConfiguration, index: Int) {
        let inChannels = index == 0 ? 1 : config.subsamplingConvChannels[index - 1]
        let outChannels = config.subsamplingConvChannels[index]
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 2, padding: 0,
            bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: outChannels, eps: config.rmsNormEps, bias: false)
    }

    /// - Parameters:
    ///   - x: `[B, T, F, C]`; `invalid` is `[B, T]`, true for padding frames.
    func callAsFunction(_ x: MLXArray, invalid: MLXArray) -> (MLXArray, MLXArray) {
        var x = MLX.where(invalid[0..., 0..., .newAxis, .newAxis], MLXArray(0, dtype: x.dtype), x)
        x = padded(x, widths: [0, 1, 1, 0])
        x = conv(x)
        let invalidOut = invalid[0..., .stride(by: 2)][0..., ..<x.dim(1)]
        return (relu(norm(x)), invalidOut)
    }
}

final class SubSampleConvProjection: Module {
    @ModuleInfo var layer0: SSCPConvBlock
    @ModuleInfo var layer1: SSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(_ config: AudioConfiguration, inputFeatures: Int = 128) {
        _layer0.wrappedValue = SSCPConvBlock(config, index: 0)
        _layer1.wrappedValue = SSCPConvBlock(config, index: 1)
        var freq = inputFeatures
        for _ in 0 ..< 2 { freq = (freq + 2 - 3) / 2 + 1 }
        _inputProjLinear.wrappedValue = Linear(
            freq * config.subsamplingConvChannels.last!, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ mel: MLXArray, invalid: MLXArray) -> (MLXArray, MLXArray) {
        var (x, mask) = layer0(mel.expandedDimensions(axis: -1), invalid: invalid)
        (x, mask) = layer1(x, invalid: mask)
        x = x.reshaped(x.dim(0), x.dim(1), -1)
        return (inputProjLinear(x), mask)
    }
}

final class ConformerFeedForward: Module, UnaryLayer {
    let clipping: Float
    let residualWeight: Float

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffw1: ClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var ffw2: ClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: RMSNorm

    init(_ config: AudioConfiguration) {
        clipping = config.gradientClipping
        residualWeight = config.residualWeight
        let h = config.hiddenSize
        _preLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: 1e-6)
        _ffw1.wrappedValue = ClippableLinear(h, h * 4, clipping: config.useClippedLinears)
        _ffw2.wrappedValue = ClippableLinear(h * 4, h, clipping: config.useClippedLinears)
        _postLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = preLayerNorm(MLX.clip(x, min: -clipping, max: clipping))
        h = ffw2(silu(ffw1(h)))
        h = postLayerNorm(MLX.clip(h, min: -clipping, max: clipping))
        return x + h * residualWeight
    }
}

final class AudioAttention: Module {
    let heads: Int
    let headDim: Int
    let chunkSize: Int
    let maxPast: Int
    let maxFuture: Int
    let contextSize: Int
    let invalidLogit: Float
    let softcap: Float
    let qScale: Float
    let kScale: Float
    /// `[1, 1, hidden / 2]` inverse timescales of the sinusoidal relative positions.
    let _invTimescales: MLXArray

    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo var post: ClippableLinear

    init(_ config: AudioConfiguration) {
        let hidden = config.hiddenSize
        heads = config.numAttentionHeads
        headDim = hidden / heads
        chunkSize = config.attentionChunkSize
        maxPast = max(0, config.attentionContextLeft - 1)
        maxFuture = config.attentionContextRight
        contextSize = chunkSize + maxPast + maxFuture
        invalidLogit = config.attentionInvalidLogitsValue
        softcap = config.attentionLogitCap
        qScale = Float(pow(Double(headDim), -0.5) / log(2.0))
        kScale = Float(log(1 + M_E) / log(2.0))

        let count = hidden / 2
        let increment = log(10_000.0) / Double(max(count - 1, 1))
        _invTimescales = MLX.exp(MLXArray(0 ..< count).asType(.float32) * Float(-increment))
            .reshaped(1, 1, count)

        let clip = config.useClippedLinears
        _relativeKProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        _perDimScale.wrappedValue = MLXArray.zeros([headDim])
        _qProj.wrappedValue = ClippableLinear(hidden, heads * headDim, clipping: clip)
        _kProj.wrappedValue = ClippableLinear(hidden, heads * headDim, clipping: clip)
        _vProj.wrappedValue = ClippableLinear(hidden, heads * headDim, clipping: clip)
        _post.wrappedValue = ClippableLinear(hidden, hidden, clipping: clip)
    }

    private func padTime(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        var widths = [IntOrPair](repeating: 0, count: x.ndim)
        widths[1] = .init((left, right))
        return padded(x, widths: widths)
    }

    /// `[B, T, ...]` → `[B, blocks, chunk, ...]`.
    private func toBlocks(_ x: MLXArray) -> MLXArray {
        let T = x.dim(1)
        let blocks = (T + chunkSize - 1) / chunkSize
        let x = padTime(x, left: 0, right: blocks * chunkSize - T)
        return x.reshaped([x.dim(0), blocks, chunkSize] + Array(x.shape.dropFirst(2)))
    }

    /// `[B, T, ...]` → `[B, blocks, context, ...]` overlapping key/value windows.
    private func blockContext(_ x: MLXArray) -> MLXArray {
        let x = padTime(x, left: maxPast, right: maxFuture + chunkSize - 1)
        let blocks = (x.dim(1) - contextSize) / chunkSize + 1
        let indices = (MLXArray(Int32(0) ..< Int32(blocks)) * Int32(chunkSize))[0..., .newAxis]
            + MLXArray(Int32(0) ..< Int32(contextSize))[.newAxis, 0...]
        return x[0..., indices]
    }

    /// Content (`ac`) + relative-position (`bd`) logits: `[B, N, U, W, C]`.
    private func relativeLogits(queries: MLXArray, keys: MLXArray) -> MLXArray {
        let (B, U, W, N, H) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3), queries.dim(4))
        let C = keys.dim(2)

        let positions = MLXArray(Array(stride(from: maxPast, through: -maxFuture, by: -1)).map { Int32($0) })
        let span = positions.dim(0)
        let scaled = positions.asType(.float32).reshaped(1, span, 1) * _invTimescales
        var sinEmb = concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1).asType(queries.dtype)
        sinEmb = relativeKProj(sinEmb.asType(relativeKProj.weight.dtype))
            .reshaped(span, heads, headDim).asType(queries.dtype)

        let q = queries.transposed(0, 3, 1, 2, 4)  // [B, N, U, W, H]
        let termAC = matmul(q, keys.transposed(0, 3, 1, 4, 2))  // [B, N, U, W, C]
        var termBD = matmul(q.reshaped(B, N, U * W, H), sinEmb.transposed(1, 2, 0))
            .reshaped(B, N, U, W, span)

        // Relative shift: pad to C + 1, flatten, drop the tail, refold.
        termBD = padded(termBD, widths: [0, 0, 0, 0, .init((0, C + 1 - span))])
        termBD = termBD.reshaped(B, N, U, W * (C + 1))[.ellipsis, ..<(W * C)].reshaped(B, N, U, W, C)
        return termAC + termBD
    }

    func callAsFunction(_ x: MLXArray, invalid: MLXArray, localMask: MLXArray) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        var q = qProj(x).asType(.float32).reshaped(B, T, heads, headDim)
        var k = kProj(x).asType(.float32).reshaped(B, T, heads, headDim)
        let v = vProj(x).asType(.float32).reshaped(B, T, heads, headDim)
        q = q * (softplus(perDimScale) * qScale)
        k = k * kScale

        let queryBlocks = toBlocks(q)
        let keyBlocks = blockContext(k)
        let valueBlocks = blockContext(v)
        let U = queryBlocks.dim(1)

        let validContext = blockContext(.!invalid)  // [B, U, C]
        let condition = validContext[0..., .newAxis, 0..., .newAxis, 0...]
            .&& localMask[.newAxis, .newAxis, .newAxis, 0..., 0...]

        var logits = relativeLogits(queries: queryBlocks, keys: keyBlocks)
        logits = MLX.tanh(logits / softcap) * softcap
        logits = MLX.where(condition, logits, MLXArray(invalidLogit, dtype: logits.dtype))
        let probs = softmax(logits, axis: -1)

        // einsum("bnuwc,bucnh->buwnh")
        let context = matmul(probs, valueBlocks.transposed(0, 3, 1, 2, 4))  // [B, N, U, W, H]
            .transposed(0, 2, 3, 1, 4)
            .reshaped(B, U * chunkSize, heads, headDim)[0..., ..<T]
        return post(context.reshaped(B, T, heads * headDim))
    }
}

final class ConformerLightConv1d: Module, UnaryLayer {
    let clipping: Float
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: RMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: ClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: RMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: ClippableLinear

    init(_ config: AudioConfiguration) {
        clipping = config.gradientClipping
        causalPadding = config.convKernelSize - 1
        let h = config.hiddenSize
        _preLayerNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _linearStart.wrappedValue = ClippableLinear(h, h * 2, clipping: config.useClippedLinears)
        _depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: 1, outputChannels: h, kernelSize: config.convKernelSize, bias: false)
        _convNorm.wrappedValue = RMSNorm(dimensions: h, eps: config.rmsNormEps)
        _linearEnd.wrappedValue = ClippableLinear(h, h, clipping: config.useClippedLinears)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linearStart(preLayerNorm(x))
        let halves = h.split(parts: 2, axis: -1)
        h = halves[0] * sigmoid(halves[1])
        h = padded(h, widths: [0, .init((causalPadding, 0)), 0])
        h = convGeneral(h, depthwiseConv1d.weight, strides: 1, padding: 0, groups: h.dim(-1))
        h = silu(convNorm(MLX.clip(h, min: -clipping, max: clipping)))
        return linearEnd(h) + x
    }
}

final class ConformerBlock: Module {
    let clipping: Float

    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: AudioAttention
    @ModuleInfo var lconv1d: ConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: RMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: RMSNorm
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm

    init(_ config: AudioConfiguration) {
        clipping = config.gradientClipping
        let h = config.hiddenSize
        _feedForward1.wrappedValue = ConformerFeedForward(config)
        _selfAttn.wrappedValue = AudioAttention(config)
        _lconv1d.wrappedValue = ConformerLightConv1d(config)
        _feedForward2.wrappedValue = ConformerFeedForward(config)
        _normPreAttn.wrappedValue = RMSNorm(dimensions: h, eps: 1e-6)
        _normPostAttn.wrappedValue = RMSNorm(dimensions: h, eps: 1e-6)
        _normOut.wrappedValue = RMSNorm(dimensions: h, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray, invalid: MLXArray, localMask: MLXArray) -> MLXArray {
        var x = feedForward1(x)
        let attn = selfAttn(
            normPreAttn(MLX.clip(x, min: -clipping, max: clipping)), invalid: invalid, localMask: localMask)
        x = x + normPostAttn(MLX.clip(attn, min: -clipping, max: clipping))
        x = x * (.!invalid)[0..., 0..., .newAxis].asType(x.dtype)
        x = feedForward2(lconv1d(x))
        return normOut(MLX.clip(x, min: -clipping, max: clipping))
    }
}

/// The Gemma 4 audio encoder (`audio_tower`).
final class AudioTower: Module {
    let config: AudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") var subsampleConvProjection: SubSampleConvProjection
    @ModuleInfo var layers: [ConformerBlock]
    @ModuleInfo(key: "output_proj") var outputProj: Linear?

    init(_ config: AudioConfiguration) {
        self.config = config
        _subsampleConvProjection.wrappedValue = SubSampleConvProjection(config)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in ConformerBlock(config) }
        _outputProj.wrappedValue = config.outputProjDims.map { Linear(config.hiddenSize, $0, bias: true) }
    }

    /// `[chunk, context]` mask of keys a query may see: strictly within the past horizon
    /// (or future horizon, which is 0 here).
    func localMask() -> MLXArray {
        let past = max(0, config.attentionContextLeft - 1)
        let future = config.attentionContextRight
        let context = config.attentionChunkSize + past + future
        let dist = MLXArray(Int32(0) ..< Int32(config.attentionChunkSize))[0..., .newAxis] + Int32(past)
            - MLXArray(Int32(0) ..< Int32(context))[.newAxis, 0...]
        let pastValid = (dist .>= 0) .&& (dist .< Int32(past))
        let futureValid = (dist .< 0) .&& (-dist .< Int32(future))
        return pastValid .|| futureValid
    }

    /// - Parameters:
    ///   - mel: `[B, T, 128]` log-mel features; `invalid` is `[B, T]`, true for padding frames.
    /// - Returns: `[B, T/4, outputProjDims]` encodings (padding zeroed) and their `[B, T/4]` invalid mask.
    func callAsFunction(_ mel: MLXArray, invalid: MLXArray) -> (MLXArray, MLXArray) {
        var (x, mask) = subsampleConvProjection(mel, invalid: invalid)
        let local = localMask()
        for layer in layers {
            x = layer(x, invalid: mask, localMask: local)
        }
        if let outputProj { x = outputProj(x) }
        mask = mask[0..., ..<x.dim(1)]
        x = MLX.where(mask[0..., 0..., .newAxis], MLXArray(0, dtype: x.dtype), x)
        return (x, mask)
    }
}
