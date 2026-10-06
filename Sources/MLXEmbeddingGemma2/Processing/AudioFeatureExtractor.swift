// Copyright © 2026
//
// Port of transformers `Gemma4AudioFeatureExtractor` (USM log-mel): 16 kHz mono, 20 ms periodic
// Hann frames every 10 ms with semicausal left padding, |rFFT(512)| → 128 HTK mel bins
// (0–8 kHz, unnormalized) → log(mel + 1e-3). Frames that end in padding are masked.

import Foundation
import MLX

public struct AudioFeatureExtractor: Sendable {
    public var samplingRate = 16_000
    public var featureSize = 128
    public var frameLength = 320
    public var hopLength = 160
    public var fftLength = 512
    public var minFrequency: Double = 0
    public var maxFrequency: Double = 8000
    public var melFloor: Float = 1e-3
    /// Longest clip kept, in samples (30 s); longer audio is truncated.
    public var maxLength = 480_000
    /// Batch padding granularity, in samples.
    public var padToMultipleOf = 128

    public init() {}

    /// `[fftLength / 2 + 1, featureSize]` triangular HTK mel filters, as `mel_filter_bank(norm=None, mel_scale="htk")`.
    func melFilters() -> [Float] {
        func hzToMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let bins = fftLength / 2 + 1
        let fftFreqs = (0 ..< bins).map { Double($0) * (Double(samplingRate) / 2) / Double(bins - 1) }
        let (lo, hi) = (hzToMel(minFrequency), hzToMel(maxFrequency))
        let filterFreqs = (0 ..< featureSize + 2).map {
            melToHz(lo + Double($0) * (hi - lo) / Double(featureSize + 1))
        }
        var out = [Float](repeating: 0, count: bins * featureSize)
        for b in 0 ..< bins {
            for m in 0 ..< featureSize {
                let down = -(filterFreqs[m] - fftFreqs[b]) / (filterFreqs[m + 1] - filterFreqs[m])
                let up = (filterFreqs[m + 2] - fftFreqs[b]) / (filterFreqs[m + 2] - filterFreqs[m + 1])
                out[b * featureSize + m] = Float(max(0, min(down, up)))
            }
        }
        return out
    }

    /// Log-mel features for a batch of 16 kHz mono clips, right-padded to the longest.
    ///
    /// - Returns: `features` `[B, T, 128]` (zeroed where invalid) and `mask` `[B, T]` (true = real frame).
    public func callAsFunction(_ clips: [[Float]]) -> (features: MLXArray, mask: MLXArray) {
        let clips = clips.map { Array($0.prefix(maxLength)) }
        var length = clips.map(\.count).max() ?? 0
        if length % padToMultipleOf != 0 { length += padToMultipleOf - length % padToMultipleOf }

        let padLeft = frameLength / 2
        let unfold = frameLength + 1
        let frames = max(0, (length + padLeft - unfold) / hopLength + 1)
        let window = (0 ..< frameLength).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(frameLength)) }

        var framed = [Float](repeating: 0, count: clips.count * frames * frameLength)
        var mask = [Bool](repeating: false, count: clips.count * frames)
        for (b, clip) in clips.enumerated() {
            for t in 0 ..< frames {
                let start = t * hopLength - padLeft  // in unpadded sample coordinates
                let base = (b * frames + t) * frameLength
                for i in 0 ..< frameLength {
                    let s = start + i
                    if s >= 0, s < clip.count { framed[base + i] = clip[s] * window[i] }
                }
                // A frame is real when its last unfold sample (index unfold - 1) is real audio.
                mask[b * frames + t] = start + unfold - 1 < clip.count
            }
        }

        let x = MLXArray(framed, [clips.count, frames, frameLength])
        let magnitude = MLX.abs(MLXFFT.rfft(x, n: fftLength, axis: -1))
        let filters = MLXArray(melFilters(), [fftLength / 2 + 1, featureSize])
        let logMel = MLX.log(matmul(magnitude, filters) + melFloor)
        let maskArray = MLXArray(mask, [clips.count, frames])
        return (logMel * maskArray.expandedDimensions(axis: -1).asType(.float32), maskArray)
    }

    /// Audio soft tokens a clip of `samples` (≤ ``maxLength``) contributes: its real mel frames
    /// (those ending inside the clip) through the two stride-2 SSCP convolutions, i.e. `⌈frames / 4⌉`.
    /// Matches `replace_audio_token` on the batch mask, independent of batch padding.
    public func softTokens(samples: Int) -> Int {
        // Frame t is real when t·hop − frameLength/2 + frameLength < samples.
        let reach = samples - (frameLength - frameLength / 2) - 1
        let frames = reach < 0 ? 0 : reach / hopLength + 1
        return (frames + 3) / 4
    }
}
