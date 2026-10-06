// Copyright © 2026
//
// Frame sampling of transformers `EmbeddingGemma2VideoProcessor`: resample to `fps` (default 1)
// using the source rate, then cap to `maxFrames` (uniform or truncate). Each kept frame goes
// through ``ImageProcessor`` at the video soft-token budget (default 140).

import Foundation

/// Decoded video frames plus the metadata needed for rate-based sampling.
public struct VideoClip: Sendable {
    /// Every decoded frame at the source rate (or an already-sampled subset when `fps` is `nil`).
    public var frames: [RGBImage]
    /// Source frame rate; `nil` disables rate-based sampling (only the `maxFrames` cap applies).
    public var fps: Double?
    /// Clip duration in seconds; defaults to `frames.count / fps`.
    public var duration: Double?

    public init(frames: [RGBImage], fps: Double?, duration: Double? = nil) {
        self.frames = frames
        self.fps = fps
        self.duration = duration ?? fps.map { Double(frames.count) / $0 }
    }
}

public struct VideoProcessor: Sendable {
    public enum OverflowStrategy: String, Sendable { case uniform, truncate }

    /// Target sampling rate in frames per second; `nil` keeps every frame.
    public var fps: Double? = 1
    public var maxFrames: Int? = 32
    public var overflowStrategy: OverflowStrategy? = .uniform
    public var image = ImageProcessor(maxSoftTokens: 140)

    public init() {}

    /// Indices of the frames to keep, as `EmbeddingGemma2VideoProcessor.sample_frames`.
    public func sampleIndices(frameCount: Int, sourceFPS: Double?, duration: Double?) -> [Int] {
        var indices: [Int]
        if let fps, let sourceFPS, let duration {
            let step = sourceFPS / fps
            let count = max(1, Int(duration * fps))
            indices = (0 ..< count).map { min(frameCount - 1, Int(Double($0) * step)) }
        } else {
            indices = Array(0 ..< frameCount)
        }
        if let overflowStrategy, let maxFrames, indices.count > maxFrames {
            switch overflowStrategy {
            case .truncate:
                indices = Array(indices.prefix(maxFrames))
            case .uniform:
                // np.linspace(0, n - 1, maxFrames, dtype=int): `i * step`, last point exact, truncated.
                let last = Double(indices.count - 1)
                let step = maxFrames > 1 ? last / Double(maxFrames - 1) : 0
                indices = (0 ..< maxFrames).map {
                    indices[$0 == maxFrames - 1 && maxFrames > 1 ? indices.count - 1 : Int(Double($0) * step)]
                }
            }
        }
        return indices
    }

    /// Patchified kept frames of `clip`.
    public func callAsFunction(_ clip: VideoClip) throws -> [PatchifiedImage] {
        guard !clip.frames.isEmpty else { throw EmbeddingGemma2Error.invalidInput("video has no frames") }
        return try sampleIndices(frameCount: clip.frames.count, sourceFPS: clip.fps, duration: clip.duration)
            .map { try image(clip.frames[$0]) }
    }
}
