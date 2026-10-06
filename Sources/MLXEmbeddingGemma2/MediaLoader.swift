// Copyright © 2026
//
// File decoding for frontends. These read from disk, so sandboxed callers must hold access to the
// URL (e.g. `startAccessingSecurityScopedResource()`) for the duration of the call; the returned
// values are fully materialized and safe to hand to ``EmbeddingSession`` later.

@preconcurrency import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum MediaLoader {
    public enum Kind: Sendable { case image, audio, video }

    public enum LoadError: LocalizedError {
        case unreadable(URL)
        case noTrack(URL, AVMediaType)

        public var errorDescription: String? {
            switch self {
            case .unreadable(let url): "Could not decode \(url.lastPathComponent)."
            case .noTrack(let url, let type): "\(url.lastPathComponent) has no \(type.rawValue) track."
            }
        }
    }

    /// Media kind from the file's type, or `nil` for anything else.
    public static func kind(of url: URL) -> Kind? {
        guard let type = UTType(filenameExtension: url.pathExtension) else { return nil }
        if type.conforms(to: .image) { return .image }
        if type.conforms(to: .audio) { return .audio }
        if type.conforms(to: .movie) || type.conforms(to: .video) { return .video }
        return nil
    }

    /// One content part for `url`, decoded by its kind.
    public static func content(_ url: URL, video: VideoProcessor = VideoProcessor()) async throws -> EmbeddingContent {
        switch kind(of: url) {
        case .image: return .image(try image(url))
        case .audio: return .audio(try await audio(url))
        case .video: return .video(try await self.video(url, processor: video))
        case nil: throw LoadError.unreadable(url)
        }
    }

    public static func image(_ url: URL) throws -> RGBImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw LoadError.unreadable(url) }
        return RGBImage(cg)
    }

    /// Decodes and resamples to 16 kHz mono float samples.
    public static func audio(_ url: URL, sampleRate: Double = 16_000) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw LoadError.noTrack(url, .audio)
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false,
        ])
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? LoadError.unreadable(url) }
        var samples: [Float] = []
        while let buffer = output.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(buffer) {
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / MemoryLayout<Float>.size)
            chunk.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            samples += chunk
        }
        if reader.status == .failed { throw reader.error ?? LoadError.unreadable(url) }
        return samples
    }

    /// Decodes only the frames `processor` would keep (rate sampling + frame cap), so long videos
    /// never sit in memory at full frame rate. The returned clip carries no `fps`, so the processor
    /// keeps all of its frames.
    public static func video(_ url: URL, processor: VideoProcessor = VideoProcessor()) async throws -> VideoClip {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw LoadError.noTrack(url, .video)
        }
        let duration = try await asset.load(.duration).seconds
        let fps = Double(try await track.load(.nominalFrameRate))
        let frameCount = max(1, Int((duration * fps).rounded()))
        let indices = processor.sampleIndices(frameCount: frameCount, sourceFPS: fps, duration: duration)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [RGBImage] = []
        for index in indices {
            let time = CMTime(seconds: Double(index) / fps, preferredTimescale: 600)
            let (cg, _) = try await generator.image(at: time)
            frames.append(RGBImage(cg))
        }
        return VideoClip(frames: frames, fps: nil)
    }
}
