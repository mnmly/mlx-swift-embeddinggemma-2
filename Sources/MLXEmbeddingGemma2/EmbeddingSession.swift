// Copyright © 2026

import Foundation
import MLX

/// All knobs for an embedding session. The defaults are what both the CLI and the app use.
public struct EmbeddingSessionConfiguration: Sendable {
    /// Hugging Face repo, resolved through the hub cache (downloaded if missing).
    public var repo: String = EmbeddingGemma2Model.defaultRepo
    /// Explicit checkpoint directory; overrides ``repo``.
    public var directory: URL?
    /// Modality towers to load; text-only sessions skip ~0.7 GB of tower weights.
    public var towers: EmbeddingGemma2Towers = .all
    /// Weight dtype override (`nil` keeps bfloat16; never `.float16`).
    public var dtype: DType?
    /// Soft-token budget per image (70, 140, 280, 560 or 1120).
    public var imageSoftTokens = 280
    /// Soft-token budget per video frame.
    public var videoSoftTokens = 140
    /// Video sampling rate (frames per second) and frame cap.
    public var videoFPS: Double = 1
    public var videoMaxFrames = 32

    // Batching. Batches are packed by cost, not count: within a lookahead window inputs are sorted by
    // modality and length so similar inputs share a batch (little padding), then packed greedily
    // until any budget below would be exceeded. An input over budget on its own runs alone.

    /// Most inputs per forward pass.
    public var batchSize = 16
    /// Most padded text-encoder positions per forward pass (`inputs × longest sequence`).
    public var maxBatchTokens = 8192
    /// Most images + video frames per forward pass through the vision tower.
    public var maxBatchVisionFrames = 8
    /// Most audio samples per forward pass through the audio tower (16 kHz; default 2 minutes).
    public var maxBatchAudioSamples = 16_000 * 120
    /// Inputs loaded and preprocessed ahead of the GPU (also the window sorted for packing).
    public var lookahead = 256
    /// Stop filling a lookahead window early once its preprocessed media exceeds this many bytes.
    public var lookaheadBytes = 512 << 20
    /// Concurrent loads + preprocessing (`nil` = active processor count).
    public var parallelism: Int?
    /// Bound on MLX's reusable buffer cache in bytes (`nil` = MLX default, which grows unbounded
    /// across varying batch shapes).
    public var cacheLimit: Int? = 512 << 20

    public init() {}
}

/// Loads EmbeddingGemma 2 once and embeds batches of mixed-modality inputs.
///
/// The shared driver behind the `embeddinggemma2` CLI and the macOS app: it owns the model,
/// processor and all hyperparameters, and takes *materialized* media (``RGBImage``, PCM samples,
/// decoded frames), never paths. Frontends decode files (see ``MediaLoader``) after establishing
/// any sandbox access they need.
///
/// `@unchecked Sendable`: the contract is one driver at a time (the CLI's main thread or the app's
/// single detached task). Do not call `embed` concurrently from two tasks.
public final class EmbeddingSession: @unchecked Sendable {
    public let configuration: EmbeddingSessionConfiguration
    public let model: EmbeddingGemma2Model
    public private(set) var processor: EmbeddingGemma2Processor

    /// Full embedding width (768).
    public var dimension: Int { model.configuration.textConfig.embeddingDim }
    /// Matryoshka widths the model was trained for.
    public static let matryoshkaDimensions = [128, 256, 512, 768]

    private init(configuration: EmbeddingSessionConfiguration, model: EmbeddingGemma2Model,
                 processor: EmbeddingGemma2Processor)
    {
        self.configuration = configuration
        self.model = model
        self.processor = processor
    }

    public static func load(_ configuration: EmbeddingSessionConfiguration = .init()) async throws -> EmbeddingSession {
        if let limit = configuration.cacheLimit { Memory.cacheLimit = limit }
        let directory: URL
        if let explicit = configuration.directory {
            directory = explicit
        } else {
            directory = try await resolveModelDirectory(repo: configuration.repo)
        }
        let model = try EmbeddingGemma2Model.load(
            directory: directory, towers: configuration.towers, dtype: configuration.dtype)
        var processor = try await EmbeddingGemma2Processor.load(directory: directory, configuration: model.configuration)
        processor.image.maxSoftTokens = configuration.imageSoftTokens
        processor.video.image.maxSoftTokens = configuration.videoSoftTokens
        processor.video.fps = configuration.videoFPS
        processor.video.maxFrames = configuration.videoMaxFrames
        return EmbeddingSession(configuration: configuration, model: model, processor: processor)
    }

    /// Loads one input on demand; called concurrently, roughly one lookahead window ahead of the GPU.
    public typealias InputLoader = @Sendable (_ index: Int) async throws -> EmbeddingInput

    /// Embeds `count` inputs produced lazily by `load`, streaming results as batches finish.
    ///
    /// Built for large jobs (thousands of items): only a lookahead window is ever decoded in memory,
    /// batches are packed by token / frame / sample budgets, CPU preprocessing of the next batch
    /// overlaps GPU work on the current one, and a failing input (unreadable file, bad media) is
    /// reported for that index without stopping the run.
    ///
    /// - Parameters:
    ///   - count: Number of inputs; `load` is called once for each index in `0 ..< count`.
    ///   - dimension: Matryoshka truncation (128/256/512); `nil` keeps all 768 dimensions.
    ///   - load: Produces input `index` (e.g. decodes a file with ``MediaLoader``).
    ///   - onResult: Called once per index, in completion order (not index order), with the unit
    ///     vector or the error for that input.
    /// - Throws: `CancellationError` if the calling task is cancelled; per-input failures are
    ///   delivered through `onResult` instead.
    public func embed(
        count: Int, dimension: Int? = nil, load: @escaping InputLoader,
        onResult: (_ index: Int, _ result: Result<[Float], any Error>) -> Void
    ) async throws {
        let c = configuration
        var nextStart = 0
        func startWindow() -> Task<[(Int, Result<PreparedInput, any Error>)], Never>? {
            guard nextStart < count else { return nil }
            let range = nextStart ..< min(nextStart + max(1, c.lookahead), count)
            let processor = processor
            let parallelism = c.parallelism ?? ProcessInfo.processInfo.activeProcessorCount
            let byteBudget = c.lookaheadBytes
            // The window may stop early on bytes; record where the next one starts once it finishes.
            return Task.detached(priority: .userInitiated) {
                await Self.loadWindow(range, parallelism: parallelism, byteBudget: byteBudget, processor: processor, load: load)
            }
        }

        var window = startWindow()
        var pending: (indices: [Int], embeddings: MLXArray)?
        while let current = window {
            let loaded = await current.value
            nextStart = (loaded.map(\.0).max() ?? nextStart) + 1
            window = startWindow()  // prefetch the next window while this one runs on the GPU

            var ready: [(Int, PreparedInput)] = []
            for (index, result) in loaded {
                switch result {
                case .success(let item): ready.append((index, item))
                case .failure(let error): onResult(index, .failure(error))
                }
            }
            for batch in pack(ready) {
                try Task.checkCancellation()
                do {
                    var embeddings = try model(processor.collate(batch.map(\.1))).embeddings
                    if let dimension, dimension < embeddings.dim(1) {
                        embeddings = Self.normalize(embeddings[0..., ..<dimension])
                    }
                    asyncEval(embeddings)
                    if let pending { deliver(pending, onResult) }
                    pending = (batch.map(\.0), embeddings)
                } catch {
                    for (index, _) in batch { onResult(index, .failure(error)) }
                }
            }
        }
        if let pending { deliver(pending, onResult) }
    }

    /// Embeds `inputs`, returning one unit vector per input in input order.
    ///
    /// Convenience over ``embed(count:dimension:load:onResult:)`` for inputs already in memory.
    ///
    /// - Parameters:
    ///   - inputs: Items to embed; each may mix text and media.
    ///   - dimension: Matryoshka truncation (128/256/512); the prefix is re-normalized.
    ///     `nil` keeps all 768 dimensions. Compare queries and documents at the same width.
    /// - Throws: The first input's error if any input fails.
    public func embed(_ inputs: [EmbeddingInput], dimension: Int? = nil) async throws -> [[Float]] {
        var out = [Result<[Float], any Error>?](repeating: nil, count: inputs.count)
        try await embed(count: inputs.count, dimension: dimension, load: { inputs[$0] }) { out[$0] = $1 }
        return try out.map { try $0!.get() }
    }

    /// Convenience for one input.
    public func embed(_ input: EmbeddingInput, dimension: Int? = nil) async throws -> [Float] {
        try await embed([input], dimension: dimension)[0]
    }

    private func deliver(
        _ batch: (indices: [Int], embeddings: MLXArray), _ onResult: (Int, Result<[Float], any Error>) -> Void
    ) {
        let rows = batch.embeddings.asArray(Float.self).chunked(batch.embeddings.dim(1))
        for (index, row) in zip(batch.indices, rows) { onResult(index, .success(row)) }
    }

    /// Loads + prepares a window concurrently, stopping early once `byteBudget` of media is held.
    private static func loadWindow(
        _ range: Range<Int>, parallelism: Int, byteBudget: Int, processor: EmbeddingGemma2Processor,
        load: @escaping InputLoader
    ) async -> [(Int, Result<PreparedInput, any Error>)] {
        var results: [(Int, Result<PreparedInput, any Error>)] = []
        var bytes = 0
        var next = range.lowerBound
        while next < range.upperBound, bytes < byteBudget {
            let chunk = next ..< min(next + max(1, parallelism), range.upperBound)
            next = chunk.upperBound
            await withTaskGroup(of: (Int, Result<PreparedInput, any Error>).self) { group in
                for index in chunk {
                    group.addTask {
                        do { return (index, .success(try processor.prepare(try await load(index)))) }
                        catch { return (index, .failure(error)) }
                    }
                }
                for await result in group {
                    if case .success(let item) = result.1 { bytes += item.byteCount }
                    results.append(result)
                }
            }
        }
        return results
    }

    /// Groups prepared inputs into batches within the configured budgets, similar inputs together.
    func pack(_ items: [(Int, PreparedInput)]) -> [[(Int, PreparedInput)]] {
        let c = configuration
        // Modality first (text-only, audio, vision) so towers batch together, then by length.
        func key(_ p: PreparedInput) -> (Int, Int) {
            (p.visionFrames > 0 ? 2 : p.audioSamples > 0 ? 1 : 0, p.tokenCount)
        }
        let sorted = items.sorted { key($0.1) < key($1.1) }
        var batches: [[(Int, PreparedInput)]] = []
        var batch: [(Int, PreparedInput)] = []
        var longest = 0, frames = 0, samples = 0
        for item in sorted {
            let p = item.1
            let fits = batch.count < c.batchSize
                && (batch.count + 1) * max(longest, p.tokenCount) <= c.maxBatchTokens
                && frames + p.visionFrames <= c.maxBatchVisionFrames
                && samples + p.audioSamples <= c.maxBatchAudioSamples
            if !fits, !batch.isEmpty {
                batches.append(batch)
                batch = []
                longest = 0; frames = 0; samples = 0
            }
            batch.append(item)
            longest = max(longest, p.tokenCount)
            frames += p.visionFrames
            samples += p.audioSamples
        }
        if !batch.isEmpty { batches.append(batch) }
        return batches
    }

    /// Releases MLX's cached buffers (call between large jobs; cadence is the frontend's choice).
    public func clearCache() { Memory.clearCache() }

    /// Dot product of two unit vectors (their cosine similarity).
    public static func similarity(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Candidate indices sorted by descending similarity to `query`.
    public static func rank(query: [Float], candidates: [[Float]]) -> [(index: Int, score: Float)] {
        candidates.enumerated().map { ($0.offset, similarity(query, $0.element)) }.sorted { $0.score > $1.score }
    }

    static func normalize(_ x: MLXArray) -> MLXArray {
        x / MLX.maximum(MLX.sqrt((x * x).sum(axis: -1, keepDims: true)), 1e-9)
    }
}
