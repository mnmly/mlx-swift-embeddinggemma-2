// Copyright © 2026
//
// Port of transformers `EmbeddingGemma2Processor` + the checkpoint's chat template: an optional
// task prompt (the template's system message) followed by the content parts in order, each media
// part expanded to its soft-token placeholders, then <bos> … <eos> and right padding.

import Foundation
import MLX
import Tokenizers

/// One part of an embedding input. Parts are concatenated in order into a single embedding.
public enum EmbeddingContent: Sendable {
    case text(String)
    case image(RGBImage)
    /// 16 kHz mono samples in `[-1, 1]`.
    case audio([Float])
    case video(VideoClip)
}

/// One item to embed: an optional task prompt plus ordered, possibly mixed, content.
public struct EmbeddingInput: Sendable {
    /// Task prefix such as `"task: search result | query: "`; see ``EmbeddingPrompt``.
    public var prompt: String?
    public var contents: [EmbeddingContent]

    public init(prompt: String? = nil, _ contents: [EmbeddingContent]) {
        self.prompt = prompt
        self.contents = contents
    }

    public static func text(_ text: String, prompt: String? = nil) -> EmbeddingInput {
        EmbeddingInput(prompt: prompt, [.text(text)])
    }
}

/// Task prompts from the checkpoint's `config_sentence_transformers.json`.
public enum EmbeddingPrompt {
    public static let query = "task: search result | query: "
    public static let document = "title: none | text: "
    public static let classification = "task: classification | query: "
    public static let clustering = "task: clustering | query: "
    public static let codeRetrieval = "task: code retrieval | query: "
    public static let factChecking = "task: fact checking | query: "
    public static let questionAnswering = "task: question answering | query: "
    public static let sentenceSimilarity = "task: sentence similarity | query: "
}

/// Turns ``EmbeddingInput``s into one padded ``EmbeddingGemma2Inputs`` batch.
public struct EmbeddingGemma2Processor: Sendable {
    public let tokenizer: any Tokenizer
    public let configuration: EmbeddingGemma2Configuration
    public var image = ImageProcessor()
    public var video = VideoProcessor()
    public var audio = AudioFeatureExtractor()
    /// Upper bound on text tokens per segment (media is never truncated).
    public var maxTextTokens = 8192

    let bosId: Int
    let eosId: Int

    public init(tokenizer: any Tokenizer, configuration: EmbeddingGemma2Configuration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        bosId = tokenizer.convertTokenToId("<bos>") ?? 2
        eosId = tokenizer.convertTokenToId("<eos>") ?? 1
    }

    public static func load(directory: URL, configuration: EmbeddingGemma2Configuration) async throws
        -> EmbeddingGemma2Processor
    {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return EmbeddingGemma2Processor(tokenizer: tokenizer, configuration: configuration)
    }

    /// Token ids of plain text (no `<bos>`/`<eos>`).
    public func tokenize(_ text: String) -> [Int] {
        text.isEmpty ? [] : Array(tokenizer.encode(text: text, addSpecialTokens: false).prefix(maxTextTokens))
    }

    /// Tokenizes and preprocesses one input (CPU only). Safe to call concurrently.
    ///
    /// Adjacent text (prompt included) is tokenized as one string, as the chat template renders it;
    /// each media part is expanded to its soft-token placeholders.
    public func prepare(_ input: EmbeddingInput) throws -> PreparedInput {
        let c = configuration
        var ids = [bosId]
        var images: [PatchifiedImage] = []
        var frames: [PatchifiedImage] = []
        var clips: [[Float]] = []
        var pending = input.prompt ?? ""
        for part in input.contents {
            if case .text(let text) = part {
                pending += text
                continue
            }
            ids += tokenize(pending)
            pending = ""
            switch part {
            case .text:
                break
            case .image(let rgb):
                let p = try image(rgb)
                images.append(p)
                ids += [c.boiTokenId] + Array(repeating: c.imageTokenId, count: p.softTokens) + [c.eoiTokenId]
            case .video(let clip):
                for p in try video(clip) {
                    frames.append(p)
                    ids += [c.boiTokenId] + Array(repeating: c.videoTokenId, count: p.softTokens) + [c.eoiTokenId]
                }
            case .audio(let samples):
                let clip = Array(samples.prefix(audio.maxLength))
                let n = audio.softTokens(samples: clip.count)
                guard n > 0 else { throw EmbeddingGemma2Error.invalidInput("audio clip is too short (\(clip.count) samples)") }
                clips.append(clip)
                ids += [c.boaTokenId] + Array(repeating: c.audioTokenId, count: n) + [c.eoaTokenId]
            }
        }
        ids += tokenize(pending) + [eosId]
        return PreparedInput(ids: ids, images: images, frames: frames, clips: clips)
    }

    /// Pads prepared inputs into one model batch. Media soft-token counts travel with the batch, so
    /// the forward pass needs no GPU→CPU synchronization.
    public func collate(_ items: [PreparedInput]) -> EmbeddingGemma2Inputs {
        let c = configuration
        var batch = EmbeddingGemma2Inputs(inputIds: MLXArray(0), attentionMask: MLXArray(0))
        let images = items.flatMap(\.images)
        let frames = items.flatMap(\.frames)
        let clips = items.flatMap(\.clips)
        if !images.isEmpty {
            (batch.pixelValues, batch.imagePositionIds) = stack(images, dimension: image.patchDimension)
            batch.imageTokenCounts = images.map(\.softTokens)
        }
        if !frames.isEmpty {
            (batch.pixelValuesVideos, batch.videoPositionIds) = stack(frames, dimension: video.image.patchDimension)
            batch.videoTokenCounts = frames.map(\.softTokens)
        }
        if !clips.isEmpty {
            (batch.inputFeatures, batch.inputFeaturesMask) = audio(clips)
            batch.audioTokenCounts = clips.map { audio.softTokens(samples: $0.count) }
        }

        let L = items.map(\.ids.count).max() ?? 0
        var flat = [Int32](repeating: Int32(c.textConfig.padTokenId), count: items.count * L)
        var valid = [Int32](repeating: 0, count: items.count * L)
        for (b, item) in items.enumerated() {
            for (t, id) in item.ids.enumerated() {
                flat[b * L + t] = Int32(id)
                valid[b * L + t] = 1
            }
        }
        batch.inputIds = MLXArray(flat, [items.count, L])
        batch.attentionMask = MLXArray(valid, [items.count, L])
        return batch
    }

    /// Prepares (in parallel) and collates `inputs` into one padded batch.
    public func callAsFunction(_ inputs: [EmbeddingInput]) throws -> EmbeddingGemma2Inputs {
        try collate(prepareAll(inputs).map { try $0.get() })
    }

    /// ``prepare(_:)`` for each input, concurrently, keeping per-input failures.
    public func prepareAll(_ inputs: [EmbeddingInput]) -> [Result<PreparedInput, any Error>] {
        guard inputs.count > 1 else { return inputs.map { input in Result { try prepare(input) } } }
        var out = [Result<PreparedInput, any Error>](
            repeating: .failure(EmbeddingGemma2Error.invalidInput("not prepared")), count: inputs.count)
        out.withUnsafeMutableBufferPointer { buffer in
            nonisolated(unsafe) let results = buffer
            DispatchQueue.concurrentPerform(iterations: inputs.count) { i in
                results[i] = Result { try prepare(inputs[i]) }
            }
        }
        return out
    }

    private func stack(_ items: [PatchifiedImage], dimension: Int) -> (MLXArray, MLXArray) {
        let P = items[0].positions.count / 2
        return (
            // `rescale_factor` multiply (not a division), matching the reference bit-for-bit.
            MLXArray(items.flatMap(\.patches), [items.count, P, dimension]).asType(.float32) * Float(1.0 / 255.0),
            MLXArray(items.flatMap(\.positions), [items.count, P, 2])
        )
    }
}

/// One input after tokenization and media preprocessing, ready to be batched by
/// ``EmbeddingGemma2Processor/collate(_:)``.
public struct PreparedInput: Sendable {
    /// Token ids including `<bos>`, `<eos>` and expanded media placeholders.
    public let ids: [Int]
    let images: [PatchifiedImage]
    let frames: [PatchifiedImage]
    let clips: [[Float]]

    /// Sequence length the text encoder sees.
    public var tokenCount: Int { ids.count }
    /// Images plus video frames this input sends through the vision tower.
    public var visionFrames: Int { images.count + frames.count }
    /// Audio samples this input sends through the audio tower.
    public var audioSamples: Int { clips.reduce(0) { $0 + $1.count } }
    /// Approximate host memory held by the preprocessed media.
    var byteCount: Int {
        (images + frames).reduce(0) { $0 + $1.patches.count + $1.positions.count * 4 } + audioSamples * 4 + ids.count * 8
    }
}

extension Array {
    func chunked(_ n: Int) -> [[Element]] {
        stride(from: 0, to: count, by: n).map { Array(self[$0 ..< Swift.min($0 + n, count)]) }
    }
}
