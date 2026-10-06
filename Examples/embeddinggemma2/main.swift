// Copyright © 2026
//
// embeddinggemma2 — embed and search text, images, audio and video with EmbeddingGemma 2.

import ArgumentParser
import Foundation
import MLX
import MLXEmbeddingGemma2

@main
struct EmbeddingGemma2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embeddinggemma2",
        abstract: "Multimodal embeddings with EmbeddingGemma 2 on MLX.",
        discussion: """
            ITEMs are file paths (images, audio and video are recognized by extension) or literal text \
            prefixed with "text:". Join parts with "+" to embed them together, e.g. \
            "text:A photo of a cat+cat.jpg".
            """,
        subcommands: [Embed.self, Search.self, Bench.self],
        defaultSubcommand: Search.self)
}

struct ModelOptions: ParsableArguments {
    @Option(help: "Checkpoint directory (defaults to the Hugging Face cache of \(EmbeddingGemma2Model.defaultRepo)).")
    var model: String?

    @Option(help: "Matryoshka width: 128, 256, 512 or 768.")
    var dim: Int = 768

    @Option(help: "Soft-token budget per image (70, 140, 280, 560, 1120).")
    var imageTokens: Int = 280

    @Option(help: "Most inputs per forward pass.")
    var batchSize: Int = 16

    @Option(help: "Most padded text positions per forward pass (inputs × longest sequence).")
    var batchTokens: Int = 8192

    func session(needsMedia: Bool) async throws -> EmbeddingSession {
        var config = EmbeddingSessionConfiguration()
        config.directory = model.map { URL(filePath: $0) }
        config.towers = needsMedia ? .all : []
        config.imageSoftTokens = imageTokens
        config.batchSize = batchSize
        config.maxBatchTokens = batchTokens
        return try await EmbeddingSession.load(config)
    }
}

enum PromptName: String, ExpressibleByArgument, CaseIterable {
    case query, document, classification, clustering, code, factcheck, qa, similarity, none

    var text: String? {
        switch self {
        case .query: EmbeddingPrompt.query
        case .document: EmbeddingPrompt.document
        case .classification: EmbeddingPrompt.classification
        case .clustering: EmbeddingPrompt.clustering
        case .code: EmbeddingPrompt.codeRetrieval
        case .factcheck: EmbeddingPrompt.factChecking
        case .qa: EmbeddingPrompt.questionAnswering
        case .similarity: EmbeddingPrompt.sentenceSimilarity
        case .none: nil
        }
    }
}

/// Parses an ITEM into content parts, decoding files.
func parseItem(_ item: String, video: VideoProcessor) async throws -> [EmbeddingContent] {
    var parts: [EmbeddingContent] = []
    for piece in item.components(separatedBy: "+") {
        if piece.hasPrefix("text:") {
            parts.append(.text(String(piece.dropFirst(5))))
        } else {
            let url = URL(filePath: piece)
            guard MediaLoader.kind(of: url) != nil else {
                throw ValidationError("Unrecognized file type: \(piece) (prefix literal text with \"text:\")")
            }
            parts.append(try await MediaLoader.content(url, video: video))
        }
    }
    return parts
}

func needsMedia(_ items: [String]) -> Bool {
    items.flatMap { $0.components(separatedBy: "+") }.contains { !$0.hasPrefix("text:") }
}

/// Prints to stderr (progress and per-item failures), keeping stdout for results.
func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

struct Embed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Stream one JSON line per ITEM: {\"index\", \"item\", \"embedding\"}.",
        discussion: """
            Built for large jobs: items are decoded lazily, batched by token budget, and written as \
            each batch finishes (completion order; use "index" to restore input order). Failing items \
            are reported on stderr and skipped. Read thousands of items with --items-file (one ITEM \
            per line, "-" for stdin).
            """)

    @OptionGroup var options: ModelOptions
    @Option(help: "Task prompt.") var prompt: PromptName = .document
    @Option(help: "File with one ITEM per line (\"-\" for stdin).") var itemsFile: String?
    @Argument(help: "Items to embed.") var items: [String] = []

    func run() async throws {
        var all = items
        if let itemsFile {
            let text = itemsFile == "-"
                ? String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
                : try String(contentsOfFile: itemsFile, encoding: .utf8)
            all += text.split(whereSeparator: \.isNewline).map(String.init).filter { !$0.isEmpty }
        }
        guard !all.isEmpty else { throw ValidationError("No items: pass ITEMs or --items-file.") }

        let session = try await options.session(needsMedia: needsMedia(all))
        let video = session.processor.video
        let prompt = prompt.text
        let total = all.count
        var done = 0, failed = 0
        let start = ContinuousClock.now
        try await session.embed(count: total, dimension: options.dim, load: { [all] i in
            EmbeddingInput(prompt: prompt, try await parseItem(all[i], video: video))
        }) { index, result in
            done += 1
            switch result {
            case .success(let vector):
                let json = try? JSONSerialization.data(withJSONObject: ["index": index, "item": all[index], "embedding": vector])
                if let json { print(String(decoding: json, as: UTF8.self)) }
            case .failure(let error):
                failed += 1
                log("error: item \(index) (\(all[index].prefix(80))): \(error.localizedDescription)")
            }
            if done % 100 == 0 || done == total {
                let seconds = (ContinuousClock.now - start) / .seconds(1)
                log(String(format: "embedded %d/%d (%d failed) · %.1f items/s", done, total, failed, Double(done) / seconds))
            }
        }
    }
}

struct Search: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Rank ITEMs by similarity to a query.")

    @OptionGroup var options: ModelOptions
    @Option(name: .shortAndLong, help: "The query (an ITEM, e.g. \"text:red planet\" or a file).")
    var query: String
    @Option(help: "Prompt for the query.") var queryPrompt: PromptName = .query
    @Option(help: "Prompt for the items.") var documentPrompt: PromptName = .document
    @Argument(help: "Items to rank.") var items: [String]

    func run() async throws {
        let session = try await options.session(needsMedia: needsMedia(items + [query]))
        let video = session.processor.video
        let q = try await session.embed(
            EmbeddingInput(prompt: queryPrompt.text, try await parseItem(query, video: video)), dimension: options.dim)
        var docs: [EmbeddingInput] = []
        for item in items {
            docs.append(EmbeddingInput(prompt: documentPrompt.text, try await parseItem(item, video: video)))
        }
        let vectors = try await session.embed(docs, dimension: options.dim)
        for (rank, hit) in EmbeddingSession.rank(query: q, candidates: vectors).enumerated() {
            print(String(format: "%2d  %.4f  %@", rank + 1, hit.score, items[hit.index]))
        }
    }
}

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Time text, image and audio embedding.")

    @OptionGroup var options: ModelOptions
    @Option(help: "Timed iterations per case.") var iterations: Int = 10

    func run() async throws {
        let session = try await options.session(needsMedia: true)
        let sentence = "The quick brown fox jumps over the lazy dog. "
        let text128 = String(repeating: sentence, count: 13)  // ≈128 tokens
        let text512 = String(repeating: sentence, count: 51)  // ≈512 tokens
        var rng = SystemRandomNumberGenerator()
        let image = RGBImage(width: 640, height: 480, pixels: (0 ..< 640 * 480 * 3).map { _ in UInt8.random(in: 0 ... 255, using: &rng) })
        let audio = (0 ..< 16_000 * 10).map { sinf(Float($0) * 0.05) * 0.3 }

        let cases: [(String, [EmbeddingInput])] = [
            ("text 128 tok ×1", [.text(text128)]),
            ("text 128 tok ×8", Array(repeating: .text(text128), count: 8)),
            ("text 512 tok ×1", [.text(text512)]),
            ("text 512 tok ×8", Array(repeating: .text(text512), count: 8)),
            ("image 640×480 ×1", [EmbeddingInput([.image(image)])]),
            ("audio 10 s ×1", [EmbeddingInput([.audio(audio)])]),
        ]
        print("case                 total ms  preprocess  forward   per-item")
        for (name, inputs) in cases {
            _ = try await session.embed(inputs)  // warm-up
            var total: [Double] = [], prep: [Double] = [], forward: [Double] = []
            for _ in 0 ..< iterations {
                let t0 = ContinuousClock.now
                let batch = try session.processor(inputs)
                eval(batch.inputIds, batch.attentionMask)
                let t1 = ContinuousClock.now
                eval(try session.model(batch).embeddings)
                let t2 = ContinuousClock.now
                prep.append((t1 - t0) / .milliseconds(1))
                forward.append((t2 - t1) / .milliseconds(1))
                total.append((t2 - t0) / .milliseconds(1))
            }
            func median(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
            print(name.padding(toLength: 20, withPad: " ", startingAt: 0)
                + String(format: " %8.2f  %10.2f  %7.2f  %8.2f", median(total), median(prep), median(forward),
                         median(total) / Double(inputs.count)))
        }

        // Throughput through the streaming path (lazy load, budget packing, overlapped preprocessing).
        let sentences = (0 ..< 2000).map { i in
            String(repeating: sentence, count: 1 + i % 24)  // ~10 – 260 tokens, mixed lengths
        }
        let streamCases: [(String, Int, EmbeddingSession.InputLoader)] = [
            ("2000 texts, mixed length", 2000, { .text(sentences[$0], prompt: EmbeddingPrompt.document) }),
            ("128 images 640×480", 128, { _ in EmbeddingInput([.image(image)]) }),
            ("64 audio clips 10 s", 64, { _ in EmbeddingInput([.audio(audio)]) }),
        ]
        print("\nstreaming                   items/s    total s")
        for (name, count, load) in streamCases {
            let start = ContinuousClock.now
            var ok = 0
            try await session.embed(count: count, load: load) { _, result in if case .success = result { ok += 1 } }
            let seconds = (ContinuousClock.now - start) / .seconds(1)
            precondition(ok == count, "\(count - ok) inputs failed")
            print(name.padding(toLength: 26, withPad: " ", startingAt: 0)
                + String(format: " %8.1f  %8.2f", Double(count) / seconds, seconds))
        }
    }
}
