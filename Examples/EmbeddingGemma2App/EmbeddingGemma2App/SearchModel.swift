import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import MLXEmbeddingGemma2
import Observation
import UniformTypeIdentifiers

/// Where an item's content comes from. Files are referenced, not decoded, until their batch is
/// embedded, so a corpus of thousands of photos never holds their pixels in memory.
enum ItemSource: Sendable {
    case text(String)
    case file(URL)

    /// The embedding input (decodes files now; non-sandboxed app, so file URLs stay readable).
    func input() async throws -> EmbeddingInput {
        switch self {
        case .text(let text):
            return .text(text, prompt: EmbeddingPrompt.document)
        case .file(let url):
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            return EmbeddingInput(prompt: EmbeddingPrompt.document, [try await MediaLoader.content(url)])
        }
    }
}

/// One thing in the corpus and its cached 768-d embedding.
struct CorpusItem: Identifiable, Sendable {
    enum Kind: String, Sendable { case text, image, audio, video }

    let id = UUID()
    let kind: Kind
    let title: String
    let detail: String
    let source: ItemSource
    let thumbnail: CGImage?
    var embedding: [Float]?
    var score: Float?
    /// Why this item couldn't be embedded, if it failed.
    var failure: String?
}

/// Serializes access to the shared ``EmbeddingSession`` off the main actor (one driver at a time).
actor Embedder {
    let session: EmbeddingSession

    init(session: EmbeddingSession) { self.session = session }

    func embed(_ inputs: [EmbeddingInput]) async throws -> [[Float]] {
        try await session.embed(inputs)
    }

    /// Streams `sources` through the session, handing results over in chunks (not per item) so the
    /// UI updates a few times per second instead of thousands of times.
    func embed(
        _ sources: [ItemSource], deliver: @escaping @Sendable ([(Int, Result<[Float], any Error>)]) async -> Void
    ) async throws {
        defer { session.clearCache() }
        var buffer: [(Int, Result<[Float], any Error>)] = []
        var deliveries: [Task<Void, Never>] = []
        var lastFlush = ContinuousClock.now
        try await session.embed(count: sources.count, load: { try await sources[$0].input() }) { index, result in
            buffer.append((index, result))
            if ContinuousClock.now - lastFlush > .milliseconds(250) {
                let chunk = buffer
                buffer.removeAll()
                lastFlush = .now
                deliveries.append(Task { await deliver(chunk) })
            }
        }
        for delivery in deliveries { await delivery.value }  // every chunk lands before returning
        await deliver(buffer)
    }
}

/// Counts for one embedding run, updated on the main actor as chunks land.
@MainActor final class Progress {
    var done = 0
    var failed = 0
}

@Observable @MainActor
final class SearchModel {
    enum Status: Equatable {
        case loading, ready, working(String), failed(String)
    }

    var items: [CorpusItem] = []
    var selection = Set<CorpusItem.ID>()
    var query = ""
    /// What the current scores are relative to (shown in the results header).
    var rankedBy: String?
    var dimension = 768
    var status: Status = .loading
    /// Problems that need the person's attention (shown as an alert, not just in the status bar).
    var alertMessage: String?
    var isImporting = false
    var isAddingText = false
    var isConfirmingClear = false
    /// Set by the window so deletions can be undone (Edit › Undo).
    var undoManager: UndoManager?

    private var embedder: Embedder?
    /// Query vector at full width; re-truncated when ``dimension`` changes.
    private var queryVector: [Float]?
    /// Items handed to the embedder and not yet back (so a second run doesn't embed them twice).
    private var inFlight = Set<CorpusItem.ID>()

    var isReady: Bool { embedder != nil }
    var isBusy: Bool { if case .working = status { true } else { false } }

    func loadModel() async {
        status = .loading
        do {
            let session = try await EmbeddingSession.load()
            embedder = Embedder(session: session)
            status = .ready
            await embedPending()
        } catch {
            status = .failed(error.localizedDescription)
            alertMessage = "EmbeddingGemma 2 couldn’t be loaded. \(error.localizedDescription) Check your network connection and relaunch to retry the download."
        }
    }

    // MARK: Corpus

    /// Adds files and folders (searched recursively for images, audio and video).
    func addFiles(_ urls: [URL]) async {
        status = .working("Scanning \(urls.count) item\(urls.count == 1 ? "" : "s")…")
        let (built, skipped) = await Self.makeItems(urls)
        items += built
        if !built.isEmpty { selection = Set(built.map(\.id)) }
        status = isReady ? .ready : .loading
        if !skipped.isEmpty {
            let shown = skipped.prefix(10).joined(separator: "\n")
            alertMessage = "\(skipped.count) file\(skipped.count == 1 ? " wasn’t" : "s weren’t") added. Supported files are images, audio and video.\n\n"
                + shown + (skipped.count > 10 ? "\n…" : "")
        }
        await embedPending()
    }

    func addText(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let item = CorpusItem(
            kind: .text, title: trimmed, detail: "\(trimmed.count) characters", source: .text(trimmed), thumbnail: nil)
        items.append(item)
        selection = [item.id]
        await embedPending()
    }

    /// Deletes the selected items (undoable).
    func deleteSelection() { delete(selection, actionName: "Delete") }

    /// Deletes every item (undoable).
    func clear() { delete(Set(items.map(\.id)), actionName: "Clear Corpus") }

    private func delete(_ ids: Set<CorpusItem.ID>, actionName: String) {
        let removed = items.enumerated().filter { ids.contains($0.element.id) }
        guard !removed.isEmpty else { return }
        items.removeAll { ids.contains($0.id) }
        selection.subtract(ids)
        undoManager?.registerUndo(withTarget: self) { model in
            MainActor.assumeIsolated { model.restore(removed, actionName: actionName) }
        }
        undoManager?.setActionName(actionName)
    }

    private func restore(_ removed: [(offset: Int, element: CorpusItem)], actionName: String) {
        for (offset, item) in removed.sorted(by: { $0.offset < $1.offset }) {
            items.insert(item, at: min(offset, items.count))
        }
        selection = Set(removed.map(\.element.id))
        rescore()
        undoManager?.registerUndo(withTarget: self) { model in
            MainActor.assumeIsolated { model.delete(Set(removed.map(\.element.id)), actionName: actionName) }
        }
        undoManager?.setActionName(actionName)
    }

    /// Embeds every item that has no vector yet, as documents, streaming results into the list.
    func embedPending() async {
        guard let embedder else { return }
        let pending = items.filter { $0.embedding == nil && $0.failure == nil && !inFlight.contains($0.id) }
        guard !pending.isEmpty else { return }
        let ids = pending.map(\.id)
        inFlight.formUnion(ids)
        defer { inFlight.subtract(ids) }

        let total = pending.count
        let progress = Progress()
        status = .working("Embedding 0 of \(total)…")
        do {
            try await embedder.embed(pending.map(\.source)) { chunk in
                await MainActor.run {
                    // One positional lookup table per chunk keeps this O(chunk) instead of O(chunk × corpus).
                    let position = Dictionary(uniqueKeysWithValues: self.items.enumerated().map { ($1.id, $0) })
                    for (i, result) in chunk {
                        progress.done += 1
                        guard let p = position[ids[i]] else { continue }  // deleted meanwhile
                        switch result {
                        case .success(let vector): self.items[p].embedding = vector
                        case .failure(let error):
                            progress.failed += 1
                            self.items[p].failure = error.localizedDescription
                        }
                    }
                    self.status = .working("Embedding \(progress.done) of \(total)…")
                }
            }
            status = .ready
            if progress.failed > 0 {
                alertMessage = "\(progress.failed) of \(total) item\(total == 1 ? "" : "s") couldn’t be embedded. They’re marked in the list."
            }
            rescore()
        } catch {
            status = .ready
            alertMessage = "Embedding stopped. \(error.localizedDescription)"
        }
    }

    // MARK: Search

    func search() async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let embedder, !text.isEmpty else { return }
        do {
            queryVector = try await embedder.embed([.text(text, prompt: EmbeddingPrompt.query)])[0]
            rankedBy = "“\(text)”"
            rescore()
        } catch {
            alertMessage = "The search couldn’t run. \(error.localizedDescription)"
        }
    }

    /// Ranks the corpus by similarity to the selected item.
    func findSimilarToSelection() {
        guard let id = selection.first, let item = items.first(where: { $0.id == id }),
            let vector = item.embedding
        else { return }
        queryVector = vector
        rankedBy = item.title
        rescore()
    }

    var canFindSimilar: Bool {
        selection.count == 1 && items.contains { selection.contains($0.id) && $0.embedding != nil }
    }

    /// Re-scores at the current Matryoshka ``dimension`` and sorts by descending similarity.
    func rescore() {
        guard let queryVector else { return }
        let q = Self.truncate(queryVector, to: dimension)
        var scored = items
        for i in scored.indices {
            scored[i].score = scored[i].embedding.map { EmbeddingSession.similarity(q, Self.truncate($0, to: dimension)) }
        }
        items = scored.sorted { ($0.score ?? -2) > ($1.score ?? -2) }  // one observable update
    }

    /// Matryoshka prefix, re-normalized (same as embedding at that width).
    static func truncate(_ v: [Float], to d: Int) -> [Float] {
        guard d < v.count else { return v }
        let prefix = Array(v.prefix(d))
        let norm = max(prefix.reduce(0) { $0 + $1 * $1 }.squareRoot(), 1e-9)
        return prefix.map { $0 / norm }
    }

    func embeddingJSON(for ids: Set<CorpusItem.ID>) -> String? {
        let vectors = items.filter { ids.contains($0.id) }.compactMap { $0.embedding.map { Self.truncate($0, to: dimension) } }
        guard !vectors.isEmpty, let data = try? JSONSerialization.data(withJSONObject: vectors.count == 1 ? vectors[0] : vectors)
        else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    // MARK: Building items (off the main actor; files are only probed, not decoded)

    /// Expands folders, keeps supported media, and builds items with thumbnails concurrently.
    nonisolated static func makeItems(_ urls: [URL]) async -> (items: [CorpusItem], skipped: [String]) {
        var files: [URL] = []
        var skipped: [String] = []
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                let walker = FileManager.default.enumerator(
                    at: url, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
                var found: [URL] = []
                while let file = walker?.nextObject() as? URL {
                    if MediaLoader.kind(of: file) != nil { found.append(file) }
                }
                files += found.sorted { $0.path().localizedStandardCompare($1.path()) == .orderedAscending }
            } else if MediaLoader.kind(of: url) != nil {
                files.append(url)
            } else {
                skipped.append(url.lastPathComponent)
            }
        }
        let built = await withTaskGroup(of: (Int, CorpusItem).self) { group in
            for (i, file) in files.enumerated() {
                group.addTask { (i, await item(for: file)) }
            }
            var out = [CorpusItem?](repeating: nil, count: files.count)
            for await (i, item) in group { out[i] = item }
            return out.compactMap { $0 }
        }
        return (built, skipped)
    }

    nonisolated static func item(for url: URL) async -> CorpusItem {
        let name = url.lastPathComponent
        switch MediaLoader.kind(of: url) {
        case .image:
            let size = imageSize(url)
            return CorpusItem(
                kind: .image, title: name, detail: size.map { "\($0.width)×\($0.height)" } ?? "Image",
                source: .file(url), thumbnail: thumbnail(url))
        case .audio, .video:
            let kind: CorpusItem.Kind = MediaLoader.kind(of: url) == .audio ? .audio : .video
            let seconds = (try? await AVURLAsset(url: url).load(.duration).seconds) ?? 0
            let detail = seconds > 0 ? Duration.seconds(seconds).formatted(.time(pattern: .minuteSecond)) : kind.rawValue.capitalized
            return CorpusItem(kind: kind, title: name, detail: detail, source: .file(url), thumbnail: nil)
        case nil:
            return CorpusItem(kind: .text, title: name, detail: "", source: .file(url), thumbnail: nil)
        }
    }

    nonisolated static func imageSize(_ url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let w = props[kCGImagePropertyPixelWidth] as? Int, let h = props[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return (w, h)
    }

    /// Fully decoded, memory-backed thumbnail (safe after file access ends).
    nonisolated static func thumbnail(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: 128,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
        ] as CFDictionary)
    }
}
