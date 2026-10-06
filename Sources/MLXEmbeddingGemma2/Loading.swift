// Copyright © 2026

import Foundation
import Hub
import MLX
import MLXNN

/// Which modality towers to keep when loading; dropped towers' weights are never materialized.
public struct EmbeddingGemma2Towers: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// Images and video.
    public static let vision = EmbeddingGemma2Towers(rawValue: 1 << 0)
    public static let audio = EmbeddingGemma2Towers(rawValue: 1 << 1)
    public static let all: EmbeddingGemma2Towers = [.vision, .audio]
}

extension EmbeddingGemma2Model {
    /// Hugging Face repo of the MLX BF16 conversion.
    public static let defaultRepo = "mlx-community/embeddinggemma-2-bf16"

    /// Loads `config.json` + `*.safetensors` from a checkpoint directory.
    ///
    /// - Parameters:
    ///   - directory: The model snapshot.
    ///   - towers: Modality towers to keep (text is always loaded).
    ///   - dtype: Cast floating-point weights to this dtype, or `nil` to keep the checkpoint's
    ///     bfloat16. Never use `.float16`: the model card warns it overflows.
    public static func load(
        directory: URL, towers: EmbeddingGemma2Towers = .all, dtype: DType? = nil
    ) throws -> EmbeddingGemma2Model {
        var config = try JSONDecoder().decode(
            EmbeddingGemma2Configuration.self, from: Data(contentsOf: directory.appending(path: "config.json")))
        if !towers.contains(.vision) { config.visionConfig = nil }
        if !towers.contains(.audio) { config.audioConfig = nil }
        let model = EmbeddingGemma2Model(config)

        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        for file in files {
            try weights.merge(MLX.loadArrays(url: file)) { a, _ in a }
        }
        weights = model.sanitize(weights)
        if let dtype {
            weights = weights.mapValues { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}

/// Newest complete snapshot of `repo` in the Hugging Face hub cache
/// (`$HF_HUB_CACHE`, `$HF_HOME/hub` or `~/.cache/huggingface/hub`), downloading it there if missing.
public func resolveModelDirectory(repo: String = EmbeddingGemma2Model.defaultRepo) async throws -> URL {
    let env = ProcessInfo.processInfo.environment
    let hubCache = env["HF_HUB_CACHE"].map { URL(filePath: $0) }
        ?? env["HF_HOME"].map { URL(filePath: $0).appending(path: "hub") }
        ?? URL(filePath: NSHomeDirectory()).appending(path: ".cache/huggingface/hub")
    let snapshots = hubCache.appending(path: "models--\(repo.replacingOccurrences(of: "/", with: "--"))/snapshots")
    let candidates = (try? FileManager.default.contentsOfDirectory(
        at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
    let complete = candidates.filter {
        FileManager.default.fileExists(atPath: $0.appending(path: "config.json").path())
            && FileManager.default.fileExists(atPath: $0.appending(path: "tokenizer.json").path())
            && FileManager.default.fileExists(atPath: $0.appending(path: "model.safetensors").path())
    }
    func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
    if let newest = complete.max(by: { modified($0) < modified($1) }) { return newest }

    return try await HubApi(downloadBase: hubCache).snapshot(from: repo, matching: ["*.json", "*.safetensors"])
}
