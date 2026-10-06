// swift-tools-version: 6.1
//
// mlx-swift-embeddinggemma2 — Apple-Silicon port of google/embeddinggemma-2 on mlx-swift.
//
// A bidirectional Gemma4-style encoder (alternating local/full attention, per-layer inputs)
// with the Gemma 4 vision and audio towers, mean-pooled into one 768-d L2-normalized space for
// text, images, audio and video. Ported from mlx-vlm's `embedding_gemma2` (revision 3d87e884);
// preprocessing follows transformers' `EmbeddingGemma2Processor`.

import PackageDescription

let package = Package(
    name: "mlx-swift-embeddinggemma2",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MLXEmbeddingGemma2", targets: ["MLXEmbeddingGemma2"]),
        .executable(name: "embeddinggemma2", targets: ["embeddinggemma2"]),
    ],
    dependencies: [
        // 0.32.2+ bundles mlx with the M5 NAX bf16 split-K GEMM fix; this model is bf16-only.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXEmbeddingGemma2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/MLXEmbeddingGemma2"
        ),
        .executableTarget(
            name: "embeddinggemma2",
            dependencies: [
                "MLXEmbeddingGemma2",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Examples/embeddinggemma2"
        ),
        .testTarget(
            name: "MLXEmbeddingGemma2Tests",
            dependencies: ["MLXEmbeddingGemma2"],
            path: "Tests/MLXEmbeddingGemma2Tests"
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation, so normal builds and downstream
// consumers don't have to resolve an extra dependency (Scripts/build_docs.sh sets the gate).
if Context.environment["SPI_GENERATE_DOCS"] == "1" || Context.environment["BUILD_DOC"] == "1" {
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
