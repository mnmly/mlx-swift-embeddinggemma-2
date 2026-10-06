# ``MLXEmbeddingGemma2``

Multimodal embeddings with Google's EmbeddingGemma 2 on Apple silicon: text, images, audio and
video, alone or mixed, in one 768-dimensional space.

## Overview

EmbeddingGemma 2 is a bidirectional Gemma 4–style encoder with the Gemma 4 vision and audio
towers. Every input becomes one L2-normalized vector, so cosine similarity is a dot product, and
you can truncate vectors to 512, 256 or 128 dimensions (Matryoshka) for cheaper storage.

``EmbeddingSession`` is the entry point. It loads the checkpoint (downloading
`mlx-community/embeddinggemma-2-bf16` into the Hugging Face cache on first use), preprocesses
inputs exactly like the reference `EmbeddingGemma2Processor`, and batches forward passes.

```swift
import MLXEmbeddingGemma2

let session = try await EmbeddingSession.load()

let query = try session.embed(.text("Which planet is known as the Red Planet?",
                                    prompt: EmbeddingPrompt.query))
let docs = try session.embed([
    .text("Mars is known as the Red Planet.", prompt: EmbeddingPrompt.document),
    EmbeddingInput(prompt: EmbeddingPrompt.document, [.image(try MediaLoader.image(photoURL))]),
])
let ranked = EmbeddingSession.rank(query: query, candidates: docs)
```

Use the same prompt family and dimension for everything you compare: a
``EmbeddingPrompt/query`` prompt for search queries and ``EmbeddingPrompt/document`` for the
indexed items. Keep weights in bfloat16 (the default). The model card warns against float16.

For large jobs (thousands of items) use the streaming form, which loads inputs lazily, packs
batches by token and frame budgets, overlaps preprocessing with GPU work and reports per-item
failures without stopping:

```swift
try await session.embed(count: urls.count, load: { i in
    EmbeddingInput(prompt: EmbeddingPrompt.document, [try await MediaLoader.content(urls[i])])
}) { index, result in
    // completion order; store result for `index`
}
```

Media is passed *materialized*: an ``RGBImage``, 16 kHz mono samples, or a ``VideoClip`` of
decoded frames. ``MediaLoader`` decodes files with ImageIO and AVFoundation. Sandboxed apps
should call it while they still have access to the file.

## Topics

### Embedding and search

- ``EmbeddingSession``
- ``EmbeddingSession/InputLoader``
- ``EmbeddingSessionConfiguration``
- ``EmbeddingInput``
- ``EmbeddingContent``
- ``EmbeddingPrompt``

### Loading media

- ``MediaLoader``
- ``RGBImage``
- ``VideoClip``

### Preprocessing

- ``EmbeddingGemma2Processor``
- ``PreparedInput``
- ``ImageProcessor``
- ``VideoProcessor``
- ``AudioFeatureExtractor``
- ``PatchifiedImage``

### Running the network directly

- ``EmbeddingGemma2Model``
- ``EmbeddingGemma2Inputs``
- ``EmbeddingGemma2Towers``
- ``resolveModelDirectory(repo:)``

### Configuration

- ``EmbeddingGemma2Configuration``
- ``TextConfiguration``
- ``VisionConfiguration``
- ``AudioConfiguration``
- ``EmbeddingGemma2Error``
