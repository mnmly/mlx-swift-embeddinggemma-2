# mlx-swift-embeddinggemma2

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) on Apple silicon with
[mlx-swift](https://github.com/ml-explore/mlx-swift): one 768-d embedding space for **text, images,
audio, video, and mixes of them**. Matryoshka truncation to 512/256/128 is supported.

The weights are [`mlx-community/embeddinggemma-2-bf16`](https://huggingface.co/mlx-community/embeddinggemma-2-bf16)
(1.5 GB, downloaded to the Hugging Face cache on first use). The network is ported from
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) `embedding_gemma2` (revision `3d87e884`), and preprocessing from
transformers' `EmbeddingGemma2Processor` (5.19).

## Library

```swift
.package(url: "https://github.com/mnmly/mlx-swift-embeddinggemma-2", from: "0.1.0")  // product: MLXEmbeddingGemma2
```

```swift
import MLXEmbeddingGemma2

let session = try await EmbeddingSession.load()

let query = try session.embed(.text("Which planet is known as the Red Planet?", prompt: EmbeddingPrompt.query))
let docs = try session.embed([
    .text("Mars is known as the Red Planet.", prompt: EmbeddingPrompt.document),
    EmbeddingInput(prompt: EmbeddingPrompt.document, [.image(try MediaLoader.image(photoURL))]),
    EmbeddingInput(prompt: EmbeddingPrompt.document, [.text("A photo of a cat"), .image(cat)]),  // mixed
    EmbeddingInput([.audio(try await MediaLoader.audio(wavURL))]),
    EmbeddingInput([.video(try await MediaLoader.video(movURL))]),
], dimension: 256)
let ranked = EmbeddingSession.rank(query: query, candidates: docs)
```

- **Large jobs.** For thousands of items, use the streaming form so only a lookahead window is
  ever decoded:
  ```swift
  try await session.embed(count: urls.count, load: { i in
      EmbeddingInput(prompt: EmbeddingPrompt.document, [try await MediaLoader.content(urls[i])])
  }) { index, result in /* completion order; Result per item */ }
  ```
  - Batches are packed by budget, not count: 16 inputs, 8,192 padded text tokens, 8 vision frames
    and 2 minutes of audio per pass, all set in `EmbeddingSessionConfiguration`. Each window is
    sorted by modality and length, so padding stays small.
  - Preprocessing of the next batch overlaps the GPU, because the forward pass needs no GPU→CPU
    syncs.
  - A failing input gets an error for its own index; the run continues.
- **Prompts.** Use `EmbeddingPrompt.query` for queries and `.document` for indexed items. The full
  catalog is in `config_sentence_transformers.json`.
- **Same width for everything you compare.** Queries and documents must be embedded at the same
  `dimension:`.
- **Materialized media.** `EmbeddingSession` takes decoded media, never paths: `RGBImage`, 16 kHz
  mono `[Float]`, or a `VideoClip` of frames. `MediaLoader` decodes files with ImageIO and
  AVFoundation; videos are sampled at 1 fps with at most 32 frames.
- **Image budget.** Images use 280 soft tokens by default; set
  `EmbeddingSessionConfiguration.imageSoftTokens` to 70/140/560/1120 to change it.
- **Text only.** `towers: []` skips loading the vision and audio towers.
- **Precision.** Keep the default bf16 weights. The model card warns against float16.

## CLI and app

```sh
make run-cli ARGS='search -q "text:a red planet" "text:Mars is red." photo.jpg talk.wav clip.mov'
make run-cli ARGS='embed --dim 256 "text:hello" photo.jpg'   # JSON lines
make run-cli ARGS='embed --items-file items.txt' > out.jsonl  # thousands: one ITEM per line, streamed
make bench
make run-app INPUT=photo.jpg                                 # SwiftUI corpus search
```

**CLI items.** An item is a file path, or literal text written as `text:…`. Join parts with `+` to
embed them together, e.g. `"text:a cat+cat.jpg"`.

**App.** The app (`Examples/EmbeddingGemma2App`, a JSON-format Xcode project) is a corpus search
window:

- Drop or add files, whole folders or text, then search with the toolbar field. Files are decoded
  only when their batch runs, and results stream in with a progress count. A corpus of 1,000
  photos embedded in about 50 s.
- Use **Find Similar** (⌘L) on any item.
- Switch the Matryoshka width at any time.

Both frontends drive the same `EmbeddingSession`.

## Parity

`make test` runs `xcodebuild test`, which needs the checkpoint in the HF cache. The parity fixtures are
not checked in. On first run, `make test` generates them into `Tests/Fixtures/` with
`Tools/gen_fixtures.py`, which runs the mlx-vlm reference and the transformers processor through `uv`
(about 1 minute plus the Python environment install). `make fixtures` regenerates them. Running
`xcodebuild test` directly without fixtures skips the parity tests.

| Case | Model on reference inputs | Full Swift pipeline from raw media |
|---|---|---|
| 6 multilingual texts (padded batch) | cos ≥ 0.9999 | token ids identical, cos ≥ 0.999 |
| image, text + image | cos ≥ 0.999 | ids/positions identical, pixels ≤ 1/255, cos ≥ 0.999 |
| audio | cos ≥ 0.999 | log-mel max \|Δ\| 1.8e-3 (float32 FFT), cos ≥ 0.999 |
| video (3 sampled frames) | cos ≥ 0.999 | ids identical, frame sampling matches, cos ≥ 0.999 |
| video file (H.264 `.mp4`, AVFoundation vs PyAV decode) | n/a | same 4 frames, ids identical, cos 0.9990 |
| heterogeneous batch (text, image, audio) | cos ≥ 0.999 | cos ≥ 0.999 |

The text retrieval example scores Venus 0.677 and Mars 0.853, the same as the reference.

**Image features.** Per-token image features agree to cos ≥ 0.995 rather than 0.9999. The matmuls
are bit-identical, but the SwiftPM-built and Python-wheel Metal kernels round a handful of
`rsqrt`/`pow` results differently, and 16 bf16 layers amplify that. Pooled embeddings are
unaffected.

## Performance

Measured on an Apple M5 Max, in bf16, as the median of 20 runs (`make bench`). Swift times include
preprocessing. The Python reference is mlx-vlm 3d87e884 with the transformers processor.

| Case | Swift total | Swift forward | Python total | Python forward |
|---|---:|---:|---:|---:|
| text 128 tok ×1 | 7.1 ms | 6.7 ms | 6.7 ms | 6.5 ms |
| text 128 tok ×8 | 16.3 ms | 15.0 ms | 15.7 ms | 14.9 ms |
| text 512 tok ×1 | 11.9 ms | 10.5 ms | 10.3 ms | 9.9 ms |
| text 512 tok ×8 | 54.4 ms | 48.9 ms | 47.3 ms | 46.0 ms |
| image 640×480 | 45.9 ms | 43.3 ms | 48.5 ms | 45.8 ms |
| audio 10 s | 28.4 ms | 27.9 ms* | 25.2 ms | 23.0 ms |

\* The Swift log-mel features are computed lazily on the GPU, so they are counted in "forward".

**Throughput** (streaming, mixed inputs):

| Workload | Result |
|---|---|
| 2,000 texts, ~10–260 tokens | ~630–650 texts/s |
| 128 images (640×480) | ~23 images/s, about the 38 ms/image vision-tower compute floor |
| 64 audio clips (10 s each) | ~64–70 clips/s |
| CLI, 4,003 items (3,000 texts + 1,000 JPEGs + 3 bad entries) | 50.7 s end to end, 2.27 GB peak RSS |

In the CLI run, the bad entries were reported on stderr and the run continued. Larger vision
batches were slower: 32 images per pass overflowed the 512 MB buffer cache, giving 63 ms/image.

The remaining text gap is tokenization: swift-transformers BPE is slower than the Rust tokenizer,
even when run in parallel across inputs.

## Docs

`make docs` builds the DocC site into `docs/`; `make docs-preview` serves a local preview.

## Licenses

- Model weights: Apache-2.0 (Google).
- Reference implementations: mlx-vlm (MIT) and transformers (Apache-2.0).
