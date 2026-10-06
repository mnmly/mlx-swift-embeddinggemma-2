# mlx-swift-embeddinggemma2

EmbeddingGemma 2 (text/image/audio/video embeddings) on mlx-swift. `EmbeddingSession` (library) is the
shared driver for the CLI (`Examples/embeddinggemma2`) and the SwiftUI app (`Examples/EmbeddingGemma2App`,
JSON `project.xcproj`). Keep model, preprocessing and hyperparameters in the library; frontends only
decode files, loop and present.

- Network: `TextEncoder.swift` / `VisionTower.swift` / `AudioTower.swift` / `EmbeddingGemma2Model.swift`
  port mlx-vlm `embedding_gemma2` + `gemma4/{vision,audio}.py` at revision 3d87e884 op-for-op, including
  dtype flow (the audio tower promotes to float32 exactly like the reference). Re-diff against it on
  upstream changes; don't "simplify" ops without re-running parity (e.g. `x*x` vs `x**2` changes rounding).
- Preprocessing: `Processing/` ports transformers `EmbeddingGemma2Processor`, `Gemma4ImageProcessor`
  (antialiased bicubic on uint8, PIL fixed-point), `Gemma4AudioFeatureExtractor`,
  `EmbeddingGemma2VideoProcessor`. Text segments between media are tokenized as one string, like the
  chat template.
- Large jobs go through `EmbeddingSession.embed(count:dimension:load:onResult:)`: lookahead window
  (lazy load + `processor.prepare` in parallel), budget packing (`pack`), `collate`, `asyncEval`
  pipelining. Keep the forward pass free of GPU→CPU syncs: media soft-token counts travel on
  `EmbeddingGemma2Inputs.*TokenCounts` (computed host-side in `prepare`). Batch defaults were
  measured (README); re-measure with `make bench` before changing them.
- Non-parameter `MLXArray` ivars on `Module`s must be `_`-prefixed (else `update(verify: .all)` fails).
- Build/test with `xcodebuild` (`make test`); `swift test` can't load MLX's Metal library. Quick compile:
  `xcrun swift build`.
- Parity fixtures are generated, not committed (`Tests/Fixtures/`, gitignored): `make test` creates them
  on first run, `make fixtures` regenerates (`cd Tools && uv run gen_fixtures.py`; never pip/python3).
  Tests read them via `fixturesDirectory` (#filePath-relative) and `XCTSkip` when missing.
- App launch input goes through `EMBEDDINGGEMMA2_INPUT` (`open --env`), not argv: AppKit treats path
  arguments as open-file events, which suppress the SwiftUI window.

## Documentation

`MLXEmbeddingGemma2` ships DocC-generated reference docs (see
`Sources/MLXEmbeddingGemma2/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public symbols are published** to the static site and (if `EMIT_LLMS_TXT=1`
is used) into `docs/llms.txt`.

When you add or modify a `public` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if the *why* is non-obvious.
- Document each parameter with `- Parameter name:` (use the **internal** name when there's an
  external label).
- Cross-reference related symbols with double-backtick links. DocC link syntax is signature-sensitive,
  and overloaded names (e.g. `embed`) need disambiguation or plain code spans.
- Add new top-level symbols under the appropriate `## Topics` group in
  `Sources/MLXEmbeddingGemma2/Documentation.docc/MLXEmbeddingGemma2.md` (grouped by user task).

Verify with `Scripts/build_docs.sh`: exit 0 and no new warnings from your changes.
