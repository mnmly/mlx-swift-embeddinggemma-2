# Lessons

1. **Don't check out tags in a shared sibling clone to read an API.**
   - Failure: ran `git checkout 0.32.3` in `../mlx-swift` (the user's working clone) to read sources; it
     left the clone on another commit with dirty submodules. It was restored right away.
   - Detection: `git status` shows `M Source/Cmlx/mlx`.
   - Prevention: read other versions with `git show <tag>:<path>`, or from `.build/checkouts`.
2. **SwiftUI app launched with path arguments shows no window.**
   - Failure: `open -n App.app --args file.png` produced only menu-bar strips. AppKit turns argv paths
     into open-file events, and SwiftUI then skips the default window.
   - Detection: `CGWindowListCopyWindowInfo` lists only 33-pt strips for the pid.
   - Prevention: pass launch input via an env var (`open --env KEY=…`).
3. **Bisect bf16 drift before blaming the port.**
   - Failure: vision features were at cos 0.9986 against the reference.
   - Detection: per-stage dumps showed the GEMMs bit-identical, with ~10 of 1.8M elements differing
     after the first float32 RMSNorm. Swift's `pow` doesn't reproduce Python's either: these are Metal
     kernel builds, not the port.
   - Prevention: dump per-stage tensors and compare stage by stage. Accept the transcendental ulp
     noise, and judge parity on pooled embeddings.
