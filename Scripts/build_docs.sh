#!/usr/bin/env bash
# Build static DocC site(s) for one or more SPM targets into ./docs
# (GitHub Pages-ready). Modeled after ml-explore/mlx-swift's
# tools/build-documentation.sh, with optional LLM-Markdown export.
#
# Usage:
#   Scripts/build_docs.sh                # build all $TARGETS into ./docs
#   Scripts/build_docs.sh preview        # local preview (first target only)
#   Scripts/build_docs.sh -f             # bypass gh-pages branch guard
#
# Required-ish env (edit defaults below or pass at call site):
#   TARGETS              Space-separated target names. Single-target packages
#                        can just set one. Default: MLXEmbeddingGemma2.
#   HOSTING_BASE_PATH    Repo name on GitHub Pages (e.g. "my-lib"). Each
#                        target is hosted at <BASE>/<Target>/ so source URLs
#                        and asset paths resolve correctly.
#   REPO_URL             https URL to the GitHub repo (no trailing slash).
#                        Enables "View on GitHub" source links per symbol.
#   REPO_BRANCH          Branch the source links point at. Default: main.
#
# Optional env:
#   OUTPUT_DIR           Default: docs
#   REQUIRE_GH_PAGES=1   Refuse to build off the gh-pages branch unless -f.
#   EMIT_MARKDOWN=1      Pass the experimental Markdown-output flags
#                        (--enable-experimental-markdown-output and
#                         --enable-experimental-markdown-output-manifest) so docc
#                        emits per-symbol .md files (+ a manifest) under
#                        <out>/<target>/. These exist only in recent swift-docc —
#                        the Xcode-bundled docc lacks them and they are probed and
#                        skipped with a warning (use a swift.org toolchain; see
#                        TOOLCHAIN below).
#   EMIT_LLMS_TXT=1      Above + concatenate the .md into <OUTPUT_DIR>/llms.txt.
#   TOOLCHAIN            Build with a specific toolchain (identifier or the alias
#                        "swift"). Honors TOOLCHAINS too. If neither is set and
#                        swiftly is installed, ~/.swiftly/env.sh is sourced so the
#                        toolchain you `swiftly use` is picked up automatically
#                        (only when EMIT_MARKDOWN / EMIT_LLMS_TXT is set).
set -euo pipefail

cd "$(dirname "$0")/.."

# If the swift-docc-plugin dependency is gated in Package.swift (the recommended
# `Context.environment[...]` pattern), set the gate var so generate-documentation
# can see the plugin. SPI_GENERATE_DOCS is the standard Swift Package Index var;
# include it in the Package.swift gate so a bare run here works unmodified.
export SPI_GENERATE_DOCS=1

TARGETS="${TARGETS:-MLXEmbeddingGemma2}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-mlx-swift-embeddinggemma-2}"
REPO_URL="${REPO_URL:-https://github.com/mnmly/mlx-swift-embeddinggemma-2}"
REPO_BRANCH="${REPO_BRANCH:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"

# Toolchain selection. The experimental Markdown flags require a recent swift-docc
# (newer than the one bundled with current Xcode). Let callers point at one without
# prefixing every invocation, and fall back to a swiftly selection if present.
if [[ -z "${TOOLCHAINS:-}" && -n "${TOOLCHAIN:-}" ]]; then
    export TOOLCHAINS="$TOOLCHAIN"
fi
# Only for the Markdown export: a plain run stays on Xcode's toolchain, since a stale swiftly
# default (e.g. Swift 6.1) can fail on dependencies that need a newer swift-tools-version.
if [[ -z "${TOOLCHAINS:-}" && -f "$HOME/.swiftly/env.sh" \
      && ( "${EMIT_MARKDOWN:-0}" == "1" || "${EMIT_LLMS_TXT:-0}" == "1" ) ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.swiftly/env.sh"
fi

# Without the Markdown export, pin Xcode's swift even if a swiftly toolchain is first on PATH.
if [[ -z "${TOOLCHAINS:-}" && "${EMIT_MARKDOWN:-0}" != "1" && "${EMIT_LLMS_TXT:-0}" != "1" ]]; then
    SWIFT=(xcrun swift)
else
    SWIFT=(swift)
fi

FORCE=0
MODE="build"
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        preview)    MODE="preview" ;;
    esac
done

if [[ "${REQUIRE_GH_PAGES:-0}" == "1" && "$MODE" == "build" && $FORCE -eq 0 ]]; then
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
    if [[ "$branch" != "gh-pages" ]]; then
        echo "Refusing to build off branch '$branch'. Use -f to override."
        exit 1
    fi
fi

export DOCC_JSON_PRETTYPRINT=YES

# Preview: first target only — `swift package preview-documentation` is
# single-target and interactive.
if [[ "$MODE" == "preview" ]]; then
    first_target="${TARGETS%% *}"
    exec "${SWIFT[@]}" package --disable-sandbox \
        preview-documentation --target "$first_target"
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Resolve the same docc the plugin will use: prefer one on PATH (swiftly puts the
# selected toolchain's binaries there), else fall back to xcrun (honors TOOLCHAINS).
# The experimental flags only exist in recent swift-docc; probe and skip-with-warning
# rather than hard-failing on an unknown option (the Xcode-bundled docc lacks them).
if [[ "${SWIFT[0]}" == "xcrun" ]]; then
    DOCC_BIN="$(xcrun --find docc 2>/dev/null || true)"
else
    DOCC_BIN="$(command -v docc 2>/dev/null || xcrun --find docc 2>/dev/null || true)"
fi
docc_supports() {
    [[ -n "$DOCC_BIN" ]] && "$DOCC_BIN" convert --help 2>&1 | grep -q -- "$1"
}

EXTRA_FLAGS=()
if [[ "${EMIT_MARKDOWN:-0}" == "1" || "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    for flag in --enable-experimental-markdown-output \
                --enable-experimental-markdown-output-manifest; do
        if docc_supports "$flag"; then
            EXTRA_FLAGS+=("$flag")
        else
            echo "warning: active docc does not support '$flag' — skipping." >&2
            echo "         Use a recent swift.org toolchain (e.g. via swiftly:" >&2
            echo "         'swiftly install main-snapshot && swiftly use main-snapshot')," >&2
            echo "         or pass TOOLCHAIN=<id>. The Xcode-bundled docc lacks it." >&2
        fi
    done
fi

# Snapshot toolchains (needed for the experimental Markdown flags) run an
# assertions-enabled swift-frontend whose debug-info round-trip verifier can crash
# during IRGen on some source files — it prints "Pass '-Xfrontend
# -disable-round-trip-debug-types' to disable this assertion" and aborts the build
# before the Markdown export runs. Disable it for the doc build; the flag is
# accepted and harmless on release toolchains, so only apply it when we actually
# requested the Markdown output (EXTRA_FLAGS non-empty).
# PLACEMENT MATTERS: `-Xswiftc` must come *before* `--allow-writing-to-directory`
# (SwiftPM rejects it after the plugin-invocation options with "Unknown option
# '-Xswiftc'"), and the swift-docc-plugin does NOT forward a *trailing* `-Xswiftc`
# to the frontend — a trailing one is silently ignored and the build still crashes.
SWIFTC_FLAGS=()
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    SWIFTC_FLAGS+=(-Xswiftc -Xfrontend -Xswiftc -disable-round-trip-debug-types)
fi

SOURCE_FLAGS=()
if [[ -n "$REPO_URL" ]]; then
    SOURCE_FLAGS+=(
        --source-service github
        --source-service-base-url "${REPO_URL%/}/blob/${REPO_BRANCH}"
        --checkout-path "$(pwd)"
    )
fi

for TARGET in $TARGETS; do
    slug="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"
    out="$OUTPUT_DIR/$TARGET"
    mkdir -p "$out"

    echo ">> Building DocC for $TARGET → $out"
    "${SWIFT[@]}" package ${SWIFTC_FLAGS[@]+"${SWIFTC_FLAGS[@]}"} \
        --allow-writing-to-directory "$out" \
        generate-documentation \
        --target "$TARGET" \
        --fallback-bundle-identifier "${HOSTING_BASE_PATH}.${slug}" \
        --output-path "$out" \
        --emit-digest \
        --disable-indexing \
        --transform-for-static-hosting \
        --hosting-base-path "${HOSTING_BASE_PATH}/${TARGET}" \
        ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
done

if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    LLMS="$OUTPUT_DIR/llms.txt"
    {
        echo "# ${HOSTING_BASE_PATH} — DocC export for LLM consumption"
        echo
        echo "Generated $(date -u +%FT%TZ) from swift-docc."
        echo "Targets: $TARGETS"
        echo
        for TARGET in $TARGETS; do
            find "$OUTPUT_DIR/$TARGET/data" -name '*.md' -type f 2>/dev/null \
                | sort \
                | while IFS= read -r f; do
                    # Skip synthesized / inherited members — DocC marks them with
                    # ::SYNTHESIZED:: in the preciseIdentifier. For SwiftUI/AppKit-
                    # heavy modules these dominate the output (the whole `View`
                    # modifier surface replicated onto every view type); dropping
                    # them keeps llms.txt to the package's own declared API — a
                    # ~75-80% cut in practice. Set KEEP_SYNTHESIZED=1 to include them.
                    if [[ "${KEEP_SYNTHESIZED:-0}" != "1" ]] && grep -q '::SYNTHESIZED::' "$f"; then
                        continue
                    fi
                    rel="${f#$OUTPUT_DIR/}"
                    echo
                    echo "---"
                    echo "## $rel"
                    echo
                    cat "$f"
                done
        done
    } > "$LLMS"
    echo "Wrote $LLMS ($(wc -l < "$LLMS" | tr -d ' ') lines)."
fi

# Write a top-level redirect index.html so the Pages root URL lands on the
# first target's documentation instead of returning 404. The DocC build emits
# everything under $OUTPUT_DIR/$TARGET/; without this redirect,
# https://<user>.github.io/<HOSTING_BASE_PATH>/ has no index and 404s even on
# successful deploys. Confirmed on SwiftSTAC (mnmly.github.io/SwiftSTAC/).
first_target="${TARGETS%% *}"
first_slug="$(echo "$first_target" | tr '[:upper:]' '[:lower:]')"
redirect_url="/${HOSTING_BASE_PATH}/${first_target}/documentation/${first_slug}/"
cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>${HOSTING_BASE_PATH}</title>
<meta http-equiv="refresh" content="0; url=${redirect_url}">
<link rel="canonical" href="${redirect_url}">
<p>Redirecting to <a href="${redirect_url}">${redirect_url}</a>.</p>
HTML

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/<Target>/index.html"
echo "or push to gh-pages."
