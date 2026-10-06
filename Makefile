# Build and run the CLI / SwiftUI frontends. Both drive the same library-side EmbeddingSession.
# xcodebuild (not `swift build`) so MLX's Metal shaders are compiled.
#
#   make run-cli ARGS='search -q "text:red planet" "text:Mars…" photo.jpg'
#   make run-app [INPUT=path]
#   make test            (generates parity fixtures on first run; needs uv)
#   make docs | docs-preview

CLI_SCHEME     ?= embeddinggemma2
APP_PROJECT    ?= Examples/EmbeddingGemma2App/EmbeddingGemma2App.xcodeproj
APP_SCHEME     ?= EmbeddingGemma2App
APP_DERIVED    ?= .xcdd-app
PACKAGE_SCHEME ?= mlx-swift-embeddinggemma2-Package
CONFIG   ?= Release
DERIVED  ?= .xcdd-rel
INPUT    ?=
ARGS     ?= --help
PRODUCTS := $(DERIVED)/Build/Products/$(CONFIG)
XCODEBUILD = xcodebuild -configuration $(CONFIG) -destination 'platform=macOS,arch=arm64' -derivedDataPath $(DERIVED)

.PHONY: all build build-cli build-app run-cli run-app bench test fixtures docs docs-preview clean

all: build

build: build-cli build-app

build-cli:
	$(XCODEBUILD) -scheme $(CLI_SCHEME) build -quiet

build-app:
	xcodebuild -project $(APP_PROJECT) -scheme $(APP_SCHEME) -configuration $(CONFIG) \
		-destination 'platform=macOS,arch=arm64' -derivedDataPath $(APP_DERIVED) build -quiet

run-cli: build-cli
	$(PRODUCTS)/$(CLI_SCHEME) $(ARGS) $(if $(INPUT),"$(INPUT)")

run-app: build-app
	open -n $(if $(INPUT),--env EMBEDDINGGEMMA2_INPUT="$(abspath $(INPUT))") $(APP_DERIVED)/Build/Products/$(CONFIG)/$(APP_SCHEME).app

bench: build-cli
	$(PRODUCTS)/$(CLI_SCHEME) bench

# Parity fixtures are generated (not checked in) from the Python reference (mlx-vlm + transformers):
# `make test` creates them on first run; `make fixtures` regenerates them.
FIXTURES := Tests/Fixtures/index.json

$(FIXTURES):
	cd Tools && uv run gen_fixtures.py

test: $(FIXTURES)
	xcodebuild -scheme $(PACKAGE_SCHEME) -destination 'platform=macOS,arch=arm64' -derivedDataPath .xcdd test

fixtures:
	cd Tools && uv run gen_fixtures.py

docs:
	Scripts/build_docs.sh

docs-preview:
	Scripts/build_docs.sh preview

clean:
	rm -rf $(DERIVED) $(APP_DERIVED) .xcdd
