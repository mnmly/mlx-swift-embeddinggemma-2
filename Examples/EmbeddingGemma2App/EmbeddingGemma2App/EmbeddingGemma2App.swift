import SwiftUI

@main
struct EmbeddingGemma2App: App {
    @State private var model = SearchModel()

    var body: some Scene {
        Window("Embedding Search", id: "main") {
            ContentView(model: model)
                .task {
                    await model.loadModel()
                    // `open -n --env EMBEDDINGGEMMA2_INPUT=a.png:b.wav EmbeddingGemma2App.app` seeds the corpus.
                    // (Not argv: AppKit treats path arguments as open-file events, which suppress the window.)
                    let paths = ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_INPUT"]?
                        .split(separator: ":").map(String.init) ?? []
                    if !paths.isEmpty { await model.addFiles(paths.map { URL(filePath: $0) }) }
                }
        }
        .defaultSize(width: 720, height: 560)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Add Files…") { model.isImporting = true }
                    .keyboardShortcut("o")
                Button("Add Text…") { model.isAddingText = true }
            }
            CommandGroup(after: .pasteboard) {
                Button("Copy Embedding") {
                    if let json = model.embeddingJSON(for: model.selection) { copy(json) }
                }
                .disabled(model.embeddingJSON(for: model.selection) == nil)
            }
            CommandMenu("Corpus") {
                Button("Search") { Task { await model.search() } }
                    .keyboardShortcut("r")
                    .disabled(!model.isReady || model.query.isEmpty)
                Button("Find Similar to Selection") { model.findSimilarToSelection() }
                    .keyboardShortcut("l")
                    .disabled(!model.canFindSimilar)
                Picker("Embedding Width", selection: $model.dimension) {
                    ForEach([128, 256, 512, 768], id: \.self) { Text("\($0)-d").tag($0) }
                }
                Divider()
                Button("Clear Corpus…") { model.isConfirmingClear = true }
                    .disabled(model.items.isEmpty)
            }
        }
    }
}
