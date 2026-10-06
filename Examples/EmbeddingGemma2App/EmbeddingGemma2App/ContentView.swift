import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Media the corpus accepts (matches the file importer and the drop target).
let supportedMediaTypes: [UTType] = [.image, .audio, .movie]

struct ContentView: View {
    @Bindable var model: SearchModel
    @State private var isDropTargeted = false
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        corpus
            .safeAreaInset(edge: .bottom, spacing: 0) { statusBar }
            .navigationTitle("Corpus")
            .navigationSubtitle(subtitle)
            .searchable(text: $model.query, placement: .toolbar, prompt: "Search the corpus")
            // Search as you type (debounced); Return searches immediately.
            .task(id: model.query) {
                try? await Task.sleep(for: .milliseconds(350))
                if !Task.isCancelled { await model.search() }
            }
            .onSubmit(of: .search) { Task { await model.search() } }
            .toolbar { toolbar }
            .onChange(of: model.dimension) { model.rescore() }
            .onAppear { model.undoManager = undoManager }
            .onChange(of: undoManager) { model.undoManager = undoManager }
            .fileImporter(
                isPresented: $model.isImporting, allowedContentTypes: supportedMediaTypes + [.folder],
                allowsMultipleSelection: true
            ) { result in
                if case .success(let urls) = result { Task { await model.addFiles(urls) } }
            }
            .sheet(isPresented: $model.isAddingText) { AddTextSheet(model: model) }
            .onDrop(of: [.fileURL, .plainText], delegate: CorpusDropDelegate(model: model, isTargeted: $isDropTargeted))
            .overlay {
                if isDropTargeted {
                    RoundedRectangle(cornerRadius: 8).strokeBorder(.tint, lineWidth: 3).padding(4)
                        .allowsHitTesting(false)
                }
            }
            .alert(
                "Something Went Wrong",
                isPresented: Binding(get: { model.alertMessage != nil }, set: { if !$0 { model.alertMessage = nil } })
            ) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(model.alertMessage ?? "")
            }
            .confirmationDialog(
                "Remove all \(model.items.count) items from the corpus?", isPresented: $model.isConfirmingClear
            ) {
                Button("Clear Corpus", role: .destructive) { model.clear() }
            } message: {
                Text("You can undo this with Edit › Undo.")
            }
            .frame(minWidth: 520, minHeight: 360)
    }

    private var subtitle: String {
        let n = model.items.count
        return n == 0 ? "Empty" : "\(n) item\(n == 1 ? "" : "s")" + (model.rankedBy.map { " · ranked by \($0)" } ?? "")
    }

    @ViewBuilder private var corpus: some View {
        if model.items.isEmpty {
            ContentUnavailableView {
                Label("No Items", systemImage: "square.and.arrow.down.on.square")
            } description: {
                Text("Drop images, audio, video, folders or text here, or choose File › Add Files….")
            } actions: {
                Button("Add Files…") { model.isImporting = true }
                Button("Add Text…") { model.isAddingText = true }
            }
        } else {
            List(selection: $model.selection) {
                ForEach(model.items) { item in
                    ItemRow(item: item)
                        .contextMenu { contextMenu(for: item) }
                }
            }
            // Provides Edit › Delete and the Delete key.
            .onDeleteCommand { model.deleteSelection() }
        }
    }

    /// Shows only actions that apply to the item.
    @ViewBuilder private func contextMenu(for item: CorpusItem) -> some View {
        if item.embedding != nil {
            Button("Find Similar") {
                model.selection = [item.id]
                model.findSimilarToSelection()
            }
            Button("Copy Embedding") {
                if let json = model.embeddingJSON(for: [item.id]) { copy(json) }
            }
            Divider()
        }
        Button("Delete", role: .destructive) {
            model.selection = [item.id]
            model.deleteSelection()
        }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem(placement: .principal) {
            Picker("Width", selection: $model.dimension) {
                ForEach([128, 256, 512, 768], id: \.self) { Text("\($0)-d").tag($0) }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(width: 90)
            .help("Embedding width used for scoring (Matryoshka)")
            .accessibilityLabel("Embedding width")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            Button { model.isImporting = true } label: { Label("Add Files…", systemImage: "plus") }
                .help("Add images, audio, video or whole folders (⌘O)")
            Button { model.isAddingText = true } label: { Label("Add Text…", systemImage: "text.badge.plus") }
                .help("Add a text passage")
            Button { model.findSimilarToSelection() } label: {
                Label("Find Similar", systemImage: "sparkle.magnifyingglass")
            }
            .disabled(!model.canFindSimilar)
            .help("Rank the corpus by similarity to the selected item (⌘L)")
        }
    }

    private var statusBar: some View {
        HStack(spacing: 6) {
            switch model.status {
            case .loading:
                ProgressView().controlSize(.small)
                Text("Loading EmbeddingGemma 2… (downloads ~1.5 GB on first launch)")
            case .working(let message):
                ProgressView().controlSize(.small)
                Text(message)
            case .ready:
                Image(systemName: "checkmark.circle").accessibilityHidden(true)
                Text("Ready")
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange).accessibilityHidden(true)
                Text(message).lineLimit(2).textSelection(.enabled).foregroundStyle(.primary)
            }
            Spacer()
        }
        .font(.callout)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }
}

/// Accepts supported media files and plain text; rejects everything else before the drop.
struct CorpusDropDelegate: DropDelegate {
    let model: SearchModel
    @Binding var isTargeted: Bool

    func validateDrop(info: DropInfo) -> Bool {
        info.hasItemsConforming(to: supportedMediaTypes + [.folder, .plainText])
    }

    func dropEntered(info: DropInfo) { isTargeted = validateDrop(info: info) }
    func dropExited(info: DropInfo) { isTargeted = false }

    func performDrop(info: DropInfo) -> Bool {
        isTargeted = false
        let files = info.itemProviders(for: [.fileURL]).filter { provider in
            provider.registeredContentTypes.contains { type in
                (supportedMediaTypes + [.folder]).contains { type.conforms(to: $0) }
            }
        }
        for provider in files {
            // Decoding happens right away in `addFiles`, while the drag grant still holds.
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in await model.addFiles([url]) }
            }
        }
        let texts = files.isEmpty ? info.itemProviders(for: [.plainText]) : []
        for provider in texts {
            _ = provider.loadObject(ofClass: String.self) { text, _ in
                guard let text else { return }
                Task { @MainActor in await model.addText(text) }
            }
        }
        return !files.isEmpty || !texts.isEmpty
    }
}

func copy(_ string: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(string, forType: .string)
}

struct ItemRow: View {
    let item: CorpusItem

    var body: some View {
        HStack(spacing: 10) {
            preview
                .frame(width: 44, height: 44)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).lineLimit(2).truncationMode(.middle)
                Text("\(item.kind.rawValue.capitalized) · \(item.detail)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let score = item.score {
                Text(score, format: .number.precision(.fractionLength(3)))
                    .font(.body.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else if let failure = item.failure {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .help(failure)
            } else if item.embedding == nil {
                ProgressView().controlSize(.small)
            }
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(item.kind.rawValue.capitalized), \(item.title), \(item.detail)")
        .accessibilityValue(accessibilityValue)
    }

    private var accessibilityValue: String {
        if let failure = item.failure { return "Couldn’t embed: \(failure)" }
        if item.embedding == nil { return "Embedding…" }
        return item.score.map { String(format: "Similarity %.3f", $0) } ?? "Not ranked"
    }

    @ViewBuilder private var preview: some View {
        if let thumbnail = item.thumbnail {
            Image(decorative: thumbnail, scale: 1).resizable().scaledToFill()
        } else {
            ZStack {
                Rectangle().fill(.quaternary)
                Image(systemName: symbol).font(.title3).foregroundStyle(.secondary)
            }
        }
    }

    private var symbol: String {
        switch item.kind {
        case .text: "text.alignleft"
        case .image: "photo"
        case .audio: "waveform"
        case .video: "film"
        }
    }
}

struct AddTextSheet: View {
    let model: SearchModel
    @State private var text = ""
    @State private var isConfirmingDiscard = false
    @Environment(\.dismiss) private var dismiss

    private var isEmpty: Bool { text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add Text").font(.headline)
            TextEditor(text: $text)
                .font(.body)
                .frame(minWidth: 380, minHeight: 160)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.separator))
                .accessibilityLabel("Passage text")
            HStack {
                Text("Press ⌘Return to add.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel", role: .cancel) {
                    if isEmpty { dismiss() } else { isConfirmingDiscard = true }
                }
                .keyboardShortcut(.cancelAction)
                // Return inserts a newline in the editor, so Add uses ⌘Return.
                Button("Add") {
                    let value = text
                    dismiss()
                    Task { await model.addText(value) }
                }
                .keyboardShortcut(.return, modifiers: .command)
                .buttonStyle(.borderedProminent)
                .disabled(isEmpty)
            }
        }
        .padding(20)
        .confirmationDialog("Discard this text?", isPresented: $isConfirmingDiscard) {
            Button("Discard", role: .destructive) { dismiss() }
        }
    }
}
