// RAGVectorDatabase.swift
// ============================================================================
// Generic VecturaKit-backed vector database engine: on-disk setup and seeding,
// embedding-engine selection, batched document embedding, and snapshot
// export/import. Domain-agnostic — hosts supply seed archives, model folders,
// and document texts, and keep their own document metadata keyed by document
// ID.
//
// Not thread-safe on its own: designed to be owned by a single actor (or other
// single concurrency domain) in the host app.
// ============================================================================

import Foundation
import VecturaEmbeddingsKit
import VecturaKit
import ZIPFoundation

// MARK: - Configuration

public struct RAGVectorDatabaseConfiguration: Sendable {
    /// VecturaKit database name; storage lives in a subdirectory with this name.
    public var name: String
    /// Embedding dimension of the sentence-transformer model below. Declared
    /// so setup never loads the model to measure it. Other engines report
    /// their own.
    public var dimension: Int
    /// Application Support subdirectory that holds the writable database.
    public var storageRootFolderName: String
    /// Folder name for the database on disk (e.g. "activities.vecturadb").
    public var databaseFolderName: String
    /// Optional bundled ZIP used to seed the database on first launch.
    public var bundledSeedZipURL: URL?
    /// Optional bundled folder used to seed the database when no ZIP exists.
    public var bundledSeedFolderURL: URL?
    /// Optional local embedding-model folder. Falls back to `remoteModelID`
    /// when nil or missing any of `requiredLocalModelFiles`.
    public var localModelFolderURL: URL?
    /// Files that must exist inside `localModelFolderURL` for it to be used.
    public var requiredLocalModelFiles: [String]
    /// Remote model repository ID used when no valid local model folder exists.
    /// It also names the sentence transformer's vector space.
    public var remoteModelID: String
    /// The engine documents and queries are embedded with. The model fields
    /// above only matter for `.sentenceTransformer`, the default.
    ///
    /// The database remembers which engine wrote its vectors. Opening it with
    /// an engine that embeds into a different space clears it, and the host
    /// re-embeds its corpus. A bundled seed from another engine is skipped the
    /// same way.
    public var embeddingEngine: RAGEmbeddingEngine

    public init(
        name: String,
        dimension: Int,
        storageRootFolderName: String = "VectorDB",
        databaseFolderName: String,
        bundledSeedZipURL: URL? = nil,
        bundledSeedFolderURL: URL? = nil,
        localModelFolderURL: URL? = nil,
        requiredLocalModelFiles: [String] = [],
        remoteModelID: String,
        embeddingEngine: RAGEmbeddingEngine = .sentenceTransformer
    ) {
        self.name = name
        self.dimension = dimension
        self.storageRootFolderName = storageRootFolderName
        self.databaseFolderName = databaseFolderName
        self.bundledSeedZipURL = bundledSeedZipURL
        self.bundledSeedFolderURL = bundledSeedFolderURL
        self.localModelFolderURL = localModelFolderURL
        self.requiredLocalModelFiles = requiredLocalModelFiles
        self.remoteModelID = remoteModelID
        self.embeddingEngine = embeddingEngine
    }
}

// MARK: - Document

/// A document to embed: a stable ID, the text that gets vectorized, and the
/// date and filter tags the document belongs to. Hosts keep any richer
/// metadata in their own lookup keyed by `id`.
public struct RAGDocument: Sendable {
    public let id: UUID
    public let text: String
    /// The document's own date — a note's capture time, an event's start —
    /// stored next to the vector so searches can narrow the corpus by date
    /// *before* ranking it. `nil` falls back to the moment of indexing, which
    /// is only the same thing for a corpus that is never re-embedded.
    public let date: Date?
    /// Filter labels stored beside the vector — a note's hashtags, an event's
    /// categories. Matched verbatim at search time, so hosts canonicalize them
    /// (case, diacritics, prefixes) the same way on both sides.
    public let tags: [String]

    public init(id: UUID, text: String, date: Date? = nil, tags: [String] = []) {
        self.id = id
        self.text = text
        self.date = date
        self.tags = tags
    }
}

/// What the index currently holds for one document: the text that was embedded
/// and the date and tags stored with its vector. Hosts diff against this to
/// re-embed only the documents whose text, date, or tags actually changed.
public struct RAGIndexedDocument: Sendable, Equatable {
    public let text: String
    public let date: Date
    public let tags: [String]

    public init(text: String, date: Date, tags: [String] = []) {
        self.text = text
        self.date = date
        self.tags = tags
    }
}

// MARK: - Engine

/// `@unchecked Sendable` carries the class's documented contract into the type
/// system: the engine has no internal synchronization, and the host keeps it
/// safe by owning it from a single actor (which Swift 6 callers could not even
/// express against a non-Sendable class).
public final class RAGVectorDatabase: @unchecked Sendable {
    /// Changes only through `switchEmbeddingEngine(to:)`.
    public private(set) var configuration: RAGVectorDatabaseConfiguration

    public private(set) var database: VecturaKit?
    public private(set) var directoryURL: URL?
    /// Incremented after snapshot imports and engine switches so UIs can reload.
    public private(set) var revision: Int = 0

    /// Kept from `setUp` so filtered searches can rank a subset of the
    /// corpus with the same embedder and settings as an unfiltered one.
    private var embedder: (any RAGEmbedder)?
    private var vecturaConfig: VecturaConfig?
    private var storage: RAGDateStampingStorage?
    private var tagStore: RAGDocumentTagStore?

    public init(configuration: RAGVectorDatabaseConfiguration) {
        self.configuration = configuration
    }

    /// The vector space of the open database (see `RAGEmbedder`), or `nil`
    /// before setup.
    public var embeddingSpaceIdentifier: String? {
        embedder?.spaceIdentifier
    }

    // MARK: - Setup

    /// Destination directory for the writable database.
    public func databaseDestinationDirectory() throws -> URL {
        let fm = FileManager.default
        let supportDir = try fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let root = supportDir.appendingPathComponent(configuration.storageRootFolderName, isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent(configuration.databaseFolderName, isDirectory: true)
    }

    public func setUp(forceReset: Bool = false) async throws {
        do {
            let engine = configuration.embeddingEngine
            let embedder = try makeEmbedder(for: engine)
            let dimension = try await vectorDimension(of: embedder, engine: engine)
            try await open(with: embedder, dimension: dimension, forceReset: forceReset)
            RAGLog.debug("✅ Successfully initialized VecturaKit")
        } catch {
            RAGLog.error("❌ Failed to initialize VecturaKit: \(error)")
            throw error
        }
    }

    /// Re-opens the database on another embedding engine, for hosts that let
    /// the user choose one.
    ///
    /// An engine that embeds into a different vector space cannot answer
    /// queries against the stored vectors, so the database is cleared:
    /// `needsEmbedding()` then reports `true`, and a host that diffs against
    /// `indexedDocuments()` finds every document missing and re-embeds it.
    /// An engine in the same space keeps the vectors.
    ///
    /// The engine is built first, so one that cannot run on this device (Apple
    /// languages none of which has a model) throws with the database
    /// untouched. A failure after that (disk, storage) leaves the database
    /// closed and still configured for the old engine, which `setUp()`
    /// re-opens. Like `setUp` and `importSnapshot`, call it while nothing else
    /// is using the database.
    public func switchEmbeddingEngine(to engine: RAGEmbeddingEngine) async throws {
        do {
            let embedder = try makeEmbedder(for: engine)
            let dimension = try await vectorDimension(of: embedder, engine: engine)
            let previousSpace = self.embedder?.spaceIdentifier
            try await open(with: embedder, dimension: dimension, forceReset: false)
            configuration.embeddingEngine = engine
            if embedder.spaceIdentifier != previousSpace {
                revision += 1
            }
            RAGLog.debug("🔀 Embedding with \(embedder.spaceIdentifier)")
        } catch {
            RAGLog.error("❌ Failed to switch embedding engine: \(error)")
            throw error
        }
    }

    /// Opens the database directory for `embedder`: vectors from another
    /// vector space are cleared (and a seed from one skipped) before anything
    /// loads.
    private func open(with embedder: any RAGEmbedder, dimension: Int, forceReset: Bool) async throws {
        // Closed first. A failure below then leaves nothing open, rather than
        // the previous engine's handle writing its vectors into a directory
        // that may already be recorded as the new engine's.
        database = nil
        self.embedder = nil
        vecturaConfig = nil
        storage = nil
        tagStore = nil

        let space = RAGEmbeddingSpaceRecord(identifier: embedder.spaceIdentifier, dimension: dimension)
        let dbDir = try prepareSeedDirectory(forceReset: forceReset, space: space)
        self.directoryURL = dbDir

        let config = try VecturaConfig(
            name: configuration.name,
            directoryURL: dbDir,
            dimension: dimension
        )

        // VecturaKit would build this provider itself, at exactly this
        // path and with these permissions; RAGKit builds it so it can wrap
        // it and stamp host dates onto the records as they are written.
        let storageDirectory = dbDir.appendingPathComponent(configuration.name, isDirectory: true)
        if !FileManager.default.fileExists(atPath: storageDirectory.path) {
            try FileManager.default.createDirectory(
                at: storageDirectory,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
        }
        let fileStorage = try FileStorageProvider(storageDirectory: storageDirectory, cacheEnabled: true)
        let storage = RAGDateStampingStorage(base: fileStorage)

        self.embedder = embedder
        self.vecturaConfig = config
        self.storage = storage
        // Lives at the directory root (not inside VecturaKit's storage
        // subdirectory, whose loader decodes every .json as a document),
        // so snapshot export/import carries it with the vectors.
        self.tagStore = RAGDocumentTagStore(
            fileURL: dbDir.appendingPathComponent("document-tags.json")
        )
        self.database = try await VecturaKit(config: config, embedder: embedder, storageProvider: storage)
        RAGLog.debug("📁 Vectura DB directory: \(dbDir.path)")

        let fm = FileManager.default
        do {
            let contents = try fm.contentsOfDirectory(at: dbDir, includingPropertiesForKeys: [.fileSizeKey], options: .skipsHiddenFiles)
            let totalBytes = try contents.reduce(0 as UInt64) { acc, url in
                let vals = try url.resourceValues(forKeys: [.fileSizeKey])
                return acc &+ UInt64(vals.fileSize ?? 0)
            }
            let mb = Double(totalBytes) / (1024 * 1024)
            RAGLog.debug("📊 Approx DB size: \(String(format: "%.2f", mb)) MB")
        } catch {
            RAGLog.warning("⚠️ Could not compute DB dir size: \(error)")
        }
    }

    // MARK: - Search

    /// Ranks the corpus against `query`, optionally restricted to documents
    /// whose stored date falls inside `dateRange` and/or that carry at least
    /// one of `tags` (compared verbatim against the tags they were indexed
    /// with).
    ///
    /// Both are *pre* filters: they narrow the corpus before ranking, so a
    /// query about one week or one tag returns that slice's best matches
    /// instead of whatever is left after filtering a global top-K — which,
    /// for a narrow slice over a large corpus, is usually nothing. Results
    /// carry the stored date as `createdAt`.
    ///
    /// - Note: A filtered search builds a throwaway engine over the matching
    ///   slice, so its text index is rebuilt per call. That is proportional to
    ///   the slice, not the corpus, and unfiltered searches are untouched.
    public func search(
        query: String,
        numResults: Int,
        threshold: Float? = nil,
        dateRange: Range<Date>? = nil,
        tags: [String]? = nil
    ) async throws -> [VecturaSearchResult] {
        guard let database else {
            throw RAGError.notInitialized
        }
        let tagFilter = (tags?.isEmpty == false) ? tags : nil
        guard dateRange != nil || tagFilter != nil else {
            return try await database.search(
                query: .text(query),
                numResults: numResults,
                threshold: threshold
            )
        }
        guard let embedder, let vecturaConfig else {
            throw RAGError.notInitialized
        }

        var inRange = try await database.getAllDocuments()
        if let dateRange {
            inRange = inRange.filter { dateRange.contains($0.createdAt) }
        }
        if let tagFilter {
            let tagged = await tagStore?.ids(withAnyOf: tagFilter) ?? []
            inRange = inRange.filter { tagged.contains($0.id) }
        }
        guard !inRange.isEmpty else {
            RAGLog.debug("🔎 No documents match the requested filters")
            return []
        }

        let scoped = try await VecturaKit(
            config: vecturaConfig,
            embedder: embedder,
            storageProvider: RAGDocumentSubsetStorage(documents: inRange)
        )
        let results = try await scoped.search(
            query: .text(query),
            numResults: numResults,
            threshold: threshold
        )

        // The text engine keeps its own copy of each document's timestamp,
        // taken before storage stamped the host date on it, so read the date
        // back from the records that were actually filtered.
        let dates = Dictionary(inRange.map { ($0.id, $0.createdAt) }) { _, latest in latest }
        return results.map { result in
            VecturaSearchResult(
                id: result.id,
                text: result.text,
                score: result.score,
                createdAt: dates[result.id] ?? result.createdAt
            )
        }
    }

    // MARK: - Embedding

    /// Embeds arbitrary text with the database's own model — the same vector a
    /// document with this text would be indexed under. Hosts use it to derive
    /// features from positions in the embedding space (similarity, colors)
    /// without writing anything into the index.
    public func embedText(_ text: String) async throws -> [Float] {
        guard let embedder else {
            throw RAGError.notInitialized
        }
        return try await embedder.embed(text: text)
    }

    /// Checks if the database needs embedding (is empty or missing).
    public func needsEmbedding() async -> Bool {
        guard let database else { return true }
        do {
            // Read storage directly rather than running a probe search, which
            // would force the embedding model to load (and possibly download).
            return try await database.getAllDocuments().isEmpty
        } catch {
            RAGLog.warning("⚠️ Could not check vector DB content: \(error)")
            return true
        }
    }

    /// Resets the database and embeds the given documents in batches.
    /// Individual document failures are logged and skipped.
    /// - Returns: The IDs of documents that embedded successfully.
    @discardableResult
    public func embedDocuments(
        _ documents: [RAGDocument],
        batchSize: Int = 20,
        progress: EmbeddingProgressTracker? = nil
    ) async throws -> [UUID] {
        guard let database else {
            throw RAGError.notInitialized
        }

        guard !documents.isEmpty else {
            RAGLog.debug("ℹ️ No documents to embed")
            return []
        }

        // Clear any stale documents (e.g. from a bundled DB with different
        // IDs) to avoid duplicates after re-embedding with deterministic IDs.
        do {
            try await database.reset()
        } catch {
            RAGLog.warning("⚠️ Could not reset vector DB before embedding: \(error)")
        }
        await tagStore?.removeAll()

        let totalBatches = (documents.count + batchSize - 1) / batchSize
        RAGLog.debug("📝 Starting embedding of \(documents.count) documents in \(totalBatches) batches")

        progress?.startEmbedding(totalBatches: totalBatches)

        var embeddedIDs: [UUID] = []
        embeddedIDs.reserveCapacity(documents.count)

        for batchIndex in 0..<totalBatches {
            try Task.checkCancellation()

            let startIdx = batchIndex * batchSize
            let endIdx = min(startIdx + batchSize, documents.count)
            let batch = Array(documents[startIdx..<endIdx])

            progress?.updateProgress(
                batch: batchIndex + 1,
                message: "Embedding batch \(batchIndex + 1) of \(totalBatches)..."
            )

            await stampDates(of: batch)

            var batchEmbeddedIDs: [UUID] = []
            for document in batch {
                do {
                    _ = try await database.addDocument(
                        text: document.text,
                        id: document.id
                    )
                    batchEmbeddedIDs.append(document.id)
                } catch {
                    RAGLog.warning("⚠️ Failed to embed document \(document.id): \(error)")
                    // Continue with other documents
                }
            }
            embeddedIDs.append(contentsOf: batchEmbeddedIDs)

            await storage?.discardPendingDates(for: batch.map(\.id))
            await stampTags(of: batch, embeddedIDs: batchEmbeddedIDs)

            RAGLog.debug("✅ Completed batch \(batchIndex + 1)/\(totalBatches)")
        }

        progress?.finishEmbedding(success: true)

        RAGLog.debug("🎉 Successfully embedded \(embeddedIDs.count) documents")
        return embeddedIDs
    }

    // MARK: - Incremental Updates

    /// Number of documents currently in the index.
    public func documentCount() async throws -> Int {
        guard let database else { throw RAGError.notInitialized }
        return try await database.getAllDocuments().count
    }

    /// The indexed text keyed by document ID, so hosts can diff their corpus
    /// against the index and re-embed only what actually changed.
    public func indexedDocumentTexts() async throws -> [UUID: String] {
        try await indexedDocuments().mapValues(\.text)
    }

    /// The indexed text, *stored date, and tags* keyed by document ID. Hosts
    /// whose documents carry dates or tags should diff against this, so a
    /// record left over from before they were known gets re-embedded with them.
    public func indexedDocuments() async throws -> [UUID: RAGIndexedDocument] {
        guard let database else { throw RAGError.notInitialized }
        let documents = try await database.getAllDocuments()
        let tags = await tagStore?.allTags() ?? [:]
        let pairs = documents.map {
            ($0.id, RAGIndexedDocument(text: $0.text, date: $0.createdAt, tags: tags[$0.id] ?? []))
        }
        return Dictionary(pairs) { _, latest in latest }
    }

    /// The indexed text for one document, or `nil` when it is not indexed.
    public func documentText(id: UUID) async throws -> String? {
        try await indexedDocument(id: id)?.text
    }

    /// The indexed text, stored date, and tags for one document, or `nil` when
    /// it is not indexed.
    public func indexedDocument(id: UUID) async throws -> RAGIndexedDocument? {
        guard let database else { throw RAGError.notInitialized }
        guard let document = try await database.getDocument(id: id) else { return nil }
        let tags = await tagStore?.tags(for: id) ?? []
        return RAGIndexedDocument(text: document.text, date: document.createdAt, tags: tags)
    }

    /// Embeds the given documents in place without resetting the database;
    /// a document whose ID is already indexed is replaced. Documents with
    /// empty text and individual embedding failures are logged and skipped.
    /// - Returns: The IDs of documents that embedded successfully.
    @discardableResult
    public func upsertDocuments(
        _ documents: [RAGDocument],
        batchSize: Int = 20,
        progress: EmbeddingProgressTracker? = nil
    ) async throws -> [UUID] {
        guard let database else {
            throw RAGError.notInitialized
        }

        let embeddable = documents.filter {
            !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        if embeddable.count != documents.count {
            RAGLog.warning("⚠️ Skipping \(documents.count - embeddable.count) documents with empty text")
        }
        guard !embeddable.isEmpty else { return [] }

        let totalBatches = (embeddable.count + batchSize - 1) / batchSize
        progress?.startEmbedding(totalBatches: totalBatches)

        var embeddedIDs: [UUID] = []
        embeddedIDs.reserveCapacity(embeddable.count)

        for batchIndex in 0..<totalBatches {
            try Task.checkCancellation()

            let startIdx = batchIndex * batchSize
            let endIdx = min(startIdx + batchSize, embeddable.count)
            let batch = Array(embeddable[startIdx..<endIdx])

            progress?.updateProgress(
                batch: batchIndex + 1,
                message: "Embedding batch \(batchIndex + 1) of \(totalBatches)..."
            )

            // Delete before re-adding: storage overwrites by ID, but the text
            // index keeps per-ID entries, so a bare re-add would double-count.
            // Deleting an ID that is not indexed is a no-op.
            do {
                try await database.deleteDocuments(ids: batch.map(\.id))
            } catch {
                RAGLog.warning("⚠️ Could not clear existing documents before upsert: \(error)")
            }

            await stampDates(of: batch)

            var batchEmbeddedIDs: [UUID] = []
            do {
                _ = try await database.addDocuments(texts: batch.map(\.text), ids: batch.map(\.id))
                batchEmbeddedIDs = batch.map(\.id)
            } catch {
                // One bad document fails the whole batch call, so retry one by one.
                for document in batch {
                    do {
                        _ = try await database.addDocument(text: document.text, id: document.id)
                        batchEmbeddedIDs.append(document.id)
                    } catch {
                        RAGLog.warning("⚠️ Failed to embed document \(document.id): \(error)")
                    }
                }
            }
            embeddedIDs.append(contentsOf: batchEmbeddedIDs)

            // A document that never made it to storage left its date behind;
            // drop it so it cannot land on some later write of the same ID.
            await storage?.discardPendingDates(for: batch.map(\.id))
            await stampTags(of: batch, embeddedIDs: batchEmbeddedIDs)
        }

        progress?.finishEmbedding(success: true)

        RAGLog.debug("🔁 Upserted \(embeddedIDs.count) of \(embeddable.count) documents")
        return embeddedIDs
    }

    /// Removes documents from the index. Unknown IDs are ignored.
    public func deleteDocuments(ids: [UUID]) async throws {
        guard let database else { throw RAGError.notInitialized }
        guard !ids.isEmpty else { return }
        try await database.deleteDocuments(ids: ids)
        await tagStore?.removeTags(for: ids)
        RAGLog.debug("🗑️ Deleted \(ids.count) documents from the index")
    }

    // MARK: - Snapshot Export/Import

    /// Creates a snapshot of the database for export.
    @discardableResult
    public func exportSnapshot(to url: URL? = nil) throws -> URL {
        guard let sourceDir = directoryURL else {
            throw NSError(
                domain: "RAGVectorDatabase",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "DB directory unknown; initialize first"]
            )
        }
        let fm = FileManager.default

        let exportRoot: URL
        if let url {
            exportRoot = url
        } else {
            exportRoot = try fm
                .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                .appendingPathComponent("Exports", isDirectory: true)
        }

        try fm.createDirectory(at: exportRoot, withIntermediateDirectories: true)

        let stamp = snapshotTimestamp()
        let snapshotDir = exportRoot.appendingPathComponent("\(configuration.name)-snapshot-\(stamp).vecturadb", isDirectory: true)

        if fm.fileExists(atPath: snapshotDir.path) {
            try fm.removeItem(at: snapshotDir)
        }

        try fm.copyItem(at: sourceDir, to: snapshotDir)

        return snapshotDir
    }

    /// Exports the database as a ZIP ready to be shipped as a bundled seed.
    /// The internal structure matches what `prepareSeedDirectory` expects on
    /// extraction.
    @discardableResult
    public func exportBundleReadyZip(named zipFileName: String, to directory: URL? = nil) throws -> URL {
        guard let sourceDir = directoryURL else {
            throw NSError(
                domain: "RAGVectorDatabase",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "DB directory unknown; initialize first"]
            )
        }
        let fm = FileManager.default

        let exportRoot: URL
        if let directory {
            exportRoot = directory
        } else {
            exportRoot = try fm
                .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                .appendingPathComponent("Exports", isDirectory: true)
        }
        try fm.createDirectory(at: exportRoot, withIntermediateDirectories: true)

        // Build the expected structure: <databaseFolderName>/<name>/*.json
        let stagingDir = fm.temporaryDirectory.appendingPathComponent("BundleExport-\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: stagingDir) }

        let innerDir = stagingDir.appendingPathComponent(configuration.databaseFolderName, isDirectory: true)
        try fm.createDirectory(at: innerDir, withIntermediateDirectories: true)

        // Copy the VecturaKit storage subdirectory (contains the .json docs)
        let storageSubdir = sourceDir.appendingPathComponent(configuration.name, isDirectory: true)
        let destSubdir = innerDir.appendingPathComponent(configuration.name, isDirectory: true)
        if fm.fileExists(atPath: storageSubdir.path) {
            try fm.copyItem(at: storageSubdir, to: destSubdir)
            // The seed says which engine embedded it, so a host set up on
            // another engine starts empty instead of adopting its vectors.
            let spaceRecord = RAGEmbeddingSpaceRecord.fileURL(in: sourceDir)
            if fm.fileExists(atPath: spaceRecord.path) {
                try fm.copyItem(at: spaceRecord, to: RAGEmbeddingSpaceRecord.fileURL(in: innerDir))
            }
        } else {
            // Fallback: copy the entire sourceDir contents
            try fm.copyItem(at: sourceDir, to: innerDir)
        }

        let zipURL = exportRoot.appendingPathComponent(zipFileName)
        if fm.fileExists(atPath: zipURL.path) {
            try fm.removeItem(at: zipURL)
        }
        try fm.zipItem(at: innerDir, to: zipURL)

        RAGLog.debug("📦 Exported bundle-ready ZIP → \(zipURL.path)")
        return zipURL
    }

    /// Imports a database snapshot from a `.vecturadb` directory or a ZIP
    /// archive containing one, then re-initializes the database.
    ///
    /// - Throws: `RAGError.embeddingSpaceMismatch`, leaving the current
    ///   database in place, when the snapshot was embedded by a different
    ///   engine than the configured one.
    public func importSnapshot(from snapshotURL: URL) async throws {
        let fm = FileManager.default
        let destDir = try databaseDestinationDirectory()

        var cleanupURL: URL?
        defer {
            if let cleanupURL {
                try? fm.removeItem(at: cleanupURL)
            }
        }

        let sourceDir: URL
        let lowercasedExtension = snapshotURL.pathExtension.lowercased()
        if lowercasedExtension == "zip" {
            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("VecturaImport-\(UUID().uuidString)", isDirectory: true)
            cleanupURL = tempDir
            try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)

            try fm.unzipItem(at: snapshotURL, to: tempDir)
            let contents = try fm.contentsOfDirectory(
                at: tempDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )

            if let vecturaFolder = contents.first(where: { $0.hasDirectoryPath && $0.pathExtension == "vecturadb" }) {
                sourceDir = vecturaFolder
            } else if let firstDir = contents.first(where: { $0.hasDirectoryPath }) {
                sourceDir = firstDir
            } else {
                throw NSError(
                    domain: "RAGVectorDatabase",
                    code: 1002,
                    userInfo: [NSLocalizedDescriptionKey: "No .vecturadb folder found inside \(snapshotURL.lastPathComponent)."]
                )
            }
        } else {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: snapshotURL.path, isDirectory: &isDir), isDir.boolValue else {
                throw NSError(
                    domain: "RAGVectorDatabase",
                    code: 1003,
                    userInfo: [NSLocalizedDescriptionKey: "Snapshot at \(snapshotURL.path) is neither a directory nor a .zip archive."]
                )
            }
            sourceDir = snapshotURL
        }

        if sourceDir.resolvingSymlinksInPath() == destDir.resolvingSymlinksInPath() {
            RAGLog.debug("ℹ️ Snapshot source matches destination — nothing to import.")
            return
        }

        // Setup would clear another engine's vectors, but only after they had
        // replaced the current database, so turn them away here instead.
        let configuredSpace = try (embedder ?? makeEmbedder(for: configuration.embeddingEngine)).spaceIdentifier
        if let snapshotSpace = storedSpaceIdentifier(in: sourceDir), snapshotSpace != configuredSpace {
            throw RAGError.embeddingSpaceMismatch(snapshot: snapshotSpace, database: configuredSpace)
        }

        if fm.fileExists(atPath: destDir.path) {
            try fm.removeItem(at: destDir)
        }
        try fm.copyItem(at: sourceDir, to: destDir)

        database = nil
        try await setUp()
        revision += 1

        RAGLog.debug("📥 Imported Vectura DB snapshot from: \(snapshotURL.lastPathComponent)")
    }

    // MARK: - Private Helpers

    /// Hands the batch's dates to storage so they land on the records these
    /// documents are about to be written into. Documents without a date keep
    /// VecturaKit's own timestamp.
    private func stampDates(of batch: [RAGDocument]) async {
        guard let storage else { return }
        let dates = batch.reduce(into: [UUID: Date]()) { dates, document in
            if let date = document.date { dates[document.id] = date }
        }
        guard !dates.isEmpty else { return }
        await storage.stampNextSave(of: dates)
    }

    /// Records the tags of the documents that actually reached the index, so a
    /// failed embed cannot leave its tags claiming a document that isn't there.
    /// Written per batch, not per run, so a cancellation mid-embed loses tags
    /// only for documents whose vectors were also never written.
    private func stampTags(of batch: [RAGDocument], embeddedIDs: [UUID]) async {
        guard let tagStore else { return }
        let embedded = Set(embeddedIDs)
        let updates = batch.reduce(into: [UUID: [String]]()) { updates, document in
            if embedded.contains(document.id) { updates[document.id] = document.tags }
        }
        await tagStore.setTags(updates)
    }

    /// Prepares the database directory for vectors in `space`: the existing
    /// database when its vectors are in that space, otherwise the bundled
    /// seed when it is, otherwise an empty directory. Either way the space is
    /// recorded, so the next setup can tell.
    private func prepareSeedDirectory(forceReset: Bool = false, space: RAGEmbeddingSpaceRecord) throws -> URL {
        let fm = FileManager.default
        let destDir = try databaseDestinationDirectory()

        if !forceReset,
           fm.fileExists(atPath: destDir.path),
           let contents = try? fm.contentsOfDirectory(
                at: destDir,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
           ),
           !contents.isEmpty {
            // VecturaKit appends config.name to directoryURL, so documents
            // live in a subdirectory named after the DB name. Ensure this
            // subdir exists, renaming a mismatched seeded subdirectory if
            // needed.
            try ensureStorageSubdirectoryMatchesDBName(in: destDir)
            if let stored = storedSpaceIdentifier(in: destDir), stored != space.identifier {
                // Another engine's vectors: this engine's queries would rank
                // them as noise, or fail on the dimension. Start over, and
                // the host re-embeds.
                RAGLog.warning("🔀 Vectura DB holds vectors from \(stored), not \(space.identifier); clearing it")
            } else {
                try space.write(to: destDir)
                RAGLog.debug("📦 Using existing Vectura DB at: \(destDir.path)")
                return destDir
            }
        }

        if fm.fileExists(atPath: destDir.path) {
            try fm.removeItem(at: destDir)
        }

        try seed(destDir)
        if let seeded = storedSpaceIdentifier(in: destDir), seeded != space.identifier {
            RAGLog.warning("🔀 Bundled seed holds vectors from \(seeded), not \(space.identifier); starting empty")
            try fm.removeItem(at: destDir)
            try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
        }
        try space.write(to: destDir)
        return destDir
    }

    /// Fills the (absent) database directory from the bundled seed ZIP or
    /// folder, or creates it empty when there is no seed.
    private func seed(_ destDir: URL) throws {
        let fm = FileManager.default

        if let zipURL = configuration.bundledSeedZipURL {
            let tempUnzip = destDir.deletingLastPathComponent().appendingPathComponent("__unzipped__", isDirectory: true)
            if fm.fileExists(atPath: tempUnzip.path) { try? fm.removeItem(at: tempUnzip) }
            try fm.createDirectory(at: tempUnzip, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: tempUnzip) }

            try fm.unzipItem(at: zipURL, to: tempUnzip)
            let topLevel = try fm.contentsOfDirectory(
                at: tempUnzip,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )

            if let vecturaFolder = topLevel.first(where: { $0.hasDirectoryPath && $0.pathExtension == "vecturadb" }) {
                try fm.copyItem(at: vecturaFolder, to: destDir)
            } else {
                try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
                for item in topLevel {
                    let target = destDir.appendingPathComponent(item.lastPathComponent, isDirectory: item.hasDirectoryPath)
                    if fm.fileExists(atPath: target.path) { try? fm.removeItem(at: target) }
                    try fm.copyItem(at: item, to: target)
                }
            }

            // VecturaKit appends config.name to directoryURL, so documents
            // must live in a subdirectory named after the DB name. Rename if
            // the seed ZIP used a different internal directory name.
            try ensureStorageSubdirectoryMatchesDBName(in: destDir)

            RAGLog.debug("🗜️ Seeded Vectura DB from bundled zip → \(destDir.path)")
            return
        }

        if let folderURL = configuration.bundledSeedFolderURL {
            try fm.copyItem(at: folderURL, to: destDir)
            try ensureStorageSubdirectoryMatchesDBName(in: destDir)
            RAGLog.debug("📦 Seeded Vectura DB from bundled folder → \(destDir.path)")
        } else {
            // No bundled seed found — create an empty directory.
            // Runtime embedding via embedDocuments will populate it.
            try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
            RAGLog.debug("📁 Created empty Vectura DB directory (no bundled DB found) → \(destDir.path)")
        }
    }

    /// The vector space `directory`'s vectors were embedded in: the one its
    /// space record names, or, for a database written before records existed,
    /// the sentence transformer's (the only engine there was). `nil` when the
    /// directory holds no vectors, which any engine may adopt.
    private func storedSpaceIdentifier(in directory: URL) -> String? {
        if let recorded = RAGEmbeddingSpaceRecord.storedIdentifier(in: directory) {
            return recorded
        }
        guard holdsVectors(directory) else { return nil }
        return RAGSentenceEmbedder.spaceIdentifier(forModelID: configuration.remoteModelID)
    }

    /// Whether any subdirectory holds VecturaKit's per-document `.json` files,
    /// by the same test `ensureStorageSubdirectoryMatchesDBName` uses.
    private func holdsVectors(_ directory: URL) -> Bool {
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return entries.contains { entry in
            guard (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return false }
            let children = (try? fm.contentsOfDirectory(atPath: entry.path)) ?? []
            return children.contains { $0.hasSuffix(".json") }
        }
    }

    /// Builds the engine a configuration names. Only an engine that cannot
    /// run on this device throws, and it does so before anything on disk
    /// changes.
    private func makeEmbedder(for engine: RAGEmbeddingEngine) throws -> any RAGEmbedder {
        switch engine {
        case .sentenceTransformer:
            // Not VecturaEmbeddingsKit's SwiftEmbedder: its GPU path leaks a
            // compiled graph per input shape and a padded batch can take
            // gigabytes. See RAGSentenceEmbedder. Named by the remote ID even
            // when the bundled copy loads, so both are the same space.
            return RAGSentenceEmbedder(modelSource: embedderSource(), modelID: configuration.remoteModelID)
        case .naturalLanguage(let languages):
            return try RAGNaturalLanguageEmbedder(languages: languages)
        case .custom(let embedder):
            return embedder
        }
    }

    /// The engine's vector length. The sentence transformer's comes from the
    /// configuration, so setup never loads (or downloads) the model just to
    /// measure it; the other engines report their own.
    private func vectorDimension(of embedder: any RAGEmbedder, engine: RAGEmbeddingEngine) async throws -> Int {
        if case .sentenceTransformer = engine {
            return configuration.dimension
        }
        return try await embedder.dimension
    }

    /// VecturaKit resolves its storage directory as `directoryURL / config.name`.
    /// If the seed ZIP used a different internal directory name (e.g. a
    /// previous build used "tarot-rag-db" while the config uses "activities"),
    /// the pre-embedded documents live under the wrong subdirectory and
    /// VecturaKit creates an empty one instead.
    ///
    /// This method detects the mismatch and renames the existing subdirectory
    /// so VecturaKit can find the pre-embedded data without re-embedding.
    private func ensureStorageSubdirectoryMatchesDBName(in destDir: URL) throws {
        let fm = FileManager.default
        let expectedSubdir = destDir.appendingPathComponent(configuration.name, isDirectory: true)

        // Already correct — nothing to do.
        if fm.fileExists(atPath: expectedSubdir.path) { return }

        // Look for a single subdirectory that contains .json document files.
        let entries = try fm.contentsOfDirectory(
            at: destDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        let subdirs = entries.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }

        for subdir in subdirs {
            let children = (try? fm.contentsOfDirectory(atPath: subdir.path)) ?? []
            let hasJSONFiles = children.contains { $0.hasSuffix(".json") }
            if hasJSONFiles {
                try fm.moveItem(at: subdir, to: expectedSubdir)
                RAGLog.debug("📦 Renamed seeded storage subdir \"\(subdir.lastPathComponent)\" → \"\(configuration.name)\"")
                return
            }
        }
    }

    private func embedderSource() -> VecturaModelSource {
        if let localURL = configuration.localModelFolderURL {
            let fm = FileManager.default
            let hasAllFiles = configuration.requiredLocalModelFiles.allSatisfy { fileName in
                fm.fileExists(atPath: localURL.appendingPathComponent(fileName).path)
            }
            guard hasAllFiles else {
                RAGLog.debug("ℹ️ Local model folder is missing files; using remote id \(configuration.remoteModelID)")
                return .id(configuration.remoteModelID)
            }
            RAGLog.debug("ℹ️ Using local model folder at: \(localURL.path)")
            return .folder(localURL)
        }
        RAGLog.debug("ℹ️ Local model folder not found; using remote id \(configuration.remoteModelID)")
        return .id(configuration.remoteModelID)
    }

    private func snapshotTimestamp() -> String {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyyyMMdd-HHmmss"
        return df.string(from: Date())
    }
}
