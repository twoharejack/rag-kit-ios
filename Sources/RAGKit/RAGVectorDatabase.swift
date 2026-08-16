// RAGVectorDatabase.swift
// ============================================================================
// Generic VecturaKit-backed vector database engine: on-disk setup and seeding,
// embedder resolution, batched document embedding, and snapshot export/import.
// Domain-agnostic — hosts supply seed archives, model folders, and document
// texts, and keep their own document metadata keyed by document ID.
//
// Not thread-safe on its own: designed to be owned by a single actor (or other
// single concurrency domain) in the host app.
// ============================================================================

import Foundation
import VecturaEmbeddingsKit
import VecturaKit
import ZIPFoundation

// MARK: - Configuration

public struct RAGVectorDatabaseConfiguration {
    /// VecturaKit database name; storage lives in a subdirectory with this name.
    public var name: String
    /// Embedding dimension (must match the embedding model).
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
    public var remoteModelID: String

    public init(
        name: String,
        dimension: Int,
        storageRootFolderName: String = "VectorDB",
        databaseFolderName: String,
        bundledSeedZipURL: URL? = nil,
        bundledSeedFolderURL: URL? = nil,
        localModelFolderURL: URL? = nil,
        requiredLocalModelFiles: [String] = [],
        remoteModelID: String
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
    }
}

// MARK: - Document

/// A document to embed: a stable ID plus the text that gets vectorized.
/// Hosts keep any richer metadata in their own lookup keyed by `id`.
public struct RAGDocument {
    public let id: UUID
    public let text: String

    public init(id: UUID, text: String) {
        self.id = id
        self.text = text
    }
}

// MARK: - Engine

public final class RAGVectorDatabase {
    public let configuration: RAGVectorDatabaseConfiguration

    public private(set) var database: VecturaKit?
    public private(set) var directoryURL: URL?
    /// Incremented after snapshot imports so UIs can reload.
    public private(set) var revision: Int = 0

    public init(configuration: RAGVectorDatabaseConfiguration) {
        self.configuration = configuration
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
            let dbDir = try prepareSeedDirectory(forceReset: forceReset)
            self.directoryURL = dbDir

            let config = try VecturaConfig(
                name: configuration.name,
                directoryURL: dbDir,
                dimension: configuration.dimension
            )

            let embedder = SwiftEmbedder(modelSource: embedderSource())
            self.database = try await VecturaKit(config: config, embedder: embedder)
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

            RAGLog.debug("✅ Successfully initialized VecturaKit")

        } catch {
            RAGLog.error("❌ Failed to initialize VecturaKit: \(error)")
            throw error
        }
    }

    // MARK: - Search

    public func search(query: String, numResults: Int, threshold: Float? = nil) async throws -> [VecturaSearchResult] {
        guard let database else {
            throw RAGError.notInitialized
        }
        return try await database.search(
            query: .text(query),
            numResults: numResults,
            threshold: threshold
        )
    }

    // MARK: - Embedding

    /// Checks if the database needs embedding (is empty or missing).
    public func needsEmbedding() async -> Bool {
        guard let database else { return true }
        do {
            // Try a simple search to check if DB has content
            let results = try await database.search(query: .text("test"), numResults: 1, threshold: 0.0)
            return results.isEmpty
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

            for document in batch {
                do {
                    _ = try await database.addDocument(
                        text: document.text,
                        id: document.id
                    )
                    embeddedIDs.append(document.id)
                } catch {
                    RAGLog.warning("⚠️ Failed to embed document \(document.id): \(error)")
                    // Continue with other documents
                }
            }

            RAGLog.debug("✅ Completed batch \(batchIndex + 1)/\(totalBatches)")
        }

        progress?.finishEmbedding(success: true)

        RAGLog.debug("🎉 Successfully embedded \(embeddedIDs.count) documents")
        return embeddedIDs
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

    private func prepareSeedDirectory(forceReset: Bool = false) throws -> URL {
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
            RAGLog.debug("📦 Using existing Vectura DB at: \(destDir.path)")
            return destDir
        }

        if fm.fileExists(atPath: destDir.path) {
            try fm.removeItem(at: destDir)
        }

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
            return destDir
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
        return destDir
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
