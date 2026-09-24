// RAGEmbeddingSpaceRecord.swift
// ============================================================================
// Records which vector space a database directory's vectors were embedded in.
// ============================================================================

import Foundation

/// The vector space a database directory's vectors belong to, written as a
/// JSON file at the directory root (beside the tag sidecar) so snapshots and
/// bundle-ready seeds carry it with the vectors.
///
/// Databases written before engines could be switched have no record. The
/// sentence transformer was the only engine then, so `RAGVectorDatabase`
/// reads a missing record as that model's space.
struct RAGEmbeddingSpaceRecord: Codable, Equatable {
    static let fileName = "embedding-space.json"

    /// The engine's `RAGEmbedder.spaceIdentifier`.
    let identifier: String
    /// Vector length, kept for anyone reading the file.
    let dimension: Int

    static func fileURL(in directory: URL) -> URL {
        directory.appendingPathComponent(fileName)
    }

    /// The record in `directory`, `nil` when there is none, and a sentinel
    /// that matches no engine when the file exists but cannot be read (so an
    /// unreadable record clears the database rather than vouching for it).
    static func storedIdentifier(in directory: URL) -> String? {
        let url = fileURL(in: directory)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let data = try? Data(contentsOf: url),
              let record = try? JSONDecoder().decode(Self.self, from: data) else {
            return "unreadable \(fileName)"
        }
        return record.identifier
    }

    func write(to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: Self.fileURL(in: directory), options: .atomic)
    }
}
