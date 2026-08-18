// RAGDocumentTagStore.swift
// ============================================================================
// Sidecar store that keeps each document's filter tags beside the vector DB.
// ============================================================================

import Foundation

/// Keeps the tags of every indexed document in one JSON sidecar inside the
/// database directory, so they travel with the vectors through snapshot
/// export/import. `VecturaDocument` has no metadata slot to write into —
/// `createdAt` was the last one, and `RAGDateStampingStorage` already spends
/// it on dates — so tags live next to the records rather than inside them.
///
/// A lost or stale sidecar is not corruption: `indexedDocuments()` then
/// reports the affected documents with no tags, hosts that diff against it
/// see a mismatch, and their next reconcile re-upserts the tags.
actor RAGDocumentTagStore {
    private let fileURL: URL
    private var tagsByID: [UUID: [String]]

    init(fileURL: URL) {
        self.fileURL = fileURL
        if let data = try? Data(contentsOf: fileURL),
           let stored = try? JSONDecoder().decode([String: [String]].self, from: data) {
            tagsByID = stored.reduce(into: [:]) { tags, entry in
                if let id = UUID(uuidString: entry.key) { tags[id] = entry.value }
            }
        } else {
            tagsByID = [:]
        }
    }

    func tags(for id: UUID) -> [String] {
        tagsByID[id] ?? []
    }

    func allTags() -> [UUID: [String]] {
        tagsByID
    }

    /// IDs of documents carrying at least one of `tags`.
    func ids(withAnyOf tags: [String]) -> Set<UUID> {
        let wanted = Set(tags)
        return Set(tagsByID.filter { !wanted.isDisjoint(with: $0.value) }.keys)
    }

    /// Records each document's tags; an empty tag list removes the entry so
    /// the sidecar never accumulates blanks.
    func setTags(_ updates: [UUID: [String]]) {
        guard !updates.isEmpty else { return }
        for (id, tags) in updates {
            if tags.isEmpty {
                tagsByID.removeValue(forKey: id)
            } else {
                tagsByID[id] = tags
            }
        }
        save()
    }

    func removeTags(for ids: [UUID]) {
        var changed = false
        for id in ids where tagsByID.removeValue(forKey: id) != nil {
            changed = true
        }
        if changed { save() }
    }

    func removeAll() {
        guard !tagsByID.isEmpty else { return }
        tagsByID = [:]
        save()
    }

    private func save() {
        // String keys so the sidecar is a plain JSON object; UUID keys would
        // flatten into an alternating array under Codable.
        let stored = tagsByID.reduce(into: [String: [String]]()) { tags, entry in
            tags[entry.key.uuidString] = entry.value
        }
        do {
            let data = try JSONEncoder().encode(stored)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            RAGLog.warning("⚠️ Could not save document tags: \(error)")
        }
    }
}
