// JudgmentCache.swift
// ============================================================================
// The grades a judge has already given, kept on disk beside the set they
// grade, so a re-run only asks about pairs the judge has not seen.
// ============================================================================

import CryptoKit
import Foundation

/// One judge's grades for one set, read from and written back to a JSON file.
///
/// Grades are keyed by the prompt's version and the exact query and document
/// text, not by IDs. Editing a note or a query therefore re-grades it, a new
/// prompt re-grades everything, and a pair that several engines return is
/// graded once. The same retrieval results always get the same judged scores,
/// so a change to an engine shows up as a change in its scores and not as a
/// judge's second opinion. Each entry also names its query and document,
/// so the file doubles as a record of what the judge decided.
actor JudgmentCache {
    struct Entry: Codable, Sendable {
        let query: String
        let document: String
        let grade: Int
        let key: String
    }

    let fileURL: URL
    private var entries: [String: Entry]
    private var unsaved = false

    init(fileURL: URL) throws {
        self.fileURL = fileURL
        if FileManager.default.fileExists(atPath: fileURL.path) {
            let stored = try JSONDecoder().decode([Entry].self, from: Data(contentsOf: fileURL))
            entries = Dictionary(stored.map { ($0.key, $0) }) { first, _ in first }
        } else {
            entries = [:]
        }
    }

    var count: Int { entries.count }

    func grade(query: String, document: String) -> Int? {
        entries[Self.key(query: query, document: document)]?.grade
    }

    func record(_ grade: Int, query: EvalSet.Query, document: EvalSet.Document) {
        let key = Self.key(query: query.text, document: document.text)
        entries[key] = Entry(query: query.id, document: document.id, grade: grade, key: key)
        unsaved = true
    }

    /// Writes one grade per line, sorted by query and document, so a re-run
    /// that adds grades shows up in a diff as only the added lines.
    func save() throws {
        guard unsaved else { return }
        let sorted = entries.values.sorted { ($0.query, $0.document, $0.key) < ($1.query, $1.document, $1.key) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let lines = try sorted.map { String(decoding: try encoder.encode($0), as: UTF8.self) }
        let json = "[\n" + lines.joined(separator: ",\n") + "\n]\n"
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(json.utf8).write(to: fileURL, options: .atomic)
        unsaved = false
    }

    static func key(query: String, document: String) -> String {
        let material = "\(JudgePrompt.version)\u{1F}\(query)\u{1F}\(document)"
        return SHA256.hash(data: Data(material.utf8))
            .prefix(12)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
