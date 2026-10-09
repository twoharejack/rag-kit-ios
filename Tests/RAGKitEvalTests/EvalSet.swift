// EvalSet.swift
// ============================================================================
// The documents and queries a retrieval eval runs on: the bundled notes
// fixture, or any set in the same JSON shape named by RAGKIT_EVAL_SET.
// ============================================================================

import CryptoKit
import Foundation
import NaturalLanguage

/// A corpus, the queries to run against it, and, optionally, hand labels.
///
/// Labels are optional because the LLM judges grade every note an engine
/// returns either way: a host can point the eval at its own notes and a list
/// of queries and get judged scores without labeling anything. Where labels
/// exist, the report also scores the engines against them and checks the
/// judges against them.
struct EvalSet: Decodable, Sendable {
    struct Document: Decodable, Sendable {
        let id: String
        let text: String
    }

    struct Query: Decodable, Sendable {
        let id: String
        let text: String
        /// Groups queries in the report: "keyword", "paraphrase", "question",
        /// "short", "non-english", "cross-language", or anything a set uses.
        let kind: String?
        /// Hand-labeled grades by document ID, on the judges' 0–3 scale.
        /// Documents left out count as 0.
        let relevant: [String: Int]?
    }

    let name: String
    /// BCP-47 codes of the languages the documents are written in, which is
    /// what Apple's engine is configured with.
    let languages: [String]
    let documents: [Document]
    let queries: [Query]

    var nlLanguages: [NLLanguage] { languages.map(NLLanguage.init(rawValue:)) }

    var hasLabels: Bool { queries.contains { $0.relevant?.isEmpty == false } }

    /// Built on each call, so look up through one copy.
    var documentsByID: [String: Document] {
        Dictionary(documents.map { ($0.id, $0) }) { first, _ in first }
    }

    private enum CodingKeys: String, CodingKey {
        case name, languages, documents, queries
    }

    // MARK: - Loading

    /// The fixture shipped with the tests.
    static let bundledURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/notes.json")

    static func load(from url: URL) throws -> EvalSet {
        let set = try JSONDecoder().decode(EvalSet.self, from: Data(contentsOf: url))
        try set.validate()
        return set
    }

    /// Catches the mistakes a hand-written set makes (a duplicated ID, a label
    /// naming a document that does not exist) before they turn into quietly
    /// wrong scores.
    func validate() throws {
        func fail(_ message: String) -> EvalSetError { EvalSetError(set: name, message: message) }

        guard !documents.isEmpty else { throw fail("has no documents") }
        guard !queries.isEmpty else { throw fail("has no queries") }
        let documentIDs = documents.map(\.id)
        if let duplicate = Self.firstDuplicate(in: documentIDs) {
            throw fail("has two documents with ID \(duplicate)")
        }
        if let duplicate = Self.firstDuplicate(in: queries.map(\.id)) {
            throw fail("has two queries with ID \(duplicate)")
        }
        let known = Set(documentIDs)
        for query in queries {
            for (id, grade) in query.relevant ?? [:] {
                guard known.contains(id) else {
                    throw fail("query \(query.id) labels unknown document \(id)")
                }
                guard (0...3).contains(grade) else {
                    throw fail("query \(query.id) grades \(id) \(grade), outside 0–3")
                }
            }
        }
    }

    private static func firstDuplicate(in ids: [String]) -> String? {
        var seen = Set<String>()
        return ids.first { !seen.insert($0).inserted }
    }

    // MARK: - IDs

    /// The UUID a document is indexed under. Derived from its ID, so the same
    /// document gets the same UUID in every engine and every run.
    static func uuid(for documentID: String) -> UUID {
        var bytes = Array(SHA256.hash(data: Data(documentID.utf8)).prefix(16))
        bytes[6] = (bytes[6] & 0x0F) | 0x50 // version 5 (name-based)
        bytes[8] = (bytes[8] & 0x3F) | 0x80 // RFC 4122 variant
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }
}

struct EvalSetError: Error, CustomStringConvertible {
    let set: String
    let message: String
    var description: String { "Eval set \"\(set)\" \(message)" }
}
