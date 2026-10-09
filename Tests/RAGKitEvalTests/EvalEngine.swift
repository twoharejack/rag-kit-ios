// EvalEngine.swift
// ============================================================================
// The engines an eval compares, and one engine's pass over a set: index every
// document in a throwaway database, run every query, keep the rankings.
// ============================================================================

import Foundation
import NaturalLanguage
import RAGKit

enum EvalEngine: String, CaseIterable, Sendable {
    /// Words only, no model: `LexicalEmbedder`'s hashed word counts in the
    /// vector half of VecturaKit's hybrid score and BM25 in the other. The
    /// floor the embedding engines have to clear.
    case lexical
    /// all-MiniLM-L6-v2 with first-token pooling: the default engine, and what
    /// every shipped database and seed was embedded with.
    case minilm
    /// all-MiniLM-L6-v2 with mean pooling, the way it was trained.
    case minilmMean = "minilm-mean"
    /// Apple's NaturalLanguage models, configured with the set's languages.
    case apple
    /// The app's Core Spotlight index. Not in the defaults: what an xctest
    /// process indexes shows up in the system's own search until the run
    /// removes it, and Spotlight matches a note's meaning only once the
    /// system has processed it, which a fresh index has not.
    case spotlight

    static let defaults: [EvalEngine] = [.lexical, .minilm, .minilmMean, .apple]

    var summary: String {
        switch self {
        case .lexical: "words only: hashed word counts and BM25, no model"
        case .minilm: "all-MiniLM-L6-v2, first-token pooling (the default)"
        case .minilmMean: "all-MiniLM-L6-v2, mean pooling"
        case .apple: "Apple's NaturalLanguage models"
        case .spotlight: "Core Spotlight (CSUserQuery) on a fresh index"
        }
    }

    func embeddingEngine(languages: [NLLanguage]) -> RAGEmbeddingEngine {
        switch self {
        case .lexical: .custom(LexicalEmbedder())
        case .minilm, .minilmMean: .sentenceTransformer
        case .apple: .naturalLanguage(languages: languages)
        case .spotlight: .spotlight
        }
    }
}

/// What one engine returned for every query in a set.
struct EngineRun: Sendable {
    let engine: EvalEngine
    /// Document IDs per query ID, best first.
    let rankings: [String: [String]]
    /// Documents the engine could not index, such as a language Apple's
    /// engine has no model for on this device. They can never be retrieved.
    let unindexed: [String]
    let indexingTime: Duration
    let searchTime: Duration
}

extension EvalEngine {
    /// Indexes `set` into a database of its own, runs every query for the
    /// top `depth` documents, and deletes the database again.
    func run(on set: EvalSet, depth: Int) async throws -> EngineRun {
        // A storage root of its own, so the eval never opens a host's
        // database and two runs never share one.
        let configuration = RAGVectorDatabaseConfiguration(
            name: "ragkit-eval",
            dimension: 384,
            storageRootFolderName: "RAGKitEval-\(rawValue)-\(UUID().uuidString)",
            databaseFolderName: "\(rawValue).vecturadb",
            remoteModelID: "sentence-transformers/all-MiniLM-L6-v2",
            embeddingEngine: embeddingEngine(languages: set.nlLanguages),
            sentenceTransformerPooling: self == .minilmMean ? .mean : .firstToken
        )
        let database = RAGVectorDatabase(configuration: configuration)
        do {
            let run = try await index(set, into: database, depth: depth)
            try await Self.tearDown(database)
            return run
        } catch {
            try? await Self.tearDown(database)
            throw error
        }
    }

    private func index(_ set: EvalSet, into database: RAGVectorDatabase, depth: Int) async throws -> EngineRun {
        let clock = ContinuousClock()
        let documentIDs = Dictionary(uniqueKeysWithValues: set.documents.map { (EvalSet.uuid(for: $0.id), $0.id) })

        let indexStart = clock.now
        try await database.setUp(forceReset: true)
        let embedded = try await database.embedDocuments(
            set.documents.map { RAGDocument(id: EvalSet.uuid(for: $0.id), text: $0.text) }
        )
        let indexingTime = clock.now - indexStart
        let embeddedIDs = Set(embedded)
        let unindexed = set.documents.map(\.id).filter { !embeddedIDs.contains(EvalSet.uuid(for: $0)) }

        var rankings: [String: [String]] = [:]
        let searchStart = clock.now
        for query in set.queries {
            // `depth` results, as a host asks for them. VecturaKit's hybrid
            // score only combines each half's top 2 × numResults candidates,
            // so the number asked for shapes the ranking itself: over the
            // whole corpus instead, first-token MiniLM's judged nDCG@10 fell
            // from 0.66 to 0.62 while the other engines held.
            let results = try await database.search(query: query.text, numResults: depth)
            rankings[query.id] = Self.ranking(of: results, depth: depth, documentIDs: documentIDs)
        }
        return EngineRun(
            engine: self,
            rankings: rankings,
            unindexed: unindexed,
            indexingTime: indexingTime,
            searchTime: clock.now - searchStart
        )
    }

    /// The document IDs of a search, as nearly the same in every run as
    /// VecturaKit allows.
    ///
    /// VecturaKit breaks exact ties between scores in whatever order its
    /// dictionaries iterate, which changes from one database instance to the
    /// next. Taken as returned, the lexical engine's top 10 differed on 22 of
    /// 40 queries between two identical runs, and luck decided whether a
    /// relevant note tied with another came first. So:
    ///
    /// - A result scored exactly 0 is left out. It is no evidence at all: no
    ///   shared word and no similarity (on the lexical engine a note without
    ///   the query's words, on Apple's a note in another script).
    /// - Equal scores are ordered by document ID.
    ///
    /// What is left is a tie across the last place, whose members VecturaKit
    /// returns arbitrarily, and a note tied at the edge of one half's
    /// candidate list, which is a candidate on some runs and not on others.
    /// Across three runs that changed one of 160 rankings, in its last place
    /// (three notes tied for two places), which costs a judge call now and
    /// then until each note that can land there has been graded.
    static func ranking(of results: [VecturaSearchResult], depth: Int, documentIDs: [UUID: String]) -> [String] {
        results
            .filter { $0.score != 0 }
            .compactMap { result in documentIDs[result.id].map { (id: $0, score: result.score) } }
            .sorted { ($0.score, $1.id) > ($1.score, $0.id) }
            .prefix(depth)
            .map(\.id)
    }

    /// Deletes the eval's database. A Spotlight one is first switched to
    /// another engine, which is what takes its items out of the system's
    /// index.
    private static func tearDown(_ database: RAGVectorDatabase) async throws {
        if case .spotlight = database.configuration.embeddingEngine {
            try await database.switchEmbeddingEngine(to: .custom(LexicalEmbedder()))
        }
        let root = try database.databaseDestinationDirectory().deletingLastPathComponent()
        try FileManager.default.removeItem(at: root)
    }
}

// MARK: - Lexical baseline

/// Embeds a text as the counts of its words, hashed into a fixed number of
/// buckets: two texts score by the words they share and nothing else. Under
/// VecturaKit's hybrid score that makes a words-only engine, BM25 beside a
/// bag-of-words cosine, which is what the embedding engines are measured
/// against. It needs no model, so the harness tests run on it too.
///
/// A constant vector would not do: VecturaKit ranks the vector half's top
/// candidates first and scores a BM25-only match as if its cosine were 0, so
/// arbitrary documents would outrank real keyword matches.
struct LexicalEmbedder: RAGEmbedder {
    let spaceIdentifier = "ragkit-eval:lexical-v1"
    let dimension = 1024

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map(vector(for:))
    }

    private func vector(for text: String) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        for word in Self.words(in: text) {
            vector[Int(Self.fnv1a(word) % UInt64(dimension))] += 1
        }
        // A text without a word ("…") still needs a direction for cosine.
        if vector.allSatisfy({ $0 == 0 }) { vector[0] = 1 }
        return vector
    }

    static func words(in text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
    }

    /// FNV-1a, not `Hasher`, whose seed changes every process: the same word
    /// has to land in the same bucket in every run.
    private static func fnv1a(_ word: String) -> UInt64 {
        word.utf8.reduce(0xcbf2_9ce4_8422_2325) { hash, byte in
            (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
        }
    }
}
