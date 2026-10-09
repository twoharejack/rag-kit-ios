// RAGSpotlightIndex.swift
// ============================================================================
// Apple's Core Spotlight as a RAGKit engine: documents become items in the
// app's own on-device Spotlight index, and searches go through CSUserQuery,
// which matches a query's words and, where the system offers it, its meaning.
// ============================================================================

import CoreSpotlight
import Foundation
import NaturalLanguage
import UniformTypeIdentifiers

extension RAGEmbeddingEngine {
    /// Whether this device can index into Spotlight at all, which is what
    /// `.spotlight` needs. Not every device can (Core Spotlight's own
    /// `isIndexingAvailable()`).
    public static var isSpotlightAvailable: Bool {
        CSSearchableIndex.isIndexingAvailable()
    }
}

/// What a `RAGVectorDatabase` configured with `RAGEmbeddingEngine.spotlight`
/// runs on in place of VecturaKit: the app's Core Spotlight index.
///
/// Spotlight embeds and ranks inside the system, so the database keeps no
/// vectors and no model of its own. Every document is one `CSSearchableItem`:
/// its text as `textContent`, its date as `contentCreationDate`, and its tags
/// in a custom attribute that filters can test but a typed query never
/// matches. A search is a `CSUserQuery` with ranked results, and the date and
/// tag filters run inside Spotlight as filter queries, so they narrow the
/// corpus before it is ranked, as they do on the vector engines.
///
/// What Spotlight cannot hand back is kept beside it, one record file per
/// document in the database directory: the text, date and tags each document
/// was indexed with. Spotlight never returns an item's text, and hosts diff
/// their corpus against exactly this (`indexedDocuments()`). The records are
/// also the last word on what a search may return, so an item Spotlight still
/// holds but the records do not is never answered.
///
/// The records only say what Spotlight holds while Spotlight keeps it, so the
/// two are tied together three ways:
///
/// - Items never expire. Spotlight's default is a month, after which an item
///   would quietly leave the index and its record would still vouch for it.
/// - A generation token is stored both in the database directory and as the
///   index's client state, which lives inside Spotlight and is lost with it.
///   On open, a token that does not match (a restored backup, an index the
///   system rebuilt, the app's index deleted) starts both over, empty, and
///   the host's next diff re-indexes every document.
/// - The index's delegate drops the records of whatever Spotlight asks the
///   app to re-index, all of them when it has lost the index.
///
/// Measured on macOS 27 (M1 Max), with a dozen short notes:
///
/// - Words match as soon as `indexSearchableItems` returns: a query for a
///   word in one note found it on the next call.
/// - Meaning matches depend on the system. The query's embedding is made
///   in-process in under a millisecond once the model is loaded (the log
///   read "Enabling semantic search … isGMOptedIn: YES"), but each item's is
///   made later by Spotlight's own pipeline (`spotlightknowledged`) as a
///   background task the system classes as intensive. The scheduler held it
///   back while the Mac was in use ("Must Not Proceed", its intensive budget
///   spent), and 45 minutes after indexing no note matched a query that
///   shares no word with it. So a document is found by its words straight
///   away, and by its meaning once the system has processed it, which on a
///   phone in use may not be the same day. With the eval's 54 notes,
///   meaning matches came within seconds of indexing on some runs, and on
///   others within the same hour, on the same index, not at all.
/// - Ranked results come back in batches, not in order, and sorted
///   descending (`CSUserQuery.Item`'s `>`) the best comes first. Only the
///   first `maxRankedResultCount` are ranked; the rest still come back, and
///   sort in among them unranked. See `search`.
/// - Filter queries on `domainIdentifier`, on `contentCreationDate` against
///   `$time.iso(…)`, and on a multi-valued custom attribute (`==` matches any
///   value; `||` inside one filter, several filters AND together) all
///   applied before ranking.
///
/// Spotlight ranks; it does not score. A search's results carry
/// `score(forRank:)` in place of a similarity: 1 for the best, falling
/// gently with each place, so code that blends scores or holds them to a
/// fraction of the best keeps Spotlight's order and stays within 0…1.
///
/// Items in the app's index also appear in the system's own search (Spotlight
/// on the Home Screen, Siri Suggestions) under the app, as everything an app
/// indexes into Spotlight does, unless the person turns that off for the app
/// in Settings.
final class RAGSpotlightIndex {
    /// Names Spotlight in the database directory's space record. It only says
    /// that the documents live in Spotlight rather than in VecturaKit's files:
    /// nothing outside Spotlight can compare its vectors.
    static let spaceIdentifier = "core-spotlight-v1"

    /// The custom attribute a document's tags are stored under. Searchable,
    /// so filter queries can test it, but not by default, so a typed query
    /// never matches a document by its tags.
    private static let tagsKey = CSCustomAttributeKey(
        keyName: "ragkit_document_tags",
        searchable: true,
        searchableByDefault: false,
        unique: false,
        multiValued: true
    )!

    private static let recordsFolderName = "spotlight-records"
    private static let recordExtension = "record"
    private static let stateFileName = "spotlight-index.json"

    /// Every item of this database carries it, so filters keep searches to
    /// it and one delete removes it all. Dots nest Spotlight's domains, so the
    /// database name has none.
    let domainIdentifier: String
    private let index: CSSearchableIndex
    private var records: RAGSpotlightRecords?
    /// Held here because the index holds its delegate weakly.
    private var delegate: Delegate?
    /// Whether this index has run its throwaway query (see `warmUpQuery()`).
    private var isWarm = false

    /// - Throws: `RAGError.spotlightUnavailable` when this device cannot index
    ///   into Spotlight. Nothing on disk or in Spotlight changes before then.
    init(databaseName: String) throws {
        guard CSSearchableIndex.isIndexingAvailable() else {
            throw RAGError.spotlightUnavailable
        }
        domainIdentifier = Self.domainIdentifier(forDatabaseNamed: databaseName)
        // The index's name is what its client state is kept under, so each
        // database has a generation of its own.
        index = CSSearchableIndex(name: domainIdentifier)
    }

    static func domainIdentifier(forDatabaseNamed name: String) -> String {
        "RAGKit." + name.replacingOccurrences(of: ".", with: "_")
    }

    // MARK: - Opening

    /// Loads the records in `directory`, and starts both them and the
    /// database's items in Spotlight over when Spotlight no longer holds the
    /// generation they describe (or `startOver` says to).
    func open(in directory: URL, startOver: Bool) async throws {
        let records = try RAGSpotlightRecords(
            folder: directory.appendingPathComponent(Self.recordsFolderName, isDirectory: true),
            fileExtension: Self.recordExtension
        )
        let stateURL = directory.appendingPathComponent(Self.stateFileName)
        let recorded = (try? Data(contentsOf: stateURL)).flatMap { try? JSONDecoder().decode(State.self, from: $0) }
        let indexed = try await fetchLastClientState().map { String(decoding: $0, as: UTF8.self) }

        if startOver || recorded == nil || recorded?.generation != indexed {
            if !records.isEmpty || indexed != nil {
                RAGLog.warning("🔀 Spotlight no longer holds what this database recorded; starting its index over")
            }
            try await deleteAllItems()
            try records.removeAll()
            let generation = UUID().uuidString
            // Spotlight's copy first: a crash before the directory's is
            // written leaves the two unequal, and the next open starts over
            // again rather than trusting records Spotlight never confirmed.
            try await storeClientState(Data(generation.utf8))
            let state = try JSONEncoder().encode(State(generation: generation))
            try state.write(to: stateURL, options: .atomic)
        }

        self.records = records
        let delegate = Delegate(domainIdentifier: domainIdentifier, records: records)
        index.indexDelegate = delegate
        self.delegate = delegate
        RAGLog.debug("🔦 Spotlight index \(domainIdentifier) holds \(records.count) documents")
    }

    /// Spotlight's query model runs in the app's process, adds to its memory,
    /// and may be unloaded at any time; `CSUserQuery.prepare()` asks for it,
    /// and a query that comes before it has loaded runs on words alone.
    /// Developers have reported that the first query after `prepare()` can
    /// still miss it ("Text embedding generation timeout") while the ones
    /// after work, so an index's first search runs one throwaway query,
    /// filtered to match nothing, before its own. Both wait for a search, not
    /// for setup: a host that only keeps the index current (a background
    /// launch) never loads the model.
    private static func warmUpQuery() async {
        let context = CSUserQueryContext()
        context.fetchAttributes = []
        context.filterQueries = ["domainIdentifier == \"RAGKit.__warm-up__\""]
        context.maxSuggestionCount = 0
        let query = CSUserQuery(userQueryString: "warm up", userQueryContext: context)
        do {
            for try await _ in query.responses {}
        } catch {
            RAGLog.debug("ℹ️ Spotlight warm-up query ended: \(error)")
        }
    }

    // MARK: - Reading

    var documentCount: Int {
        records?.count ?? 0
    }

    func indexedDocuments() -> [UUID: RAGIndexedDocument] {
        (records?.all() ?? [:]).mapValues(\.indexed)
    }

    func indexedDocument(id: UUID) -> RAGIndexedDocument? {
        records?.record(for: id)?.indexed
    }

    // MARK: - Writing

    /// Indexes `documents` into Spotlight, replacing any item with the same
    /// ID, and records each one Spotlight took. Documents with empty text are
    /// skipped. A batch Spotlight refuses is retried one document at a time,
    /// so one bad document cannot keep the rest out.
    /// - Returns: The IDs Spotlight indexed.
    func upsert(
        _ documents: [RAGDocument],
        batchSize: Int,
        progress: EmbeddingProgressTracker?
    ) async throws -> [UUID] {
        guard let records else { throw RAGError.notInitialized }
        let indexable = documents.filter {
            !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        if indexable.count != documents.count {
            RAGLog.warning("⚠️ Skipping \(documents.count - indexable.count) documents with empty text")
        }
        guard !indexable.isEmpty else { return [] }

        let batchSize = max(batchSize, 1)
        let totalBatches = (indexable.count + batchSize - 1) / batchSize
        progress?.startEmbedding(totalBatches: totalBatches)

        var indexedIDs: [UUID] = []
        indexedIDs.reserveCapacity(indexable.count)
        for batchIndex in 0..<totalBatches {
            try Task.checkCancellation()
            let batch = Array(indexable[(batchIndex * batchSize)..<min((batchIndex + 1) * batchSize, indexable.count)])
            progress?.updateProgress(
                batch: batchIndex + 1,
                message: "Indexing batch \(batchIndex + 1) of \(totalBatches)..."
            )

            // A document without a date is dated when it is indexed, as the
            // vector engines date it.
            let now = Date()
            let entries = batch.map { document in
                (id: document.id, record: RAGSpotlightRecords.Record(
                    text: document.text,
                    date: document.date ?? now,
                    tags: document.tags
                ))
            }

            var indexed: [UUID: RAGSpotlightRecords.Record] = [:]
            do {
                try await indexItems(entries.map { item(id: $0.id, record: $0.record) })
                for entry in entries { indexed[entry.id] = entry.record }
            } catch {
                RAGLog.warning("⚠️ Spotlight refused a batch (\(error)); indexing it one document at a time")
                for entry in entries {
                    do {
                        try await indexItems([item(id: entry.id, record: entry.record)])
                        indexed[entry.id] = entry.record
                    } catch {
                        RAGLog.warning("⚠️ Failed to index document \(entry.id) in Spotlight: \(error)")
                    }
                }
            }
            try records.set(indexed)
            indexedIDs.append(contentsOf: entries.map(\.id).filter { indexed[$0] != nil })
        }

        progress?.finishEmbedding(success: true)
        RAGLog.debug("🔁 Indexed \(indexedIDs.count) of \(indexable.count) documents in Spotlight")
        return indexedIDs
    }

    /// Takes the documents out of Spotlight, then out of the records.
    /// Unknown IDs are ignored.
    func delete(ids: [UUID]) async throws {
        guard let records else { throw RAGError.notInitialized }
        guard !ids.isEmpty else { return }
        try await deleteItems(withIdentifiers: ids.map(identifier(for:)))
        try records.remove(ids)
        RAGLog.debug("🗑️ Deleted \(ids.count) documents from the Spotlight index")
    }

    /// Takes every one of the database's documents out of Spotlight and out
    /// of the records.
    func removeAll() async throws {
        guard let records else { throw RAGError.notInitialized }
        try await deleteAllItems()
        try records.removeAll()
    }

    /// Takes a database's items out of Spotlight when it moves to another
    /// engine. Spotlight would otherwise keep them, still found by the
    /// system's search, with nothing left to keep them current. Its records
    /// go with the database directory.
    static func removeItems(ofDatabaseNamed name: String) async {
        let domain = domainIdentifier(forDatabaseNamed: name)
        do {
            try await CSSearchableIndex(name: domain).deleteItems(inDomain: domain)
            RAGLog.debug("🗑️ Removed \(domain) from Spotlight")
        } catch {
            RAGLog.warning("⚠️ Could not remove \(domain) from Spotlight: \(error)")
        }
    }

    // MARK: - Search

    struct Hit {
        let id: UUID
        let text: String
        let score: Float
        let date: Date
    }

    /// Spotlight's best matches for `query`, best first, among the documents
    /// dated inside `dateRange` and carrying at least one of `tags`. Both
    /// filters run inside Spotlight, before ranking, and are checked again
    /// against the records: Spotlight compares dates to the second.
    func search(
        query: String,
        numResults: Int,
        threshold: Float?,
        dateRange: Range<Date>?,
        tags: [String]?
    ) async throws -> [Hit] {
        guard let records else { throw RAGError.notInitialized }
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        // An empty query string lists every item Spotlight holds.
        guard !query.isEmpty, numResults > 0 else { return [] }
        let tags = (tags?.isEmpty == false) ? tags : nil

        CSUserQuery.prepare()
        if !isWarm {
            isWarm = true
            await Self.warmUpQuery()
        }
        let filters = filterQueries(dateRange: dateRange, tags: tags)
        // Every match ranked. Spotlight returns the matches past
        // maxRankedResultCount as well, unranked, and they sort in among the
        // ranked ones: with three ranked, the one note holding every word of
        // "sourdough OR starter OR the" sorted below notes holding only
        // "the". Ranking all of them was no slower.
        let rankLimit = max(records.count, numResults)

        // The query, and each of its words on its own, side by side. The
        // words cost little: with them the search took 0.56 s, without them
        // 0.49 s.
        async let wholeQuery = Self.rankedIdentifiers(matching: query, filters: filters, rankLimit: rankLimit)
        let wordMatches = try await withThrowingTaskGroup(of: [String].self) { group in
            for word in Self.wordQueries(query) where word.caseInsensitiveCompare(query) != .orderedSame {
                group.addTask {
                    try await Self.rankedIdentifiers(matching: word, filters: filters, rankLimit: rankLimit)
                }
            }
            var lists: [[String]] = []
            for try await list in group {
                lists.append(list)
            }
            return lists
        }
        let ranked = Self.merge(wholeQuery: try await wholeQuery, wordMatches: wordMatches, documentCount: records.count)
        try Task.checkCancellation()

        let wanted = tags.map(Set.init)
        var hits: [Hit] = []
        for identifier in ranked {
            guard let id = documentID(forIdentifier: identifier),
                  let record = records.record(for: id) else { continue }
            if let dateRange, !dateRange.contains(record.date) { continue }
            if let wanted, wanted.isDisjoint(with: record.tags) { continue }
            let score = Self.score(forRank: hits.count)
            if let threshold, score < threshold { break }
            hits.append(Hit(id: id, text: record.text, score: score, date: record.date))
            if hits.count == numResults { break }
        }
        RAGLog.debug("🔦 Spotlight matched \(ranked.count) items, answered \(hits.count)")
        return hits
    }

    /// The identifiers of the items Spotlight matches to `text`, best first.
    private static func rankedIdentifiers(matching text: String, filters: [String], rankLimit: Int) async throws -> [String] {
        let context = CSUserQueryContext()
        context.fetchAttributes = []
        context.filterQueries = filters
        context.enableRankedResults = true
        context.maxRankedResultCount = rankLimit
        context.maxSuggestionCount = 0
        var found: [CSUserQuery.Item] = []
        for try await response in CSUserQuery(userQueryString: text, userQueryContext: context).responses {
            if case .item(let item) = response {
                found.append(item)
            }
        }
        // Descending is best first. Ascending put the note holding every word
        // of "sourdough OR starter OR the" below six that hold only "the".
        return found.sorted(by: >).map(\.item.uniqueIdentifier)
    }

    /// A Spotlight query for each distinct word of `query`, in order: at
    /// most eight, so a pasted paragraph does not fan out into a hundred
    /// queries. Words are found by `NLTagger`, not at spaces, so Chinese and
    /// Japanese queries have words too.
    ///
    /// Spotlight matches a query word only where it begins a word of the
    /// note, so "tomatoes" misses a note about a tomato, and "eating" one
    /// that says "eat". A word whose dictionary form differs is searched as
    /// both: "tomatoes OR tomato". Not a form shorter than three letters:
    /// "go", from "goes", also begins "good" and "got", and pushed the
    /// right note for "where my money goes each month" out of the top 10.
    ///
    /// A word the query quotes is searched quoted, and so is its dictionary
    /// form: `"tomatoes" OR "tomato"`. Quoted, Spotlight matches a word only
    /// whole and never reads it as a date, so a host that quotes its words
    /// for that keeps it in the word searches too. Left bare, `art` from
    /// `"art"` found the notes about an artichoke, an article and an artist,
    /// and `september` every note captured in September.
    static func wordQueries(_ query: String) -> [String] {
        let tagger = NLTagger(tagSchemes: [.lemma])
        tagger.string = query
        let quotedRanges = quotedRanges(in: query)
        var seen = Set<String>()
        var queries: [String] = []
        tagger.enumerateTags(
            in: query.startIndex..<query.endIndex,
            unit: .word,
            scheme: .lemma,
            options: [.omitWhitespace, .omitPunctuation, .omitOther]
        ) { lemma, range in
            let word = String(query[range])
            guard seen.insert(word.lowercased()).inserted else { return true }
            let isQuoted = quotedRanges.contains { $0.contains(range.lowerBound) }
            let form: (String) -> String = isQuoted ? { "\"\($0)\"" } : { $0 }
            if let lemma = lemma?.rawValue, lemma.count >= 3, lemma.lowercased() != word.lowercased() {
                queries.append("\(form(word)) OR \(form(lemma))")
            } else {
                queries.append(form(word))
            }
            return queries.count < maxQueryWords
        }
        return queries
    }

    static let maxQueryWords = 8

    /// The stretches of `query` between pairs of straight double quotes. A
    /// quote mark left without a partner opens nothing.
    private static func quotedRanges(in query: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var opening: String.Index?
        for index in query.indices where query[index] == "\"" {
            if let start = opening {
                ranges.append(query.index(after: start)..<index)
                opening = nil
            } else {
                opening = index
            }
        }
        return ranges
    }

    /// One ranking from Spotlight's answer to the whole query and to each of
    /// its words.
    ///
    /// Spotlight's own answer is precise and short. It matches a note by its
    /// words only when every word of the query begins a word of the note
    /// ("tax return" finds the tax note, "tax return due" does not), and by
    /// its meaning at most one note, and only a close one. On the eval's 40
    /// queries over 54 notes it never answered with more than one note, and
    /// often with none.
    ///
    /// So its answer leads, and behind it come the notes that hold some of
    /// the query's words. Each word counts by BM25's IDF, so a rare word
    /// ("sourdough") counts for far more than a common one ("the"), which
    /// counts for almost nothing; notes of equal weight go by how high the
    /// words' own searches ranked them. With Spotlight matching nothing by
    /// meaning, this took nDCG@10 on those queries, against hand labels,
    /// from 0.12 to 0.51.
    ///
    /// A note the words alone found has to weigh at least 0.4 of the rarest
    /// word's IDF (``fillFloor``), or it is left out: a note holding only
    /// "in" and "the" does not answer "in the garden". Without the floor,
    /// such notes filled every place the rarer words left. On the eval's 40
    /// queries, with meaning off, it took the notes the judges graded
    /// unrelated from 5.0 to 3.3 per top 10. nDCG@10 against the labels
    /// stayed the same (0.51), and against the judges it fell by less than
    /// 0.005, as a few notes graded "related" went. At a half it also lost
    /// relevant ones: the dog-training note for "teaching the new dog to
    /// come when called".
    ///
    /// BM25's length normalization is left out. Spotlight does not say how
    /// often a note holds a word, so a long note could never earn back the
    /// length it is penalized for: the one note about the Skye ferry fell
    /// from first to third for a question about it. Nearly empty notes rose
    /// instead, "todo" for the "to" in "flight to Chengdu". Over the 40
    /// queries it gained nothing.
    static func merge(wholeQuery: [String], wordMatches: [[String]], documentCount: Int) -> [String] {
        let documents = Double(documentCount)
        var weight: [String: Double] = [:]
        var placing: [String: Double] = [:]
        var rarest = 0.0
        for matches in wordMatches where !matches.isEmpty {
            let holding = Double(matches.count)
            // BM25's IDF, in the form that never turns negative.
            let idf = log(1 + max(documents - holding + 0.5, 0) / (holding + 0.5))
            rarest = max(rarest, idf)
            for (rank, identifier) in matches.enumerated() {
                weight[identifier, default: 0] += idf
                placing[identifier, default: 0] += 1 / Double(10 + rank)
            }
        }
        let floor = rarest * fillFloor
        let byWords = weight.keys
            .filter { weight[$0, default: 0] >= floor }
            .sorted {
                (weight[$0, default: 0], placing[$0, default: 0], $1) > (weight[$1, default: 0], placing[$1, default: 0], $0)
            }
        var seen = Set<String>()
        return (wholeQuery + byWords).filter { seen.insert($0).inserted }
    }

    /// How much of the rarest word's IDF a note the words alone found must
    /// weigh to be answered. See `merge`.
    static let fillFloor = 0.4

    /// The score a result carries for its place in Spotlight's ranking, best
    /// first from 0: 1, then 1 / (1 + rank / 10) — a half at the eleventh
    /// place, a quarter at the thirty-first. Never zero, so a positive
    /// threshold keeps a ranked prefix.
    static func score(forRank rank: Int) -> Float {
        1 / (1 + Float(max(rank, 0)) / 10)
    }

    private func filterQueries(dateRange: Range<Date>?, tags: [String]?) -> [String] {
        var filters = ["domainIdentifier == \(Self.quoted(domainIdentifier))"]
        if let dateRange {
            // Whole seconds, widened outward; the records narrow it back.
            let lower = Self.isoTime(dateRange.lowerBound.timeIntervalSince1970.rounded(.down))
            let upper = Self.isoTime(dateRange.upperBound.timeIntervalSince1970.rounded(.up))
            filters.append("contentCreationDate >= $time.iso(\(lower)) && contentCreationDate <= $time.iso(\(upper))")
        }
        if let tags, !tags.isEmpty {
            let key = Self.tagsKey.keyName
            filters.append("(" + tags.map { "\(key) == \(Self.quoted($0))" }.joined(separator: " || ") + ")")
        }
        return filters
    }

    private static func quoted(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }

    private static func isoTime(_ secondsSince1970: TimeInterval) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: Date(timeIntervalSince1970: secondsSince1970))
    }

    // MARK: - Items

    private func identifier(for id: UUID) -> String {
        "\(domainIdentifier):\(id.uuidString)"
    }

    private func documentID(forIdentifier identifier: String) -> UUID? {
        Self.documentID(forIdentifier: identifier, inDomain: domainIdentifier)
    }

    fileprivate static func documentID(forIdentifier identifier: String, inDomain domain: String) -> UUID? {
        let prefix = domain + ":"
        guard identifier.hasPrefix(prefix) else { return nil }
        return UUID(uuidString: String(identifier.dropFirst(prefix.count)))
    }

    private func item(id: UUID, record: RAGSpotlightRecords.Record) -> CSSearchableItem {
        let attributes = CSSearchableItemAttributeSet(contentType: .text)
        attributes.textContent = record.text
        attributes.contentCreationDate = record.date
        if !record.tags.isEmpty {
            attributes.setValue(record.tags as NSArray, forCustomKey: Self.tagsKey)
        }
        let item = CSSearchableItem(
            uniqueIdentifier: identifier(for: id),
            domainIdentifier: domainIdentifier,
            attributeSet: attributes
        )
        item.expirationDate = .distantFuture
        return item
    }

    // MARK: - Core Spotlight calls

    // Each completion-handler API is wrapped by hand: where Swift also sees
    // its synchronous form (the handler is optional), a bare call can bind to
    // that one and return before Spotlight has done anything.

    private func indexItems(_ items: [CSSearchableItem]) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            index.indexSearchableItems(items) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

    private func deleteItems(withIdentifiers identifiers: [String]) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            index.deleteSearchableItems(withIdentifiers: identifiers) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

    private func deleteAllItems() async throws {
        try await index.deleteItems(inDomain: domainIdentifier)
    }

    private func fetchLastClientState() async throws -> Data? {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data?, Error>) in
            index.fetchLastClientState { state, error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: state)
                }
            }
        }
    }

    /// Writes `state` as the index's client state, through an empty batch.
    private func storeClientState(_ state: Data) async throws {
        index.beginBatch()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            index.endBatch(withClientState: state) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

    /// The generation the database directory's records belong to.
    private struct State: Codable {
        let generation: String
    }

    // MARK: - Delegate

    /// Hears Spotlight ask for documents again, which it does when it has
    /// lost them: their records go, so the host's next diff finds them
    /// missing and indexes them again. Called on Spotlight's own queue, which
    /// is why the records lock themselves.
    private final class Delegate: NSObject, CSSearchableIndexDelegate {
        let domainIdentifier: String
        let records: RAGSpotlightRecords

        init(domainIdentifier: String, records: RAGSpotlightRecords) {
            self.domainIdentifier = domainIdentifier
            self.records = records
        }

        func searchableIndex(
            _ searchableIndex: CSSearchableIndex,
            reindexAllSearchableItemsWithAcknowledgementHandler acknowledgementHandler: @escaping () -> Void
        ) {
            do {
                try records.removeAll()
                RAGLog.warning("🔀 Spotlight asked for every document of \(domainIdentifier) again")
            } catch {
                RAGLog.warning("⚠️ Could not clear Spotlight records: \(error)")
            }
            acknowledgementHandler()
        }

        func searchableIndex(
            _ searchableIndex: CSSearchableIndex,
            reindexSearchableItemsWithIdentifiers identifiers: [String],
            acknowledgementHandler: @escaping () -> Void
        ) {
            let ids = identifiers.compactMap {
                RAGSpotlightIndex.documentID(forIdentifier: $0, inDomain: domainIdentifier)
            }
            do {
                try records.remove(ids)
            } catch {
                RAGLog.warning("⚠️ Could not drop Spotlight records: \(error)")
            }
            acknowledgementHandler()
        }
    }
}

private extension CSSearchableIndex {
    func deleteItems(inDomain domain: String) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            deleteSearchableItems(withDomainIdentifiers: [domain]) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }
}

// MARK: - Records

/// One file per document, holding the text, date and tags it was indexed into
/// Spotlight with. One file each rather than one for all, so saving a document
/// writes that document: a single file of every text would be rewritten whole
/// on each save. Not `.json`, which the database reads as VecturaKit's own
/// document files when it looks for them in a subdirectory.
///
/// Locked rather than isolated, because Spotlight's delegate reaches it from
/// its own queue while the database works with it from its owner's.
final class RAGSpotlightRecords: @unchecked Sendable {
    struct Record: Codable, Equatable {
        let text: String
        let date: Date
        let tags: [String]

        var indexed: RAGIndexedDocument {
            RAGIndexedDocument(text: text, date: date, tags: tags)
        }
    }

    private let folder: URL
    private let fileExtension: String
    private let lock = NSLock()
    private var records: [UUID: Record]

    init(folder: URL, fileExtension: String) throws {
        self.folder = folder
        self.fileExtension = fileExtension
        let fm = FileManager.default
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        var loaded: [UUID: Record] = [:]
        let decoder = JSONDecoder()
        for url in try fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        where url.pathExtension == fileExtension {
            guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent) else { continue }
            // An unreadable record is a document to index again, not an
            // error: dropping it makes the host's diff find it missing.
            guard let data = try? Data(contentsOf: url),
                  let record = try? decoder.decode(Record.self, from: data) else {
                try? fm.removeItem(at: url)
                continue
            }
            loaded[id] = record
        }
        records = loaded
    }

    var count: Int {
        lock.withLock { records.count }
    }

    var isEmpty: Bool {
        lock.withLock { records.isEmpty }
    }

    func all() -> [UUID: Record] {
        lock.withLock { records }
    }

    func record(for id: UUID) -> Record? {
        lock.withLock { records[id] }
    }

    func set(_ updates: [UUID: Record]) throws {
        guard !updates.isEmpty else { return }
        let encoder = JSONEncoder()
        try lock.withLock {
            for (id, record) in updates {
                try encoder.encode(record).write(to: fileURL(for: id), options: .atomic)
                records[id] = record
            }
        }
    }

    func remove(_ ids: [UUID]) throws {
        guard !ids.isEmpty else { return }
        let fm = FileManager.default
        try lock.withLock {
            for id in ids where records.removeValue(forKey: id) != nil {
                let url = fileURL(for: id)
                if fm.fileExists(atPath: url.path) {
                    try fm.removeItem(at: url)
                }
            }
        }
    }

    func removeAll() throws {
        let fm = FileManager.default
        try lock.withLock {
            records = [:]
            for url in try fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            where url.pathExtension == fileExtension {
                try fm.removeItem(at: url)
            }
        }
    }

    private func fileURL(for id: UUID) -> URL {
        folder.appendingPathComponent(id.uuidString).appendingPathExtension(fileExtension)
    }
}
