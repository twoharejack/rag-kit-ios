// RetrievalEvaluator.swift
// ============================================================================
// One eval from end to end: every engine over the set, every judge over the
// pooled results, and the scores both add up to.
// ============================================================================

import CryptoKit
import Foundation

struct EvalOptions: Sendable {
    /// How many documents each engine returns per query: what nDCG and
    /// recall look at, and how deep the judges' pools go.
    var depth = 10
    var precisionCutoff = 5
    /// Documents per judge call. Each call repeats the instructions, so
    /// larger batches cost less; much larger ones and a judge's attention
    /// starts to drift down the list.
    var batchSize = 10
    var judgeConcurrency = 4
}

/// What one judge did during an eval.
struct JudgeActivity: Sendable {
    /// Pairs whose grade came from the cache.
    var cached = 0
    /// Pairs graded by calls made in this run.
    var graded = 0
    var calls = 0
    var failures: [String] = []
}

/// The engines' rankings and the judges' grades, which every score in the
/// report is computed from.
struct EvalResult: Sendable {
    let set: EvalSet
    let options: EvalOptions
    let runs: [EngineRun]
    let judges: [String]
    /// Grades by judge, query ID and document ID, for every document in the
    /// query's pool.
    let judgments: [String: [String: [String: Int]]]
    let activity: [String: JudgeActivity]
}

struct RetrievalEvaluator {
    let set: EvalSet
    let options: EvalOptions
    let judges: [any RelevanceJudge]
    /// Where each judge's `JudgmentCache` lives, one file per judge.
    let cacheDirectory: URL

    func run(engines: [EvalEngine]) async throws -> EvalResult {
        try await evaluate(retrieve(with: engines))
    }

    /// Every engine's rankings, one engine at a time, with exact ties settled
    /// the same way every run (see `EvalEngine.ranking(of:depth:documentIDs:)`),
    /// so a re-run has next to nothing new to grade.
    func retrieve(with engines: [EvalEngine]) async throws -> [EngineRun] {
        var runs: [EngineRun] = []
        for engine in engines {
            log("Indexing and searching with \(engine.rawValue)")
            runs.append(try await engine.run(on: set, depth: options.depth))
        }
        return runs
    }

    /// Has every judge grade the pooled rankings, the judges side by side,
    /// and gathers the result.
    func evaluate(_ runs: [EngineRun]) async throws -> EvalResult {
        let pools = pools(of: runs)
        var judgments: [String: [String: [String: Int]]] = [:]
        var activity: [String: JudgeActivity] = [:]
        try await withThrowingTaskGroup(of: (String, [String: [String: Int]], JudgeActivity).self) { group in
            for judge in judges {
                group.addTask {
                    let cache = try JudgmentCache(
                        fileURL: cacheDirectory.appendingPathComponent("\(judge.name).json")
                    )
                    let (grades, judgeActivity) = try await grade(pools, with: judge, cache: cache)
                    return (judge.name, grades, judgeActivity)
                }
            }
            for try await (name, grades, judgeActivity) in group {
                judgments[name] = grades
                activity[name] = judgeActivity
            }
        }
        return EvalResult(
            set: set,
            options: options,
            runs: runs,
            judges: judges.map(\.name),
            judgments: judgments,
            activity: activity
        )
    }

    // MARK: - Pooling

    /// Per query, every document any engine returned plus every labeled one:
    /// what the judges grade. Grading the labeled documents too means an
    /// engine that misses one still loses it from its judged nDCG, and the
    /// judges can be checked against the labels.
    ///
    /// The order is a hash of the IDs, so where a document sits in front of a
    /// judge says nothing about how any engine ranked it.
    func pools(of runs: [EngineRun]) -> [(query: EvalSet.Query, documents: [EvalSet.Document])] {
        let documentsByID = set.documentsByID
        return set.queries.map { query in
            var ids = Set(runs.flatMap { $0.rankings[query.id] ?? [] })
            ids.formUnion((query.relevant ?? [:]).filter { $0.value > 0 }.keys)
            let documents = ids
                .compactMap { documentsByID[$0] }
                .sorted { Self.shuffleKey(query.id, $0.id) < Self.shuffleKey(query.id, $1.id) }
            return (query, documents)
        }
    }

    private static func shuffleKey(_ queryID: String, _ documentID: String) -> String {
        SHA256.hash(data: Data("\(queryID)/\(documentID)".utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Judging

    private struct Batch: Sendable {
        let query: EvalSet.Query
        let documents: [EvalSet.Document]
    }

    private func grade(
        _ pools: [(query: EvalSet.Query, documents: [EvalSet.Document])],
        with judge: any RelevanceJudge,
        cache: JudgmentCache
    ) async throws -> ([String: [String: Int]], JudgeActivity) {
        var grades: [String: [String: Int]] = [:]
        var activity = JudgeActivity()
        var batches: [Batch] = []

        for (query, documents) in pools {
            var missing: [EvalSet.Document] = []
            for document in documents {
                if let grade = await cache.grade(query: query.text, document: document.text) {
                    grades[query.id, default: [:]][document.id] = grade
                    activity.cached += 1
                } else {
                    missing.append(document)
                }
            }
            for start in stride(from: 0, to: missing.count, by: options.batchSize) {
                let end = min(start + options.batchSize, missing.count)
                batches.append(Batch(query: query, documents: Array(missing[start..<end])))
            }
        }

        let pairs = batches.reduce(0) { $0 + $1.documents.count }
        log("\(judge.name): \(activity.cached) pairs cached, \(pairs) to grade in \(batches.count) calls")
        guard !batches.isEmpty else { return (grades, activity) }

        func absorb(_ batch: Batch, _ outcome: Result<[String: Int], Error>) async throws {
            activity.calls += 1
            switch outcome {
            case .success(let batchGrades):
                for document in batch.documents {
                    guard let grade = batchGrades[document.id] else { continue }
                    grades[batch.query.id, default: [:]][document.id] = grade
                    await cache.record(grade, query: batch.query, document: document)
                    activity.graded += 1
                }
                // Saved as it goes, so an interrupted run keeps what it paid for.
                try await cache.save()
            case .failure(let error):
                activity.failures.append("\(batch.query.id): \(error)")
            }
            if activity.calls % 10 == 0 || activity.calls == batches.count {
                log("\(judge.name): \(activity.calls)/\(batches.count) calls")
            }
        }

        // One call alone first: a judge that cannot answer at all (not signed
        // in, a model this account cannot use) fails once, not once per batch.
        let first = await grade(batches[0], with: judge)
        try await absorb(batches[0], first)
        if case .failure = first {
            log("\(judge.name) could not grade; skipping its other \(batches.count - 1) calls")
            return (grades, activity)
        }

        var pending = batches.dropFirst()
        try await withThrowingTaskGroup(of: (Batch, Result<[String: Int], Error>).self) { group in
            for _ in 0..<options.judgeConcurrency {
                guard let batch = pending.popFirst() else { break }
                group.addTask { (batch, await grade(batch, with: judge)) }
            }
            for try await (batch, outcome) in group {
                try await absorb(batch, outcome)
                if let next = pending.popFirst() {
                    group.addTask { (next, await grade(next, with: judge)) }
                }
            }
        }
        return (grades, activity)
    }

    /// Grades one batch, by document ID, trying up to three times: now and
    /// then a judge's answer misses a note or breaks its JSON, or the call
    /// stalls until `JudgeProcess.timeout` (in one run, four of Claude's 116
    /// calls stalled twice in a row, then answered in seconds when replayed).
    private func grade(_ batch: Batch, with judge: any RelevanceJudge) async -> Result<[String: Int], Error> {
        let passages = batch.documents.enumerated().map { index, document in
            JudgedPassage(label: "N\(index + 1)", text: document.text)
        }
        var lastError: Error = CancellationError()
        for attempt in 1...3 {
            do {
                if attempt > 1 { try await Task.sleep(for: .seconds(5 * (attempt - 1))) }
                let grades = try await judge.grade(query: batch.query.text, passages: passages)
                return .success(Dictionary(uniqueKeysWithValues: zip(batch.documents, passages).compactMap { document, passage in
                    grades[passage.label].map { (document.id, $0) }
                }))
            } catch is CancellationError {
                return .failure(CancellationError())
            } catch {
                lastError = error
            }
        }
        return .failure(lastError)
    }

    private func log(_ message: String) {
        print("[RAGKit eval] \(message)")
    }
}

// MARK: - Scores

/// One engine's scores under one source of grades, averaged over the queries
/// the source finds anything relevant for.
struct MetricScores: Sendable {
    let ndcg: Double?
    let mrr: Double?
    let precision: Double?
    let recall: Double?
}

/// How one judge's grades line up with the hand labels.
struct LabelAgreement: Sendable {
    /// Of the pairs labeled relevant (2 or more), the share the judge also
    /// grades relevant: a judge that misreads the task fails this first.
    let labeledRelevantJudgedRelevant: Double?
    /// Of the pooled pairs the labels leave out, the share the judge grades
    /// relevant: relevant notes the labels missed, or a lenient judge.
    let unlabeledJudgedRelevant: Double?
    /// Treating a document without a label as a 0.
    let agreement: GradeAgreement?
}

extension EvalResult {
    var precisionCutoff: Int { min(options.precisionCutoff, options.depth) }

    /// `judge`'s grades for a query's pool.
    func grades(of judge: String, for query: EvalSet.Query) -> [String: Int] {
        judgments[judge]?[query.id] ?? [:]
    }

    /// The mean of the judges' grades per document, or the labels when no
    /// judge ran: what the report counts as relevant when it shows one view.
    func consensusGrades(for query: EvalSet.Query) -> [String: Double] {
        guard !judges.isEmpty else { return (query.relevant ?? [:]).mapValues(Double.init) }
        var sums: [String: (total: Int, count: Int)] = [:]
        for judge in judges {
            for (document, grade) in grades(of: judge, for: query) {
                sums[document, default: (0, 0)].total += grade
                sums[document, default: (0, 0)].count += 1
            }
        }
        return sums.mapValues { Double($0.total) / Double($0.count) }
    }

    func scores(
        of run: EngineRun,
        queries: [EvalSet.Query]? = nil,
        grades: (EvalSet.Query) -> [String: Int]
    ) -> MetricScores {
        let queries = queries ?? set.queries
        let ranked = queries.map { (run.rankings[$0.id] ?? [], grades($0)) }
        return MetricScores(
            ndcg: RetrievalMetrics.mean(ranked.map { RetrievalMetrics.ndcg(ranked: $0, grades: $1, k: options.depth) }),
            mrr: RetrievalMetrics.mean(ranked.map { RetrievalMetrics.reciprocalRank(ranked: $0, grades: $1) }),
            precision: RetrievalMetrics.mean(ranked.map { RetrievalMetrics.precision(ranked: $0, grades: $1, k: precisionCutoff) }),
            recall: RetrievalMetrics.mean(ranked.map { RetrievalMetrics.recall(ranked: $0, grades: $1, k: options.depth) })
        )
    }

    func judgedScores(of run: EngineRun, judge: String, queries: [EvalSet.Query]? = nil) -> MetricScores {
        scores(of: run, queries: queries) { grades(of: judge, for: $0) }
    }

    func labeledScores(of run: EngineRun, queries: [EvalSet.Query]? = nil) -> MetricScores? {
        guard set.hasLabels else { return nil }
        return scores(of: run, queries: queries) { $0.relevant ?? [:] }
    }

    /// nDCG averaged over the judges, or against the labels when no judge
    /// ran: the one number per engine the report and the floors use.
    func headlineNDCG(of run: EngineRun, queries: [EvalSet.Query]? = nil) -> Double? {
        guard !judges.isEmpty else { return labeledScores(of: run, queries: queries)?.ndcg }
        return RetrievalMetrics.mean(judges.map { judgedScores(of: run, judge: $0, queries: queries).ndcg })
    }

    func labelAgreement(of judge: String) -> LabelAgreement? {
        guard set.hasLabels else { return nil }
        var pairs: [(label: Int, judged: Int)] = []
        for query in set.queries {
            let labels = query.relevant ?? [:]
            for (document, grade) in grades(of: judge, for: query) {
                pairs.append((labels[document] ?? 0, grade))
            }
        }
        let threshold = RetrievalMetrics.relevantGrade
        func share(of subset: [(label: Int, judged: Int)]) -> Double? {
            subset.isEmpty ? nil : Double(subset.filter { $0.judged >= threshold }.count) / Double(subset.count)
        }
        return LabelAgreement(
            labeledRelevantJudgedRelevant: share(of: pairs.filter { $0.label >= threshold }),
            unlabeledJudgedRelevant: share(of: pairs.filter { $0.label == 0 }),
            agreement: GradeAgreement(pairs.map { ($0.label, $0.judged) })
        )
    }

    func agreement(between first: String, and second: String) -> GradeAgreement? {
        var pairs: [(Int, Int)] = []
        for query in set.queries {
            let other = grades(of: second, for: query)
            for (document, grade) in grades(of: first, for: query) {
                if let otherGrade = other[document] { pairs.append((grade, otherGrade)) }
            }
        }
        return GradeAgreement(pairs)
    }

    /// The query kinds in the order the set first uses them.
    var kinds: [String] {
        var seen = Set<String>()
        return set.queries.compactMap { $0.kind }.filter { seen.insert($0).inserted }
    }
}
