// RetrievalEvalTests.swift
// ============================================================================
// The retrieval eval itself, opt-in through the environment, and the harness
// tests that keep it honest on every `swift test` without a model or an LLM.
// ============================================================================

import Foundation
import Testing

@Suite("Retrieval eval")
struct RetrievalEvalTests {
    /// Lowest nDCG@10 each engine may score on the bundled set, averaged over
    /// the judges (or against the labels when no judge runs; labeled scores
    /// run a little higher). Set 0.05 under what the engines scored, judged
    /// by claude-sonnet-5-5 and codex-gpt-5.6-sol, when these were written
    /// (0.43, 0.67, 0.76, 0.66): room for a judge model to be swapped, small
    /// enough to catch a regression. Raise one when its engine improves.
    static let ndcgFloors: [EvalEngine: Double] = [
        .lexical: 0.38,
        .minilm: 0.62,
        .minilmMean: 0.71,
        .apple: 0.61,
    ]

    /// Of the documents labeled relevant, the share a judge must grade
    /// relevant too, below which its grades are not worth scoring with.
    static let minimumJudgeAgreement = 0.85

    @Test(
        "Retrieval accuracy, judged by LLMs",
        .enabled(
            if: EvalSettings.isRequested,
            "Set RAGKIT_EVAL_JUDGES=claude,codex (and optionally RAGKIT_EVAL_ENGINES) to run it; see Docs/Evals.md"
        ),
        .timeLimit(.minutes(60))
    )
    func retrievalAccuracy() async throws {
        let settings = try EvalSettings.fromEnvironment()
        let set = try EvalSet.load(from: settings.setURL)
        let evaluator = RetrievalEvaluator(
            set: set,
            options: settings.options,
            judges: settings.judges,
            cacheDirectory: settings.cacheDirectory
        )
        let result = try await evaluator.run(engines: settings.engines)

        let report = EvalReport(result)
        let files = try report.write(to: settings.outputDirectory)
        print(report.markdown)
        print("[RAGKit eval] Report: \(files.markdown.path)\n[RAGKit eval] Rankings and grades: \(files.json.path)")

        for judge in result.judges {
            let activity = result.activity[judge] ?? JudgeActivity()
            #expect(
                activity.failures.isEmpty,
                "\(judge) failed \(activity.failures.count) of \(activity.calls) calls, first: \(activity.failures.first ?? "")"
            )
            if let agreement = result.labelAgreement(of: judge)?.labeledRelevantJudgedRelevant {
                #expect(
                    agreement >= Self.minimumJudgeAgreement,
                    "\(judge) graded only \(Int(agreement * 100))% of the labeled-relevant documents relevant"
                )
            }
        }

        // Floors only mean something on the set they were measured on, at the
        // depth they were measured at.
        guard settings.isBundledSet, settings.options.depth == EvalOptions().depth else { return }
        for run in result.runs {
            guard let floor = Self.ndcgFloors[run.engine], let ndcg = result.headlineNDCG(of: run) else { continue }
            #expect(ndcg >= floor, "\(run.engine.rawValue) scored nDCG@10 \(String(format: "%.2f", ndcg)), under its floor of \(floor)")
        }
    }
}

@Suite("Eval harness")
struct EvalHarnessTests {
    @Test("The bundled set is valid and fully labeled")
    func bundledSet() throws {
        let set = try EvalSet.load(from: EvalSet.bundledURL)
        #expect(set.documents.count >= 50)
        #expect(set.queries.allSatisfy { $0.relevant?.values.contains { $0 >= 2 } == true })
        #expect(Set(set.documents.map { EvalSet.uuid(for: $0.id) }).count == set.documents.count)
    }

    @Test("A set that labels a missing document is rejected")
    func invalidSet() throws {
        let json = """
            {"name": "broken", "languages": ["en"],
             "documents": [{"id": "a", "text": "A note."}],
             "queries": [{"id": "q", "text": "note", "relevant": {"b": 3}}]}
            """
        let set = try JSONDecoder().decode(EvalSet.self, from: Data(json.utf8))
        #expect(throws: EvalSetError.self) { try set.validate() }
    }

    /// A judge that answers with the hand labels makes the judged scores and
    /// the labeled ones the same numbers, which checks the pooling, the
    /// mapping of grades back to documents, and the cache end to end without
    /// a model or an LLM.
    @Test("Judged scores match the labels when the judge is the labels")
    func labelJudgeRoundTrip() async throws {
        let set = try EvalSet.load(from: EvalSet.bundledURL)
        let cacheDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RAGKitEval-cache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: cacheDirectory) }

        let judge = LabelJudge(set: set)
        let evaluator = RetrievalEvaluator(set: set, options: EvalOptions(), judges: [judge], cacheDirectory: cacheDirectory)
        let result = try await evaluator.run(engines: [.lexical])
        let run = try #require(result.runs.first)

        let judged = result.judgedScores(of: run, judge: judge.name)
        let labeled = try #require(result.labeledScores(of: run))
        #expect(abs((judged.ndcg ?? -1) - (labeled.ndcg ?? -2)) < 1e-9)
        #expect(abs((judged.mrr ?? -1) - (labeled.mrr ?? -2)) < 1e-9)
        #expect(result.labelAgreement(of: judge.name)?.labeledRelevantJudgedRelevant == 1)
        #expect(result.activity[judge.name]?.failures.isEmpty == true)

        // Words alone find a note that shares the query's words.
        let keyword = set.queries.filter { $0.kind == "keyword" }
        #expect((result.labeledScores(of: run, queries: keyword)?.mrr ?? 0) >= 0.8)

        // Judging the same rankings again asks the judge nothing: every pair
        // is in the cache, read back from disk by a new evaluator.
        let rerun = try await RetrievalEvaluator(
            set: set,
            options: EvalOptions(),
            judges: [judge],
            cacheDirectory: cacheDirectory
        ).evaluate(result.runs)
        #expect(rerun.activity[judge.name]?.calls == 0)
        #expect(rerun.activity[judge.name]?.cached == result.activity[judge.name]?.graded)

        let report = EvalReport(rerun).markdown
        #expect(report.contains("| lexical |"))
        #expect(report.contains("## Judged by the LLMs"))
    }
}

/// Grades with the set's hand labels, 0 for anything unlabeled.
private struct LabelJudge: RelevanceJudge {
    let name = "labels"
    let labelsByQuery: [String: [String: Int]]
    let documentIDsByText: [String: String]

    init(set: EvalSet) {
        labelsByQuery = Dictionary(set.queries.map { ($0.text, $0.relevant ?? [:]) }) { first, _ in first }
        documentIDsByText = Dictionary(set.documents.map { ($0.text, $0.id) }) { first, _ in first }
    }

    func grade(query: String, passages: [JudgedPassage]) async throws -> [String: Int] {
        let labels = labelsByQuery[query] ?? [:]
        return Dictionary(uniqueKeysWithValues: passages.map { passage in
            (passage.label, documentIDsByText[passage.text].flatMap { labels[$0] } ?? 0)
        })
    }
}
