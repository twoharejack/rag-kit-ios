// RetrievalMetricsTests.swift
// ============================================================================
// The metrics, the engines' tie handling, and the judges' prompt and answer
// parsing, against values worked out by hand.
// ============================================================================

import Foundation
import RAGKit
import Testing

@Suite("Retrieval metrics")
struct RetrievalMetricsTests {
    let grades = ["a": 3, "b": 2, "c": 0]

    @Test("nDCG against a hand-worked ranking")
    func ndcg() throws {
        // DCG = (2²−1)/log₂2 + (2³−1)/log₂3; ideal = (2³−1)/log₂2 + (2²−1)/log₂3.
        let expected = (3 + 7 / log2(3.0)) / (7 + 3 / log2(3.0))
        let value = try #require(RetrievalMetrics.ndcg(ranked: ["b", "a", "c"], grades: grades, k: 10))
        #expect(abs(value - expected) < 1e-12)
        #expect(RetrievalMetrics.ndcg(ranked: ["a", "b"], grades: grades, k: 10) == 1)
    }

    @Test("nDCG's ideal ranking includes relevant documents no engine retrieved")
    func ndcgCountsMissedDocuments() throws {
        let value = try #require(RetrievalMetrics.ndcg(ranked: ["a", "x", "y"], grades: grades, k: 10))
        #expect(value < 1)
        #expect(RetrievalMetrics.ndcg(ranked: ["a"], grades: ["a": 0], k: 10) == nil)
    }

    @Test("nDCG only reads the top k")
    func ndcgCutoff() {
        #expect(RetrievalMetrics.ndcg(ranked: ["c", "a", "b"], grades: grades, k: 1) == 0)
    }

    @Test("Reciprocal rank counts the first document graded 2 or more")
    func reciprocalRank() {
        #expect(RetrievalMetrics.reciprocalRank(ranked: ["c", "x", "b"], grades: grades) == 1.0 / 3)
        #expect(RetrievalMetrics.reciprocalRank(ranked: ["c", "x"], grades: grades) == 0)
        #expect(RetrievalMetrics.reciprocalRank(ranked: ["a"], grades: ["a": 1]) == nil)
    }

    @Test("Precision and recall")
    func precisionAndRecall() {
        #expect(RetrievalMetrics.precision(ranked: ["a", "c", "x", "y", "b", "z"], grades: grades, k: 5) == 0.4)
        #expect(RetrievalMetrics.recall(ranked: ["a", "c", "x"], grades: grades, k: 10) == 0.5)
        #expect(RetrievalMetrics.recall(ranked: ["a", "c", "b"], grades: grades, k: 2) == 0.5)
        #expect(RetrievalMetrics.precision(ranked: ["a"], grades: ["a": 1], k: 5) == nil)
    }

    @Test("The mean skips queries a metric is undefined for")
    func mean() {
        #expect(RetrievalMetrics.mean([1, nil, 0.5]) == 0.75)
        #expect(RetrievalMetrics.mean([nil, nil]) == nil)
    }

    @Test("Agreement and Cohen's kappa against a hand-worked example")
    func agreement() throws {
        // Relevant or not: (T,T) (F,F) (T,F) (F,F): observed 3/4; base rates
        // 1/2 and 1/4, so chance = 1/8 + 3/8 = 1/2 and κ = (3/4 − 1/2) / (1/2).
        let agreement = try #require(GradeAgreement([(3, 3), (0, 0), (2, 1), (1, 0)]))
        #expect(agreement.pairs == 4)
        #expect(agreement.exact == 0.5)
        #expect(agreement.withinOne == 1)
        #expect(abs((agreement.kappa ?? 0) - 0.5) < 1e-12)
        #expect(GradeAgreement([(3, 2), (2, 3)])?.kappa == nil)
        #expect(GradeAgreement([]) == nil)
    }
}

@Suite("Engine rankings")
struct EngineRankingTests {
    let ids = ["a", "b", "c", "d", "e"]

    func results(_ scores: [(String, Float)]) -> [VecturaSearchResult] {
        scores.map { VecturaSearchResult(id: EvalSet.uuid(for: $0.0), text: "", score: $0.1, createdAt: Date()) }
    }

    var documentIDs: [UUID: String] {
        Dictionary(uniqueKeysWithValues: ids.map { (EvalSet.uuid(for: $0), $0) })
    }

    @Test("Ties are ordered by ID and results scored 0 are left out")
    func ties() {
        let ranked = EvalEngine.ranking(of: results([("c", 0.5), ("b", 0.5), ("a", 0.9), ("d", 0)]), depth: 3, documentIDs: documentIDs)
        #expect(ranked == ["a", "b", "c"])
    }

    @Test("Equal scores are ordered by ID down to the last place")
    func cutoffTie() {
        let tied = results([("a", 0.9), ("d", 0.4), ("c", 0.4), ("b", 0.4)])
        #expect(EvalEngine.ranking(of: tied, depth: 3, documentIDs: documentIDs) == ["a", "b", "c"])
    }
}

@Suite("Judge prompt")
struct JudgePromptTests {
    let passages = [
        JudgedPassage(label: "N1", text: "Ferns like damp soil."),
        JudgedPassage(label: "N2", text: "The train leaves at nine."),
    ]

    @Test("The prompt carries the query, every note and its label")
    func render() {
        let prompt = JudgePrompt.render(query: "watering a fern", passages: passages)
        #expect(prompt.contains("<query>watering a fern</query>"))
        #expect(prompt.contains("<note id=\"N1\">\nFerns like damp soil.\n</note>"))
        #expect(prompt.contains("<note id=\"N2\">"))
    }

    @Test("Grades parse from bare JSON and from JSON wrapped in prose or a fence")
    func parse() throws {
        let bare = #"{"grades":[{"note":"N1","grade":3},{"note":"N2","grade":0}]}"#
        #expect(try JudgePrompt.parseGrades(bare, labels: ["N1", "N2"]) == ["N1": 3, "N2": 0])

        let fenced = "Here you go:\n```json\n\(bare)\n```"
        #expect(try JudgePrompt.parseGrades(fenced, labels: ["N1", "N2"]) == ["N1": 3, "N2": 0])

        let extra = #"{"grades":[{"note":"N1","grade":2},{"note":"N9","grade":3}]}"#
        #expect(try JudgePrompt.parseGrades(extra, labels: ["N1"]) == ["N1": 2])
    }

    @Test("An answer that skips a note, leaves the scale, or is not JSON is refused")
    func refuse() {
        #expect(throws: JudgeError.self) {
            try JudgePrompt.parseGrades(#"{"grades":[{"note":"N1","grade":3}]}"#, labels: ["N1", "N2"])
        }
        #expect(throws: JudgeError.self) {
            try JudgePrompt.parseGrades(#"{"grades":[{"note":"N1","grade":4}]}"#, labels: ["N1"])
        }
        #expect(throws: JudgeError.self) {
            try JudgePrompt.parseGrades("I can't grade these.", labels: ["N1"])
        }
    }

    @Test("Claude's JSON envelope yields its structured output, its text, or its error")
    func claudeEnvelope() throws {
        let structured = #"{"type":"result","is_error":false,"result":"ignored","structured_output":{"grades":[{"note":"N1","grade":1}]}}"#
        #expect(try JudgePrompt.parseGrades(ClaudeCLIJudge.answer(in: structured), labels: ["N1"]) == ["N1": 1])

        let text = #"{"type":"result","is_error":false,"result":"{\"grades\":[{\"note\":\"N1\",\"grade\":2}]}"}"#
        #expect(try JudgePrompt.parseGrades(ClaudeCLIJudge.answer(in: text), labels: ["N1"]) == ["N1": 2])

        let failure = #"{"type":"result","is_error":true,"result":"Not logged in"}"#
        #expect(throws: JudgeError.self) { try ClaudeCLIJudge.answer(in: failure) }
    }

    @Test("Unknown judges and engines are named in the error")
    func settings() {
        #expect(throws: JudgeError.self) { try JudgeSpec.judges(from: "gemini") }
        #expect(throws: SettingsError.self) { try EvalSettings.parseEngines("minilm,bert") }
        #expect((try? EvalSettings.parseEngines(nil)) == EvalEngine.defaults)
        #expect((try? EvalSettings.parseEngines("all")) == EvalEngine.allCases)
    }
}
