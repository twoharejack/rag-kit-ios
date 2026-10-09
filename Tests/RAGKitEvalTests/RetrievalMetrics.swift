// RetrievalMetrics.swift
// ============================================================================
// Ranking metrics over graded relevance, and agreement between two graders.
// Pure functions of document IDs and 0–3 grades, shared by the judged scores
// and the hand-labeled ones.
// ============================================================================

import Foundation

enum RetrievalMetrics {
    /// Where the binary metrics (MRR, precision, recall) start counting a
    /// document as relevant: "substantially about the query".
    static let relevantGrade = 2

    /// Normalized discounted cumulative gain of the top `k`, with a gain of
    /// 2^grade − 1. The ideal ranking is built from every graded document,
    /// not only the retrieved ones, so an engine that misses a relevant
    /// document pays for it. `nil` when no document grades above 0.
    static func ndcg(ranked: [String], grades: [String: Int], k: Int) -> Double? {
        let ideal = dcg(grades.values.sorted(by: >).prefix(k))
        guard ideal > 0 else { return nil }
        return dcg(ranked.prefix(k).map { grades[$0] ?? 0 }) / ideal
    }

    static func dcg<Grades: Sequence<Int>>(_ grades: Grades) -> Double {
        zip(grades, 1...).reduce(0) { sum, entry in
            let (grade, position) = entry
            return sum + (pow(2, Double(grade)) - 1) / log2(Double(position) + 1)
        }
    }

    /// 1 / the rank of the first relevant document, 0 when none was
    /// retrieved. `nil` when the query has no relevant document at all.
    static func reciprocalRank(ranked: [String], grades: [String: Int]) -> Double? {
        guard hasRelevant(grades) else { return nil }
        guard let index = ranked.firstIndex(where: { isRelevant($0, grades) }) else { return 0 }
        return 1 / Double(index + 1)
    }

    /// The share of the top `k` that is relevant. A query with fewer than `k`
    /// relevant documents cannot reach 1. `nil` when it has none.
    static func precision(ranked: [String], grades: [String: Int], k: Int) -> Double? {
        guard hasRelevant(grades), k > 0 else { return nil }
        return Double(ranked.prefix(k).filter { isRelevant($0, grades) }.count) / Double(k)
    }

    /// The share of the relevant documents that made the top `k`. `nil` when
    /// there are none.
    static func recall(ranked: [String], grades: [String: Int], k: Int) -> Double? {
        let relevant = grades.filter { $0.value >= relevantGrade }.count
        guard relevant > 0 else { return nil }
        return Double(ranked.prefix(k).filter { isRelevant($0, grades) }.count) / Double(relevant)
    }

    /// The mean of the values that exist, `nil` when none does.
    static func mean(_ values: some Sequence<Double?>) -> Double? {
        let present = values.compactMap { $0 }
        guard !present.isEmpty else { return nil }
        return present.reduce(0, +) / Double(present.count)
    }

    private static func hasRelevant(_ grades: [String: Int]) -> Bool {
        grades.values.contains { $0 >= relevantGrade }
    }

    private static func isRelevant(_ id: String, _ grades: [String: Int]) -> Bool {
        (grades[id] ?? 0) >= relevantGrade
    }
}

/// How closely two graders agree over the same (query, document) pairs.
struct GradeAgreement: Equatable, Sendable {
    let pairs: Int
    /// The share of pairs given the same grade.
    let exact: Double
    /// The share of pairs whose grades differ by at most one.
    let withinOne: Double
    /// Cohen's κ on relevant (grade ≥ 2) or not: their agreement beyond
    /// what their base rates alone would produce. `nil` when chance
    /// agreement is already total (both call every pair relevant, or none).
    let kappa: Double?

    init?(_ grades: [(Int, Int)]) {
        guard !grades.isEmpty else { return nil }
        let count = Double(grades.count)
        pairs = grades.count
        exact = Double(grades.filter { $0.0 == $0.1 }.count) / count
        withinOne = Double(grades.filter { abs($0.0 - $0.1) <= 1 }.count) / count

        let threshold = RetrievalMetrics.relevantGrade
        let first = grades.map { $0.0 >= threshold }
        let second = grades.map { $0.1 >= threshold }
        let observed = Double(zip(first, second).filter { $0 == $1 }.count) / count
        let firstRate = Double(first.filter { $0 }.count) / count
        let secondRate = Double(second.filter { $0 }.count) / count
        let chance = firstRate * secondRate + (1 - firstRate) * (1 - secondRate)
        kappa = chance < 1 ? (observed - chance) / (1 - chance) : nil
    }
}
