// MMRDiversifier.swift
// ============================================================================
// Maximal Marginal Relevance re-ranking for search results: balances query
// relevance against similarity to already-selected results so the final set
// stays diverse. Doc-doc similarity is Jaccard over word tokens, optionally
// sharpened by a host-supplied concept key (two results with the same key are
// treated as duplicates).
// ============================================================================

import Foundation
import VecturaKit

public enum MMRDiversifier {
    /// Diversifies ranked results with MMR, using each result's query score as
    /// sim(query, doc) and Jaccard token overlap for doc-doc similarity.
    /// - Parameter conceptKey: Optional canonicalizer; results mapping to the
    ///   same key are treated as maximally similar to each other.
    public static func diversify(
        _ results: [VecturaSearchResult],
        k: Int,
        lambda: Double = 0.7,
        conceptKey: ((String) -> String)? = nil
    ) -> [VecturaSearchResult] {
        guard !results.isEmpty else { return [] }
        let selectedIndices = selectionOrder(
            texts: results.map(\.text),
            queryScores: results.map(\.score),
            k: k,
            lambda: lambda,
            conceptKey: conceptKey
        )
        return selectedIndices.map { results[$0] }
    }

    /// Returns the indices of the selected texts, in MMR selection order.
    /// Assumes `texts`/`queryScores` are already sorted by descending
    /// relevance (the first element seeds the selection).
    public static func selectionOrder(
        texts: [String],
        queryScores: [Float],
        k: Int,
        lambda: Double = 0.7,
        conceptKey: ((String) -> String)? = nil
    ) -> [Int] {
        guard !texts.isEmpty, texts.count == queryScores.count, k > 0 else { return [] }

        let limit = min(k, texts.count)
        let candidateTokens = texts.map(Self.tokens)
        let candidateConcepts = conceptKey.map { key in texts.map(key) }
        var selectedIndices = [0]

        while selectedIndices.count < limit {
            var bestIdx = -1
            var bestScore = -Double.infinity

            for i in texts.indices where !selectedIndices.contains(i) {
                let rel = Double(queryScores[i])

                let maxTextSimilarity = selectedIndices
                    .map { selectedIndex in
                        Self.jaccard(candidateTokens[i], candidateTokens[selectedIndex])
                    }
                    .max() ?? 0.0
                let sharesConcept = candidateConcepts.map { concepts in
                    selectedIndices.contains { selectedIndex in
                        concepts[i] == concepts[selectedIndex]
                    }
                } ?? false
                let maxSimilarity = max(maxTextSimilarity, sharesConcept ? 1.0 : 0.0)

                let mmrScore = lambda * rel - (1 - lambda) * maxSimilarity
                if mmrScore > bestScore {
                    bestScore = mmrScore
                    bestIdx = i
                }
            }

            if bestIdx >= 0 {
                selectedIndices.append(bestIdx)
            } else {
                break
            }
        }

        return selectedIndices
    }

    private static func tokens(_ s: String) -> Set<String> {
        let seps = CharacterSet.alphanumerics.inverted
        return Set(s
            .lowercased()
            .components(separatedBy: seps)
            .filter { $0.count > 2 })
    }

    private static func jaccard(_ a: Set<String>, _ b: Set<String>) -> Double {
        let inter = a.intersection(b).count
        let uni = a.union(b).count
        return uni == 0 ? 0 : Double(inter) / Double(uni)
    }
}
