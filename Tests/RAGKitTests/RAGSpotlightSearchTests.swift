// RAGSpotlightSearchTests.swift
// ============================================================================
// How the Spotlight engine splits a query into words and merges Spotlight's
// answers into one ranking. Pure functions: nothing here touches Spotlight.
// ============================================================================

import Testing
@testable import RAGKit

@Suite("Spotlight search")
struct RAGSpotlightSearchTests {
    @Test("Each word is a query, with its dictionary form when that differs")
    func wordQueries() {
        let queries = RAGSpotlightIndex.wordQueries("When can the tomatoes go outside?")
        #expect(queries.contains("tomatoes OR tomato"))
        #expect(queries.contains("the"))
        #expect(RAGSpotlightIndex.wordQueries("where my money goes").contains("goes"))
        #expect(!queries.contains { $0.contains("?") })
        #expect(RAGSpotlightIndex.wordQueries("the The THE") == ["the"])
        #expect(RAGSpotlightIndex.wordQueries("去成都的航班").count > 1)
        let long = (1...20).map { "word\($0)" }.joined(separator: " ")
        #expect(RAGSpotlightIndex.wordQueries(long).count == RAGSpotlightIndex.maxQueryWords)
    }

    @Test("A quoted word is searched quoted, with its dictionary form")
    func quotedWordQueries() {
        #expect(RAGSpotlightIndex.wordQueries("\"art\" \"class\"") == ["\"art\"", "\"class\""])
        #expect(RAGSpotlightIndex.wordQueries("\"september\"") == ["\"september\""])
        let mixed = RAGSpotlightIndex.wordQueries("When can the \"tomatoes\" go outside?")
        #expect(mixed.contains("\"tomatoes\" OR \"tomato\""))
        #expect(mixed.contains("the"))
        #expect(RAGSpotlightIndex.wordQueries("\"tax return\" due") == ["\"tax\"", "\"return\"", "due"])
        // A quote mark without a partner quotes nothing.
        #expect(RAGSpotlightIndex.wordQueries("\"art class") == ["art", "class"])
    }

    @Test("Spotlight's answer to the whole query leads")
    func wholeQueryLeads() {
        let merged = RAGSpotlightIndex.merge(
            wholeQuery: ["c"],
            wordMatches: [["a", "b", "c"], ["a", "b"]],
            documentCount: 10
        )
        #expect(merged == ["c", "a", "b"])
    }

    @Test("A note holding a rare word outranks one holding only common ones")
    func rarityWeighs() {
        // "the" is in 8 of 10 notes, "sourdough" in one.
        let the = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "s"]
        let merged = RAGSpotlightIndex.merge(wholeQuery: [], wordMatches: [the, ["s"]], documentCount: 10)
        #expect(merged.first == "s")
        // "trip" is in 3 of 10, "weekend" in 2: one of them is enough.
        let trips = RAGSpotlightIndex.merge(wholeQuery: [], wordMatches: [["a", "b", "c"], ["a", "d"]], documentCount: 10)
        #expect(trips == ["a", "d", "b", "c"])
    }

    @Test("A note holding only words far commoner than the rarest is left out")
    func commonWordsAloneDoNotFill() {
        // "in the garden": "the" in 8 of 10, "in" in 7, "garden" in one.
        let the = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "g"]
        let inWord = ["x1", "x2", "x3", "y1", "y2", "y3", "g"]
        let merged = RAGSpotlightIndex.merge(wholeQuery: [], wordMatches: [inWord, the, ["g"]], documentCount: 10)
        #expect(merged == ["g"])
        // The whole query's answer is never held to it.
        let led = RAGSpotlightIndex.merge(wholeQuery: ["x5"], wordMatches: [the, ["g"]], documentCount: 10)
        #expect(led == ["x5", "g"])
    }

    @Test("Equal weight goes by the words' own ranking, then identifier")
    func ties() {
        let merged = RAGSpotlightIndex.merge(wholeQuery: [], wordMatches: [["b", "a"]], documentCount: 10)
        #expect(merged == ["b", "a"])
        let even = RAGSpotlightIndex.merge(wholeQuery: [], wordMatches: [["b"], ["a"]], documentCount: 10)
        #expect(even == ["a", "b"])
    }
}
