// RAGEmbedder.swift
// ============================================================================
// The one contract every embedding engine RAGKit indexes with fulfils, and the
// choice of engine a database is configured with.
// ============================================================================

import Foundation
import NaturalLanguage
import VecturaKit

/// An embedding engine a `RAGVectorDatabase` can index and search with.
///
/// It is VecturaKit's `VecturaEmbedder` (`embed(texts:)`, `embed(text:)`,
/// `dimension`) plus the one thing a persistent index needs on top: the name
/// of the vector space the engine writes into. Two engines' vectors cannot be
/// compared, even when their lengths happen to agree, so the database records
/// this name beside its vectors. When an engine with another name opens it,
/// the database clears the old vectors instead of ranking one model's queries
/// against another model's documents.
///
/// `RAGSentenceEmbedder` and `RAGNaturalLanguageEmbedder` both conform, and
/// the database only ever talks to this protocol, so a host can bring its own
/// engine through `RAGEmbeddingEngine.custom`. Spotlight
/// (`RAGEmbeddingEngine.spotlight`) is the one engine that is not an embedder:
/// it keeps its vectors inside the system.
public protocol RAGEmbedder: VecturaEmbedder {
    /// Names the vector space this engine embeds into: the model, plus
    /// anything else that moves where a text lands (its language, its
    /// revision, how a long text is pooled). Two engines with the same
    /// identifier must produce interchangeable vectors, and any change that
    /// moves vectors must change the identifier.
    var spaceIdentifier: String { get }
}

/// Which engine a `RAGVectorDatabase` embeds documents and queries with.
/// Every case but `.spotlight` resolves to a `RAGEmbedder`, so indexing,
/// search, and `embedText` work the same whichever of them is chosen. Only the
/// vectors differ, and so do the scores they produce: see the README before
/// reusing a search threshold across engines.
public enum RAGEmbeddingEngine: Sendable {
    /// The BERT sentence-transformer named by the configuration's model fields
    /// (all-MiniLM-L6-v2 in the shipped apps), run on the CPU by
    /// `RAGSentenceEmbedder`. A model that is not bundled downloads the first
    /// time it embeds. MiniLM was trained on English.
    case sentenceTransformer
    /// Apple's own embedding models (`NLEmbedding`, `NLContextualEmbedding`)
    /// for the languages the corpus is written in, run by
    /// `RAGNaturalLanguageEmbedder`. Nothing to bundle or fetch from Hugging
    /// Face. Each text is embedded by the model for its script, so a corpus
    /// can mix languages and scripts. Empty means the device's preferred
    /// languages, so a change there that adds a script re-embeds.
    case naturalLanguage(languages: [NLLanguage])
    /// Any other engine, built by the host.
    case custom(any RAGEmbedder)
    /// Apple's Core Spotlight: documents go into the app's own on-device
    /// Spotlight index, and searches run through `CSUserQuery`, which matches
    /// a query's words and, once the system has processed a document, its
    /// meaning, in whatever languages the system's search reads. Nothing to
    /// bundle, download, or embed in the app. The database keeps no vectors:
    /// `embedText` throws `RAGError.spotlightHasNoVectors`, scores stand for
    /// Spotlight's rank order rather than a similarity, and the documents
    /// also appear in the system's own search under the app. Check
    /// `RAGEmbeddingEngine.isSpotlightAvailable` first. See the README.
    case spotlight
}
