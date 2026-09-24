import Foundation

public enum RAGError: LocalizedError {
    case notInitialized
    case embedderUnavailable
    /// Apple has no embedding model for this language (a BCP-47 code), or
    /// this device has not downloaded it yet.
    case naturalLanguageModelUnavailable(language: String)
    /// A snapshot was embedded in a different vector space than the one the
    /// database is configured for, so its vectors cannot answer its queries.
    case embeddingSpaceMismatch(snapshot: String, database: String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "RAG vector database is not initialized. Check console for startup errors."
        case .embedderUnavailable:
            return "On-device embedder unavailable (requires iOS 18.0+)."
        case .naturalLanguageModelUnavailable(let language):
            return "Apple's embedding model for \"\(language)\" is not available on this device."
        case .embeddingSpaceMismatch(let snapshot, let database):
            return "The snapshot was embedded with \(snapshot), but the database embeds with \(database)."
        }
    }
}
