import Foundation

public enum RAGError: LocalizedError {
    case notInitialized
    case embedderUnavailable

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "RAG vector database is not initialized. Check console for startup errors."
        case .embedderUnavailable:
            return "On-device embedder unavailable (requires iOS 18.0+)."
        }
    }
}
