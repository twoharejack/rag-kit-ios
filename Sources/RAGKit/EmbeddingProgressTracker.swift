import Combine
import Foundation

/// Observable object that tracks embedding progress for UI display.
public final class EmbeddingProgressTracker: ObservableObject {
    @MainActor @Published public private(set) var isEmbedding: Bool = false
    @MainActor @Published public private(set) var progress: Double = 0.0
    @MainActor @Published public private(set) var currentBatch: Int = 0
    @MainActor @Published public private(set) var totalBatches: Int = 0
    @MainActor @Published public private(set) var statusMessage: String = ""

    public init() {}

    public func startEmbedding(totalBatches: Int) {
        Task { @MainActor in
            isEmbedding = true
            progress = 0.0
            currentBatch = 0
            self.totalBatches = totalBatches
            statusMessage = "Preparing to embed documents..."
        }
    }

    public func updateProgress(batch: Int, message: String) {
        Task { @MainActor in
            currentBatch = batch
            progress = totalBatches > 0 ? Double(batch) / Double(totalBatches) : 0.0
            statusMessage = message
        }
    }

    public func finishEmbedding(success: Bool) {
        Task { @MainActor in
            isEmbedding = false
            progress = success ? 1.0 : 0.0
            statusMessage = success ? "Embedding complete" : "Embedding failed"
        }
    }
}
