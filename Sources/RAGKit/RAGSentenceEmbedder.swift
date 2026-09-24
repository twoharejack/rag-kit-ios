// RAGSentenceEmbedder.swift
// ============================================================================
// The sentence encoder every RAGVectorDatabase embeds with: a BERT-family
// swift-embeddings model, run on the CPU, a couple of texts per pass.
// ============================================================================

import CoreML
import Embeddings
import Foundation
import VecturaEmbeddingsKit

/// Embeds text with a BERT-family sentence-transformer (all-MiniLM-L6-v2 and
/// its relatives) on the CPU, at most `maxBatchSize` texts per forward pass.
///
/// It stands in for VecturaEmbeddingsKit's `SwiftEmbedder`, which runs the same
/// model under swift-embeddings' default `.cpuAndGPU` compute policy. There,
/// MLTensor hands every operation to MPSGraph, and for a 22M-parameter encoder
/// that costs far more memory than the model itself. Measured with MiniLM on
/// an M1 Max:
///
/// - A padded batch keeps every intermediate tensor alive at once. Sixteen
///   texts at the 512-token cap peaked at 8.8 GB, and a single one at 626 MB —
///   on a phone, a kill at the per-app memory limit.
/// - Every distinct input shape compiles and caches its own graph, ~10 MB
///   each, process-wide. Nothing releases them, not even dropping the model,
///   so each new text length is a permanent leak.
///
/// On the CPU the same work peaks at 159 MB for one 512-token text and 161 MB
/// for two, nothing accumulates across shapes, and it is faster as well —
/// 0.04 s per short text against 0.8 s while the GPU compiles a new shape.
///
/// Vectors are unchanged. The model, the 512-token truncation and the
/// first-token pooling are exactly what `SwiftEmbedder` produced for BERT, so
/// databases and bundled seeds embedded before still answer new queries.
public actor RAGSentenceEmbedder: VecturaEmbedder {
    /// Two texts at the 512-token cap cost no more memory than one; four
    /// doubled the peak.
    public static let defaultMaxBatchSize = 2

    private let modelSource: VecturaModelSource
    private let maxBatchSize: Int
    private var modelTask: Task<Bert.ModelBundle, Error>?
    private var cachedDimension: Int?

    /// - Parameters:
    ///   - modelSource: A local folder or Hugging Face ID of a BERT-architecture
    ///     sentence-transformer.
    ///   - maxBatchSize: The most texts encoded in one forward pass. Larger
    ///     requests are split, so memory is bounded however many texts a
    ///     caller hands over at once.
    public init(
        modelSource: VecturaModelSource,
        maxBatchSize: Int = RAGSentenceEmbedder.defaultMaxBatchSize
    ) {
        self.modelSource = modelSource
        self.maxBatchSize = max(1, maxBatchSize)
    }

    public var dimension: Int {
        get async throws {
            if let cachedDimension { return cachedDimension }
            let probe = try await embed(text: "dimension")
            cachedDimension = probe.count
            return probe.count
        }
    }

    public func embed(texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let model = try await loadedModel()

        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for start in stride(from: 0, to: texts.count, by: maxBatchSize) {
            let batch = Array(texts[start..<min(start + maxBatchSize, texts.count)])
            let tensor = try model.batchEncode(batch, computePolicy: .cpuOnly)
            try await vectors.append(contentsOf: Self.rows(of: tensor))
        }
        return vectors
    }

    public func embed(text: String) async throws -> [Float] {
        let model = try await loadedModel()
        let tensor = try model.encode(text, computePolicy: .cpuOnly)
        return await tensor.cast(to: Float.self).shapedArray(of: Float.self).scalars
    }

    // MARK: - Model

    /// Loads the model once. Callers that arrive while it is loading share
    /// that load instead of each building a copy.
    private func loadedModel() async throws -> Bert.ModelBundle {
        if let modelTask {
            return try await modelTask.value
        }
        let source = modelSource
        let task = Task { try await Self.loadModel(from: source) }
        modelTask = task
        do {
            return try await task.value
        } catch {
            // Leave a failed load retryable; a remote download can fail transiently.
            modelTask = nil
            throw error
        }
    }

    private static func loadModel(from source: VecturaModelSource) async throws -> Bert.ModelBundle {
        switch source {
        case .folder(let url, _):
            return try await Bert.loadModelBundle(from: url)
        case .id(let id, _):
            return try await Bert.loadModelBundle(from: id)
        }
    }

    /// Splits an `[N, D]` embedding tensor into N vectors.
    private static func rows(of tensor: MLTensor) async throws -> [[Float]] {
        let shape = tensor.shape
        guard shape.count == 2, let width = shape.last, width > 0 else {
            throw VecturaError.invalidInput("Expected embeddings shaped [N, D], got \(shape)")
        }
        let scalars = await tensor.cast(to: Float.self).shapedArray(of: Float.self).scalars
        return stride(from: 0, to: scalars.count, by: width).map {
            Array(scalars[$0..<($0 + width)])
        }
    }
}
