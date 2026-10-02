// RAGSentenceEmbedder.swift
// ============================================================================
// The default engine a RAGVectorDatabase embeds with: a BERT-family
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
/// By default the vectors are unchanged: the model, the 512-token truncation
/// and the first-token pooling are exactly what `SwiftEmbedder` produced for
/// BERT, so databases and bundled seeds embedded before still answer new
/// queries. ``Pooling/mean`` reads the model the way sentence-transformers
/// trained it, and ranks far better; see ``Pooling``.
public actor RAGSentenceEmbedder: RAGEmbedder {
    /// How a text's token vectors become its one vector.
    public enum Pooling: String, Sendable, CaseIterable {
        /// The first token's (`[CLS]`) vector, as `SwiftEmbedder` read every
        /// BERT model. What every database and bundled seed embedded before
        /// pooling was a choice holds, so it stays the default.
        case firstToken
        /// The mean of every token's vector, padding left out:
        /// sentence-transformers' own pooling, the one all-MiniLM-L6-v2 and
        /// its relatives were trained for. Their first token was never trained
        /// to stand for the text, and read alone it barely tells texts apart.
        /// Measured with all-MiniLM-L6-v2 on 76 notes and a description of
        /// each: the right note's cosine to its description had a median of
        /// 0.78 and the others 0.66 (90th percentile 0.74) with the first
        /// token, and 0.42 against 0.09 (0.23) with the mean; mean reciprocal
        /// rank under VecturaKit's hybrid score rose from 0.76 to 0.80. The
        /// scores land lower, so a threshold tuned on first-token vectors does
        /// not carry over (see the README).
        case mean
    }

    /// Two texts at the 512-token cap cost no more memory than one; four
    /// doubled the peak.
    public static let defaultMaxBatchSize = 2

    /// The model's name, and its pooling when that is not the first token's.
    /// Truncation never varies, so the two decide where a text lands.
    public nonisolated let spaceIdentifier: String
    public nonisolated let pooling: Pooling

    /// What the model reads of a text, in tokens: BERT's position limit, and
    /// what swift-embeddings' `encode` and `batchEncode` cut to.
    private static let maxTokens = 512

    private let modelSource: VecturaModelSource
    private let maxBatchSize: Int
    private var modelTask: Task<Bert.ModelBundle, Error>?
    private var cachedDimension: Int?

    /// - Parameters:
    ///   - modelSource: A local folder or Hugging Face ID of a BERT-architecture
    ///     sentence-transformer.
    ///   - modelID: The model's Hugging Face ID, which names its vector space.
    ///     Defaults to the ID of an `.id` source, or the folder name of a
    ///     `.folder` one. Pass it when loading a local copy, so the copy and
    ///     the download count as the same space.
    ///   - maxBatchSize: The most texts encoded in one forward pass. Larger
    ///     requests are split, so memory is bounded however many texts a
    ///     caller hands over at once.
    ///   - pooling: How token vectors become the text's vector. A database
    ///     embedded with one pooling is cleared and re-embedded when it is
    ///     opened with the other.
    public init(
        modelSource: VecturaModelSource,
        modelID: String? = nil,
        maxBatchSize: Int = RAGSentenceEmbedder.defaultMaxBatchSize,
        pooling: Pooling = .firstToken
    ) {
        self.modelSource = modelSource
        self.maxBatchSize = max(1, maxBatchSize)
        self.pooling = pooling
        self.spaceIdentifier = Self.spaceIdentifier(
            forModelID: modelID ?? Self.defaultModelID(of: modelSource),
            pooling: pooling
        )
    }

    /// The vector space of `modelID`'s vectors under `pooling`.
    /// `RAGVectorDatabase` also assigns the first-token space to databases
    /// that predate space records, all of which this engine (or
    /// `SwiftEmbedder`, which made the same vectors) wrote that way.
    static func spaceIdentifier(forModelID modelID: String, pooling: Pooling = .firstToken) -> String {
        switch pooling {
        case .firstToken:
            return "sentence-transformer:\(modelID)"
        case .mean:
            return "sentence-transformer-mean:\(modelID)"
        }
    }

    private static func defaultModelID(of source: VecturaModelSource) -> String {
        switch source {
        case .id(let id, _):
            return id
        case .folder(let url, _):
            return url.lastPathComponent
        }
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
            // CoreML's own scope, which is all swift-embeddings' `computePolicy:`
            // argument does, but that argument only exists from 0.0.30 and
            // hosts on WhisperKit resolve 0.0.26. See Package.swift.
            let tensor = try withMLTensorComputePolicy(.cpuOnly) { try Self.encode(batch, with: model, pooling: pooling) }
            try await vectors.append(contentsOf: Self.rows(of: tensor))
        }
        return vectors
    }

    public func embed(text: String) async throws -> [Float] {
        let model = try await loadedModel()
        let tensor = try withMLTensorComputePolicy(.cpuOnly) { () throws -> MLTensor in
            switch pooling {
            case .firstToken:
                return try model.encode(text)
            case .mean:
                return try Self.meanPooled([text], with: model)
            }
        }
        return await tensor.cast(to: Float.self).shapedArray(of: Float.self).scalars
    }

    // MARK: - Pooling

    /// `[N, D]`: one vector per text, pooled the way this engine pools.
    private static func encode(_ texts: [String], with model: Bert.ModelBundle, pooling: Pooling) throws -> MLTensor {
        switch pooling {
        case .firstToken:
            return try model.batchEncode(texts)
        case .mean:
            return try meanPooled(texts, with: model)
        }
    }

    /// `[N, D]`: each text's token vectors averaged over its own tokens, the
    /// `[CLS]` and `[SEP]` among them, the padding of a shorter text left out,
    /// as sentence-transformers' `Pooling(mode="mean")` reads them. The same
    /// tokens, padding and attention mask as swift-embeddings' `batchEncode`,
    /// which pools by taking the first token instead.
    private static func meanPooled(_ texts: [String], with model: Bert.ModelBundle) throws -> MLTensor {
        let batch = try model.tokenizer.tokenizeTextsPaddingToLongest(texts, padTokenId: 0, maxLength: maxTokens)
        let inputIDs = MLTensor(shape: batch.shape, scalars: batch.tokens)
        let attentionMask = MLTensor(shape: batch.shape, scalars: batch.attentionMask)
        let tokenVectors = model.model(inputIds: inputIDs, attentionMask: attentionMask).sequenceOutput
        let summed = (tokenVectors * attentionMask.expandingShape(at: 2)).sum(alongAxes: 1, keepRank: false)
        let tokenCounts = attentionMask.sum(alongAxes: 1, keepRank: true)
        return summed / tokenCounts
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
