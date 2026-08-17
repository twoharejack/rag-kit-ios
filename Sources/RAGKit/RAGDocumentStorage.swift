// RAGDocumentStorage.swift
// ============================================================================
// Storage providers that let RAGKit put a host-supplied date on every vector
// record and then rank a date-bounded subset of the corpus.
// ============================================================================

import Foundation
import VecturaKit

/// Persists documents with the host's own date instead of the moment they were
/// embedded.
///
/// `VecturaDocument` already carries a `createdAt` that is stored with the
/// vector, returned by `getAllDocuments()`, and copied onto every search
/// result — but VecturaKit builds those documents inside `addDocuments`, and
/// stamps `Date()` on them with no seam for the caller to supply anything.
/// Intercepting the save is the last point where the host's date can still
/// reach the record that gets written, and writing it into `createdAt` keeps
/// the date *inside* the vector store, so it survives snapshot export/import
/// and needs no sidecar file to stay in step.
///
/// Dates are handed over per document ID and consumed by the save of that same
/// ID, so an interleaved write of some other document can never pick up the
/// wrong date.
actor RAGDateStampingStorage: VecturaStorage {
    private let base: any VecturaStorage
    private var pendingDates: [UUID: Date] = [:]

    init(base: any VecturaStorage) {
        self.base = base
    }

    /// Registers the dates to stamp onto the next save of these documents.
    func stampNextSave(of dates: [UUID: Date]) {
        pendingDates.merge(dates) { _, latest in latest }
    }

    /// Drops dates whose save never happened, so a failed embed cannot leak
    /// its date onto a later, unrelated write of the same ID.
    func discardPendingDates(for ids: [UUID]) {
        for id in ids { pendingDates.removeValue(forKey: id) }
    }

    private func stamped(_ document: VecturaDocument) -> VecturaDocument {
        guard let date = pendingDates.removeValue(forKey: document.id) else { return document }
        return VecturaDocument(
            id: document.id,
            text: document.text,
            embedding: document.embedding,
            createdAt: date
        )
    }

    // MARK: - VecturaStorage

    func createStorageDirectoryIfNeeded() async throws {
        try await base.createStorageDirectoryIfNeeded()
    }

    func loadDocuments() async throws -> [VecturaDocument] {
        try await base.loadDocuments()
    }

    func saveDocument(_ document: VecturaDocument) async throws {
        try await base.saveDocument(stamped(document))
    }

    func saveDocuments(_ documents: [VecturaDocument]) async throws {
        try await base.saveDocuments(documents.map { stamped($0) })
    }

    func deleteDocument(withID id: UUID) async throws {
        pendingDates.removeValue(forKey: id)
        try await base.deleteDocument(withID: id)
    }

    func updateDocument(_ document: VecturaDocument) async throws {
        try await base.updateDocument(stamped(document))
    }

    func getTotalDocumentCount() async throws -> Int {
        try await base.getTotalDocumentCount()
    }

    func getDocument(id: UUID) async throws -> VecturaDocument? {
        try await base.getDocument(id: id)
    }

    func documentExists(id: UUID) async throws -> Bool {
        try await base.documentExists(id: id)
    }
}

/// An in-memory storage view over a fixed slice of an existing corpus.
///
/// Handing this to a throwaway `VecturaKit` is what makes a filter a *pre*
/// filter: the search engine only ever sees the documents that passed the
/// filter, so the top-K it returns is the best of that slice rather than
/// whatever survives filtering a global top-K afterwards. Ranking stays
/// identical to an unfiltered search because it is the same engine over a
/// smaller corpus.
actor RAGDocumentSubsetStorage: VecturaStorage {
    /// Held in order so equal scores break ties the same way on every run.
    private var documents: [VecturaDocument]

    init(documents: [VecturaDocument]) {
        self.documents = documents
    }

    // MARK: - VecturaStorage

    func createStorageDirectoryIfNeeded() async throws {}

    func loadDocuments() async throws -> [VecturaDocument] {
        documents
    }

    func saveDocument(_ document: VecturaDocument) async throws {
        if let index = documents.firstIndex(where: { $0.id == document.id }) {
            documents[index] = document
        } else {
            documents.append(document)
        }
    }

    func deleteDocument(withID id: UUID) async throws {
        documents.removeAll { $0.id == id }
    }

    func updateDocument(_ document: VecturaDocument) async throws {
        try await saveDocument(document)
    }

    func getTotalDocumentCount() async throws -> Int {
        documents.count
    }

    func getDocument(id: UUID) async throws -> VecturaDocument? {
        documents.first { $0.id == id }
    }

    func documentExists(id: UUID) async throws -> Bool {
        documents.contains { $0.id == id }
    }
}
