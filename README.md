# RAGKit

Generic on-device RAG engine extracted from the Nimue app: a VecturaKit-backed
vector database with bundled-seed setup, local/remote embedding-model
resolution, batched document embedding with progress reporting, incremental
upsert/delete for live corpora, snapshot export/import, and MMR result
diversification.

## What lives here

- `RAGVectorDatabase` — the engine. Owns the on-disk VecturaKit database:
  seeding from a bundled `.vecturadb` ZIP or folder, storage-subdirectory
  repair, search, batched `embedDocuments`, and snapshot export/import.
  Configured through `RAGVectorDatabaseConfiguration`; not thread-safe on its
  own — own it from a single actor.

  Two write paths cover the two corpus shapes:
  - `embedDocuments` resets the database and re-embeds everything — for a
    static corpus shipped or rebuilt as a unit (a bundled seed).
  - `upsertDocuments` / `deleteDocuments` change the index in place — for a
    live corpus that changes one document at a time (user content). Hosts diff
    against `indexedDocumentTexts()` (or `documentText(id:)` for one document)
    to re-embed only what changed.
- `RAGDocument` — id + text pair to embed. Hosts keep richer metadata in their
  own lookup keyed by the document ID.
- `MMRDiversifier` — Maximal Marginal Relevance re-ranking with an optional
  host-supplied concept key for duplicate collapsing.
- `EmbeddingProgressTracker` — `ObservableObject` progress for embedding UI.
- `RAGError`, `RAGLog` — shared error surface and self-contained logging
  (route via `RAGLog.handler` if desired).

`VecturaKit` is re-exported, so `import RAGKit` also provides `VecturaKit`,
`VecturaSearchResult`, etc.

## What deliberately does not live here

- Embedding models and database seeds. The host app bundles (or downloads)
  the sentence-transformer model folder and any pre-embedded database ZIP and
  passes their URLs in through the configuration — the same split
  SupertonicTTS uses for its ONNX models.
- Domain logic. Document schemas, embedding-text construction, query
  building, and result filtering stay in the host app.
