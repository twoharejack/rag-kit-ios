# RAGKit

Generic on-device RAG engine extracted from the Nimue app: a VecturaKit-backed
vector database with bundled-seed setup, switchable embedding engines (a
sentence-transformer, or Apple's own multilingual models), batched document
embedding with progress reporting, incremental upsert/delete for live
corpora, snapshot export/import, and MMR result diversification.

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
    against `indexedDocuments()` (or `indexedDocument(id:)` for one document)
    to re-embed only what changed.

  `search(query:numResults:threshold:dateRange:)` takes an optional date
  range. It is a *pre* filter: the corpus is narrowed to the documents dated
  inside the range and the same engine ranks what is left, so a question about
  one week returns that week's best matches instead of whatever survives
  filtering a global top-K.
- `RAGDocument` — id, text, and the document's own `date`, stored beside the
  vector so date filtering needs no sidecar file and survives snapshot
  export/import. Hosts keep richer metadata in their own lookup keyed by the
  document ID.
- `RAGEmbedder` — the protocol every embedding engine conforms to:
  VecturaKit's `VecturaEmbedder` plus a `spaceIdentifier` naming the vector
  space it writes into. The database only ever talks to this protocol. See
  [Embedding engines](#embedding-engines).
- `RAGSentenceEmbedder` — the default engine: a BERT-family
  sentence-transformer (all-MiniLM-L6-v2) run on the CPU, at most two texts
  per forward pass. It replaces VecturaEmbeddingsKit's `SwiftEmbedder`, whose
  GPU path held every intermediate of a padded batch at once (16 texts at 512
  tokens peaked at 8.8 GB) and cached a compiled graph for every input shape,
  which was never released. It produces the same vectors, so existing
  databases and seeds stay valid.
- `RAGNaturalLanguageEmbedder` — Apple's on-device models (`NLEmbedding`,
  `NLContextualEmbedding`) for a corpus in any mix of languages. Nothing to
  bundle or fetch from Hugging Face.
- `MMRDiversifier` — Maximal Marginal Relevance re-ranking with an optional
  host-supplied concept key for duplicate collapsing.
- `EmbeddingProgressTracker` — `ObservableObject` progress for embedding UI.
- `RAGError`, `RAGLog` — shared error surface and self-contained logging
  (route via `RAGLog.handler` if desired).

`VecturaKit` is re-exported, so `import RAGKit` also provides `VecturaKit`,
`VecturaSearchResult`, etc.

## Embedding engines

`RAGVectorDatabaseConfiguration.embeddingEngine` picks one:

```swift
.sentenceTransformer                                   // default: MiniLM, the configuration's model fields
.naturalLanguage(languages: [.english, .simplifiedChinese])  // Apple's models; [] = device languages
.custom(myEmbedder)                                    // any RAGEmbedder
```

To let the user change it at runtime, call
`database.switchEmbeddingEngine(to:)` (like `setUp`, while nothing else is
using the database).

**Vectors from two engines never meet.** The database writes the engine's
`spaceIdentifier` into `embedding-space.json` beside its vectors. When it is
opened with an engine whose identifier differs, it clears itself:
`needsEmbedding()` turns `true`, and a host that diffs against
`indexedDocuments()` sees every document missing and re-embeds it. Two
engines' vectors can happen to share a length, so a dimension check alone
would let such a search return noise instead of failing. The record also
travels with snapshots and bundle-ready ZIPs. A bundled seed from another
engine is skipped, and `importSnapshot` refuses a snapshot from one with
`RAGError.embeddingSpaceMismatch`, leaving the current database in place.
Databases written before the record existed are read as the sentence
transformer's, the only engine there was then, so they open unchanged.

### Apple's models, in any language

Apple's two embedding APIs each cover part of the problem:

- `NLEmbedding.sentenceEmbedding(for:)`: one model per language, seven
  languages. Only English was installed on the development Mac, and an app
  cannot request the others.
- `NLContextualEmbedding`: one model per script, shared by its languages:
  Latin (20 languages), Chinese/Japanese/Korean, Cyrillic, Arabic, Indic, and
  Thai. Latin and CJK were installed; the rest are system downloads the app can
  request.

`RAGNaturalLanguageEmbedder` gives each script the corpus uses its own
block of the vector. A text is embedded by its script's model into that
block, and the other blocks stay zero. A query therefore only ever scores
against documents from its own model, and a mixed English/Chinese corpus
works. A Latin block that serves English alone uses the English
`NLEmbedding`; every other block uses its script's `NLContextualEmbedding`.
Measured on macOS 27 (ten notes and 24 queries per language, mean reciprocal
rank of the right note):

| Notes in | English `NLEmbedding` | `NLContextualEmbedding` | This engine |
|----------|-----------------------|-------------------------|-------------|
| English  | 0.85                  | 0.75                    | 0.83 (`[.english]`) |
| French   | 0.54                  | 0.83                    | 0.86 (`[.english, .french]`) |
| Chinese  | 0.37                  | 0.74                    | 0.69 (`[.english, .simplifiedChinese]`) |
| Six languages mixed | 0.85       | —                       | 0.95 |

What it cannot do is cross languages: an English query found the matching
French or Chinese note first in 0 of 7 trials with either Apple model. A text
in a script with no block goes to the first language's block, where it is
indexed (and keyword-searchable) but ranks poorly. Configure every language
the corpus is written in.

Models that are not on the device are listed by `languagesNeedingDownload()`
and fetched by `requestMissingAssets()`. Until then, texts in those languages
fail to embed and are left out of the index, and the host's next reconcile
picks them up. `RAGNaturalLanguageEmbedder.support(for:)` tells a settings
screen whether a language is ready, downloadable, or unsupported.

### Scores differ by engine

VecturaKit's hybrid score is `0.5 × cosine + 0.5 × min(BM25 / 10, 1)`, so a
threshold tuned for one engine does not carry over to another. Apple's
engine centers each block (it subtracts the model's mean over fixed reference
sentences). Raw contextual vectors all point the same way: unrelated notes
averaged 0.70 against a query and the right one 0.74, which left a threshold
nothing to separate. After centering, the cosine of the right note
averaged 0.23–0.43 and of the others 0.10–0.29, depending on the language,
and notes in another script score 0. Tune thresholds per engine.

## What deliberately does not live here

- Embedding models and database seeds. The host app bundles (or downloads)
  the sentence-transformer model folder and any pre-embedded database ZIP and
  passes their URLs in through the configuration — the same split
  SupertonicTTS uses for its ONNX models. Apple's models ship with the OS, or
  the system downloads them on the app's request.
- Domain logic. Document schemas, embedding-text construction, query
  building, and result filtering stay in the host app.
