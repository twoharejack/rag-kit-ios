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
works. Every block uses its script's `NLContextualEmbedding`. The English
`NLEmbedding` only stands in for the Latin model, in a block that serves
English alone, while the Latin model is not on the device.

Each block is centered on a reference mean, and then the few directions its
reference texts vary along most, text length first among them, are projected
out. Without that second step every short text points the same way: a
one-word query scored 0.6–0.8 against notes like "Okay." and under 0.1
against the notes it was about. Measured on macOS 27, as the mean reciprocal
rank of the right note by embedding alone (ten notes and 24 queries per
language), except for the last row, which is mean average precision on 71
English notes, 15 of them nearly empty, with 34 one- and two-word queries:

| Notes in | MiniLM | Apple, scheme 1 | Apple, scheme 2 (now) |
|----------|--------|-----------------|-----------------------|
| English  | 0.90   | 0.83 (sentence model) | 0.80 (Latin model) |
| French   | 0.76   | 0.86            | 0.93 |
| Chinese  | 0.49   | 0.69            | 0.81 |
| Six languages mixed | 0.86 | 0.95  | 0.95 |
| Each language with six nearly empty notes added | — | 0.26–0.41 (contextual), 0.83 (sentence) | 0.80–0.93 |
| English notes, hashtag-like queries | 0.19 | 0.36 (sentence) / 0.08 (Latin) | 0.58 |

Cross-language search stays weak. Within the Latin script it improved: an
English query ranked the matching French note at 0.77 (full search) against
0.68 before. Across scripts the cosine is exactly 0, so only a shared keyword
can match. A text in a script with no block goes to the first language's
block, where it is indexed (and keyword-searchable) but ranks poorly.
Configure every language the corpus is written in.

Models that are not on the device are listed by `languagesNeedingDownload()`
and fetched by `requestMissingAssets()`. Until then, texts in those languages
fail to embed and are left out of the index, and the host's next reconcile
picks them up. The exception is English alone, which the sentence model
embeds in the meantime. Its block changes model when the download lands, so
open the database on a new engine then (`switchEmbeddingEngine(to:)`), which
re-embeds. `RAGNaturalLanguageEmbedder.support(for:)` tells a settings
screen whether a language is ready, downloadable, or unsupported.

Working out a block's directions takes about 150 texts through the model,
some two seconds on an M1 Max. They are kept in the Caches directory, keyed
by the model's identity and revision, so later launches reuse them.

[Docs/Languages.md](Docs/Languages.md) compares the two engines language by
language. It also covers:

- choosing the language list
- keyword search in scripts without spaces
- how much of a long text each engine reads
- where search cutoffs land in each language

### Scores differ by engine

VecturaKit's hybrid score is `0.5 × cosine + 0.5 × min(BM25 / 10, 1)`, so a
threshold tuned for one engine does not carry over to another. Apple's
engine centers each block (it subtracts the model's mean over fixed reference
sentences) and takes its length directions out. Raw contextual vectors all
point the same way: unrelated notes averaged 0.70 against a query and the
right one 0.74, which left a threshold nothing to separate. After both
steps, the cosine of the right note averaged 0.35–0.60 and of the others
0.17–0.46, depending on the language, and notes in another script score 0.
That is higher than scheme 1 scored (0.23–0.43 and 0.10–0.29), so a cutoff
tuned on scheme 1 now lets more through. Tune thresholds per engine.

## What deliberately does not live here

- Embedding models and database seeds. The host app bundles (or downloads)
  the sentence-transformer model folder and any pre-embedded database ZIP and
  passes their URLs in through the configuration — the same split
  SupertonicTTS uses for its ONNX models. Apple's models ship with the OS, or
  the system downloads them on the app's request.
- Domain logic. Document schemas, embedding-text construction, query
  building, and result filtering stay in the host app.
