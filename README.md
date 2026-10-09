# RAGKit

Generic on-device RAG engine extracted from the Nimue app: a VecturaKit-backed
vector database with bundled-seed setup, switchable engines (a
sentence-transformer, Apple's own multilingual models, or Apple's Spotlight
index), batched document embedding with progress reporting, incremental
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
  databases and seeds stay valid. `Pooling.mean` (opt-in, through
  `RAGVectorDatabaseConfiguration.sentenceTransformerPooling`) reads the
  model the way it was trained and ranks far better; see
  [MiniLM's pooling](#minilms-pooling).
- `RAGNaturalLanguageEmbedder` — Apple's on-device models (`NLEmbedding`,
  `NLContextualEmbedding`) for a corpus in any mix of languages. Nothing to
  bundle or fetch from Hugging Face.
- `RAGEmbeddingEngine.spotlight` — the app's own Core Spotlight index in
  place of VecturaKit: Spotlight indexes the documents and ranks them, by
  their words and by their meaning. See
  [Apple's Spotlight index](#apples-spotlight-index).
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
.sentenceTransformer                                   // default: MiniLM, the configuration's model fields and pooling
.naturalLanguage(languages: [.english, .simplifiedChinese])  // Apple's models; [] = device languages
.custom(myEmbedder)                                    // any RAGEmbedder
.spotlight                                             // the app's Core Spotlight index; no vectors in the app
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

### MiniLM's pooling

A BERT model returns one vector per token, and something has to make them
one vector for the text. `RAGSentenceEmbedder` has always taken the first
token's (`[CLS]`), as VecturaEmbeddingsKit's `SwiftEmbedder` did, and that
stays the default so existing databases and bundled seeds keep answering.
But all-MiniLM-L6-v2 and its sentence-transformers relatives were trained
to be read by the mean of their token vectors; their first token was never
trained to stand for the text, and it barely tells texts apart. Set
`sentenceTransformerPooling: .mean` for the mean (padding left out, `[CLS]`
and `[SEP]` in, as sentence-transformers reads it). It is a vector space of
its own: a database switched to it is cleared and re-embedded, and a
first-token seed is skipped.

Measured with all-MiniLM-L6-v2 on 76 notes (64 English, the rest in ten
other languages), with a one-line description of each as the query, under
VecturaKit's hybrid score:

| | First token | Mean |
|---|---|---|
| Cosine of the right note (median) | 0.78 | 0.42 |
| Cosine of the other notes (median / 90th percentile) | 0.66 / 0.74 | 0.09 / 0.23 |
| Mean reciprocal rank of the right note | 0.76 | 0.80 |
| Mean reciprocal rank, eight questions about one note each | 0.52 | 0.73 |

The cutoff has to move with it. With first-token vectors the hybrid score of
a note that shares no word with the query sits near 0.3–0.4 whatever it is
about, so the 0.25 cutoff the tarot app tuned lets every note through. With
the mean, on the same notes and with each note cut into passages of about
250 words:

| Hybrid cutoff | Right notes kept (descriptions) | Unrelated notes passing (one-word queries) |
|---|---|---|
| 0.12 | 91% | 6% |
| 0.15 | 87% | 1% |
| 0.20 | 82% | 0% |
| 0.25 | 72% | 0% |

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

| Notes in | MiniLM (first token) | Apple, scheme 1 | Apple, scheme 2 (now) |
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

### Apple's Spotlight index

`.spotlight` runs the database on the app's own on-device Spotlight index
instead of VecturaKit. Each document becomes one `CSSearchableItem`: its text
as `textContent`, its date as `contentCreationDate`, and its tags in a custom
attribute that filters can test but a typed query never matches. A search is
a `CSUserQuery` with ranked results, which matches the query's words and,
where the system offers it, its meaning, in the languages the system's
search reads. Nothing is bundled, downloaded, or embedded in the app. Check
`RAGEmbeddingEngine.isSpotlightAvailable` first: not every device can index.

The rest of the database works as it does on the vector engines:
`upsertDocuments`, `deleteDocuments`, `indexedDocuments()` for diffing,
`documentCount()`, `needsEmbedding()`, `switchEmbeddingEngine(to:)`. The date
and tag filters of `search` run inside Spotlight as filter queries, so they
narrow the corpus before it is ranked. What Spotlight cannot hand back (it
never returns an item's text) is kept beside it, one record file per
document in the database directory, and the records are also the last word
on what a search may return.

The records only describe Spotlight while Spotlight keeps the items, so the
two are tied together. Items never expire (Spotlight's default is a month). A
generation token is stored both in the directory and as the index's client
state, which lives inside Spotlight: when they disagree on open (a restored
backup, an index the system rebuilt, the app's index deleted), both start
over, empty, and the host's next diff re-indexes everything. And when
Spotlight asks the app to re-index items, which it does after losing them,
their records go.

What differs:

- **No vectors.** `embedText` throws `RAGError.spotlightHasNoVectors`, and so
  do snapshot export and import. A host that derives features from vectors
  needs another embedder for them.
- **Scores are ranks.** Spotlight ranks; it does not score. A result's score
  is `1 / (1 + rank / 10)`: 1 for the first, a half at the eleventh, a
  quarter at the thirty-first. It keeps Spotlight's order and stays within
  0…1 for code that blends scores or compares them to the best one, but a
  threshold only caps how many results come back.
- **Spotlight's own answer is short.** It matches a note by its words only
  when every word of the query begins a word of the note ("tax return"
  finds the tax note, "tax return due" does not), and by meaning at most one
  note. On the eval's 40 queries over 54 notes it never answered with more
  than one note, and often with none. So RAGKit also searches each word of
  the query on its own, and also its dictionary form ("tomatoes OR tomato"),
  side by side with the whole query. Spotlight's answer leads, and the notes
  holding the most and rarest of the words (BM25's IDF) fill the rest.
  Against [the eval](Docs/Evals.md)'s hand labels, that took nDCG@10 from
  0.12 to 0.51 while Spotlight matched nothing by meaning. While it did,
  runs scored 0.44 and 0.54 before and 0.66 and 0.56 after (without the
  dictionary forms, which added 0.02 with meaning off). A search takes
  about half a second, a little more than the whole query alone.
- **Meaning comes and goes.** Measured on macOS 27 with a dozen short notes,
  45 minutes after indexing no note matched a query that shared no word with
  it: each item's embedding is made by the system's Spotlight pipeline
  (`spotlightknowledged`) as an intensive background task, which the
  scheduler held back while the Mac was in use. With the eval's notes,
  meaning matches came within seconds of indexing on some runs ("bugs eating
  my flowers" found the note about aphids on a rose), and on others within
  the same hour, on the same index, not at all. The words always match.
- **The system's search shows them.** Everything an app indexes into
  Spotlight can appear in the system's own search under the app, unless the
  person turns that off for the app in Settings.
- **Results come in batches.** RAGKit waits for the whole answer and sorts
  it by rank, best first: `CSUserQuery.Item`'s `>`, not `<`, which put the
  note holding every word of "sourdough OR starter OR the" below six notes
  holding only "the". Spotlight ranks only `maxRankedResultCount` matches
  and returns the rest unranked, mixed in among them, so RAGKit has every
  match ranked.

Leaving `.spotlight` for another engine removes the database's items from
Spotlight as well as the records from the directory.

### Scores differ by engine

VecturaKit's hybrid score is `0.5 × cosine + 0.5 × min(BM25 / 10, 1)`, so a
threshold tuned for one engine does not carry over to another. Spotlight's
scores are ranks, not similarities (see above). Apple's
engine centers each block (it subtracts the model's mean over fixed reference
sentences) and takes its length directions out. Raw contextual vectors all
point the same way: unrelated notes averaged 0.70 against a query and the
right one 0.74, which left a threshold nothing to separate. After both
steps, the cosine of the right note averaged 0.35–0.60 and of the others
0.17–0.46, depending on the language, and notes in another script score 0.
That is higher than scheme 1 scored (0.23–0.43 and 0.10–0.29), so a cutoff
tuned on scheme 1 now lets more through. Tune thresholds per engine.

## Measuring retrieval

`Tests/RAGKitEvalTests` scores the engines on a corpus of 54 notes and 40
queries, with Claude (`claude`) and OpenAI's models (`codex`) grading every
note each engine returns:

```bash
RAGKIT_EVAL_JUDGES=claude,codex swift test --filter RetrievalEvalTests
```

On it, mean-pooled MiniLM ranks best: a judged nDCG@10 of 0.76, against
0.67 for first-token MiniLM, 0.66 for Apple's models, and 0.43 for words
alone. Apple's models lead on one-word queries and on notes in other
languages.

A plain `swift test` runs only the harness, which needs no model or LLM. The
judges' grades are cached in the repository, so a re-run calls them only about
notes they have not graded. [Docs/Evals.md](Docs/Evals.md) covers the
results, the options, running it on your own notes, and its limits.

## What deliberately does not live here

- Embedding models and database seeds. The host app bundles (or downloads)
  the sentence-transformer model folder and any pre-embedded database ZIP and
  passes their URLs in through the configuration — the same split
  SupertonicTTS uses for its ONNX models. Apple's models ship with the OS, or
  the system downloads them on the app's request; Spotlight's are the
  system's own.
- Domain logic. Document schemas, embedding-text construction, query
  building, and result filtering stay in the host app.
