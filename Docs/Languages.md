# Languages

RAGKit's two built-in embedding engines differ most in what they do with text
that is not English. This guide covers:

- which engine suits which languages
- how to configure Apple's engine for a corpus
- what the keyword half of search does in each script
- how scores and cutoffs shift from language to language
- what does not work at all

Every number comes from one test run, described in
[How this was measured](#how-this-was-measured). Rankings are given as mean
reciprocal rank (MRR) of the right note:

- 1.0 means the right note always came first.
- 0.5 means it came second on average.
- 0.29 is chance with ten notes.

## Choosing an engine

| The corpus is in | Use | Why |
|---|---|---|
| English only | Either | MiniLM 0.94 against Apple's 0.92, a gap within noise. Apple's needs no 90 MB download. |
| Another Latin-script language (French, German, Spanish, …) | Apple | Ties or beats MiniLM, and leads on the embedding alone: 0.86 vs 0.76 in French. |
| Chinese, Japanese, Korean | Apple | 0.73–0.83 against MiniLM's 0.51–0.69. MiniLM's vocabulary lacks most Chinese characters. |
| Russian, Arabic, Hindi, Thai, or another language on Apple's list | Apple, once the model is downloaded | Apple's side was not measured (models not installed). MiniLM's embedding alone scored 0.29–0.49 on these languages (chance is 0.29), and its full search scored 0.35 in Thai. |
| Several languages | Apple, with every language listed | 1.00 against MiniLM's 0.93 on a six-language set. |
| Queries in one language for notes in another | Neither | See [Cross-language search](#cross-language-search). |

## Which languages each engine covers

### MiniLM

`RAGSentenceEmbedder` runs all-MiniLM-L6-v2: one model for all text, trained on
English. Its vocabulary is 30,522 lowercase English word pieces, and accents
are stripped before lookup. Other scripts reach the model letter by letter or
as unknown tokens. Measured on the test notes:

| Text in | Unknown tokens | Tokens per character | What the model sees |
|---|---|---|---|
| English, French, German, Spanish | 0% | 0.21–0.35 | Words and word pieces (`comprimés` → com, pr, ime, s) |
| Russian, Arabic | 0% | 0.83 | Mostly single letters |
| Hindi | 4.5% | 0.59 | Mostly single letters, with some vowel signs and the virama stripped |
| Korean | 3.5% | 1.45 | Syllables broken into their letters (jamo) |
| Japanese | 25% | 0.95 | Kana letter by letter; two in three kanji unknown |
| Chinese | 66% | 0.97 | Three in four characters unknown |
| Thai | 71% | 0.05 | Each stretch between spaces is one unknown token |

### Apple

Apple has one contextual model (`NLContextualEmbedding`) per script, plus an
English sentence model (`NLEmbedding`). RAGKit uses them as follows:

| Model | Languages | On the test Mac |
|---|---|---|
| English sentence model | English | Installed |
| Latin | Croatian, Czech, Danish, Dutch, English, Finnish, French, German, Hungarian, Indonesian, Italian, Norwegian Bokmål, Polish, Portuguese, Romanian, Slovak, Spanish, Swedish, Turkish, Vietnamese | Installed |
| Chinese, Japanese, Korean | Chinese (Simplified, Traditional), Japanese, Korean | Installed |
| Cyrillic | Bulgarian, Kazakh, Russian, Ukrainian | Needs a download |
| Arabic | Arabic, Najdi Arabic | Needs a download |
| Indic | Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu | Needs a download |
| Thai | Thai | Needs a download |

**Model details.** Every model has 512 dimensions, and the contextual ones
read 256 tokens per call. Apple also lists sentence models for French,
German, Italian, Portuguese, Spanish and Simplified Chinese. None of them was
installed on the test Mac, and an app cannot request them, so RAGKit does not
use them.

**What is installed varies.** The last column shows one macOS 27 Mac. Check a
device at runtime with `RAGNaturalLanguageEmbedder.support(for:)`, which
returns one of:

- `.ready`
- `.needsDownload`
- `.unsupported`

**No Apple model:** Greek, Hebrew, Persian, Catalan, Malay, Filipino, Serbian,
Lithuanian, Latvian, Estonian, Slovenian, Icelandic, Armenian, Georgian,
Khmer, Lao, Burmese, Sinhala, Mongolian, Amharic, Swahili, and others. Neither
engine was tested on these languages.

## Search in the note's own language

Ten notes and 24 queries per language, with queries in the notes' language.
Each cell is the MRR of the full hybrid search, with the embedding's
score alone in parentheses. Apple's engine was configured with that language
alone, so English used the English sentence model. The mixed row lists all
six languages.

| Language | MiniLM | Apple | Right note had a keyword match |
|---|---|---|---|
| English | 0.94 (0.90) | 0.92 (0.83) | 75% |
| French | 0.92 (0.76) | 0.92 (0.86) | 67% |
| German | 0.82 (0.69) | 0.86 (0.72) | 54% |
| Spanish | 0.80 (0.62) | 0.86 (0.67) | 58% |
| Chinese (Simplified) | 0.51 (0.49) | 0.73 (0.69) | 8% |
| Japanese | 0.62 (0.60) | 0.80 (0.75) | 8% |
| Korean | 0.69 (0.32) | 0.83 (0.69) | 58% |
| Russian | 0.75 (0.49) | Not installed | 58% |
| Arabic | 0.77 (0.29) | Not installed | 71% |
| Hindi | 0.75 (0.45) | Not installed | 58% |
| Thai | 0.35 (0.35) | Not installed | 8% |
| Six languages mixed | 0.93 (0.86) | 1.00 (0.95) | 65% |

How to read the table:

- **Share of queries answered first:**
  - MiniLM: 88% (English), 38% (Chinese), 17% (Thai).
  - Apple: 83% (English), 58% (Chinese), 67% (Japanese), and every query
    in the mixed set.
- **MiniLM leans on keywords outside English.** In Korean, Russian, Arabic
  and Hindi its embedding alone scored 0.29–0.49, and word matches lifted
  its full search to 0.69–0.77. In Thai there are almost no word matches
  either (see [Keyword search by script](#keyword-search-by-script)).
- **Noise.** With 24 queries, a difference under about 0.05 is noise.

## Configuring Apple's engine

```swift
.naturalLanguage(languages: [])                             // the device's preferred languages
.naturalLanguage(languages: [.english])                     // the English sentence model
.naturalLanguage(languages: [.english, .french])            // one Latin block, contextual model
.naturalLanguage(languages: [.english, .simplifiedChinese]) // a Latin and a CJK block: 1,024 dimensions
```

The list decides which blocks the vector has, one per script model. A text is
embedded into its script's block, and every other block stays zero.

### One language covers its whole script

Texts are routed by script model, not by exact language:

- Japanese notes scored the same (0.80) with `[.english, .simplifiedChinese]`
  as with `[.japanese]`.
- German notes scored the same (0.86) with `[.english, .french]` as with
  `[.german]`.

### English alone is the exception

A Latin block that serves only English uses the English sentence model. It is
the better of Apple's two models for English, and weaker for any other
Latin-script text that turns up:

| Notes | `[.english]` | `[.english, .french]` |
|---|---|---|
| English | 0.92 (0.83) | 0.86 (0.77) |
| French | 0.84 (0.55) | 0.92 (0.86) |
| German | 0.66 (0.42) | 0.86 (0.72) |

If the corpus may hold other Latin-script languages, list at least one of
them.

### Unlisted scripts go to the first block

Texts in a script with no block are still indexed and keyword-searchable,
but they are embedded by the first block's model, which does not read their
script:

- Chinese notes under `[.english]`: 0.44 (0.37)
- Russian notes under `[.english]`: 0.67, mostly from keywords (0.39)

Put the corpus's main language first, and list every script the corpus uses.

### The space identifier changes with the blocks

`spaceIdentifier` names the blocks' models and their order:

- **Same blocks, same index.** Reordering languages that share one block
  (French and German) keeps it.
- **Different blocks, re-embed.** Any of these means a new vector space:
  - adding or removing a block
  - reordering blocks
  - changing a block's model (English alone, then English plus another
    Latin-script language)

  The database clears itself, and the host re-embeds its corpus.

`switchEmbeddingEngine(to:)` clears the index only in the second case.

### Unsupported languages are skipped

They are dropped with a logged warning. If no language on the list has a
model, the initializer throws `RAGError.naturalLanguageModelUnavailable`.

### Missing models

A listed language whose model is not downloaded still gets its block, so the
space does not change when the download finishes. Until then:

- **Indexing.** `upsertDocuments` skips texts in that language, logs them,
  and returns only the IDs that embedded. A host that diffs against
  `indexedDocuments()` retries the skipped texts on its next reconcile.
- **Search.** A query in that language makes `search` throw
  `RAGError.naturalLanguageModelUnavailable`. Catch it, then either fall back
  to plain keyword search or offer the download.
- **Downloading.** `languagesNeedingDownload()` lists those languages.
  `requestMissingAssets()` asks the system to download their models; the
  download is Apple's, not the app's.

### Device languages

`devicePreferredLanguages` (what `[]` resolves to) maps the device's locales
to NaturalLanguage languages:

- **Chinese.** A Hant script, or a TW, HK or MO region, maps to Traditional
  Chinese.
- **Norwegian.** `no` and `nn` map to Bokmål.
- **Punjabi.** A `pa` locale maps to `.punjabi`, which Apple names by its
  script (`"pa-Guru"`). A bare `NLLanguage("pa")` has no model, so pass
  `.punjabi` when building a list by hand.

### Recommended setup

A setup that works:

1. Put the language the user writes in first. NoteNow uses its summary
   language.
2. Follow it with `devicePreferredLanguages`.
3. When either changes, build a new engine and call
   `switchEmbeddingEngine(to:)`.

## Keyword search by script

VecturaKit's hybrid score is `0.5 × cosine + 0.5 × min(BM25 / 10, 1)`. Its
keyword tokenizer lowercases, folds diacritics, and splits at every character
that is not a letter or digit:

| Text | Keyword tokens |
|---|---|
| Vérifier si la pharmacie a les comprimés contre l'allergie. | verifier · si · la · pharmacie · a · les · comprimes · contre · l · allergie |
| Nachsehen, ob die Apotheke die Allergietabletten vorrätig hat. | nachsehen · ob · die · apotheke · die · allergietabletten · vorratig · hat |
| 약국에 알레르기 약이 있는지 확인한다. | 약국에 · 알레르기 · 약이 · 있는지 · 확인한다 |
| 看看药店有没有抗过敏药。 | 看看药店有没有抗过敏药 |
| 薬局にアレルギーの薬の在庫があるか確認する。 | 薬局にアレルギーの薬の在庫があるか確認する |
| ดูว่าร้านขายยามียาแก้แพ้หรือไม่ | ดูว่าร้านขายยามียาแก้แพ้หรือไม่ |
| #groceries #购物 #買い物 | groceries · 购物 · 買い物 |

### Scripts with spaces

Latin, Cyrillic, Arabic, Devanagari and Korean get word matches, but only on
exact forms:

- A German compound (`allergietabletten`) does not match `Allergie`.
- A Korean particle keeps `약이` apart from `약`.
- Differently inflected Russian or Arabic words do not match.

The right note got a keyword share for 54–75% of queries.

### Chinese, Japanese and Thai

These scripts put no spaces between words. Each stretch between punctuation
or spaces is one token, so a query matches only an identical stretch. In
practice that means a whole hashtag: every keyword match in these languages
came from a one-word query equal to a note's hashtag (旅行 finding #旅行). The
right note got a keyword share for 8% of queries (2 of 24).

Search in these languages therefore runs on the embedding alone: a note's
hybrid score is just half its cosine (`0.5 × cosine`). A cutoff tuned on
English removes many of these notes (see the next section). Hosts that need
substring matching in these scripts have to run it themselves, alongside
RAGKit.

## Scores and cutoffs by language

A search cutoff (`threshold:`) is compared with the hybrid score. The two
engines score on different scales, and each scale shifts with the language.
The cutoffs below, 0.12 for Apple and 0.25 for MiniLM, are the ones NoteNow
uses. Each query has 9 wrong notes.

| Language | Apple: median score, right / wrong note | Apple at 0.12: right notes kept, wrong notes let through | MiniLM: median score, right / wrong note | MiniLM at 0.25: right notes kept, wrong notes let through |
|---|---|---|---|---|
| English | 0.30 / 0.13 | 100%, 5.0 | 0.47 / 0.31 | 100%, 8.8 |
| French | 0.30 / 0.15 | 100%, 6.2 | 0.46 / 0.34 | 100%, 9.0 |
| German | 0.20 / 0.09 | 62%, 3.7 | 0.42 / 0.35 | 100%, 9.0 |
| Spanish | 0.27 / 0.15 | 88%, 5.5 | 0.45 / 0.36 | 100%, 9.0 |
| Chinese | 0.15 / 0.05 | 54%, 2.0 | 0.38 / 0.36 | 100%, 9.0 |
| Japanese | 0.11 / 0.01 | 38%, 1.1 | 0.36 / 0.32 | 100%, 8.9 |
| Korean | 0.15 / 0.04 | 58%, 1.5 | 0.49 / 0.41 | 100%, 9.0 |
| Six languages mixed | 0.31 / 0.00 | 90%, 0.8 | 0.51 / 0.32 | 100%, 8.9 |

### Apple

One cutoff cannot fit every language:

- **English and French.** 0.12 keeps every right note.
- **Chinese, Japanese and Korean.** 0.12 drops about half of the right notes.
  These notes get no keyword share, and the CJK model's centered cosines run
  lower: 0.16–0.23 for the right note, against 0.30–0.43 in Latin-script
  languages.
- **Mixed scripts.** A corpus in several scripts is cleaner, because a note
  in another script has an embedding score of exactly 0.

Use a cutoff per script, or rank only and cap the number of results.

### MiniLM

With RAGKit's first-token pooling, MiniLM's vectors crowd together: unrelated
notes averaged 0.63–0.84 cosine. So 0.25 filtered almost nothing, and ranking
did all the work:

- **English and French.** About 0.38 kept 90% of the right notes and let
  5–7% of the wrong ones through.
- **The other languages.** A cutoff that kept 90% of the right notes let
  42–86% of the wrong ones through.

These test sets are small. Tune any cutoff on the real corpus.

## How much of a long text is read

Each engine reads only the start of a long text:

| Language | MiniLM: 512 tokens | Apple: contextual model, two 256-token passes |
|---|---|---|
| English | ≈ 2,400 characters | ≈ 1,800 characters |
| French, Spanish | ≈ 1,550–1,650 | ≈ 1,800 |
| German | ≈ 1,450 | ≈ 1,650 |
| Chinese | ≈ 530 | ≈ 620 |
| Japanese | ≈ 540 | ≈ 710 |
| Korean | ≈ 350 | ≈ 880 |
| Russian, Arabic | ≈ 620 | Not installed |
| Hindi | ≈ 870 | Not installed |

A block using the English sentence model reads the first 512 words instead,
about 3,000 characters of English. The estimates come from tokens per
character on the test notes.

In Chinese, Japanese and Korean, only the first few hundred characters count.
Put a title or summary first in the text you embed.

## Cross-language search

Neither engine reliably finds a note written in one language from a query in
another.

| English queries, notes in | MiniLM | Apple, `[.english, <language>]` |
|---|---|---|
| French | 0.63 | 0.68 |
| German | 0.62 | 0.68 |
| Spanish | 0.58 | 0.61 |
| Chinese | 0.46 | 0.29 |
| Japanese | 0.53 | 0.29 |
| Korean | 0.31 | 0.29 |
| Russian, Arabic, Hindi, Thai | 0.28–0.41 | Not installed |
| Six languages mixed: 7 English queries for non-English notes | 0.57 | 0.18 |

### Apple, another script

Blocks never overlap, so the cosine is exactly 0. Only a shared keyword, such
as a name or "ATP", can match. 0.29 is chance.

### Apple, the same script

The Latin model does place an English query near the matching French note:
the right note came first 54% of the time when every note was French. But the
scores are tiny (median 0.04, so only 12% passed 0.12), and notes in the
query's own language tend to outrank it. In the mixed set, 0 of 7 English
queries ranked their note first, an MRR of 0.18, below chance.

### MiniLM

MiniLM shares word pieces across Latin-script languages. That carries names
and some cognates between them (the right note came first for 46% of English
queries over French notes), and very little else.

### When it matters

If cross-language search matters, bring a multilingual model through
`.custom(_:)`, or index a translation beside each text.

## Short queries and language detection

The Apple engine routes each text by the top three guesses of
`NLLanguageRecognizer`, mapped to script models. On the 24 short queries per
language:

- **Exact language.** The top guess was right 19–24 times.
- **Script model.** The top guess mapped to the right model 24 of 24 times,
  in all 11 languages.

A query misread as a neighbouring language of the same script lands in the
same block. Only the language hint passed to the model changes.

## How this was measured

- **Setup.** Run on 2026-09-24 with macOS 27.0 on an M1 Max, RAGKit 6bb12d8
  and VecturaKit e5b7cd8.
- **Corpus.**
  - Ten everyday notes, each a title, hashtags and 3–5 sentences
    (200–450 characters in English).
  - 24 short queries, including four one-word ones.
  - All written in English, then translated into French, German, Spanish,
    Chinese, Japanese, Korean, Russian, Arabic, Hindi and Thai.
  - A second set mixes six languages (French, Spanish, English, German,
    Chinese, Japanese), with 20 queries in the note's language and 7 in
    English.
- **Search.** VecturaKit's hybrid search, compiled from source with RAGKit's
  default settings. Every note was ranked, with no cutoff. "Embedding alone"
  ranks by cosine.
- **Apple.** The shipped `RAGNaturalLanguageEmbedder`. The Cyrillic, Arabic,
  Indic and Thai models were not installed and were not downloaded for the
  test.
- **MiniLM.** The all-MiniLM-L6-v2 file the app downloads, run in PyTorch with
  RAGKit's pooling (first token, 512-token cap) and Hugging Face's tokenizer.
  The app tokenizes with swift-transformers, which normalizes text the same
  way but was not run here.
- **Scoring.** MRR counts ties at their expected rank. With 24 queries per
  language, differences under about 0.05 are noise.
