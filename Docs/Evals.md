# Retrieval evals

`Tests/RAGKitEvalTests` measures how well each engine finds the right notes.
Two LLMs grade every note an engine returns: Claude, through Claude Code's
`claude` command, and OpenAI's models, through the Codex CLI's `codex`
command. The eval:

1. indexes a corpus with each engine and runs every query, keeping the top 10;
2. pools, per query, every note any engine returned plus the hand-labeled
   ones;
3. has each judge grade every pooled note 0–3 against the query;
4. scores each engine on those grades, and on the hand labels where a set
   has them.

Hand labels alone fall short in two ways. Writing them is the expensive part
of an eval, and they miss relevant notes their author did not think of; a
judge grades whatever an engine actually returns. With judges, a host can
run the eval on its own corpus with nothing but a list of queries.

## Results

On the bundled set (54 notes, 40 queries), judged by `claude-sonnet-5-5` and
`codex-gpt-5.6-sol`, nDCG@10 averaged over the two judges:

| Engine | All | Keyword | Paraphrase | Question | Short | Non-English | Cross-language |
|---|---|---|---|---|---|---|---|
| Words only (`lexical`) | 0.43 | 0.92 | 0.20 | 0.70 | 0.07 | 0.41 | 0.04 |
| MiniLM, first token (`minilm`) | 0.67 | 0.94 | 0.52 | 0.86 | 0.33 | 0.79 | 0.44 |
| MiniLM, mean (`minilm-mean`) | **0.76** | 0.97 | **0.72** | **0.90** | 0.50 | 0.84 | 0.37 |
| Apple (`apple`) | 0.66 | 0.97 | 0.41 | 0.73 | **0.66** | **0.88** | 0.19 |

- Every engine finds a note that shares the query's words. The engines part
  ways on everything else.
- Mean pooling is the best engine overall, and by far on paraphrases (0.72
  against 0.52 for the first token and 0.41 for Apple). It found the right
  note first for 5 of the 10 paraphrases; first-token MiniLM and Apple each
  managed 1.
- Apple's models lead on one-word topic queries ("health", "programming") and
  on notes in other languages.
- No engine handles cross-language search: an English query for a French,
  Spanish or Chinese note. MiniLM's top 10 held the note for 2 (mean) or 3
  (first token) of the 4 queries, first only once; Apple's held it once, in
  tenth place.
- The judges agree with the hand labels and with each other. Claude graded
  98% of the labeled-relevant notes relevant and Codex 100%. On the 868
  pooled pairs, the two gave the same grade 95% of the time and never
  differed by more than one (κ 0.97).

These agree with what the README measured by other means: mean pooling ranks
better than the first token, and Apple's models suit short queries and other
languages.

## Running it

A plain `swift test` runs only the harness: the metrics against hand-worked
values, the judges' prompt and answer parsing, and one pass of the whole
pipeline on the words-only engine with a stand-in judge that answers with the
labels. It needs no model and no LLM and takes under a second. The eval
itself is skipped unless the environment asks for it:

```bash
RAGKIT_EVAL_JUDGES=claude,codex swift test --filter RetrievalEvalTests
```

That runs the four default engines and both judges on the bundled set, prints
the report, and fails if a judge call failed, if a judge disagrees with the
labels too often to be trusted, or if an engine scores under its floor
(`RetrievalEvalTests.ndcgFloors`, 0.05 under the results above). With the
committed grades it calls neither judge and takes about 30 seconds.

Without judges, every score is against the hand labels, and no LLM is called:

```bash
RAGKIT_EVAL_ENGINES=lexical,minilm,minilm-mean,apple swift test --filter RetrievalEvalTests
```

In Xcode, set the same variables on the test scheme's Run → Arguments →
Environment Variables.

| Variable | Default | |
|---|---|---|
| `RAGKIT_EVAL_JUDGES` | none | `claude`, `codex`, or both, each optionally with a model: `claude:opus,codex:gpt-5.6-terra`. |
| `RAGKIT_EVAL_ENGINES` | `lexical,minilm,minilm-mean,apple` | Any of those and `spotlight`, or `all`. |
| `RAGKIT_EVAL_SET` | the bundled `Fixtures/notes.json` | A set of your own; see [Your own corpus](#your-own-corpus). |
| `RAGKIT_EVAL_OUTPUT` | `.build/ragkit-eval` | Where `report.md` and `results.json` are written. |
| `RAGKIT_EVAL_DEPTH` | 10 | Results per query: what nDCG and recall read, and how deep the pools go. |
| `RAGKIT_EVAL_JUDGE_CONCURRENCY` | 4 | Judge calls in flight at once, per judge. |

The eval needs:

- `claude` signed in (`claude` once, interactively). The judge defaults to
  `claude-sonnet-5-5`.
- `codex` signed in, with a ChatGPT account or an API key (`codex login`).
  The judge defaults to `gpt-5.6-sol`.
- For the MiniLM engines, a one-time download of all-MiniLM-L6-v2 (90 MB)
  from Hugging Face.

Both commands are looked up on `PATH`, then in `/opt/homebrew/bin`,
`/usr/local/bin`, `~/.local/bin` and `~/.claude/local`, since Xcode gives tests
a `PATH` without them.

## The engines

| Name | Engine |
|---|---|
| `lexical` | Words only, no model: hashed word counts in the vector half of VecturaKit's hybrid score, BM25 in the other. The floor the others have to clear. |
| `minilm` | `.sentenceTransformer`, first-token pooling: the default, and what shipped databases hold. |
| `minilm-mean` | `.sentenceTransformer` with `sentenceTransformerPooling: .mean`. |
| `apple` | `.naturalLanguage` with the set's languages. |
| `spotlight` | `.spotlight`, not a default. The eval's notes appear in the Mac's own Spotlight search until the run removes them, and its scores move between runs: Spotlight's meaning matches come and go with the system's state (see the README). With none, it scored a judged nDCG@10 of 0.48 (0.51 against the labels), ahead of `lexical` on questions and non-English queries, with next to nothing on paraphrases or one-word topics. With them, an earlier version scored 0.56–0.66 against the labels. |

Each engine gets a database of its own under Application Support
(`RAGKitEval-<engine>-<UUID>`), deleted when its queries are done.

Each search asks for 10 results, as a host would. The number matters:
VecturaKit's hybrid score combines only each half's top 20 candidates for 10
results. Ranked over the whole corpus instead, first-token MiniLM scored 0.62
rather than 0.67.

VecturaKit also breaks exact ties between scores arbitrarily, differently in
each database. Taken as returned, the words-only engine's top 10 changed on
22 of 40 queries between two identical runs. So the eval leaves out results
scored exactly 0, which have no evidence for them at all, and orders equal
scores by document ID. What is left, across three runs, was one of the 160
rankings changing in its last place, where three notes tie for two places.

## What the scores mean

Grades follow the judges' scale:

| Grade | Meaning |
|---|---|
| 3 | Exactly what the search is after. |
| 2 | Substantially about it, but only partly answers it or buries the answer. |
| 1 | Related, or shares words with the query, but would not satisfy the search. |
| 0 | Nothing to do with it. |

- **nDCG@10**: how close the top 10 come to the best possible order of
  everything graded for the query, weighting a 3 well above a 2. Notes the
  engine missed count against it. This is the headline number, and the one
  the floors use.
- **MRR**: 1 / the rank of the first note graded 2 or more.
- **P@5**: the share of the top 5 graded 2 or more. A query with one relevant
  note can reach only 0.2, so read it across engines, not on its own.
- **Recall@10** (labels only): the share of the labeled-relevant notes in the
  top 10.

The report also lists, per query, the rank at which each engine returned the
first relevant note. `results.json` holds every ranking and every grade.

The bundled set has six kinds of query:

- **keyword**: shares its words with the note.
- **paraphrase**: shares none of them.
- **question**: a question the note answers. One is about a long note whose
  answer sits in the middle; in another, python means the snake.
- **short**: a single-word topic, like a hashtag.
- **non-english**: French, Spanish, German and Chinese queries for notes in
  the same language.
- **cross-language**: English queries for notes in another language.

Four nearly empty notes ("Okay.", "todo") are in the corpus because they
trip up embedding engines.

## The judges

Each judge sees the query and up to ten notes at a time, labeled N1, N2, … in
an order hashed from their IDs, so neither the label nor the position says
which engine returned a note or where it ranked. The prompt is
`JudgePrompt.render`. The judge answers with a grade per label, held to a JSON
schema by both CLIs.

Each judge runs as a bare model:

- **Claude Code**: `claude --print` with the judge's own system prompt in place
  of Claude Code's, no tools, no MCP servers, no settings files or hooks, and
  no saved session.
- **Codex**: `codex exec`, read-only and ephemeral, with `--ignore-user-config`.
  A model in `~/.codex/config.toml` that a ChatGPT sign-in cannot use (Codex
  refuses those with a 400) never applies, and the judge is the same model on
  every machine. Sign-in still comes from `CODEX_HOME`.

Both run in an empty scratch directory, so neither finds a project, a
`CLAUDE.md` or an `AGENTS.md` to read.

A call is given up after two minutes and tried up to three times. In the
first run, 4 of Claude's 116 calls stalled twice in a row with no answer; the
same prompts answered in seconds when replayed.

A judge is only worth scoring with if it agrees with the labels, so the eval
fails when one grades fewer than 85% of the labeled-relevant notes relevant.
The report shows that rate for each judge, and how often each one calls an
unlabeled note relevant. It also shows how often the two judges give the same
grade and their Cohen's κ on relevant or not.

### The cache

Every grade is saved to `Fixtures/notes.judgments/<judge>.json`, one per
line, keyed by the exact query and note text and the prompt's version.
Re-runs only call the judges about pairs they have not graded, and the same
rankings always get the same judged scores. A change in an engine's scores is
then the engine's doing, not a judge's second opinion. Commit the cache files:
a checkout that has them runs the eval on the bundled set without calling
either judge, unless an engine change returns notes they have not graded.

- Editing a note or a query re-grades only that note or query.
- Bump `JudgePrompt.version` after changing the prompt, so nothing graded under
  the old wording is reused.
- Changing a judge's model starts that judge's cache over, in a file of its
  own.

With an empty cache, the bundled set is about 1,000 pairs: 116 calls per
judge. Claude's calls took about 20 seconds each and Codex's less. With four
calls in flight per judge and the two judges side by side, a cold run takes
roughly ten minutes, most of it Claude's.

## Your own corpus

A set is one JSON file:

```json
{
  "name": "my-notes",
  "languages": ["en", "de"],
  "documents": [
    { "id": "standup-0412", "text": "Standup: blocked on the API keys…" }
  ],
  "queries": [
    { "id": "blocked", "text": "what was I blocked on", "kind": "paraphrase",
      "relevant": { "standup-0412": 3 } }
  ]
}
```

`kind` and `relevant` are optional. Without labels the report has judged
scores only, and the judges' agreement with each other is the check on them.
`languages` configures Apple's engine. Grades are cached beside the file, in
`my-notes.judgments/`. The engine floors only apply to the bundled set.

```bash
RAGKIT_EVAL_SET=~/eval/my-notes.json RAGKIT_EVAL_JUDGES=claude,codex swift test --filter RetrievalEvalTests
```

The judges read every query and every pooled note, so running them sends that
text to Anthropic and to OpenAI. Leave `RAGKIT_EVAL_JUDGES` unset to score a
private corpus against its labels without sending anything anywhere.

## Limits

- **Pooling.** A note no engine returned and no label names is never graded,
  so it cannot count against any engine. Adding an engine can only add notes
  to the pools, which can lower the others' nDCG.
- **The judges are models.** They can share a blind spot that both the
  agreement check and κ would miss. The labels are the check on that, and on
  a set without labels nothing is.
- **Small numbers.** With 4 to 10 queries per kind, one query moves a kind's
  score by a tenth or more. Read the kind breakdown as direction, not as a
  measurement.
