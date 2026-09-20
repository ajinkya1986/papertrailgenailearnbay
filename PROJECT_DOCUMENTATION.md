# Paper Trail (Python Edition) — Full Technical Documentation
### Line-by-line walkthrough of every file

This documents the **pure-Python rewrite**: `build_corpus.py`, `retriever.py`,
`generator.py`, `app.py`, `templates/index.html`, and `smoke_test.py`. No
JavaScript runs anywhere in this version — Flask renders complete HTML pages
server-side, and every logical step (retrieval, generation, citation-linking)
is Python you can read top to bottom.

---

## 1. How the four deliverables map to files now

| Deliverable | File(s) | Class/function |
|---|---|---|
| 1. Document preprocessing | `build_corpus.py` | `parse_sections`, `split_into_sentences`, `chunk_section`, `build` |
| 2. Retrieval system | `retriever.py` | `Retriever.__init__`, `Retriever.retrieve` |
| 3. Answer generation | `generator.py` | `build_prompt`, `generate_answer` |
| 4. Source attribution | `retriever.py` (`Retriever.highlight`) + `app.py` (`linkify_citations`) + `templates/index.html` | — |
| Web interface | `app.py` (Flask routes) + `templates/index.html` (Jinja2) | `index()` |

---

## 2. Architecture (Python version)

```
STARTUP (once, when `python app.py` runs)
  app.py line 23-24:  Flask(__name__)  +  Retriever()
                              │
                       retriever.py __init__:
                       chunks.json (76 chunks, built earlier by
                       build_corpus.py) → TfidfVectorizer.fit_transform()
                       → a fitted vocabulary + an (76 × n_terms) TF-IDF matrix,
                       held in memory for the life of the process.

PER REQUEST (every GET or POST to "/")
  browser ──POST question──► app.py: index()
                                  │
                                  ├─► retriever.retrieve(question, k=8)
                                  │     (vectorizes the question against the
                                  │      *already-fitted* vocabulary, computes
                                  │      cosine similarity to all 76 chunks,
                                  │      returns the top 8)
                                  │
                                  ├─► retriever.highlight() for each of the 8
                                  │     (marks up query-term matches with <mark>)
                                  │
                                  ├─► generator.generate_answer(question, top-8)
                                  │     (builds a prompt, POSTs to
                                  │      api.anthropic.com, parses the reply)
                                  │
                                  ├─► app.linkify_citations(answer, 8)
                                  │     (turns "[3]" into a clickable <a> tag)
                                  │
                                  └─► render_template("index.html", ...)
                                        (Jinja2 fills in the answer HTML,
                                         the 8 source cards, and the status line)
                                  │
  browser ◄──full rendered HTML page──┘
```

The critical difference from the browser/JS version: **the TF-IDF index is
built exactly once, at process startup** (`retriever = Retriever()` at
`app.py` line 24, executed at import time, before Flask starts serving
requests), not once per page load. This is possible precisely because a
server process stays alive between requests, whereas a browser tab's
JavaScript had to rebuild the index every time the page was opened.

---

## 3. `build_corpus.py` — document preprocessing, line by line

```python
 14  import json
 15  import re
 16  from pathlib import Path
 17
 18  BASE_DIR = Path(__file__).parent  # this script's own directory, not the caller's cwd
```
Standard library only — `json` to write `chunks.json`, `re` for the sentence
splitter and section parser. **This changed from an earlier version**: the
first draft used bare relative strings (`"papers/transformer.txt"`,
`"chunks.json"`), which only worked if you happened to run the script from
inside its own folder — running `python build_corpus.py` from any other
directory raised `FileNotFoundError`, on every OS equally (this was never a
Windows-vs-Mac issue; relative paths are resolved against the *current
working directory* on all three). `BASE_DIR = Path(__file__).parent` fixes
this by resolving every path relative to *where this file lives on disk*,
not where the terminal happens to be `cd`'d into — the same pattern
`retriever.py` already used for `chunks.json` (§4). This was caught and
fixed by actually testing the failure mode: running the script from `/tmp`
and confirming it still finds `papers/*.txt` and writes `chunks.json` back
into the script's own directory.

```python
 20  PAPERS = [
 21      {
 22          "id": "transformer",
 23          "file": BASE_DIR / "papers" / "transformer.txt",
 24          "title": "Attention Is All You Need",
 25          "short": "Transformer",
 26          "year": 2017,
 27      },
       ... (rag, gpt3 entries follow the same shape)
 42  ]
```
A plain list of dicts — the "manifest" of the corpus. `id` becomes the
prefix of every chunk id (e.g. `transformer-05-00`); `short`/`year` are what
the UI displays; `file` points at the curated `.txt` source for that paper,
now built with the `/` operator on a `Path` object (`BASE_DIR / "papers" /
"transformer.txt"`) rather than a hand-typed string. `Path.__truediv__`
(what `/` calls here) is deliberately cross-platform: it joins segments
with whatever separator the current OS actually uses (`\` on Windows, `/`
on macOS/Linux) — you never write a separator yourself, so the same source
line produces a correct path on all three. Adding a 4th paper to this
system is literally: drop a new `.txt` file into `papers/`, add one dict
here, re-run the script.

```python
 42  TARGET_WORDS = 130   # soft target chunk size
 43  MAX_WORDS = 190       # hard cap before we force a split
 44  SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')
```
Module-level constants, so they're visible in one place and reusable as
default arguments below. `SENT_SPLIT_RE` is a regex with **zero-width
assertions**: `(?<=[.!?])` is a *lookbehind* — it requires the preceding
character to be `.`, `!`, or `?`, but does not consume it (so it stays part
of the sentence before the split). `\s+` is the actual text that gets
matched and removed (the whitespace between sentences). `(?=[A-Z(])` is a
*lookahead* — it requires (without consuming) that the next character be an
uppercase letter or an open parenthesis, which is how the regex tells "end
of sentence" apart from a decimal point or abbreviation followed by a
lowercase word. This is a heuristic, not a full sentence tokenizer (it will
occasionally get an abbreviation like "Fig." wrong), but it's good enough
for the clean, curated prose these `.txt` files contain, and — importantly —
whichever way it splits, it never *deletes or duplicates* any character
other than the matched whitespace, which is checked by re-joining chunks
with `" ".join(...)` later.

```python
 47  def parse_sections(raw_text):
 48      """Split the curated text on 'SECTION: <title>' headers."""
 49      parts = re.split(r'\nSECTION:\s*(.+?)\n', raw_text)
 50      # parts[0] is the preamble (title/authors) before the first SECTION marker
 51      preamble = parts[0].strip()
 52      sections = [("Front Matter", preamble)] if preamble else []
 53      for i in range(1, len(parts), 2):
 54          title = parts[i].strip()
 55          body = parts[i + 1].strip() if i + 1 < len(parts) else ""
 56          sections.append((title, body))
 57      return sections
```
Line 49 is the crux: `re.split` with a **capturing group** in the pattern
returns the delimiter text (the captured section title) interleaved with
the text *between* delimiters. So for input like
`"TITLE: X\nSECTION: Abstract\nfoo\nSECTION: Intro\nbar"`, `parts` comes back
as `["TITLE: X", "Abstract", "foo", "Intro", "bar"]` — index 0 is everything
before the first marker, then alternating (title, body, title, body, ...).
Line 52 handles the title/author block at the very top of each `.txt` file
(before any `SECTION:` marker) by giving it a synthetic section name,
`"Front Matter"`, so it isn't silently dropped — this is where the
`transformer-00-00` chunk (paper title + author list) comes from. The `for`
loop at line 53 steps through `parts` two-at-a-time (`range(1, len(parts),
2)`) pairing each captured title with the body text that follows it.

```python
 60  def split_into_sentences(text):
 61      text = text.replace("\n", " ")
 62      text = re.sub(r'\s+', ' ', text).strip()
 63      if not text:
 64          return []
 65      return SENT_SPLIT_RE.split(text)
```
Line 61 collapses newlines to spaces (the curated `.txt` files sometimes
wrap a paragraph across lines); line 62 then collapses *any* run of
whitespace — including the spaces just created — down to a single space and
trims the ends. Only *after* the text is a single clean line do we apply the
sentence-boundary regex (line 65) — running the regex on un-normalized text
with inconsistent spacing would make the lookahead/lookbehind assertions
unreliable.

```python
 68  def chunk_section(sentences, target=TARGET_WORDS, hard_max=MAX_WORDS):
 74      chunks = []
 75      current, current_words = [], 0
 76      for sent in sentences:
 77          w = len(sent.split())
 78          if current and current_words + w > hard_max:
 79              chunks.append(" ".join(current))
 80              current, current_words = [], 0
 81          current.append(sent)
 82          current_words += w
 83          if current_words >= target:
 84              chunks.append(" ".join(current))
 85              current, current_words = [], 0
 86      if current:
 87          chunks.append(" ".join(current))
 88      return chunks
```
This is a **greedy bin-packing loop** — walk the sentences in order,
accumulating them into `current`. Two `chunks.append(...)` points close out
a chunk:
- **Line 78–80**: *before* adding the next sentence, if doing so would push
  the running total over `hard_max` **and** `current` is non-empty (the
  `current and ...` guard prevents closing an empty chunk, which matters for
  the edge case of a single sentence longer than `hard_max` on its own — see
  the docstring), close the chunk now, so the oversized sentence starts a
  fresh one instead of being appended past the cap.
- **Line 83–85**: *after* adding a sentence, if the running total has
  reached `target`, close the chunk here — this is the normal, common exit,
  producing chunks right around 130 words.
- **Line 86–87**: after the loop, whatever sentences are left in `current`
  (the section's last chunk, which may be under `target` words) get flushed
  as a final chunk — without this, the last few sentences of every section
  would be silently dropped.

```python
 91  def build():
 94      for paper in PAPERS:
 95          raw = open(paper["file"], encoding="utf-8").read()
 96          sections = parse_sections(raw)
 97          n_before = len(all_chunks)
 98          for sec_idx, (sec_title, sec_body) in enumerate(sections):
 99              sentences = split_into_sentences(sec_body)
100              if not sentences:
101                  continue
102              for local_idx, chunk_text in enumerate(chunk_section(sentences)):
103                  chunk_id = f"{paper['id']}-{sec_idx:02d}-{local_idx:02d}"
104                  all_chunks.append({...})
```
Three nested loops: papers → sections within a paper → chunks within a
section. `sec_idx` and `local_idx` (both zero-padded to 2 digits via
`:02d`) are what make chunk ids both **unique** and **human-readable** —
`transformer-05-00` unambiguously means "Transformer paper, 6th section
(0-indexed), 1st chunk of that section" — you can go straight from an id in
a citation back to roughly where in the paper it came from, without opening
`chunks.json`.

```python
116  with open(BASE_DIR / "chunks.json", "w", encoding="utf-8") as f:
117      json.dump(all_chunks, f, indent=2)
```
`indent=2` makes the output file human-readable (useful while debugging
chunking — you can `cat chunks.json` and actually read it), at the cost of a
slightly larger file than a minified dump. Since this file is loaded once at
Flask startup and never over a network to a browser (unlike the earlier
browser-only version, which had to embed and transmit it), file size here
doesn't matter enough to sacrifice readability for it.

**Validated output** (see the shell log from earlier in this build):
76 chunks total (Transformer: 28, RAG: 27, GPT-3: 21), word counts
19–190, average 105.4.

---

## 4. `retriever.py` — retrieval, line by line

```python
 13  from __future__ import annotations
```
Lets the type hints below (`List[dict]`, `-> List[RetrievedChunk]`) be
written normally even though `RetrievedChunk` and `List` are resolved
lazily as strings rather than evaluated immediately — mostly a forward-
compatibility/cleanliness choice here, not load-bearing for such a small
file, but good practice for anyone reading or extending this code.

```python
 21  from sklearn.feature_extraction.text import TfidfVectorizer
 22  from sklearn.metrics.pairwise import cosine_similarity
```
The two scikit-learn pieces this whole module leans on. `TfidfVectorizer`
does tokenization, vocabulary building, TF-IDF weighting, and L2
normalization all in one fitted object — everything the hand-rolled JS
`buildIndex`/`tokenize`/`vectorizeQuery` functions did manually in the
browser version, scikit-learn does with `.fit_transform()`.

```python
 24  CHUNKS_PATH = Path(__file__).parent / "chunks.json"
```
`Path(__file__).parent` is the directory `retriever.py` itself lives in —
using this instead of a bare `"chunks.json"` relative path means the app
finds its data file correctly regardless of *which directory* you happen to
run `python app.py` from (a common source of "file not found" bugs in
scripts that assume the current working directory).

```python
 27  @dataclass
 28  class RetrievedChunk:
 29      """One retrieval hit: the chunk plus how well it matched the query."""
 30      id: str
 31      paper_id: str
 ...
 37      score: float
```
`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from the
type-annotated fields, so instead of a plain dict (as `build_corpus.py`
uses, and as the JS version used everywhere) this is a proper typed object:
`result.score` and `result.paper_short` are attribute accesses your editor
can autocomplete and type-check, catching typos like `result.papershort` at
development time rather than at runtime with a silent `None`/`KeyError`.
This is a genuine improvement the Python rewrite made over the original
JS/dict-everywhere approach.

```python
 48  def __init__(self, chunks_path: Path = CHUNKS_PATH):
 49      self.chunks: List[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
 50      if not self.chunks:
 51          raise ValueError(f"No chunks loaded from {chunks_path}")
 53      texts = [c["text"] for c in self.chunks]
```
Loads the whole `chunks.json` into memory as a list of plain dicts (this is
the *raw* corpus, kept around so `retrieve()` can look up `paper_id`,
`section`, etc. by index later); line 50–51 fails loudly and immediately
if the file is missing or empty, rather than letting the app boot into a
broken state where every question silently returns zero results — an
empty corpus is a startup-time configuration error, not a per-request one,
so it should raise at startup.

```python
 60  self.vectorizer = TfidfVectorizer(
 61      stop_words="english",
 62      ngram_range=(1, 2),
 63      min_df=1,
 64  )
 65  self.matrix = self.vectorizer.fit_transform(texts)
```
- `stop_words="english"` drops a built-in list of ~318 very common English
  words ("the", "of", "and", ...) from the vocabulary before TF-IDF
  weighting — these carry essentially no topical signal and would otherwise
  dominate raw term-frequency counts.
- `ngram_range=(1, 2)` is the one substantive algorithmic upgrade over the
  browser version: the vocabulary includes both single words ("multi",
  "head", "attention") **and** adjacent word-pairs ("multi head", "head
  attention"). This lets a question containing the literal phrase
  "multi-head attention" match a chunk that also contains that exact
  two-word sequence more strongly than one that just happens to contain
  "multi" and "attention" separately, far apart, in an unrelated context —
  a real (if partial) fix for the kind of phrase-blindness discussed as a
  limitation in the original documentation.
- `min_df=1` (the default, stated explicitly for clarity) means a term only
  needs to appear in at least 1 chunk to enter the vocabulary — appropriate
  for a corpus this small, where a term appearing in even a single chunk
  might still be exactly the term a question is looking for.
- `fit_transform(texts)` does two things in one call: **fit** learns the
  vocabulary and IDF weights from `texts`; **transform** immediately
  converts those same 76 texts into a sparse matrix using the
  just-learned weights. The result, `self.matrix`, has shape
  `(76, n_terms)` — one row per chunk.

```python
 67  def retrieve(self, query: str, k: int = 8) -> List[RetrievedChunk]:
 69      query = (query or "").strip()
 70      if not query:
 71          return []
```
`(query or "")` guards against `query` being `None` (defensive, in case a
caller passes it) before `.strip()`, which would otherwise raise
`AttributeError` on `None`. An empty/whitespace-only query returns an empty
list rather than raising or crashing — the caller (`app.py`) never actually
hits this path because it checks for a blank question itself first, but
`retrieve()` doesn't *rely* on that — it's independently safe to call with
a blank string.

```python
 73  query_vec = self.vectorizer.transform([query])
 74  sims = cosine_similarity(query_vec, self.matrix)[0]
```
Critically, this calls `.transform()`, **not** `.fit_transform()`. Fitting
again on just the question would learn a brand-new, incompatible vocabulary
and IDF weights from a single short sentence — nonsensical, and
incomparable to `self.matrix`. `.transform()` reuses the vocabulary and IDF
weights already learned from the 76-chunk corpus, projecting the question
into the *same* vector space the chunks live in, which is the only way
their cosine similarity means anything. `cosine_similarity(A, B)` returns a
matrix of pairwise similarities; since `query_vec` is a single row, the
result has shape `(1, 76)`, and `[0]` unwraps it to a flat length-76 array.

```python
 76  k = min(k, len(self.chunks))
 77  top_indices = sims.argsort()[::-1][:k]
```
Line 76 clamps `k` so a caller asking for more results than exist in the
corpus doesn't cause an out-of-range slice further down (it won't crash
either way — Python slicing is forgiving of an over-long slice — but this
makes the intent explicit and matches the browser version's behavior).
`sims.argsort()` returns the indices that would sort `sims`
**ascending**; `[::-1]` reverses that to descending order; `[:k]` takes the
first `k` — i.e., the indices of the `k` largest similarity scores, highest
first.

```python
 80  for idx in top_indices:
 81      c = self.chunks[idx]
 82      results.append(RetrievedChunk(
 83          id=c["id"], ... , score=float(sims[idx]),
 90      ))
```
For each of those top indices, look up the original chunk dict by position
(`self.chunks[idx]` and `self.matrix`'s rows are in the same order because
both were built from the same `texts` list, in the same iteration order —
this positional correspondence is the load-bearing invariant of the whole
class) and package it into a typed `RetrievedChunk`. `float(sims[idx])`
converts a NumPy scalar (`numpy.float64`) to a plain Python `float` — this
matters because Flask/Jinja's `"%.3f"|format(...)` filter and JSON
serialization elsewhere expect native Python numeric types, not NumPy's.

```python
 94  def highlight(self, text: str, query: str) -> str:
101      query_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9\-]+", query) if len(t) > 2}
102      if not query_terms:
103          return _escape(text)
105      pieces = re.split(r"(\s+)", text)
106      out = []
107      for piece in pieces:
108          bare = re.sub(r"[^a-zA-Z0-9\-]", "", piece).lower()
109          if bare in query_terms:
110              out.append(f"<mark>{_escape(piece)}</mark>")
111          else:
112              out.append(_escape(piece))
113      return "".join(out)
```
This is the source-attribution highlighting: which words in a retrieved
passage actually matched the user's question, visually marked. Line 101
extracts alphanumeric/hyphen "words" from the raw query with `re.findall`
and keeps only those longer than 2 characters (filtering out short/noisy
tokens like "is", "a", "an" that would highlight almost every word in every
passage and defeat the purpose). Line 105's `re.split(r"(\s+)", text)` —
note the **capturing parentheses around `\s+`** — is the trick that
preserves the original whitespace in the output list (an uncaptured
`re.split(r"\s+", text)` would *discard* the whitespace, and re-joining the
pieces would collapse all spacing to nothing); with the capture group,
`pieces` alternates between words and the exact whitespace that separated
them, and `"".join(pieces)` at the end reproduces the original spacing
exactly. For each `piece`, line 108 strips it down to a bare lowercase
alphanumeric form (stripping punctuation like a trailing comma) purely for
*comparison* against `query_terms` — but it's the **original** `piece`
(with its punctuation and capitalization intact) that gets wrapped in
`<mark>` and appended, so a matched word keeps its real appearance in the
rendered page; only the comparison, not the display, is normalized.

```python
116  def _escape(s: str) -> str:
117      return (
118          s.replace("&", "&amp;")
119           .replace("<", "&lt;")
120           .replace(">", "&gt;")
121           .replace('"', "&quot;")
122           .replace("'", "&#39;")
123      )
```
A minimal manual HTML-escaper (module-level function, not a method, since
it needs no access to `self`). It's called on *every* piece of chunk text
before it's wrapped in `<mark>` — this is the guard against a paper excerpt
(or, in principle, a user's question, which flows into the comparison but
not directly into escaped output here) containing characters like `<` that
would otherwise be interpreted as the start of an HTML tag when the
resulting string is inserted into the template with Jinja's `|safe` filter
(which deliberately tells Jinja "trust this string, don't auto-escape it
again" — meaning the escaping has to have already happened correctly by
the time `|safe` is applied, which is exactly what this function
guarantees).

---

## 5. `generator.py` — answer generation, line by line

**This file changed significantly since the first version of this project**:
it started as a single hardcoded call to Claude, and is now provider-agnostic
— Claude, ChatGPT, Kimi (Moonshot AI), OpenRouter, or any future
OpenAI-compatible API, selected by an environment variable with zero code
changes. This walkthrough documents the current version.

```python
 41  MAX_TOKENS = 1000
```
One constant shared by every provider, rather than repeated per provider —
if this ever needs to change, it changes in one place regardless of which
provider is active.

```python
 44  class GenerationError(Exception):
 45      """Raised for any failure calling or parsing an LLM API response."""
```
Unchanged in spirit from the single-provider version: one exception type
for every possible failure, from any provider — a missing key, a network
error, a non-200 status, or a malformed response — so `app.py`'s `except
GenerationError` clause (§6) doesn't need to know or care which provider
was active when the failure happened.

```python
 48  @dataclass
 49  class GenerationResult:
 50      answer: str
 51      n_sources_used: int
 52      provider: str
 53      model: str
```
**Two fields were added here**: `provider` and `model` record *which*
provider and *which* model string actually produced this answer. This
matters once more than one provider exists — without it, nothing in the
returned value would tell you (or a log, or a UI) whether an answer came
from Claude or from Kimi. Adding fields to a `@dataclass` is a breaking
change for any code that constructs one directly with positional-only
assumptions — this exact thing broke `smoke_test.py`'s mock (§8 in the
project's revision history), which built a `GenerationResult` with only the
original two fields, and had to be updated to supply all four.

```python
 67  PROVIDERS = {
 68      "anthropic": {
 69          "format": "anthropic",
 70          "base_url": "https://api.anthropic.com/v1/messages",
 71          "api_key_env": "ANTHROPIC_API_KEY",
 72          "default_model": "claude-sonnet-4-6",
 73      },
 74      "openai": { ... "format": "openai", ... "api_key_env": "OPENAI_API_KEY", ... },
 80      "kimi": { ... "format": "openai", ... "api_key_env": "MOONSHOT_API_KEY", ... },
 93      "openrouter": { ... "format": "openai", ... "api_key_env": "OPENROUTER_API_KEY", ... },
104  }
```
This dict is the entire "provider registry" — the single source of truth
for how to reach each provider. The `"format"` key is the important design
decision here: **only two distinct request/response shapes actually exist**
across all four providers. Anthropic's Messages API puts `system` as a
top-level request field and returns a list of typed content blocks; OpenAI,
Kimi, and OpenRouter all speak the same "chat completions" shape (a
`messages` list where the system prompt is just another message, and a
`choices[0].message.content` response) — this is a real, documented
convention many hosted-LLM vendors converged on, not a coincidence this
project discovered. Because of that, adding OpenRouter (and, before it,
Kimi) required **zero new request-building code** — just one more dict
entry pointing `"format": "openai"` at the shared function (`_call_openai_
compatible`, below). This is the payoff of noticing the shared shape early:
a 5th provider (DeepSeek, Groq, a self-hosted Ollama server) is *also* just
one more dict entry, as the comment above the dict (lines 56–66, not shown)
spells out with a worked example.

```python
112  def build_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
```
Completely unchanged by the multi-provider rework — this function builds
plain text, with no knowledge of HTTP, headers, or response shapes, so it
doesn't need a per-provider version at all. This is a small illustration of
a bigger principle: separating *what to ask* (prompt construction) from
*how to ask it* (the HTTP mechanics below) means a change to one never
risks breaking the other.

```python
129  def _call_anthropic(prompt: str, api_key: str, model: str, base_url: str) -> str:
```
The original single-provider implementation, now demoted to a private
helper (leading underscore — a Python convention meaning "internal to this
module, not part of its public API") and parameterized: `api_key`, `model`,
and `base_url` are now *arguments* instead of hardcoded constants, because
the same function signature has to work whether it's called with a real
key or a test double, and — architecturally — because a function that reads
its own globals is harder to unit-test in isolation than one that receives
everything it needs as arguments. The body (build the Anthropic-shaped
request, check `status_code`, extract `text` blocks) is identical to the
original single-provider version described in earlier documentation.

```python
154  def _call_openai_compatible(prompt: str, api_key: str, model: str, base_url: str) -> str:
159      headers={
160          "Authorization": f"Bearer {api_key}",
161          "Content-Type": "application/json",
162      },
163      json={
164          "model": model,
165          "max_tokens": MAX_TOKENS,
166          "messages": [
167              {"role": "system", "content": SYSTEM_PROMPT},
168              {"role": "user", "content": prompt},
169          ],
170      },
```
The genuinely new function. Same parameter signature as `_call_anthropic`
(deliberately — `generate_answer` below picks between them without caring
which it's calling), but a different authentication header (`Authorization:
Bearer <key>`, the OpenAI-family convention, versus Anthropic's `x-api-key`)
and a different request body shape: `system` is not a separate top-level
field here — it's just the *first message* in the `messages` list, tagged
with `"role": "system"`. This one structural difference is the entire
reason two functions exist instead of one with an `if` branch buried inside
it — the two request shapes differ enough (different headers, different
place for the system prompt, different response parsing below) that
keeping them as separate functions is more readable than one function full
of conditionals.

```python
175  data = response.json()
176  choices = data.get("choices") or []
177  if not choices:
178      raise GenerationError("No choices returned by the model.")
179  content = choices[0].get("message", {}).get("content")
180  if not content:
181      raise GenerationError("No text content returned by the model.")
```
The OpenAI-family response shape is `{"choices": [{"message": {"content":
"..."}}]}` — a list of `choices` (relevant when a caller requests multiple
candidate completions, `n > 1`, which this project never does, so there is
always at most one) each wrapping a `message` object. `data.get("choices")
or []` defends against `choices` being present but `null` (not just
absent) — `.get(..., [])`'s default only applies when the key is *missing
entirely*; `or []` additionally catches the key being present with a falsy
value. `choices[0].get("message", {}).get("content")` chains two `.get()`
calls with fallback defaults (`{}` then implicitly `None`) so a
differently-shaped or truncated response degrades to a clear
`GenerationError` two lines later, rather than raising an unrelated
`KeyError` or `TypeError` from deep inside this function.

```python
185  def generate_answer(
186      question: str,
187      retrieved: List[RetrievedChunk],
188      provider: Optional[str] = None,
189      model: Optional[str] = None,
190  ) -> GenerationResult:
```
The public entry point — the only function of this module `app.py` calls
(§6) — gained two new *optional* parameters. Because both default to
`None`, every existing call site (`generate_answer(question, retrieved)`,
exactly as `app.py` still calls it) keeps working unchanged; the new
parameters exist purely so tests, or a future "pick your model" UI
control, can override the environment-variable defaults for a single call
without touching any environment variables at all — this is precisely how
`test_generator_providers.py` exercises the `openai`/`kimi` paths without
ever needing `LLM_PROVIDER` to be set globally.

```python
203  provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).strip().lower()
204  if provider not in PROVIDERS:
205      raise GenerationError(
206          f"Unknown LLM_PROVIDER '{provider}'. Valid options: {', '.join(PROVIDERS)}"
207      )
```
**This line fixed a real, user-hit bug.** The original version compared
`provider` against `PROVIDERS`'s keys with no normalization — setting
`LLM_PROVIDER=OPENROUTER` (uppercase, as PowerShell's tab-completion or a
copy-pasted example might produce) failed the `in PROVIDERS` check
silently... except the *old* code path (before `"openrouter"` even existed
as a key) had already defaulted to `"anthropic"` one line earlier via
`os.environ.get("LLM_PROVIDER", "anthropic")`, and — critically — that
default only applies when the key is *entirely absent*, not when it's
present-but-wrong-case; so what actually happened when this bug was hit
live was more subtle than a clean "unknown provider" error. Adding
`.strip().lower()` means `OPENROUTER`, `OpenRouter`, `openrouter`, and
`  openrouter  ` (stray whitespace from a copy-paste) are now all treated
identically, and — just as importantly — an actually-unrecognized provider
now reliably reaches the `raise` on line 205 instead of ever silently
falling through to a different provider's default. This is a good example
of a bug that was invisible in code review (the logic *looks* correct) and
only surfaced from an actual user's terminal session — see §9.

```python
209  model = model or os.environ.get("LLM_MODEL") or cfg["default_model"]
```
A three-way fallback chain, evaluated left to right: an explicit `model=`
argument wins if given; otherwise the `LLM_MODEL` environment variable, if
set, lets a user override just the model without touching code (e.g. `set
LLM_MODEL=kimi-k2.6` to pick up a newer model release without a code
change); otherwise the provider's own `default_model` from the registry.

```python
224  try:
225      if cfg["format"] == "anthropic":
226          answer = _call_anthropic(prompt, api_key, model, cfg["base_url"])
227      else:
228          answer = _call_openai_compatible(prompt, api_key, model, cfg["base_url"])
229  except requests.exceptions.RequestException as exc:
230      raise GenerationError(f"Network error calling {provider}: {exc}") from exc
```
The dispatch point: one `if/else` on the registry's `"format"` value
decides which of the two request-building functions actually runs — this
is the only place in the whole module that branches on provider *shape*;
everywhere else operates on `cfg` generically. `except requests.exceptions.
RequestException` here (wrapping *both* possible calls in one `try`) is
deliberately provider-agnostic too — a DNS failure or timeout looks
identical regardless of which provider's URL it was trying to reach, so one
shared handler is correct, not a missed case.

```python
232  return GenerationResult(
233      answer=answer, n_sources_used=len(retrieved), provider=provider, model=model
234  )
```
The final `provider`/`model` values — after normalization and fallback
resolution — are what get recorded on the result, not whatever the caller
originally passed in (which might have been `None`, or wrong-case, or
absent entirely). This means `result.provider` always tells you the truth
about what was actually used, which matters for logging/debugging a
multi-provider system: "which provider actually served this answer" is a
question worth being able to answer with certainty later.

## 6. `app.py` — the Flask app, line by line

```python
 16  import re
 18  from flask import Flask, render_template, request
 20  from generator import GenerationError, generate_answer
 21  from retriever import Retriever
```
`Flask` is the application object type itself; `render_template` looks up
a file in the `templates/` folder and fills it in with the Jinja2 templating
engine (bundled with Flask); `request` gives access to the incoming HTTP
request (its method, and — for a POST — its form fields). The two
project-internal imports pull in exactly the two functions/classes `app.py`
actually calls (`generate_answer`, `GenerationError`) rather than
`import generator` and referring to `generator.generate_answer` everywhere
— a style choice that also, incidentally, is *why* the `smoke_test.py`
monkeypatch (`app_module.generate_answer = fake_generate_ok`) works: because
`from generator import generate_answer` binds the name `generate_answer`
into `app.py`'s own module namespace, overwriting that name from outside
(as the test does) correctly redirects what `index()` calls, without
touching the original function in `generator.py` at all.

```python
 23  app = Flask(__name__)
 24  retriever = Retriever()
```
`Flask(__name__)` creates the application object; passing `__name__` lets
Flask correctly locate resources (like the `templates/` folder) relative to
this file regardless of how the script is invoked. **Line 24 runs at
import time** — the moment Python executes `app.py` (whether via
`python app.py` or via `import app` in `smoke_test.py`), `Retriever()`'s
`__init__` runs immediately, loading `chunks.json` and fitting the
TF-IDF vectorizer, *before* any request has been served and *before*
`if __name__ == "__main__":` at the bottom is even reached. This is also
exactly why `smoke_test.py` needs no special setup to get a working
retriever — simply `import app as app_module` triggers this line.

```python
 36  CITATION_RE = re.compile(r"\[(\d+)\]")
```
Compiled once, at module load, rather than inside `linkify_citations` (which
runs once per request) — a minor but real efficiency point: recompiling the
same regex pattern on every single request would be wasted work,
especially under load with many concurrent requests.

```python
 39  def linkify_citations(answer_text: str, n_sources: int) -> str:
 51      escaped = (
 52          answer_text.replace("&", "&amp;")
 53          .replace("<", "&lt;")
 54          .replace(">", "&gt;")
 55      )
```
Escapes the model's raw answer text *first*, before any citation-parsing —
this ordering matters: if the model's answer happened to contain literal
`<` or `&` characters, they must become `&lt;`/`&amp;` before this string is
ever inserted into HTML via Jinja's `|safe` filter (which explicitly tells
Jinja "don't escape this, I've already handled it" — used on
`answer_html` in the template). Only 3 characters are escaped here (not the
full 5-character set `_escape()` in `retriever.py` handles, which also
covers `"` and `'`) because this string is inserted as HTML *body* content
(inside a `<div>`), not as an HTML attribute value, so quote characters
don't need escaping in this particular context — whereas `retriever.py`'s
`_escape()` is reused for values that could end up compared/matched
character by character and is written more conservatively.

```python
 57  def _replace(match: re.Match) -> str:
 58      n = int(match.group(1))
 59      if 1 <= n <= n_sources:
 60          return f'<a href="#source-{n}" class="cite">[{n}]</a>'
 61      return match.group(0)
 63  return CITATION_RE.sub(_replace, escaped)
```
`CITATION_RE.sub(_replace, escaped)` calls `_replace` once for **every**
match of `\[(\d+)\]` found in `escaped`, replacing each match with
whatever `_replace` returns. Because the regex has no lookahead/lookbehind
tricks for adjacency, a string like `"[1][4]"` naturally produces **two
separate, independent matches** — `"[1]"` and `"[4]"` — each triggering its
own call to `_replace`, so there's no special-casing needed for
multi-citations at all (this is the exact bug class the browser/JS version
hit and had to fix — see the earlier documentation's §6 — solved here from
the start by using the same simple single-bracket pattern). Inside
`_replace`, `match.group(1)` is the captured digit string (from the
parentheses in the pattern), converted to an `int`; the bounds check
`1 <= n <= n_sources` only turns numbers that correspond to an actually-
retrieved source into links — a citation like `[9]` when only 8 sources
were retrieved is left as plain, unlinked text (`match.group(0)`, the whole
original matched substring, e.g. `"[9]"`) rather than becoming a dead link
to a card that was never rendered.

```python
 66  @app.route("/", methods=["GET", "POST"])
 67  def index():
 68      question = ""
 69      answer_html = None
 70      sources = []
 71      status = None
 72      status_is_error = False
```
`@app.route("/", methods=["GET", "POST"])` registers this one function to
handle *both* HTTP methods at the root URL — a `GET` (loading the page
fresh, e.g. by typing the URL or refreshing) and a `POST` (submitting the
form) both land in the same `index()` function, which branches on
`request.method` below. Lines 68–72 initialize every value the template
needs with a sensible default *before* checking the request method, so a
plain `GET` (which skips the whole `if request.method == "POST":` block)
still has something valid to hand to `render_template` — an empty
question, no answer yet, no sources yet, no status message, matching the
initial empty-state the page should show.

```python
 74  if request.method == "POST":
 75      question = request.form.get("question", "").strip()
 77      if not question:
 78          status, status_is_error = "Type a question first.", True
```
`request.form` is a dict-like object of the submitted form fields (from
`<input name="question">` in the template); `.get("question", "")`
defaults to an empty string if the field is somehow missing entirely
(defensive, though the template always sends it), and `.strip()` removes
leading/trailing whitespace so a question of just spaces is treated the
same as no question at all. Line 78 is a **tuple assignment** —
`status, status_is_error = X, True` sets both variables in one line — a
small Python idiom worth recognizing.

```python
 79  else:
 80      retrieved = retriever.retrieve(question, k=TOP_K)
 81      weak_match = all(r.score == 0 for r in retrieved)
```
`all(r.score == 0 for r in retrieved)` is a generator expression fed
directly into the built-in `all()` — this evaluates to `True` only if
*every* one of the 8 retrieved chunks has a similarity score of exactly
`0.0`, i.e. the question shares no vocabulary at all with the corpus. This
is a lazy, memory-efficient way to check a condition across a whole
collection without first building an intermediate list.

```python
 83  sources = [
 84      {
 85          "n": i,
 86          "paper_short": r.paper_short,
 ...
 90          "snippet_html": retriever.highlight(r.text, question),
 91          "chunk_id": r.id,
 92      }
 93      for i, r in enumerate(retrieved, start=1)
 94  ]
```
A **list comprehension** building one dict per retrieved chunk, again
numbered from 1 (matching the numbering used in `generator.build_prompt`
and in `linkify_citations`'s `#source-{n}` anchors — all three places must
agree on 1-based, rank-order numbering for citations to actually resolve to
the correct card). Each dict is deliberately built as a plain dict here
(not the `RetrievedChunk` dataclass) because this is specifically the shape
Jinja2 templates read most naturally (`s.paper_short`, `s.score`, etc. —
Jinja can access both dict keys and object attributes with the same `.`
syntax, so this choice is mostly about keeping the "data for the template"
separate from the "data used in application logic").
`retriever.highlight(r.text, question)` is called here, inside the request
handler, per source, per request — this is the one part of the pipeline
that's genuinely per-request-per-source work (as opposed to `Retriever`'s
one-time index build), and it's cheap enough (simple regex work over
~100-word strings) that this isn't a performance concern even under load.

```python
 96  try:
 97      result = generate_answer(question, retrieved)
 98      answer_html = linkify_citations(result.answer, len(retrieved))
 99      status = (
100          "Answered, but no retrieved passage actually matched this "
101          "question's vocabulary — treat the answer with caution and "
102          "check the sources below."
103          if weak_match else
104          f"Done. Answer grounded in {len(retrieved)} retrieved passages."
105      )
106      status_is_error = weak_match
107  except GenerationError as exc:
108      status = f"Generation error: {exc}"
109      status_is_error = True
```
Lines 99–105 are a Python **conditional expression** (`A if condition else
B`), spanning multiple lines for readability, choosing between two status
messages based on `weak_match` — and critically, this choice is made
**after** generation succeeds, using the `weak_match` flag computed back at
line 81 and still in scope. This directly fixes the exact bug the earlier
(JS) version had and that the smoke test caught: a weak-match warning
computed before generation is not silently overwritten by a generic "Done"
message once generation finishes — this Python version got it right from
the start, informed by that earlier debugging experience.
`except GenerationError as exc` catches specifically the custom exception
type from `generator.py` (not a bare `except:` or `except Exception:`,
which would also swallow unrelated bugs — e.g. a typo in
`linkify_citations` — silently) and turns it into a status message rather
than crashing the whole request with a 500 error page — this is the
"graceful degradation" behavior the smoke test explicitly verifies (a
failed generation still returns HTTP 200 with the retrieved sources still
visible, not a blank error page).

```python
111  return render_template(
112      "index.html",
113      question=question,
114      answer_html=answer_html,
115      sources=sources,
116      status=status,
117      status_is_error=status_is_error,
118      sample_questions=SAMPLE_QUESTIONS,
119      n_chunks=len(retriever.chunks),
120  )
```
`render_template` looks for `index.html` inside the `templates/` folder
(Flask's convention) and calls it with these keyword arguments becoming
variables inside the template's Jinja2 syntax (`{{ question }}`,
`{% for s in sources %}`, etc. — see §7). This single `return` statement is
reached on *every* request — `GET`, `POST` with a blank question, `POST`
with a real question that generates successfully, and `POST` where
generation fails — always rendering the *same* template, just with
different variable values, which is what makes the whole page workable
with zero client-side JavaScript: every state the page can be in is just a
different set of template variables computed server-side.

```python
123  if __name__ == "__main__":
124      app.run(debug=True)
```
This block only runs when `app.py` is executed directly (`python app.py`),
**not** when it's imported (as `smoke_test.py` does via
`import app as app_module`) — this is the standard Python idiom for
"only start the actual server in this mode," letting the same file be both
a runnable script and an importable module for testing.
`debug=True` enables Flask's auto-reloader (the server restarts itself
automatically when you edit and save a `.py` file — visible in the earlier
server log as `Restarting with watchdog`) and a detailed interactive
in-browser traceback on unhandled errors. This is convenient for
development but is explicitly **not** meant for production use (Flask
itself prints exactly this warning at startup, visible in the earlier
`server.log` output: `"This is a development server. Do not use it in a
production deployment."`) — a real deployment should run this app under a
production WSGI server such as Gunicorn instead, with `debug=False`.

---

## 7. `templates/index.html` — the Jinja2 template

This isn't Python, but it's where all the server-computed values actually
reach the page, so it's worth walking through the templating-specific
parts (the CSS is presentation-only and not repeated here).

```html
<input id="question" name="question" type="text" value="{{ question|e }}" ...>
```
`{{ question|e }}` inserts the `question` variable through Jinja's
explicit escape filter (`|e`) — belt-and-suspenders here, since Jinja
auto-escapes `{{ }}` expressions by default for `.html` templates anyway,
but stating `|e` explicitly documents that this value is untrusted
user input and must never be inserted raw. This is what makes a
previously-submitted question re-appear in the input box after the page
reloads (a nice small UX touch that costs nothing extra, since `question`
is already being passed to the template regardless).

```html
{% for q in sample_questions %}
<form method="post" action="/">
  <input type="hidden" name="question" value="{{ q }}">
  <button type="submit" class="sample-chip" title="{{ q }}">{{ q if q|length <= 60 else q[:57] + '…' }}</button>
</form>
{% endfor %}
```
Each sample question is rendered as its **own separate `<form>`** with a
single hidden field pre-filled with the full question text and a submit
button showing a (possibly truncated) label. Clicking any of these buttons
submits that tiny form directly to `/`, which is indistinguishable from the
user having typed that question into the main box and pressed the main
Ask button — this is the pure-HTML replacement for what used to be a
JavaScript `onclick` handler that filled in the input and triggered a
`fetch()` call. `{{ q if q|length <= 60 else q[:57] + '…' }}` is a Jinja
conditional expression doing the same "truncate to 60 chars with an
ellipsis" the old JS version did with `q.slice(0, 57) + "…"` — the same
logic, just Python/Jinja syntax instead of JavaScript.

```html
{% if answer_html %}
<div class="answer-block">{{ answer_html|safe }}</div>
{% endif %}
```
`{% if answer_html %}` only renders the answer block at all if there is an
answer to show (on a fresh `GET`, `answer_html` is `None`, and this whole
block is skipped). `|safe` is the Jinja filter that means "do not
auto-escape this — trust that it's already valid, safe HTML" — this is
exactly, and only, appropriate here because `answer_html` was built by
`linkify_citations()` in `app.py`, which already escaped the model's raw
text *before* inserting the `<a class="cite">` tags (§6) — if this template
inserted the model's *raw* unescaped answer with `|safe`, that would be a
real HTML/script-injection vulnerability; the smoke test's XSS-guard
assertion (§8) exists specifically to keep this invariant honest.

```html
{% for s in sources %}
<div class="source-card" id="source-{{ s.n }}">
  ...
  <span class="source-score">cos={{ "%.3f"|format(s.score) }}</span>
  ...
  <div class="source-snippet">{{ s.snippet_html|safe }}</div>
  ...
</div>
{% endfor %}
```
`id="source-{{ s.n }}"` generates the exact same `source-1`, `source-2`, ...
ids that `linkify_citations` generates links to — these two pieces of code
(one in `app.py`, one in the template) have to agree on this naming scheme
for citation-jumping to work at all, which is why both are documented
together here. `"%.3f"|format(s.score)` is Jinja's way of applying Python's
old-style `%` string formatting as a filter, producing e.g. `0.288` from
the float `0.28800000...` — the same 3-decimal precision the browser
version showed. `s.snippet_html|safe` inserts the `<mark>`-highlighted
excerpt HTML built by `Retriever.highlight()` — safe for the same reason as
`answer_html` above: it was built by code that escapes every piece of text
before wrapping any of it in HTML tags.

```html
.source-card:target{ background: var(--accent-soft); }
```
This one CSS rule (in the `<style>` block) is the entire mechanism that
replaces the JS version's `element.scrollIntoView()` + `classList.add
("flash")` + `setTimeout(...remove...)` click handler. The CSS `:target`
pseudo-class matches an element whose `id` equals the URL fragment
(`#source-3`) currently in the address bar. Clicking `<a href="#source-3">`
is a completely native browser action — no JavaScript involved at all —
that (a) scrolls the browser to bring that element into view, and (b)
changes the URL fragment, which (c) makes `:target` match that card and
apply its background-highlight style. The trade-off, honestly noted: this
highlight doesn't automatically fade back out after a couple of seconds the
way the JS version's `setTimeout`-based flash did (a fragment stays in the
URL, and thus the highlight stays applied, until you click another
citation or otherwise navigate) — a small UX difference that's the direct
cost of the "no JavaScript" constraint, and worth being able to explain if
asked why the two versions don't behave identically.

---

## 8. `smoke_test.py` — line by line

```python
 15  import app as app_module
```
Importing `app.py` as a module (under the alias `app_module`, avoiding a
name clash with Flask's own `app` object defined inside it) is what
triggers `Retriever()` to actually load `chunks.json` and fit the TF-IDF
vectorizer (§6) — by the time this import finishes, the real retrieval
index is built and ready, with no extra setup code needed in this test file.

```python
 21  def check(cond, msg):
 22      global failures
 23      status = "PASS" if cond else "FAIL"
 24      print(f"{status}: {msg}")
 25      if not cond:
 26          failures += 1
```
A minimal hand-rolled assertion helper (rather than using `pytest` or
`unittest`, to keep this runnable with zero extra dependencies beyond what
`requirements.txt` already lists) — it prints every check's result
immediately (so a full run gives a readable pass/fail log, not just a final
count) and tracks a running `failures` count via the `global` keyword
(needed because `failures` is a module-level variable being *reassigned*,
not just read, inside this function).

```python
 29  def fake_generate_ok(question, retrieved):
 30      return GenerationResult(
 31          answer='Multi-head attention runs several attention functions in parallel [1]. '
 32                 'This lets the model attend to different representation subspaces at once [1][2].',
 33          n_sources_used=len(retrieved),
 34      )
```
A **test double** standing in for the real `generate_answer` — it has the
same signature (`question, retrieved`) and the same return type
(`GenerationResult`) as the real function, so it's a drop-in replacement,
but it returns a fixed, known answer instead of making a real network call.
This specific canned answer deliberately includes both a single citation
(`[1]`) and an adjacent double citation (`[1][2]`) so the test can verify
both citations link correctly (§8 test at line 63–64) — directly testing
the exact scenario that caused a real bug in the earlier browser/JS
version.

```python
 41  client = app_module.app.test_client()
```
Flask's `test_client()` gives you an object that can make `.get(...)` and
`.post(...)` calls that go through **all** of Flask's real routing,
request-parsing, and view-function logic, and get back a real `Response`
object with a real status code and body — without needing an actual TCP
server listening on a port. This is what makes the difference between a
superficial "does the Python syntax parse" check and a genuine end-to-end
test of the actual request/response behavior a browser would experience.

```python
 57  app_module.generate_answer = fake_generate_ok
```
This line **replaces the name `generate_answer` inside the already-loaded
`app` module** with the fake function — this only works because `app.py`
did `from generator import generate_answer` (binding that name directly
into `app`'s own namespace) rather than always referring to
`generator.generate_answer` — see the note on this exact point back in §6.
After this line, any *future* call to `index()` (which looks up
`generate_answer` in its enclosing module's global namespace each time it
runs, not once at function-definition time) will call the fake, not the
real network-calling function — no changes to `app.py` or `generator.py`
were needed to make this testable this way.

```python
 65  check("Transformer" in body.split('id="source-1"')[1][:200], ...)
```
Rather than parsing the returned HTML with a proper HTML parser (which
would be more robust but adds a dependency), this test uses a pragmatic
string-splitting trick: split the whole page body on the exact string
`id="source-1"`, take everything *after* that split point (`[1]`), and look
at just the first 200 characters of it — enough to contain that specific
source card's own content but unlikely to accidentally include the *next*
card's content too. This is intentionally a lighter-weight testing
technique than full HTML parsing, appropriate for a smoke test whose job is
"catch obvious breakage," not exhaustively validate DOM structure.

```python
 84  resp = client.post("/", data={"question": "<script>alert(1)</script> attention"})
 85  body = resp.get_data(as_text=True)
 86  check("<script>alert(1)</script>" not in body, ...)
```
This test submits a question that looks like an XSS payload and asserts
the **literal, unescaped** string `<script>alert(1)</script>` never appears
in the response body — it would be escaped to
`&lt;script&gt;alert(1)&lt;/script&gt;` by Jinja's autoescaping on the
`{{ question|e }}` line in the template (§7), so this string should never
appear raw. This test doesn't currently exercise the `answer_html` /
`linkify_citations` escaping path directly (since `fake_generate_ok`
returns a fixed, safe string, not a reflection of the user's question) —
worth knowing as an honest gap: a more thorough test would also mock
`generate_answer` to *echo back* the raw question in its fake answer, to
verify `linkify_citations`'s own escaping (§6) independently of Jinja's.
This is a reasonable "what would you add next" answer if asked.

---

## 9. Testing summary (what was actually run and confirmed)

1. **Retriever validated standalone** against all 5 sample questions —
   `retriever.retrieve()` was called directly (no Flask involved) and its
   top results printed and inspected for each question.
2. **`smoke_test.py`**, 18 checks, run via `python smoke_test.py`, all
   passing — using Flask's real test client against the real app, with
   only network-calling `generate_answer` mocked out.
3. **Live connectivity check** — called `generate_answer` with a
   deliberately invalid `ANTHROPIC_API_KEY` and confirmed the request
   reaches `api.anthropic.com` and returns a structured `401
   authentication_error` (proving the request is correctly formed — headers,
   body shape, URL — end to end; only a valid key is missing in this
   sandboxed environment).
4. **Live server boot test** — actually ran `python app.py` in the
   background, then hit `http://127.0.0.1:5000/` with real `curl` `GET` and
   `POST` requests and confirmed `HTTP 200` responses and correctly-shaped
   HTML (8 `source-card` elements present), over a real TCP connection —
   not just Flask's in-process test client.
5. **`test_generator_providers.py`**, 15 checks, added when multi-provider
   support was built — verifies the `openai`/`kimi` request shapes (Bearer
   auth header, `system` as a `messages[0]` entry, correct base URLs,
   correct default models) against mocked HTTP responses, an unknown
   provider raises a clear error, a missing API key names the exact
   environment variable to set, and the `anthropic` path still round-trips
   against the real API.
6. **Directory-independence test** — ran `python build_corpus.py` from
   `/tmp` (a directory containing none of the project's files) after
   switching its path handling to `Path(__file__).parent`-relative, and
   confirmed it still found `papers/*.txt` and wrote `chunks.json` back
   into the script's own folder rather than failing or writing to `/tmp`.
7. **A real bug, found from an actual user session, not code review** —
   setting `LLM_PROVIDER=OPENROUTER` (uppercase) was silently treated as
   the default `anthropic` provider instead of either working correctly or
   raising a clear "unknown provider" error, because the provider-name
   comparison was case-sensitive. Fixed by normalizing with `.strip()
   .lower()` before the lookup (§5); confirmed fixed by re-running the
   exact failing scenario (`LLM_PROVIDER=OPENROUTER` with no
   `OPENROUTER_API_KEY` set) and checking the error message now correctly
   names `OPENROUTER_API_KEY` and `'openrouter'`, not `ANTHROPIC_API_KEY`
   and `'anthropic'`.

## 10. Cross-platform verification (Windows / macOS / Linux)

This project is only ever *executed* in this build environment on Linux —
being upfront about that limits what "tested on Windows/Mac" can honestly
mean here. What was actually done, split into what was *run* versus what
was *audited*:

**Run, for real, on Linux**: every test suite above (1–7), plus the live
server boot test.

**Audited for platform-independence** (grep + manual review of every `.py`
and `.html` file), since these are the actual ways cross-platform Python
code tends to break silently on only one OS:
- No hardcoded path separators (`\\` or string-concatenated `/`) anywhere —
  all paths are built with `pathlib`'s `/` operator or Flask/Jinja's own
  (already cross-platform) template loader.
- No `os.system`, `subprocess`, or `shell=True` calls anywhere in the
  project — nothing that could depend on which shell or OS is running it.
- Every `open()`/`read_text()` call that reads project files specifies
  `encoding="utf-8"` explicitly. This matters concretely, not just in
  theory: `generator.py` and `templates/index.html` contain real non-ASCII
  characters (an em dash `—`, an ellipsis `…`, arrows, a bullet). Python 3
  always parses `.py` *source* files as UTF-8 regardless of OS (PEP 3120),
  but *data* files read at runtime (`chunks.json`, the papers' `.txt`
  files, the Jinja template) are only safe from mojibake or
  `UnicodeDecodeError` on Windows — where the OS-locale default encoding is
  often `cp1252`, not UTF-8 — because every read explicitly requests UTF-8
  rather than relying on a platform default.
- No OS-specific modules imported anywhere (`fcntl`, `termios`, `pwd`,
  `posix`, `nt`, ...).
- Flask's built-in dev server (`app.run(debug=True)`) and `requests` are
  both pure-Python-plus-C-extension packages published with wheels for
  Windows, macOS, and Linux — nothing about how this project uses either
  library is platform-conditional.

**The one thing that *is* genuinely platform-specific, and can't be coded
around**: how you set an environment variable at the shell, which differs
across cmd.exe (`set VAR=value`), PowerShell (`$env:VAR = "value"`), and
bash/zsh (`export VAR=value`) — this is a shell syntax difference, not a
Python one, and is covered command-by-command in the companion
`CROSS_PLATFORM_SETUP.md`.

## 11. Quick-reference cheat sheet (Python version specifics)

- Retrieval: scikit-learn `TfidfVectorizer(stop_words="english",
  ngram_range=(1,2))` + `cosine_similarity`; `.fit_transform()` once at
  startup, `.transform()` per question (never re-fit per question).
- `TOP_K = 8`, same empirically-chosen value as the browser version.
- Generation: raw `requests.post` to `https://api.anthropic.com/v1/messages`,
  model `claude-sonnet-4-6`, `max_tokens=1000`, headers `x-api-key`,
  `anthropic-version: 2023-06-01`, `content-type: application/json`.
- Citation regex: `\[(\d+)\]` — deliberately matches one bracket at a time
  (not the multi-bracket pattern that caused a bug in the JS version).
- Zero JavaScript: sample questions are individual `<form>`s with hidden
  inputs; citation-jump-and-highlight uses `<a href="#source-n">` plus the
  CSS `:target` pseudo-class.
- `Retriever()` is constructed once at module import time (`app.py` line
  24), not per-request — this is the main architectural advantage of a
  server process over the earlier stateless-per-page-load browser version.
- Multi-provider generation: `LLM_PROVIDER` env var selects `anthropic` /
  `openai` / `kimi` / `openrouter` (default `anthropic`), each reading its
  own `..._API_KEY` env var; `LLM_MODEL` optionally overrides the model
  string. Provider names are matched case-insensitively.
- Two request shapes cover all four providers: `_call_anthropic` (Claude's
  Messages API) and `_call_openai_compatible` (OpenAI, Kimi, OpenRouter —
  all documented-identical "chat completions" shape).
- All file paths are built with `pathlib`'s `/` operator relative to
  `Path(__file__).parent`, never a bare relative string — this is what
  makes both `retriever.py` and `build_corpus.py` runnable from any working
  directory, on any OS.
