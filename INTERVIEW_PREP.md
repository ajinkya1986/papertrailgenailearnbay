# Interview Preparation Guide — Paper Trail (RAG QA Project)

Covers all three evaluation rounds: **Project Presentation**, **Coding
Round** (live Python), and **Subject Knowledge** (interview-style). Work
through this in order — each section builds on the last.

---

## 1. Your 90-second elevator pitch (memorize the shape, not the words)

Structure: **problem → approach → one real finding → one honest limitation
→ one improvement**. Practice saying this out loud until it takes 90
seconds without rushing.

> "I built a retrieval-augmented QA system over three foundational AI
> papers — Attention Is All You Need, the RAG paper, and GPT-3. The
> pipeline has four stages: preprocess the papers into ~130-word chunks,
> retrieve the most relevant ones for a question using TF-IDF and cosine
> similarity, generate an answer from an LLM constrained to only use those
> chunks, and attribute every claim back to its source with clickable
> citations. One finding I want to highlight: for the few-shot-learning
> question, the single best passage actually ranked 8th, not 1st, when I
> only retrieved the top 3 — because TF-IDF's weighting formula penalizes a
> term for being *common*, and 'few-shot' is the most common term in a
> paper about few-shot learning. I fixed that empirically by retrieving 8
> passages instead of 3. The honest limitation is that TF-IDF has no
> concept of paraphrase or synonyms — a production version of this would
> need dense embeddings or a hybrid retriever with re-ranking, which is
> exactly what I'd build next."

---

## 2. Know the papers cold — this is non-negotiable

Subject-knowledge interviews for a RAG project **will** test whether you
actually understand the source material, not just your code. Be able to
explain each of these **without notes**, ideally by drawing on a
whiteboard:

### Attention Is All You Need (Transformer)
- **Encoder**: 6 identical layers, each with 2 sub-layers — multi-head
  self-attention, then a position-wise feed-forward network. Each sub-layer
  wrapped in a residual connection + layer norm: `LayerNorm(x + Sublayer(x))`.
- **Decoder**: same 2 sub-layers plus a third (encoder-decoder attention),
  and the self-attention sub-layer is **masked** so position *i* can't see
  positions after it (preserves autoregression).
- **Scaled dot-product attention**: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`.
  **Be ready to explain why we divide by √d_k**: for large d_k, the dot
  products grow large in magnitude, pushing softmax into regions with
  vanishing gradients — dividing by √d_k counteracts that.
  - **Q, K, V exist because**: Query = "what am I looking for", Key = "what
    do I contain" (for matching against queries), Value = "what do I
    actually pass forward if selected" — three distinct learned projections
    of the same input, so a token can look for one thing but contribute
    different information depending on who's asking.
- **Multi-head attention**: instead of one attention function over the full
  `d_model` dimension, project Q/K/V into `h=8` smaller subspaces
  (`d_k=d_v=d_model/h=64`), attend in parallel, concatenate, project once
  more. **Why beneficial**: a single head averages over all positions,
  which inhibits attending to different types of relationships
  simultaneously; multiple heads can each specialize (one might track
  syntax, another long-distance coreference).
- **Positional encoding**: sinusoidal, `PE(pos,2i)=sin(pos/10000^(2i/d_model))`.
  Needed because the architecture has **no recurrence and no convolution**
  — nothing else in the model encodes token order at all. Chosen over
  learned embeddings (which scored nearly identically) because sinusoids
  may let the model extrapolate to sequence lengths longer than training.
- **Complexity**: self-attention is O(n²·d) per layer but O(1) sequential
  operations (fully parallel); a recurrent layer is O(n·d²) per layer but
  O(n) sequential operations (fundamentally serial). Self-attention wins on
  parallelism; wins on total compute only when sequence length n < d.

### Retrieval-Augmented Generation (RAG paper)
- **Two components**: retriever p_η(z|x) (DPR — a BERT bi-encoder producing
  query and document embeddings, matched via Maximum Inner Product Search)
  and generator p_θ(y|x,z) (BART-large). They're trained **jointly**, but
  the document encoder is **frozen** — only the query encoder and BART are
  fine-tuned (updating the document encoder would require constantly
  re-indexing all of Wikipedia).
- **RAG-Sequence vs RAG-Token**: Sequence uses *one* retrieved document to
  generate the entire output; Token can pick a *different* document per
  output token and marginalize per-token. Token performed better on
  Jeopardy question generation specifically because it can pull different
  facts from different documents into one generated sentence (the
  "Hemingway" example: one document informs "A Farewell to Arms", another
  informs "The Sun Also Rises").
- **Why this matters for your project**: your system is closer to
  RAG-Sequence in spirit — you retrieve a fixed set of chunks once, then
  generate the whole answer conditioned on all of them at once (not
  per-token marginalization, since you're calling a hosted LLM API, not
  training a model).

### Language Models are Few-Shot Learners (GPT-3)
- **Zero/One/Few-shot are about inference-time conditioning, not
  training** — no gradient updates happen in any of the three. Zero-shot:
  natural-language instruction only. One-shot: instruction + 1 example.
  Few-shot: instruction + K examples (typically 10–100, bounded by the
  2048-token context window).
- **"In-context learning"** is the paper's term for this — the model uses
  its context window as the task specification, relying on patterns learned
  during pre-training rather than any parameter update.
- **Be ready for**: "is this actually learning, or just recognizing a task
  it saw in training?" — the paper itself raises this as an open question,
  and saying so shows you read past the abstract.

---

## 3. "Why did you..." questions — have a real answer, not a shrug

These separate someone who understood their own project from someone who
followed a tutorial. Practice saying each answer in under 30 seconds.

**Q: Why TF-IDF instead of embeddings (e.g. sentence-transformers, OpenAI
embeddings)?**
> "Three reasons: zero infrastructure — no vector DB, no embedding API call,
> runs entirely in scikit-learn; full explainability — every similarity
> score is a hand-checkable dot product I can defend, not a black-box
> neural representation; and the corpus is small and lexically dense —
> someone asking about 'multi-head attention' is very likely to hit chunks
> that literally contain that phrase. The real cost showed up empirically:
> the few-shot question's ideal chunk ranked 8th because TF-IDF's IDF term
> punishes common-within-corpus vocabulary, which is exactly what a paper's
> own central topic looks like."

**Q: Why not LangChain or LlamaIndex?**
> "Deliberately avoided them — I wanted to be able to explain every line of
> the retrieval and generation logic under questioning, not depend on a
> framework abstraction I'd have to say 'it just handles that' about."

**Q: Why Flask with server-side rendering instead of a JS frontend / React?**
> "Simplicity and auditability — every interactive element, including
> citation-jump-and-highlight, works with plain HTML forms and the CSS
> `:target` selector, zero JavaScript. The honest trade-off is a full page
> reload per question instead of an async update, and the citation
> highlight doesn't auto-fade the way a JS `setTimeout` version would."

**Q: Why support multiple LLM providers (Claude/OpenAI/Kimi/OpenRouter)?**
> "Two reasons. Practically, it means the project isn't locked to one API
> key running out or one company's pricing. Architecturally, it forced a
> clean separation — I noticed OpenAI, Kimi, and OpenRouter all implement
> the identical 'chat completions' request shape, so adding each new one
> after the first was a one-entry addition to a registry dict, not new
> code. Only Anthropic's API has a genuinely different shape (system prompt
> as a top-level field, typed content blocks in the response), so that's
> the one place with provider-specific logic."

**Q: What's the biggest bug you found, and how?**
> "Two, both from actual testing, not code review. First: my citation regex
> initially tried to match multi-citations like '[1][4]' as one unit and
> mangled them into a broken link to a nonexistent source. Caught by adding
> a test case with exactly that pattern. Second: a case-sensitivity bug —
> setting LLM_PROVIDER=OPENROUTER (uppercase) silently fell back to the
> default Anthropic provider instead of erroring clearly, because the
> provider lookup was an exact string match. Found from an actual user
> session, not a test — fixed by lowercasing and stripping the value before
> comparison."

**Q: How did you validate retrieval quality?**
> "Ran the retriever standalone against the 5 required sample questions,
> printed the top-k results with similarity scores, and manually checked
> whether the passage that actually answers each question showed up — and
> found the few-shot ranking issue that way. I'd be upfront that this is
> eyeballing 5 questions, not a rigorous precision@k/recall@k evaluation
> against a labeled set — that's my top recommendation for what to add
> next."

---

## 4. Live coding round — what to be fluent in without looking anything up

The trainer said this will be a **live demonstration**. Rehearse these
specific tasks until you can do each in under 2 minutes:

1. **Add a 4th paper to the corpus.** You'd: drop a new curated `.txt` file
   into `papers/` with `SECTION:` markers, add one dict entry to the
   `PAPERS` list in `build_corpus.py`, run `python build_corpus.py`,
   restart the Flask app (which reloads `chunks.json` at startup).
2. **Change `TOP_K`.** It's a module-level constant in `app.py`
   (`TOP_K = 8`) — change it, restart, and be ready to explain the
   trade-off (higher k = better recall, more context tokens sent to the
   LLM, higher cost/latency per call).
3. **Explain `.transform()` vs `.fit_transform()` from memory** — fitting
   learns the vocabulary + IDF weights from the corpus; transforming a new
   query re-uses those learned weights rather than re-fitting on a single
   sentence (which would be meaningless and incomparable to the corpus
   vectors).
4. **Trace a request end-to-end out loud**: form submit → `index()` in
   `app.py` → `retriever.retrieve()` → `retriever.highlight()` per source →
   `generate_answer()` → `linkify_citations()` → `render_template()`.
5. **Explain what happens if `ANTHROPIC_API_KEY` is missing** — you've
   personally debugged this exact scenario for real; walk through the
   `GenerationError` raised in `generator.py`, caught in `app.py`'s
   `except GenerationError`, surfaced as a status message, with sources
   still shown (graceful degradation, not a crash).
6. **Add a print/debug statement and re-run a test** — show you're
   comfortable with the actual dev loop, not just reciting code.

---

## 5. Testing — bring this up before they ask

Most peers submitting this project probably have a `.ipynb` or a script
they ran once and eyeballed. You have:
- `smoke_test.py` — 18 assertions, using Flask's real test client
- `test_generator_providers.py` — 15 assertions covering all 4 LLM
  providers with mocked HTTP responses
- A documented, reproducible bug-fix history (the citation regex bug, the
  case-sensitivity bug, the directory-independence fix in
  `build_corpus.py`)

**Say this explicitly during your presentation**, don't wait to be asked:
*"Before I show you a feature works, let me show you how I verified it."*
This is a senior-engineer habit and is one of the most differentiating
things you can demonstrate in a student cohort.

---

## 6. Questions to ask THEM (ends your presentation on a strong note)

Pick one or two, don't ask all of them:
- "Given more time, I'd want to replace the 5 hand-picked sample questions
  with a proper labeled evaluation set and measure precision@k/recall@k —
  is that the kind of rigor you'd expect for a production RAG system?"
- "I chose TF-IDF over embeddings for explainability and zero
  infrastructure — in your experience, at what corpus size or query
  complexity does that trade-off stop being worth it?"
- "Is there a preferred vector database or re-ranking approach your
  organization typically uses, that I should get familiar with?"

---

## 7. Final pre-interview checklist

- [ ] Can recite the 90-second pitch without notes
- [ ] Can explain scaled dot-product attention and why we divide by √d_k
- [ ] Can explain RAG-Sequence vs RAG-Token and why RAG-Token won on
      Jeopardy generation specifically
- [ ] Can define zero/one/few-shot precisely (inference-time conditioning,
      no gradient updates)
- [ ] Can explain the "few-shot ranks 8th" finding and *why* it happens
      (IDF penalizes common terms)
- [ ] Can justify TF-IDF vs embeddings, Flask vs JS frontend, and
      multi-provider support, each in under 30 seconds
- [ ] Can perform all 6 live-coding tasks in section 4 without hesitation
- [ ] Have the app actually running and tested on the machine you'll
      present from — not just "it worked yesterday"
- [ ] Know your own test suite numbers (18 + 15 assertions) and can name
      one specific thing each caught
- [ ] Have 1–2 questions ready to ask the interviewer
