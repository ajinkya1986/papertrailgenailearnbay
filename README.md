# Paper Trail (Python / Flask edition)

A retrieval-augmented QA system over three AI papers — *Attention Is All You
Need*, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*,
and *Language Models are Few-Shot Learners* (GPT-3) — implemented **entirely
in Python**, with a server-rendered web interface and no client-side
JavaScript.

This is a rewrite of the original browser-only version (which did retrieval
and generation in JavaScript inside a single HTML file). Here, Flask serves
plain HTML; every request round-trips to the server, which does the
retrieval (scikit-learn TF-IDF), generation (a direct HTTPS call to Claude),
and citation-linking (a regex over the model's answer text) in Python.

## Files

| File | Role |
|---|---|
| `build_corpus.py` | Offline preprocessing: papers → sections → chunks → `chunks.json` |
| `chunks.json` | The 76 preprocessed chunks (already built — you don't need to re-run `build_corpus.py` unless you edit `papers/*.txt`) |
| `retriever.py` | TF-IDF retrieval (`Retriever` class), built on scikit-learn |
| `generator.py` | Calls the Anthropic Messages API to generate a grounded, cited answer |
| `app.py` | Flask app: routes, orchestration, citation-linking |
| `templates/index.html` | The entire UI — one Jinja2 template, zero JavaScript |
| `smoke_test.py` | Integration test of the Flask app (mocks generation, no API key needed) |
| `papers/*.txt` | The curated source text that was chunked |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # get one at https://console.anthropic.com/settings/keys
python app.py
```

Then open **http://127.0.0.1:5000/** in a browser.

## Choosing which LLM provider to use

The app supports three providers out of the box: **Anthropic (Claude)**,
**OpenAI (ChatGPT)**, and **Kimi (Moonshot AI)**. All three are configured
the same way — set `LLM_PROVIDER` and that provider's API key:

```bash
# Claude (default if LLM_PROVIDER is unset)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# ChatGPT
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Kimi (Moonshot AI)
export LLM_PROVIDER=kimi
export MOONSHOT_API_KEY=sk-...
```

You can also override the specific model without touching code:
```bash
export LLM_MODEL=kimi-k2.6
```

**Why this was easy to add:** OpenAI and Kimi both implement the same
"chat completions" REST API shape (Kimi's provider, Moonshot AI, explicitly
documents itself as OpenAI-compatible) — only the base URL, API key, and
default model name differ between them. Only Anthropic's Messages API uses
a genuinely different request/response shape (a top-level `system` field
and a list of typed content blocks in the response, versus OpenAI/Kimi's
`messages` list with a `system`-role message and `choices[0].message.content`).
`generator.py` reflects that: one `_call_anthropic` function, one shared
`_call_openai_compatible` function for everything else, and a `PROVIDERS`
dict that's the only thing you'd touch to add e.g. DeepSeek or Groq — no
other code changes needed since they also speak the OpenAI-compatible shape.

**A caveat worth knowing:** this sandboxed build environment can only reach
`api.anthropic.com` on the network, so the Anthropic path was verified
against the real API (confirmed it correctly returns a structured `401` for
an invalid key). The OpenAI and Kimi paths were verified with mocked HTTP
responses — the request shape (headers, body, URL) matches their published
API docs, but you should do one real test call yourself with a valid key
before relying on it.

## Windows / macOS / Linux setup

The commands above are Linux/macOS-flavored. For exact per-OS commands
(Windows cmd.exe, Windows PowerShell, macOS, Linux — including the
env-var-setting gotchas that differ between them), see
**`CROSS_PLATFORM_SETUP.md`**. Short version: the Python code itself is
100% cross-platform (no hardcoded path separators, explicit `utf-8`
encoding on every file read, no OS-specific calls) — only shell syntax for
creating a venv and setting environment variables differs by platform.

## Preparing for a project interview/evaluation on this codebase

See **`INTERVIEW_PREP.md`** for a full prep guide: an elevator pitch, the
paper content you should know cold (Transformer/RAG/GPT-3), "why did you
choose X" answers for every major design decision, live-coding-round drill
tasks, and a final pre-interview checklist.

## Running the tests

```bash
python smoke_test.py
```

This uses Flask's test client to drive real requests through real routing
and real retrieval, with only the Claude API call mocked out (so it needs no
API key and makes no network call). It checks: the form renders, sample
questions work, blank questions are rejected, a real question returns 8
ranked sources with correctly-linked citations, a failed generation call
degrades gracefully instead of crashing, a zero-overlap question is flagged,
and a `<script>` tag typed into the question box is never reflected back
unescaped.

## Re-running preprocessing (only needed if you edit the papers)

```bash
python build_corpus.py
```

This reads `papers/transformer.txt`, `papers/rag.txt`, `papers/gpt3.txt` and
rewrites `chunks.json`.

## Why no JavaScript at all

Every interactive element (the question box, the 5 sample-question buttons,
even the "click a citation to jump to its source") is done with plain HTML
forms and CSS. Sample questions are separate one-button `<form>`s with a
hidden `question` field; jumping to a citation uses a plain `<a href="#source-3">`
anchor link and the CSS `:target` pseudo-class to highlight the target card
— no `onclick`, no `fetch`, no DOM manipulation. The trade-off: every
question is a full page reload rather than an async update. For a small
educational QA tool this is a fine trade — simpler code, one language for
all logic, and it still works with JavaScript disabled entirely.

## Differences from the browser-only (HTML/JS) version

- Retrieval runs on the server (scikit-learn) instead of in the browser
  (hand-written JS). It also now uses **unigrams + bigrams** (`ngram_range=(1,2)`)
  instead of unigrams only, which slightly improves phrase matching (e.g.
  "multi-head attention" as one term rather than two independent ones).
- Generation calls the Anthropic API directly over HTTPS with `requests`
  and your own `ANTHROPIC_API_KEY` — this version has no built-in
  authentication, unlike the artifact-hosted version where Claude.ai's
  sandbox authenticated `api.anthropic.com` calls automatically.
- Citation linking is a server-side regex + Jinja `|safe` filter instead of
  a client-side DOM rewrite.
- No JavaScript anywhere, including no fetch calls, no client-side TF-IDF,
  and no client-side citation click handlers — replaced by plain forms and
  the CSS `:target` selector.
