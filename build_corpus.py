"""
Document preprocessing pipeline for the RAG system.

Steps:
1. Load each paper's curated text file.
2. Parse into (section_title, section_text) pairs using the "SECTION:" markers.
3. Split each section into word-bounded chunks (target ~130 words, max ~190),
   breaking only at sentence boundaries so we never cut a sentence in half.
4. Attach metadata to every chunk: paper id, paper title, section title,
   chunk index, and a stable chunk id used for citation ("source attribution").
5. Write the result to chunks.json, which is the artifact the retriever
   (TF-IDF, see build_index.py) and the web UI both consume.
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent  # this script's own directory, not the caller's cwd

PAPERS = [
    {
        "id": "transformer",
        "file": BASE_DIR / "papers" / "transformer.txt",
        "title": "Attention Is All You Need",
        "short": "Transformer",
        "year": 2017,
    },
    {
        "id": "rag",
        "file": BASE_DIR / "papers" / "rag.txt",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "short": "RAG",
        "year": 2020,
    },
    {
        "id": "gpt3",
        "file": BASE_DIR / "papers" / "gpt3.txt",
        "title": "Language Models are Few-Shot Learners",
        "short": "GPT-3",
        "year": 2020,
    },
]

TARGET_WORDS = 130   # soft target chunk size
MAX_WORDS = 190       # hard cap before we force a split
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')


def parse_sections(raw_text):
    """Split the curated text on 'SECTION: <title>' headers."""
    parts = re.split(r'\nSECTION:\s*(.+?)\n', raw_text)
    # parts[0] is the preamble (title/authors) before the first SECTION marker
    preamble = parts[0].strip()
    sections = [("Front Matter", preamble)] if preamble else []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((title, body))
    return sections


def split_into_sentences(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    return SENT_SPLIT_RE.split(text)


def chunk_section(sentences, target=TARGET_WORDS, hard_max=MAX_WORDS):
    """Greedily pack sentences into chunks close to `target` words,
    never exceeding `hard_max` words unless a single sentence is longer
    than that on its own (rare; we keep it whole rather than truncate
    mid-sentence, since a truncated sentence is worse for a reader
    checking a citation than a slightly long chunk)."""
    chunks = []
    current, current_words = [], 0
    for sent in sentences:
        w = len(sent.split())
        if current and current_words + w > hard_max:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sent)
        current_words += w
        if current_words >= target:
            chunks.append(" ".join(current))
            current, current_words = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def build():
    all_chunks = []
    stats = []
    for paper in PAPERS:
        raw = open(paper["file"], encoding="utf-8").read()
        sections = parse_sections(raw)
        n_before = len(all_chunks)
        for sec_idx, (sec_title, sec_body) in enumerate(sections):
            sentences = split_into_sentences(sec_body)
            if not sentences:
                continue
            for local_idx, chunk_text in enumerate(chunk_section(sentences)):
                chunk_id = f"{paper['id']}-{sec_idx:02d}-{local_idx:02d}"
                all_chunks.append({
                    "id": chunk_id,
                    "paper_id": paper["id"],
                    "paper_title": paper["title"],
                    "paper_short": paper["short"],
                    "year": paper["year"],
                    "section": sec_title,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                })
        stats.append((paper["short"], len(sections), len(all_chunks) - n_before))

    with open(BASE_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Total chunks: {len(all_chunks)}")
    for short, n_sections, n_chunks in stats:
        print(f"  {short}: {n_sections} sections -> {n_chunks} chunks")
    word_counts = [c["word_count"] for c in all_chunks]
    print(f"Chunk word count: min={min(word_counts)} max={max(word_counts)} "
          f"avg={sum(word_counts)/len(word_counts):.1f}")
    return all_chunks


if __name__ == "__main__":
    build()
