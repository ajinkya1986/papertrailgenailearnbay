"""
app.py
------
The web interface. A single Flask route handles both showing the empty
question form (GET) and processing a submitted question (POST), rendering
everything server-side with Jinja2. There is no client-side JavaScript in
this version -- retrieval, generation, and citation-linking are all plain
Python, and the browser only ever receives a fully-rendered HTML page.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000/
"""
import re

from flask import Flask, render_template, request

from generator import GenerationError, generate_answer
from retriever import Retriever

app = Flask(__name__)
retriever = Retriever()  # loads chunks.json and builds the TF-IDF index once at startup

TOP_K = 8

SAMPLE_QUESTIONS = [
    "What are the main components of a RAG model, and how do they interact?",
    "What are the two sub-layers in each encoder layer of the Transformer model?",
    "Explain how positional encoding is implemented in Transformers and why it is necessary.",
    "Describe the concept of multi-head attention in the Transformer architecture. Why is it beneficial?",
    "What is few-shot learning, and how does GPT-3 implement it during inference?",
]

CITATION_RE = re.compile(r"\[(\d+)\]")


def linkify_citations(answer_text: str, n_sources: int) -> str:
    """
    Turn every literal "[n]" in the model's answer into an HTML link to the
    matching source card (id="source-n"), but only for n within the range
    of sources actually retrieved -- if the model cites an out-of-range
    number, it's left as plain text instead of linking to a nonexistent card.
    Note: the answer text is escaped by Jinja's autoescaping when rendered
    with the `|safe` filter is NOT used on the raw text; instead we build
    the final HTML here explicitly and mark only this constructed string
    as safe in the template, so nothing in the model's own answer can
    inject arbitrary HTML.
    """
    escaped = (
        answer_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            return f'<a href="#source-{n}" class="cite">[{n}]</a>'
        return match.group(0)

    return CITATION_RE.sub(_replace, escaped)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer_html = None
    sources = []
    status = None
    status_is_error = False

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            status, status_is_error = "Type a question first.", True
        else:
            retrieved = retriever.retrieve(question, k=TOP_K)
            weak_match = all(r.score == 0 for r in retrieved)

            sources = [
                {
                    "n": i,
                    "paper_short": r.paper_short,
                    "year": r.year,
                    "section": r.section,
                    "score": r.score,
                    "snippet_html": retriever.highlight(r.text, question),
                    "chunk_id": r.id,
                }
                for i, r in enumerate(retrieved, start=1)
            ]

            try:
                result = generate_answer(question, retrieved)
                answer_html = linkify_citations(result.answer, len(retrieved))
                status = (
                    "Answered, but no retrieved passage actually matched this "
                    "question's vocabulary — treat the answer with caution and "
                    "check the sources below."
                    if weak_match else
                    f"Done. Answer grounded in {len(retrieved)} retrieved passages."
                )
                status_is_error = weak_match
            except GenerationError as exc:
                status = f"Generation error: {exc}"
                status_is_error = True

    return render_template(
        "index.html",
        question=question,
        answer_html=answer_html,
        sources=sources,
        status=status,
        status_is_error=status_is_error,
        sample_questions=SAMPLE_QUESTIONS,
        n_chunks=len(retriever.chunks),
    )


if __name__ == "__main__":
    app.run(debug=True)
