"""
smoke_test.py
-------------
Integration test for the actual Flask app (app.py) using Flask's built-in
test client -- this drives real HTTP requests through real Flask routing,
real Jinja rendering, and the real retriever, exactly as a browser would,
without needing a running server or a browser. Generation is monkeypatched
to a canned response so this test needs no ANTHROPIC_API_KEY and makes no
network calls, but everything else in the request/response cycle is real.

Run with: python smoke_test.py
"""
import re

import app as app_module
from generator import GenerationError, GenerationResult

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def fake_generate_ok(question, retrieved):
    return GenerationResult(
        answer='Multi-head attention runs several attention functions in parallel [1]. '
               'This lets the model attend to different representation subspaces at once [1][2].',
        n_sources_used=len(retrieved),
        provider="anthropic",
        model="claude-sonnet-4-6",
    )


def fake_generate_error(question, retrieved):
    raise GenerationError("simulated failure: API error 500: internal error")


client = app_module.app.test_client()

# --- GET / : empty form, no sources yet ---
resp = client.get("/")
check(resp.status_code == 200, "GET / returns 200")
body = resp.get_data(as_text=True)
check("Ask a question about these papers" in body, "form label is present")
check(body.count('class="sample-chip"') == 5, "5 sample-question chips rendered")
check("Ask a question to see which passages were retrieved" in body, "empty-state message shown before any question")

# --- POST / with an empty question ---
resp = client.post("/", data={"question": "   "})
check(resp.status_code == 200, "POST with blank question returns 200 (re-renders form)")
check("Type a question first" in resp.get_data(as_text=True), "blank question shows validation message")

# --- POST / with a real question, generation mocked to succeed ---
app_module.generate_answer = fake_generate_ok
resp = client.post("/", data={"question": "What are the two sub-layers in each encoder layer of the Transformer model?"})
body = resp.get_data(as_text=True)
check(resp.status_code == 200, "POST with a real question returns 200")
check(body.count('class="source-card"') == 8, "8 source cards rendered (TOP_K=8)")
check('id="source-1"' in body, "first source card has id=source-1 for anchor-link targeting")
check('href="#source-1" class="cite"' in body, "citation [1] in the answer becomes a link to #source-1")
check('href="#source-2" class="cite"' in body, "citation [2] in the answer becomes a link to #source-2")
check("Transformer" in body.split('id="source-1"')[1][:200], "top-ranked source for this question is from the Transformer paper")
check("Done. Answer grounded in 8 retrieved passages." in body, "success status message shown")

# --- POST / , generation mocked to fail ---
app_module.generate_answer = fake_generate_error
resp = client.post("/", data={"question": "What are the two sub-layers in each encoder layer of the Transformer model?"})
body = resp.get_data(as_text=True)
check(resp.status_code == 200, "POST that fails generation still returns 200 (graceful degradation)")
check("Generation error" in body, "generation failure surfaces a Generation error status")
check(body.count('class="source-card"') == 8, "sources still shown even though generation failed")

# --- POST / with a question sharing no vocabulary with the corpus ---
app_module.generate_answer = fake_generate_ok
resp = client.post("/", data={"question": "zzzzz qqqqq wwwqxyz plonk"})
body = resp.get_data(as_text=True)
check("no retrieved passage actually matched" in body,
      "zero-overlap query is flagged, and the warning is not overwritten by the generic 'Done' message")

# --- XSS / HTML-injection guard: a question containing markup must not be rendered as HTML ---
resp = client.post("/", data={"question": "<script>alert(1)</script> attention"})
body = resp.get_data(as_text=True)
check("<script>alert(1)</script>" not in body, "raw <script> tag from user input is not reflected unescaped into the page")

print("=" * 70)
print("ALL SMOKE TESTS PASSED" if failures == 0 else f"{failures} SMOKE TEST(S) FAILED")
raise SystemExit(0 if failures == 0 else 1)
