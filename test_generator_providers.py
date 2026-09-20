import os
from unittest.mock import patch, MagicMock

import generator
from retriever import Retriever

failures = 0


def check(cond, msg):
    global failures
    print(("PASS" if cond else "FAIL") + ":", msg)
    if not cond:
        failures += 1


retrieved = Retriever().retrieve("What is multi-head attention?", k=3)

# --- Unknown provider ---
os.environ["LLM_PROVIDER"] = "not-a-real-provider"
try:
    generator.generate_answer("q", retrieved)
    check(False, "unknown provider should raise")
except generator.GenerationError as e:
    check("Unknown LLM_PROVIDER" in str(e), f"unknown provider raises a clear error: {e}")

# --- Missing API key, per provider ---
for provider, env_var in [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"), ("kimi", "MOONSHOT_API_KEY")]:
    os.environ.pop(env_var, None)
    try:
        generator.generate_answer("q", retrieved, provider=provider)
        check(False, f"{provider} without a key should raise")
    except generator.GenerationError as e:
        check(env_var in str(e), f"{provider} missing key names {env_var} in the error: {e.args[0].splitlines()[0]}")

# --- OpenAI-compatible request shape (mocked, since api.openai.com/api.moonshot.ai aren't reachable here) ---
os.environ["OPENAI_API_KEY"] = "sk-fake-for-shape-test"
os.environ["MOONSHOT_API_KEY"] = "sk-fake-for-shape-test"

mock_response = MagicMock()
mock_response.status_code = 200
mock_response.json.return_value = {
    "choices": [{"message": {"content": "Mocked answer citing [1]."}}]
}

with patch("generator.requests.post", return_value=mock_response) as mock_post:
    result = generator.generate_answer("What is attention?", retrieved, provider="openai")
    check(result.answer == "Mocked answer citing [1].", "openai provider returns parsed content")
    check(result.provider == "openai" and result.model == "gpt-4o-mini", "result records provider/model used")
    call_kwargs = mock_post.call_args.kwargs
    check(call_kwargs["json"]["messages"][0] == {"role": "system", "content": generator.SYSTEM_PROMPT},
          "openai-shaped request puts system prompt as a messages[0] with role=system")
    check(call_kwargs["headers"]["Authorization"] == "Bearer sk-fake-for-shape-test",
          "openai-shaped request uses Bearer auth header")
    check(mock_post.call_args.args[0] == "https://api.openai.com/v1/chat/completions",
          "openai provider hits the correct base URL")

with patch("generator.requests.post", return_value=mock_response) as mock_post:
    result = generator.generate_answer("What is attention?", retrieved, provider="kimi")
    check(result.model == "kimi-k2.5", "kimi provider defaults to kimi-k2.5")
    check(mock_post.call_args.args[0] == "https://api.moonshot.ai/v1/chat/completions",
          "kimi provider hits Moonshot's base URL")

# --- Custom model override ---
with patch("generator.requests.post", return_value=mock_response) as mock_post:
    result = generator.generate_answer("q", retrieved, provider="kimi", model="kimi-k2.6")
    check(result.model == "kimi-k2.6", "explicit model= argument overrides the provider default")
    check(mock_post.call_args.kwargs["json"]["model"] == "kimi-k2.6", "overridden model is what's actually sent")

# --- OpenAI-compatible error path: non-200 status ---
mock_error_response = MagicMock()
mock_error_response.status_code = 401
mock_error_response.text = '{"error": {"message": "Incorrect API key"}}'
with patch("generator.requests.post", return_value=mock_error_response):
    try:
        generator.generate_answer("q", retrieved, provider="openai")
        check(False, "a 401 from OpenAI should raise")
    except generator.GenerationError as e:
        check("401" in str(e), f"OpenAI 401 surfaces as a GenerationError: {e}")

# --- Anthropic path still live-tested against the real endpoint (invalid key -> expect 401) ---
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-invalid-for-connectivity-test"
try:
    generator.generate_answer("q", retrieved, provider="anthropic")
    check(False, "invalid anthropic key should raise")
except generator.GenerationError as e:
    check("401" in str(e), f"anthropic path still reaches the real API and gets a structured 401: {e}")

print("=" * 70)
print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
raise SystemExit(0 if failures == 0 else 1)
