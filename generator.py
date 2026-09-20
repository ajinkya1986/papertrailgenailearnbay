"""
generator.py
------------
The answer-generation component of the RAG pipeline -- now provider-agnostic.

Two request/response *shapes* are supported:

1. "anthropic" -- Claude's Messages API. `system` is a top-level field;
   the response's `content` is a list of typed blocks.
2. "openai" -- the de-facto standard "chat completions" shape used by
   OpenAI itself, and also by Moonshot AI's Kimi models, DeepSeek, Groq,
   Together AI, Mistral, and most other hosted LLM APIs. `system` is just
   a message with role "system" in the same `messages` list; the response
   is `choices[0].message.content`.

Which provider is used is controlled by the LLM_PROVIDER environment
variable (default: "anthropic"). Each provider reads its API key from its
own environment variable, so you can have all three keys set at once and
just flip LLM_PROVIDER to switch between them without editing any code.

    export LLM_PROVIDER=anthropic   # uses ANTHROPIC_API_KEY
    export LLM_PROVIDER=openai      # uses OPENAI_API_KEY
    export LLM_PROVIDER=kimi        # uses MOONSHOT_API_KEY

To add another OpenAI-compatible provider (DeepSeek, Groq, Together, a
self-hosted vLLM/Ollama server, ...), you only need to add one entry to
the PROVIDERS dict below -- no other code changes are required, because
every OpenAI-shaped provider is handled by the same _call_openai_compatible
function.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import requests

from retriever import RetrievedChunk

MAX_TOKENS = 1000


class GenerationError(Exception):
    """Raised for any failure calling or parsing an LLM API response."""


@dataclass
class GenerationResult:
    answer: str
    n_sources_used: int
    provider: str
    model: str


# ---------------------------------------------------------------------------
# Provider registry. "format" picks which function below builds the request
# and parses the response for that provider. Add a new OpenAI-compatible
# provider by adding one dict here -- e.g. DeepSeek:
#   "deepseek": {
#       "format": "openai",
#       "base_url": "https://api.deepseek.com/chat/completions",
#       "api_key_env": "DEEPSEEK_API_KEY",
#       "default_model": "deepseek-chat",
#   },
# ---------------------------------------------------------------------------
PROVIDERS = {
    "anthropic": {
        "format": "anthropic",
        "base_url": "https://api.anthropic.com/v1/messages",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-6",
    },
    "openai": {
        "format": "openai",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "kimi": {
        # Moonshot AI's Kimi models speak the OpenAI chat-completions format.
        # International endpoint shown; mainland China accounts should use
        # https://api.moonshot.cn/v1/chat/completions instead (same shape,
        # different base URL and account). Moonshot ships new model IDs
        # fairly often (kimi-k2.5, kimi-k2.6, ...) -- check
        # https://platform.moonshot.ai/docs for the current list if this
        # default stops working.
        "format": "openai",
        "base_url": "https://api.moonshot.ai/v1/chat/completions",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "kimi-k2.5",
    },
    "openrouter": {
        # OpenRouter is a router in front of 400+ models from many providers,
        # also using the OpenAI chat-completions shape. The model field is
        # provider-prefixed, e.g. "anthropic/claude-sonnet-4", "openai/gpt-4o",
        # "moonshotai/kimi-k2" -- set LLM_MODEL to pick a specific one; see
        # https://openrouter.ai/models for the current catalog.
        "format": "openai",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini",
    },
}

SYSTEM_PROMPT = (
    "You are a precise research assistant for a retrieval-augmented QA demo. "
    "Never use knowledge outside the provided excerpts."
)


def build_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
    """Identical across providers -- the prompt text itself doesn't depend
    on which API it's ultimately sent through."""
    excerpts = "\n\n".join(
        f'[{i}] ({r.paper_short}, {r.year} — Section "{r.section}")\n{r.text}'
        for i, r in enumerate(retrieved, start=1)
    )
    return (
        "You are answering a question using ONLY the excerpts below, drawn from "
        "three AI research papers. Cite every claim with the excerpt number(s) it "
        'came from, in square brackets, e.g. "[2]" or "[1][4]". If the excerpts do '
        "not fully answer the question, say what is missing rather than guessing. "
        "Write in clear, well-formed prose (not bullet points) unless the question "
        f"specifically asks for a list.\n\nEXCERPTS:\n{excerpts}\n\nQUESTION: {question}"
    )


def _call_anthropic(prompt: str, api_key: str, model: str, base_url: str) -> str:
    response = requests.post(
        base_url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise GenerationError(f"API error {response.status_code}: {response.text[:300]}")
    data = response.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise GenerationError("No text content returned by the model.")
    return "\n".join(text_blocks)


def _call_openai_compatible(prompt: str, api_key: str, model: str, base_url: str) -> str:
    """Handles OpenAI itself, Kimi/Moonshot, and any other provider that
    implements the same 'chat completions' request/response shape."""
    response = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise GenerationError(f"API error {response.status_code}: {response.text[:300]}")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise GenerationError("No choices returned by the model.")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise GenerationError("No text content returned by the model.")
    return content


def generate_answer(
    question: str,
    retrieved: List[RetrievedChunk],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> GenerationResult:
    """
    provider/model let a caller override the environment-variable defaults
    per call (handy for testing, or for a UI that lets the user pick a
    model); normally both are left as None and taken from LLM_PROVIDER /
    each provider's own default_model.
    """
    # Normalized to lowercase so LLM_PROVIDER=OPENROUTER, =OpenRouter, and
    # =openrouter are all treated the same -- previously this was an exact
    # dict-key match, so setting it in any case other than lowercase (or to
    # a provider that didn't exist yet, like "openrouter" before this was
    # added) silently fell through to the "anthropic" default instead of
    # raising a clear error, which is confusing to debug from the outside.
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).strip().lower()
    if provider not in PROVIDERS:
        raise GenerationError(
            f"Unknown LLM_PROVIDER '{provider}'. Valid options: {', '.join(PROVIDERS)}"
        )
    cfg = PROVIDERS[provider]
    model = model or os.environ.get("LLM_MODEL") or cfg["default_model"]

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise GenerationError(
            f"{cfg['api_key_env']} is not set (required for provider '{provider}'). "
            f"Export it before running the app, e.g.:\n"
            f"  export {cfg['api_key_env']}=...\n"
            f"Or set LLM_PROVIDER to a provider whose key you do have."
        )
    if not retrieved:
        raise GenerationError("No retrieved chunks were passed to the generator.")

    prompt = build_prompt(question, retrieved)

    try:
        if cfg["format"] == "anthropic":
            answer = _call_anthropic(prompt, api_key, model, cfg["base_url"])
        else:
            answer = _call_openai_compatible(prompt, api_key, model, cfg["base_url"])
    except requests.exceptions.RequestException as exc:
        raise GenerationError(f"Network error calling {provider}: {exc}") from exc

    return GenerationResult(
        answer=answer, n_sources_used=len(retrieved), provider=provider, model=model
    )
