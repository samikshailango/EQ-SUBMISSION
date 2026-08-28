from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class LLMError(Exception):
    """Raised when the underlying LLM call fails in a way callers should see."""


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        use_mock: Optional[bool] = None,
        provider: Optional[str] = None,
    ):
        self.provider = provider or _detect_provider(use_mock)
        if use_mock:
            self.provider = "mock"

        self.model = model or {
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL,
            "mock": "mock",
        }[self.provider]

        self.use_mock = self.provider == "mock"
        self._client = None
        self._api_key = None

        if self.provider == "anthropic":
            import anthropic 
            self._api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        elif self.provider == "gemini":
            self._api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        """Return raw text completion for a (system, user) prompt pair."""
        if self.use_mock:
            return MockLLM.complete(system, user)
        if self.provider == "gemini":
            return self._complete_gemini(system, user, max_tokens)
        return self._complete_anthropic(system, user, max_tokens)

    def _complete_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:  
            raise LLMError(f"Anthropic call failed: {exc}") from exc

    def _complete_gemini(self, system: str, user: str, max_tokens: int) -> str:
        if not self._api_key:
            raise LLMError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY)."
            )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self._api_key}"
        )
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  
            detail = exc.read().decode("utf-8", errors="ignore")
            raise LLMError(f"Gemini call failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc: 
            raise LLMError(f"Gemini call failed: {exc}") from exc

        try:
            candidates = data["candidates"]
            parts = candidates[0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from exc


def _detect_provider(use_mock: Optional[bool]) -> str:
    if use_mock:
        return "mock"
    if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY") and _anthropic_importable():
        return "anthropic"
    return "mock"


def _anthropic_importable() -> bool:
    try:
        import anthropic 

        return True
    except ImportError:
        return False


class MockLLM:
    """
    Deterministic, offline stand-in for a real LLM.

    It is intentionally simple: it looks at which SYSTEM prompt was passed
    (planner / executor / verifier, detected by a keyword) and returns
    reasonable, deterministic text in the expected shape. The real
    computation for executor/verifier is delegated to agent.core_solver
    (see executor.py / verifier.py) exactly the way a real LLM call would
    be expected to lean on a python tool for arithmetic -- the mock's job
    here is only to stand in for the "plan" phase's free text, since that
    part has no single correct numeric answer to get wrong.
    """

    @staticmethod
    def complete(system: str, user: str) -> str:
        if "PLANNER" in system:
            return MockLLM._mock_plan(user)
        return "{}"

    @staticmethod
    def _mock_plan(user_prompt: str) -> str:
        q = user_prompt.lower()
        if re.search(r"\b\d{1,2}:\d{2}\b", q) and ("slot" in q or "free" in q or "meeting" in q):
            return (
                "1. Parse required duration in minutes.\n"
                "2. Parse each free slot into start/end times.\n"
                "3. Compute duration of each slot in minutes.\n"
                "4. Filter slots whose duration >= required duration.\n"
                "5. Format the list of fitting slots as the answer."
            )
        if re.search(r"\b\d{1,2}:\d{2}\b", q):
            return (
                "1. Parse the relevant times (HH:MM).\n"
                "2. Convert times to minutes since midnight.\n"
                "3. Compute the difference (handle midnight rollover).\n"
                "4. Convert back to hours/minutes.\n"
                "5. Validate the duration is non-negative and format the answer."
            )
        if any(k in q for k in ["times as many", "fewer", "more than", "twice", "as many as"]):
            return (
                "1. Extract each named entity's base quantity or relation.\n"
                "2. Resolve relationships into concrete values.\n"
                "3. Compute any requested sum/total.\n"
                "4. Validate all quantities are non-negative.\n"
                "5. Format the final answer."
            )
        return (
            "1. Extract the numeric quantities and the operation implied.\n"
            "2. Compute the intermediate value(s) using arithmetic.\n"
            "3. Chain any subsequent operations in order.\n"
            "4. Validate the result makes sense (non-negative where relevant).\n"
            "5. Format the final short answer."
        )
