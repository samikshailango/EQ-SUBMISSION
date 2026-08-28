from __future__ import annotations

from .llm_client import LLMClient
from .prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def make_plan(self, question: str, attempt: int = 0) -> str:
        """
        Returns a short numbered-step plan (plain text).
        `attempt` is passed in so a retried call can nudge the model to try
        a different angle (used by the agent's retry loop).
        """
        user_prompt = build_planner_user_prompt(question)
        if attempt > 0:
            user_prompt += (
                f"\n\n(Note: this is retry attempt #{attempt}. The previous attempt's "
                "solution failed verification - consider double-checking parsing of "
                "numbers/times/relationships before computing.)"
            )
        plan = self.llm.complete(PLANNER_SYSTEM_PROMPT, user_prompt)
        return plan.strip()
