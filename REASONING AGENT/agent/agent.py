from __future__ import annotations

from typing import Dict, Optional

from .executor import Executor
from .llm_client import LLMClient
from .planner import Planner
from .verifier import Verifier

MAX_RETRIES = 2


class ReasoningAgent:
    def __init__(
        self,
        use_mock_llm: Optional[bool] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.llm = LLMClient(model=model, use_mock=use_mock_llm, provider=provider)
        self.planner = Planner(self.llm)
        self.executor = Executor(self.llm)
        self.verifier = Verifier()

    def solve(self, question: str, max_retries: int = MAX_RETRIES) -> Dict:
        if not question or not question.strip():
            return _build_response(
                answer="",
                status="failed",
                reasoning="The question was empty.",
                plan="",
                checks=[],
                retries=0,
            )

        retries = 0
        plan = ""
        exec_result: Dict = {}
        verify_result: Dict = {"checks": [], "overall_passed": False, "notes": ""}

        while True:
            plan = self.planner.make_plan(question, attempt=retries)
            exec_result = self.executor.execute(question, plan)
            verify_result = self.verifier.verify(question, exec_result)

            if verify_result["overall_passed"]:
                return _build_response(
                    answer=exec_result["draft_answer"],
                    status="success",
                    reasoning=_summarize_reasoning(exec_result, verify_result, retries),
                    plan=plan,
                    checks=verify_result["checks"],
                    retries=retries,
                )

            if retries >= max_retries:
                return _build_response(
                    answer=exec_result.get("draft_answer", "Unable to determine."),
                    status="failed",
                    reasoning=_summarize_failure(exec_result, verify_result, retries),
                    plan=plan,
                    checks=verify_result["checks"],
                    retries=retries,
                )

            retries += 1


def _summarize_reasoning(exec_result: Dict, verify_result: Dict, retries: int) -> str:
    category = exec_result.get("category", "problem")
    base = f"Solved as a {category.replace('_', ' ')} problem and verified against an independent re-check."
    if retries:
        base += f" Succeeded after {retries} retry(ies)."
    return base


def _summarize_failure(exec_result: Dict, verify_result: Dict, retries: int) -> str:
    issues = exec_result.get("issues") or [verify_result.get("notes") or "verification failed"]
    return (
        f"Could not produce a verified answer after {retries} retry(ies). "
        f"Reason: {'; '.join(issues)}"
    )


def _build_response(
    answer: str,
    status: str,
    reasoning: str,
    plan: str,
    checks,
    retries: int,
) -> Dict:
    return {
        "answer": answer,
        "status": status,
        "reasoning_visible_to_user": reasoning,
        "metadata": {
            "plan": plan,
            "checks": checks,
            "retries": retries,
        },
    }


def solve(question: str, use_mock_llm: Optional[bool] = None, provider: Optional[str] = None) -> Dict:
    """Module-level convenience function: `solve(question) -> dict`."""
    return ReasoningAgent(use_mock_llm=use_mock_llm, provider=provider).solve(question)
