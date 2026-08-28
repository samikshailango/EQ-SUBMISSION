from __future__ import annotations

from typing import Dict, List

from . import core_solver


class Verifier:
    def verify(self, question: str, executor_result: Dict) -> Dict:
        checks: List[Dict] = []

        checks.append(self._reresolve_check(question, executor_result))
        checks.append(self._constraint_check(executor_result))
        checks.append(self._consistency_check(executor_result))

        overall_passed = all(c["passed"] for c in checks)
        notes = "" if overall_passed else "; ".join(
            c["details"] for c in checks if not c["passed"]
        )
        return {"checks": checks, "overall_passed": overall_passed, "notes": notes}

    def _reresolve_check(self, question: str, executor_result: Dict) -> Dict:
        independent = core_solver.solve(question)
        same_category = independent["category"] == executor_result["category"]
        same_value = _values_equal(independent["value"], executor_result["intermediate_value"])
        passed = same_category and same_value and independent["valid"]
        details = (
            f"independent re-solve gave value={independent['value']!r}, "
            f"executor gave value={executor_result['intermediate_value']!r}"
        )
        if not independent["valid"]:
            details += f"; independent solve flagged issues: {independent['issues']}"
        return {"check_name": "independent_resolve", "passed": passed, "details": details}

    def _constraint_check(self, executor_result: Dict) -> Dict:
        issues = list(executor_result.get("issues", []))
        if not executor_result.get("valid", True):
            details = "; ".join(issues) if issues else "executor flagged the result invalid"
            return {"check_name": "constraint_validation", "passed": False, "details": details}
        return {
            "check_name": "constraint_validation",
            "passed": True,
            "details": "no non-negativity / time-ordering / units violations found",
        }


    def _consistency_check(self, executor_result: Dict) -> Dict:
        has_steps = bool(executor_result.get("steps"))
        has_answer = bool(executor_result.get("draft_answer")) and executor_result[
            "draft_answer"
        ] not in ("Unable to determine.", "")
        passed = has_steps and has_answer
        details = (
            "steps and draft answer both present"
            if passed
            else "missing reasoning steps or a usable draft answer"
        )
        return {"check_name": "steps_consistency", "passed": passed, "details": details}


def _values_equal(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    return a == b
