from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twice": 2, "thrice": 3,
}

TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time_to_minutes(h: int, m: int) -> int:
    return h * 60 + m


def _minutes_to_hm(total_minutes: int) -> Tuple[int, int]:
    h, m = divmod(total_minutes, 60)
    return h, m


def _fmt_hm(total_minutes: int) -> str:
    h, m = _minutes_to_hm(total_minutes)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m or not parts:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    return " ".join(parts)


def _find_times(text: str) -> List[Tuple[int, int]]:
    return [(int(h), int(m)) for h, m in TIME_RE.findall(text)]


def _word_to_number(token: str) -> Optional[float]:
    token = token.strip().lower()
    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token])
    m = re.match(r"^(\d+(?:\.\d+)?)$", token)
    if m:
        return float(m.group(1))
    return None



def solve_time_difference(question: str) -> Dict:
    q = question.lower()
    times = _find_times(question)
    steps: List[str] = []
    issues: List[str] = []

    duration_match = re.search(
        r"lasts?\s+(?:(\d+)\s*hours?)?\s*(?:and\s*)?(?:(\d+)\s*minutes?)?", q
    )

    if len(times) >= 2:
        (h1, m1), (h2, m2) = times[0], times[1]
        start = _time_to_minutes(h1, m1)
        end = _time_to_minutes(h2, m2)
        steps.append(f"{h1:02d}:{m1:02d} = {start} min since midnight")
        steps.append(f"{h2:02d}:{m2:02d} = {end} min since midnight")
        diff = end - start
        rollover = False
        if diff < 0:
            diff += 24 * 60
            rollover = True
            steps.append("Arrival is on the next day (midnight rollover): +1440 min")
        steps.append(f"duration = {diff} min")
        if diff < 0:
            issues.append("Computed negative duration.")
        answer = _fmt_hm(diff)
        return {
            "category": "time_difference",
            "extracted": {
                "start": f"{h1:02d}:{m1:02d}",
                "end": f"{h2:02d}:{m2:02d}",
                "rollover": rollover,
            },
            "steps": steps,
            "value": diff,
            "answer": answer,
            "valid": len(issues) == 0,
            "issues": issues,
        }

    if len(times) == 1 and (
        "lasts" in q or "duration" in q or re.search(r"\d+\s*hours?|\d+\s*minutes?", q)
    ):
        (h1, m1) = times[0]
        start = _time_to_minutes(h1, m1)
        hours = int(duration_match.group(1)) if duration_match and duration_match.group(1) else 0
        minutes = int(duration_match.group(2)) if duration_match and duration_match.group(2) else 0
        if hours == 0 and minutes == 0:
            # try alternate phrasing "2 hours 10 minutes" anywhere in text
            alt = re.search(r"(\d+)\s*hours?\s*(\d+)?\s*minutes?", q)
            if alt:
                hours = int(alt.group(1))
                minutes = int(alt.group(2)) if alt.group(2) else 0
        add = hours * 60 + minutes
        end = start + add
        steps.append(f"start = {h1:02d}:{m1:02d} = {start} min")
        steps.append(f"add duration {hours}h {minutes}m = {add} min")
        end_mod = end % (24 * 60)
        eh, em = _minutes_to_hm(end_mod)
        steps.append(f"end = {start} + {add} = {end_mod} min = {eh:02d}:{em:02d}")
        return {
            "category": "time_difference",
            "extracted": {"start": f"{h1:02d}:{m1:02d}", "duration_minutes": add},
            "steps": steps,
            "value": end_mod,
            "answer": f"{eh:02d}:{em:02d}",
            "valid": True,
            "issues": [],
        }

    issues.append("Could not find two clear times (or one time + duration) in the question.")
    return {
        "category": "time_difference",
        "extracted": {"times_found": times},
        "steps": steps,
        "value": None,
        "answer": "Unable to determine.",
        "valid": False,
        "issues": issues,
    }


_BASE_RE = re.compile(
    r"([A-Z][a-zA-Z]*)\s+has\s+(\d+)\s+([a-zA-Z]+)", re.IGNORECASE
)
_MULT_RE = re.compile(
    r"([A-Z][a-zA-Z]*)\s+has\s+(twice|thrice|\d+\s*times?)\s+as\s+many\s+[a-zA-Z]*\s*as\s+([A-Z][a-zA-Z]*)",
    re.IGNORECASE,
)
_FEWER_RE = re.compile(
    r"([A-Z][a-zA-Z]*)\s+has\s+(\d+)\s+fewer\s+[a-zA-Z]*\s*than\s+([A-Z][a-zA-Z]*)",
    re.IGNORECASE,
)
_MORE_RE = re.compile(
    r"([A-Z][a-zA-Z]*)\s+has\s+(\d+)\s+more\s+[a-zA-Z]*\s*than\s+([A-Z][a-zA-Z]*)",
    re.IGNORECASE,
)
_WHO_HAS_RE = re.compile(r"([A-Z][a-zA-Z]*)\s+who\s+has\s+(\d+)", re.IGNORECASE)
_ALSO_HAS_RE = re.compile(
    r"and\s+([A-Z][a-zA-Z]*)\s+has\s+(\d+)\s+([a-zA-Z]+)", re.IGNORECASE
)
_TOTAL_HINT_RE = re.compile(r"total|altogether|in\s+all", re.IGNORECASE)
_QUESTION_SUBJECT_RE = re.compile(
    r"how\s+many\s+[a-zA-Z]*\s*does\s+([A-Z][a-zA-Z]*)\s+have", re.IGNORECASE
)


_DUAL_QTY_RE_A = re.compile(
    r"(\d+)\s+(\w+)\s+(\w+)\s+and\s+(twice|thrice|\d+\s*times?)\s+as\s+many\s+(\w+)\s+\3\s+as\s+\2",
    re.IGNORECASE,
)
_DUAL_QTY_RE_B = re.compile(
    r"(\d+)\s+(\w+)\s+and\s+(twice|thrice|\d+\s*times?)\s+as\s+many\s+(\w+)\s+as\s+\2",
    re.IGNORECASE,
)


def _match_dual_quantity(question: str) -> Optional[Tuple[float, str, float, str]]:
    """
    Detects same-subject dual-quantity phrasing such as:
      "3 red apples and twice as many green apples as red"
      "5 dogs and 3 times as many cats as dogs"
    Returns (base_amount, base_label, other_amount, other_label) or None.
    """
    m = _DUAL_QTY_RE_A.search(question)
    if m:
        base_amount = float(m.group(1))
        base_label = m.group(2)
        mult = _mult_to_number(m.group(4))
        other_label = m.group(5)
        return base_amount, base_label, mult * base_amount, other_label

    m = _DUAL_QTY_RE_B.search(question)
    if m:
        base_amount = float(m.group(1))
        base_label = m.group(2)
        mult = _mult_to_number(m.group(3))
        other_label = m.group(4)
        return base_amount, base_label, mult * base_amount, other_label

    return None


def _mult_to_number(tok: str) -> float:
    tok = tok.strip().lower()
    if tok in ("twice",):
        return 2.0
    if tok in ("thrice",):
        return 3.0
    m = re.match(r"(\d+)", tok)
    return float(m.group(1)) if m else 1.0


_RELATIONAL_STOPWORDS = {"fewer", "more", "less", "times", "as", "than"}


def solve_counting(question: str) -> Dict:
    steps: List[str] = []
    issues: List[str] = []
    values: Dict[str, float] = {}
    deps: Dict[str, Tuple[str, float, str]] = {} 
    dual_qty = _match_dual_quantity(question)

    for name, amount, noun in _BASE_RE.findall(question):
        if noun.lower() in _RELATIONAL_STOPWORDS:
            continue
        values.setdefault(name, float(amount))
    for name, amount, noun in _ALSO_HAS_RE.findall(question):
        if noun.lower() in _RELATIONAL_STOPWORDS:
            continue
        values.setdefault(name, float(amount))
    for name, amount in _WHO_HAS_RE.findall(question):
        values.setdefault(name, float(amount))

    for name, mult_tok, ref in _MULT_RE.findall(question):
        deps[name] = ("mul", _mult_to_number(mult_tok), ref)
    for name, amount, ref in _FEWER_RE.findall(question):
        deps[name] = ("sub", float(amount), ref)
    for name, amount, ref in _MORE_RE.findall(question):
        deps[name] = ("add", float(amount), ref)

    if dual_qty is not None:
        base_amount, base_label, other_amount, other_label = dual_qty
        steps.append(f"{base_label} = {base_amount:g}")
        steps.append(f"{other_label} = {other_amount:g} (relative to {base_label})")
        total = base_amount + other_amount
        steps.append(f"total = {base_amount:g} + {other_amount:g} = {total:g}")
        return {
            "category": "counting",
            "extracted": {base_label: base_amount, other_label: other_amount},
            "steps": steps,
            "value": total,
            "answer": f"{total:g}",
            "valid": total >= 0,
            "issues": [] if total >= 0 else ["Total would be negative."],
        }

    resolved: Dict[str, float] = dict(values)

    def resolve(name: str, _seen=()) -> Optional[float]:
        if name in resolved:
            return resolved[name]
        if name in _seen:
            issues.append(f"Circular relationship detected for '{name}'.")
            return None
        if name not in deps:
            return None
        op, factor, ref = deps[name]
        ref_val = resolve(ref, _seen + (name,))
        if ref_val is None:
            return None
        if op == "mul":
            val = factor * ref_val
            steps.append(f"{name} = {factor:g} x {ref} ({ref_val:g}) = {val:g}")
        elif op == "sub":
            val = ref_val - factor
            steps.append(f"{name} = {ref} ({ref_val:g}) - {factor:g} = {val:g}")
        else:  # add
            val = ref_val + factor
            steps.append(f"{name} = {ref} ({ref_val:g}) + {factor:g} = {val:g}")
        resolved[name] = val
        return val

    for name in list(deps.keys()):
        resolve(name)

    if not resolved:
        issues.append("Could not extract any named quantities from the question.")
        return {
            "category": "counting",
            "extracted": {},
            "steps": steps,
            "value": None,
            "answer": "Unable to determine.",
            "valid": False,
            "issues": issues,
        }

    for name, val in resolved.items():
        if val < 0:
            issues.append(f"{name} would be negative ({val:g}), which is impossible.")

    subj_match = _QUESTION_SUBJECT_RE.search(question)
    if subj_match and subj_match.group(1) in resolved and not _TOTAL_HINT_RE.search(question):
        subject = subj_match.group(1)
        value = resolved[subject]
        steps.append(f"answer = {subject} = {value:g}")
        answer = f"{value:g}"
    else:
        total = sum(resolved.values())
        steps.append(f"total = sum({', '.join(f'{k}={v:g}' for k, v in resolved.items())}) = {total:g}")
        value = total
        answer = f"{total:g}"

    return {
        "category": "counting",
        "extracted": {k: v for k, v in resolved.items()},
        "steps": steps,
        "value": value,
        "answer": answer,
        "valid": len(issues) == 0,
        "issues": issues,
    }

_SLOT_RE = re.compile(r"(\d{1,2}:\d{2})\s*[-\u2013\u2014]\s*(\d{1,2}:\d{2})")
_NEED_RE = re.compile(r"(\d+)\s*minutes?")


def solve_scheduling(question: str) -> Dict:
    steps: List[str] = []
    issues: List[str] = []

    need_match = _NEED_RE.search(question)
    if not need_match:
        issues.append("Could not find the required meeting duration.")
        return {
            "category": "scheduling",
            "extracted": {},
            "steps": steps,
            "value": None,
            "answer": "Unable to determine.",
            "valid": False,
            "issues": issues,
        }
    required = int(need_match.group(1))
    steps.append(f"required duration = {required} min")

    slots = _SLOT_RE.findall(question)
    if not slots:
        issues.append("Could not find any free time slots in the question.")
        return {
            "category": "scheduling",
            "extracted": {"required_minutes": required},
            "steps": steps,
            "value": None,
            "answer": "Unable to determine.",
            "valid": False,
            "issues": issues,
        }

    fitting = []
    for s, e in slots:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        length = _time_to_minutes(eh, em) - _time_to_minutes(sh, sm)
        if length < 0:
            issues.append(f"Slot {s}-{e} has negative length.")
            continue
        fits = length >= required
        steps.append(f"{s}-{e} = {length} min ({'fits' if fits else 'too short'})")
        if fits:
            fitting.append(f"{s}-{e}")

    if fitting:
        answer = ", ".join(fitting)
    else:
        answer = "No slot is long enough."

    return {
        "category": "scheduling",
        "extracted": {"required_minutes": required, "slots": [f"{a}-{b}" for a, b in slots]},
        "steps": steps,
        "value": fitting,
        "answer": answer,
        "valid": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# 4. GENERIC / MULTI-STEP ARITHMETIC
# ---------------------------------------------------------------------------

_DIFF_BETWEEN_RE = re.compile(
    r"difference\s+between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_CHAIN_RE = re.compile(
    r"then\s+(add|subtract|multiply(?:\s+by)?|divide(?:\s+by)?)\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_SIMPLE_OP_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(plus|minus|multiplied\s+by|divided\s+by|times)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_HAD_SOLD_RE = re.compile(
    r"had\s+(\d+)\s*[a-zA-Z]*.*?(?:sold|used|spent|gave\s+away)\s+(\d+|\bsome\b)", re.IGNORECASE
)
_NEGATIVE_RESULT_RE = re.compile(r"now\s+has\s+(-\d+)", re.IGNORECASE)


def solve_arithmetic(question: str) -> Dict:
    steps: List[str] = []
    issues: List[str] = []

    neg_match = _NEGATIVE_RESULT_RE.search(question)
    had_sold = _HAD_SOLD_RE.search(question)
    if neg_match:
        result = int(neg_match.group(1))
        issues.append(
            f"The question states a resulting count of {result}, which is negative and impossible "
            "for a physical item count. The scenario is inconsistent."
        )
        extracted = {}
        if had_sold:
            extracted = {"had": had_sold.group(1), "sold": had_sold.group(2)}
        return {
            "category": "arithmetic",
            "extracted": extracted,
            "steps": steps,
            "value": None,
            "answer": "Cannot be determined - the scenario is inconsistent (negative inventory).",
            "valid": False,
            "issues": issues,
        }

    diff_match = _DIFF_BETWEEN_RE.search(question)
    if diff_match:
        a, b = float(diff_match.group(1)), float(diff_match.group(2))
        value = abs(a - b)
        steps.append(f"difference between {a:g} and {b:g} = {value:g}")
        value = _apply_chain(question, value, steps)
        return {
            "category": "arithmetic",
            "extracted": {"a": a, "b": b},
            "steps": steps,
            "value": value,
            "answer": _fmt_number(value),
            "valid": value is not None and value >= 0 if isinstance(value, (int, float)) else True,
            "issues": issues,
        }

    simple_match = _SIMPLE_OP_RE.search(question)
    if simple_match:
        a = float(simple_match.group(1))
        op = simple_match.group(2).lower()
        b = float(simple_match.group(3))
        value = _apply_op(a, op, b)
        steps.append(f"{a:g} {op} {b:g} = {value:g}")
        value = _apply_chain(question, value, steps)
        return {
            "category": "arithmetic",
            "extracted": {"a": a, "op": op, "b": b},
            "steps": steps,
            "value": value,
            "answer": _fmt_number(value),
            "valid": True,
            "issues": issues,
        }

    if had_sold and had_sold.group(2).lower() != "some":
        had = int(had_sold.group(1))
        sold = int(had_sold.group(2))
        value = had - sold
        steps.append(f"{had} - {sold} = {value}")
        issues_local = []
        if value < 0:
            issues_local.append(f"Remaining count {value} is negative.")
        return {
            "category": "arithmetic",
            "extracted": {"had": had, "sold": sold},
            "steps": steps,
            "value": value,
            "answer": _fmt_number(value),
            "valid": len(issues_local) == 0,
            "issues": issues_local,
        }
    if had_sold and had_sold.group(2).lower() == "some":
        issues.append("The quantity sold is not specified ('some'), so the result cannot be computed.")
        return {
            "category": "arithmetic",
            "extracted": {"had": had_sold.group(1), "sold": "unspecified"},
            "steps": steps,
            "value": None,
            "answer": "Cannot be determined - amount sold is unspecified.",
            "valid": False,
            "issues": issues,
        }

    issues.append("Could not identify a clear arithmetic operation in the question.")
    return {
        "category": "arithmetic",
        "extracted": {},
        "steps": steps,
        "value": None,
        "answer": "Unable to determine.",
        "valid": False,
        "issues": issues,
    }


def _apply_op(a: float, op: str, b: float) -> float:
    op = op.lower()
    if op == "plus":
        return a + b
    if op == "minus":
        return a - b
    if op in ("multiplied by", "times"):
        return a * b
    if op == "divided by":
        return a / b
    raise ValueError(f"Unknown operator: {op}")


def _apply_chain(question: str, value: float, steps: List[str]) -> float:
    for op_word, num_str in _CHAIN_RE.findall(question):
        num = float(num_str)
        op_word = op_word.lower()
        if op_word == "add":
            value = value + num
        elif op_word == "subtract":
            value = value - num
        elif op_word.startswith("multiply"):
            value = value * num
        elif op_word.startswith("divide"):
            value = value / num
        steps.append(f"then {op_word} {num:g} -> {value:g}")
    return value


def _fmt_number(value) -> str:
    if value is None:
        return "Unable to determine."
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)



def detect_category(question: str) -> str:
    q = question.lower()
    times = _find_times(question)
    if ("slot" in q or "free" in q) and ("meeting" in q or "minutes" in q):
        return "scheduling"
    if len(times) >= 1 and ("leaves" in q or "arrives" in q or "starts" in q or "lasts" in q or "journey" in q or "duration" in q):
        return "time_difference"
    if any(k in q for k in ["times as many", "as many as", "fewer than", "more than", "twice", "thrice"]):
        return "counting"
    return "arithmetic"


def solve(question: str) -> Dict:
    category = detect_category(question)
    if category == "scheduling":
        return solve_scheduling(question)
    if category == "time_difference":
        return solve_time_difference(question)
    if category == "counting":
        return solve_counting(question)
    return solve_arithmetic(question)
