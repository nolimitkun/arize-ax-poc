"""The groundedness judgement, in one place.

Both poc/04 (batch evals over production spans) and poc/08 (the v1-vs-v2
experiment) have to grade groundedness, and they have to grade it the *same
way* -- otherwise the experiment cannot confirm or deny what production
measurement found. They previously disagreed badly: poc/04 ran an LLM judge over
the retrieved context and scored v1 at 0.46, while poc/08 used a keyword
heuristic that returned `not_applicable` for every question whose
`expected_behavior` wasn't `refuse_no_context` -- so v1 scored 1.00 and no prompt
change could ever move the number.

The template lives here so there is one definition. poc/04 feeds it to Phoenix
Evals (the path the AX docs prescribe for batch work); poc/08 calls
`judge_groundedness` per row, because an experiment evaluator grades one output
at a time.
"""

from __future__ import annotations

import re

import openai

from .config import THINKING_OFF, Settings

GROUNDEDNESS_TEMPLATE = """\
You are grading a customer-support answer for GROUNDEDNESS.

The assistant may only assert things supported by the retrieved documentation.
If the documentation does not cover the question, the correct behaviour is to
say so plainly -- NOT to state a plausible-sounding policy.

[Question]
{question}

[Retrieved documentation -- this is the ONLY source the assistant was given]
{retrieved_context}

[Assistant answer]
{answer}

Answer "grounded" if every factual claim about Nimbus is supported by the
retrieved documentation, or if the assistant correctly said the documentation
does not cover it.

Answer "hallucinated" if the assistant stated any specific policy, number,
timeframe, or entitlement that the retrieved documentation does not contain --
even if it sounds reasonable. Inventing a refund window is hallucination.
"""

CHOICES: dict[str, float] = {"grounded": 1.0, "hallucinated": 0.0}

CHOICE_DESCRIPTIONS: dict[str, tuple[float, str]] = {
    "grounded": (
        1.0,
        "Every factual claim about Nimbus is supported by the retrieved "
        "documentation, or the assistant correctly said the documentation does "
        "not cover the question.",
    ),
    "hallucinated": (
        0.0,
        "The assistant stated a specific policy, number, timeframe or "
        "entitlement that the retrieved documentation does not contain, however "
        "plausible it sounds.",
    ),
}

_INSTRUCTION = (
    "Reply with exactly one word on the first line -- `grounded` or "
    "`hallucinated` -- then one short sentence of justification on the next line."
)

# Words a judge sometimes prefixes its verdict with. Dropped before looking for
# the label so `Verdict: grounded` still parses.
_FILLER = frozenset({"verdict", "label", "answer", "assessment", "the", "is", "a", "it"})


def parse_verdict(text: str) -> tuple[str, float, str]:
    """Map raw judge text to (label, score, explanation).

    Exact token, never substring. `ungrounded` *contains* `grounded`, so a
    substring test scores a hallucination as a pass -- and a judge that
    volunteers that word is exactly the case worth catching, not silently
    passing.

    Anything that isn't one of the two labels returns `error` and a NaN score
    rather than defaulting. A default would inflate both variants of an
    experiment by the same amount, which hides itself: the means move together,
    the delta looks clean, and nothing signals that a fraction of the rows were
    never actually graded. NaN instead drops the row from the mean and out of
    the paired test, so the sample size falls visibly.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    first = lines[0].lower() if lines else ""
    tokens = [t for t in re.findall(r"[a-z]+", first) if t not in _FILLER]
    label = tokens[0] if tokens else ""
    if label not in CHOICES:
        return (
            "error",
            float("nan"),
            f"Unparseable judge verdict: {text.strip()[:200]!r}",
        )
    explanation = " ".join(line.strip() for line in lines[1:]).strip() or first
    return label, CHOICES[label], explanation


def judge_groundedness(
    question: str,
    answer: str,
    retrieved_context: str,
    *,
    settings: Settings,
    model: str = "deepseek-v4-pro",
) -> tuple[str, float, str]:
    """Grade one answer. Returns (label, score, explanation).

    Thinking is disabled: it costs latency an experiment runs N times over, and
    a two-way classification with the context in front of it does not need it.
    Text output rather than structured output, for the same reason poc/05's
    online judge uses text -- V4 rejects json_schema outright.
    """
    if not answer.strip():
        return "hallucinated", 0.0, "No answer produced."

    prompt = GROUNDEDNESS_TEMPLATE.format(
        question=question,
        retrieved_context=retrieved_context or "(no matching documentation found)",
        answer=answer,
    )
    client = openai.OpenAI(
        api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url
    )
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            extra_body=THINKING_OFF,
        )
    except Exception as exc:  # noqa: BLE001 - one bad row shouldn't void the run
        return "error", float("nan"), f"Judge call failed: {type(exc).__name__}: {exc}"

    return parse_verdict(response.choices[0].message.content or "")
