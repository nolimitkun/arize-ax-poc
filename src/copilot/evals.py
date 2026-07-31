"""The judgements the POC makes about its own agent, in one place.

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

# The score columns step 04 writes, and the failure each one represents. One
# definition, because "did this turn fail?" is asked in three places -- poc/04b
# deciding whether a bad session was silent, poc/07 selecting dataset rows, and
# poc/03's heuristics, which answer a deliberately narrower question. A score
# below 1.0 is a failure for all of them; NaN means never graded, which is not
# evidence of anything.
TURN_FAILURE_EVALS: dict[str, str] = {
    "eval.groundedness.score": "hallucination",
    "eval.conciseness.score": "verbosity",
    "eval.tool_selection.score": "wrong_tool",
    "eval.escalation_appropriate.score": "missing_escalation",
}

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


def parse_verdict(text: str, choices: dict[str, float] | None = None) -> tuple[str, float, str]:
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
    scale = CHOICES if choices is None else choices
    lines = [line for line in text.strip().splitlines() if line.strip()]
    first = lines[0].lower() if lines else ""
    # Letters only. No label in use here contains an underscore, and including
    # one in the character class makes `__hallucinated__` -- markdown emphasis a
    # judge emits unprompted -- tokenise as a single unknown word and score as
    # an error instead of a verdict.
    tokens = [t for t in re.findall(r"[a-z]+", first) if t not in _FILLER]
    label = tokens[0] if tokens else ""
    if label not in scale:
        return (
            "error",
            float("nan"),
            f"Unparseable judge verdict: {text.strip()[:200]!r}",
        )
    explanation = " ".join(line.strip() for line in lines[1:]).strip() or first
    return label, scale[label], explanation


def escape_braces(text: str) -> str:
    """Make arbitrary text safe to embed *inside* a `str.format` template.

    Only matters for the aligned judge in poc/06b, which splices real answers
    into the template as worked examples rather than passing them as values. An
    answer containing `{` -- a JSON snippet, a code block -- would otherwise
    make `.format()` raise KeyError on every subsequent grading call.
    """
    return text.replace("{", "{{").replace("}", "}}")


ALIGNMENT_PREAMBLE = """\
Here are previously reviewed cases with the verdict a human reviewer gave.
Match their standard -- especially where the correct verdict is not the
obvious one.
"""

_ANCHOR = "[Question]"


def build_aligned_template(
    worked_examples: list[dict[str, str]],
    base: str = GROUNDEDNESS_TEMPLATE,
    *,
    escape: bool = True,
) -> str:
    """The base template plus human-labelled worked examples.

    This is the align-evals loop in one function: the examples are the cases
    where the judge and a human disagreed, so the template is being corrected
    exactly where it is known to be wrong.

    The examples are spliced in *before* the case under grading rather than
    appended, because a judge that reads the case first and the standard second
    has already decided.

    `escape` doubles the braces in the spliced-in text, which is required for
    the local judge -- these strings become part of a `str.format` template, and
    a real answer containing JSON would otherwise raise KeyError on every later
    call. It must be off when building a template for AX's hosted evaluator:
    that one substitutes `{input.value}` itself and is not Python formatting, so
    doubled braces would render as a literal `{{` in the prompt the judge reads.
    """
    if not worked_examples:
        return base

    def clean(text: str) -> str:
        return escape_braces(text) if escape else text

    blocks = []
    for i, ex in enumerate(worked_examples, 1):
        blocks.append(
            f"--- Reviewed case {i} ---\n"
            f"Question: {clean(ex['question'])}\n"
            f"Documentation retrieved: {clean(ex['context']) or '(none)'}\n"
            f"Answer: {clean(ex['answer'])}\n"
            f"Human verdict: {ex['label']} -- {clean(ex['reason'])}\n"
        )
    section = ALIGNMENT_PREAMBLE + "\n" + "\n".join(blocks) + "\nNow grade this new case.\n"

    head, sep, tail = base.partition(_ANCHOR)
    if not sep:  # template restructured; degrade to prepending rather than failing
        return section + "\n" + base
    return head + section + "\n" + sep + tail


def _ask(
    prompt: str,
    *,
    instruction: str,
    settings: Settings,
    model: str,
    max_tokens: int = 256,
) -> str | None:
    """One text-mode judge call. None on failure, so callers score it NaN.

    Thinking is disabled: it costs latency an experiment runs N times over, and
    a small classification with the evidence in front of it does not need it.
    Text output rather than structured output, for the same reason poc/05's
    online judge uses text -- V4 rejects json_schema outright.
    """
    client = openai.OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            extra_body=THINKING_OFF,
        )
    except Exception:  # noqa: BLE001 - one bad row shouldn't void the run
        return None
    return response.choices[0].message.content or ""


def judge_groundedness(
    question: str,
    answer: str,
    retrieved_context: str,
    *,
    settings: Settings,
    model: str = "deepseek-v4-pro",
    template: str = GROUNDEDNESS_TEMPLATE,
) -> tuple[str, float, str]:
    """Grade one answer. Returns (label, score, explanation).

    `template` is overridable so poc/06b can grade the *same* rows with the
    aligned template and the original one, changing exactly one variable. It
    must accept the same three placeholders.
    """
    if not answer.strip():
        return "hallucinated", 0.0, "No answer produced."

    prompt = template.format(
        question=question,
        retrieved_context=retrieved_context or "(no matching documentation found)",
        answer=answer,
    )
    text = _ask(prompt, instruction=_INSTRUCTION, settings=settings, model=model)
    if text is None:
        return "error", float("nan"), "Judge call failed."
    return parse_verdict(text)


# --------------------------------------------------------------------------
# Session-level judgement (poc/04b)
#
# Every other evaluator here grades a single turn. A support conversation can
# be made of turns that are each individually fine and still fail as a whole --
# the user asks three times, gets three grounded answers, and leaves without
# what they came for. That is only visible with the transcript in view, which
# is what the Sessions view in AX groups for you.
# --------------------------------------------------------------------------

SESSION_TEMPLATE = """\
You are grading a complete customer-support conversation for OUTCOME.

Judge the conversation as a whole, from the customer's point of view. Do not
grade individual answers for correctness -- grade whether the customer left
with what they needed.

[Conversation]
{transcript}

Reply "resolved" if the customer's request was satisfied, or was correctly
handed to a human via an escalation.

Reply "unresolved" if the conversation ends with the request unmet and no
escalation raised -- including when the assistant was polite, accurate and
simply unable to help.

Reply "frustrated" if the customer expressed dissatisfaction, repeated
themselves because they were not understood, or pushed back on an answer.
"""

SESSION_CHOICES: dict[str, float] = {"resolved": 1.0, "unresolved": 0.5, "frustrated": 0.0}

_SESSION_INSTRUCTION = (
    "Reply with exactly one word on the first line -- `resolved`, `unresolved` "
    "or `frustrated` -- then one short sentence of justification on the next line."
)


def judge_session(
    transcript: str,
    *,
    settings: Settings,
    model: str = "deepseek-v4-pro",
) -> tuple[str, float, str]:
    """Grade one conversation. Returns (label, score, explanation).

    Three ordered outcomes rather than pass/fail, scored 1.0 / 0.5 / 0.0, so the
    mean carries the distinction between "couldn't help" and "made things
    worse". Those are different problems with different fixes.
    """
    if not transcript.strip():
        return "error", float("nan"), "Empty transcript."

    text = _ask(
        SESSION_TEMPLATE.format(transcript=transcript),
        instruction=_SESSION_INSTRUCTION,
        settings=settings,
        model=model,
        max_tokens=384,
    )
    if text is None:
        return "error", float("nan"), "Judge call failed."
    return parse_verdict(text, SESSION_CHOICES)
