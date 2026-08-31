"""Pure helpers for reference generation and sample validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonl_io import read_jsonl
from problem_statements import extract_examples
from runner import check_equal, run_solution


@dataclass(frozen=True)
class Profile:
    name: str
    instruction: str


PROFILES = (
    Profile(
        "direct",
        "Solve this ACM programming problem and provide a correct Python 3 solution.",
    ),
    Profile(
        "brute",
        "Solve this ACM programming problem. Ignore time and memory complexity and prioritize "
        "correctness. Provide a correct Python 3 solution.",
    ),
)


def extract_code(response: str, *, require_fenced: bool = False) -> str:
    text = response.strip().split("</think>")[-1].strip()
    blocks = re.findall(r"```(?:python)?\s*\n?([\s\S]*?)\n?```", text, flags=re.IGNORECASE)
    if require_fenced and not blocks:
        raise ValueError("truncated response contains no complete fenced code block")
    code = blocks[-1].strip() if blocks else text
    code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\n```\s*$", "", code).strip()
    if not code:
        raise ValueError("model returned empty code")
    compile(code, "<reference>", "exec")
    return code


def validate_reference(
    problem: dict[str, Any],
    code: str,
    timeout: float,
) -> tuple[str | None, int, list[dict[str, Any]]]:
    examples = extract_examples(problem["question"])
    mismatches = []
    for index, example in enumerate(examples, 1):
        result = run_solution(code, example["input"], timeout)
        if result.status != "ok":
            return (
                (
                    f"example {index} execution {result.status}; input={example['input'][:800]!r}; "
                    f"stderr={result.error[-800:]!r}"
                ),
                len(examples),
                [],
            )
        if not check_equal(result.output, example["expected"]):
            mismatches.append(
                {
                    "index": index,
                    "input": example["input"],
                    "expected": example["expected"],
                    "actual": result.output,
                }
            )
    if not mismatches:
        return None, len(examples), []
    first = mismatches[0]
    return (
        (
            f"example {first['index']} output mismatch; input={first['input'][:800]!r}; "
            f"expected={first['expected'][:800]!r}; actual={first['actual'][:800]!r}"
        ),
        len(examples),
        mismatches,
    )


def build_prompt(
    problem: dict[str, Any],
    profile: Profile,
    previous_code: str,
    error: str,
    *,
    code_only: bool = False,
) -> str:
    repair = ""
    if error:
        concise = "\nKeep the response concise." if "truncated" in error else ""
        repair = f"""

The previous solution failed validation:
{error[-2400:]}
{concise}

Previous solution:
{previous_code[-16000:]}

Fix the solution and provide a complete correct Python 3 solution.
"""
    output_format = ""
    if code_only:
        output_format = """

Return only one complete Python 3 code block. Do not include explanations, analysis, or text outside the code block.
"""
    return f"""
{profile.instruction}

Problem:
{problem["question"]}
{repair}{output_format}"""


def build_recovery_prompt(
    problem: dict[str, Any],
    disputed_inputs: list[str],
) -> str:
    cases = "\n\n".join(f"Input {index}:\n{stdin}" for index, stdin in enumerate(disputed_inputs, 1))
    return f"""Solve this ACM programming problem and provide a correct Python 3 solution.

Problem:
{problem["question"]}

The existing solutions disagree on these inputs:
{cases}

Return only one complete Python 3 code block. Do not include explanations, analysis, or text outside the code block.
"""


def load_checkpoint(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for record in read_jsonl(path, tolerate_malformed=True):
        latest[str(record["problem_id"]), record["profile"]] = record
    return latest
