"""Generate, execute, and label base and corner input candidates."""

from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORY_METHODS = {
    "base": "generate_base_case",
    "corner": "generate_corner_case",
}
MIN_CONSTRAINT_CASES = 20
RECOVERY_EXPECTED_THRESHOLD = 10


@dataclass(frozen=True)
class GeneratorResult:
    status: str
    cases: dict[str, list[str]]
    error: str = ""


def build_input_prompt(
    problem: dict[str, Any],
    previous_code: str = "",
    error: str = "",
    *,
    code_only: bool = False,
) -> str:
    repair = ""
    if error:
        repair = f"""

The previous generator failed validation:
{error[-2000:]}

Previous generator:
{previous_code[-16000:]}

Fix it and return a complete generator.
"""
    output_format = ""
    if code_only:
        output_format = """

Return only one complete Python code block. Do not include explanations, analysis, or text outside the code block.
"""
    return f"""Implement a self-contained Python TestGenerator class for this ACM problem.

The class must have these methods:
- generate_base_case(self) -> list[str]: return 40 diverse, valid stdin inputs.
- generate_corner_case(self) -> list[str]: return 10 diverse, valid boundary stdin inputs.

Keep test cases within a size that a brute-force solution can handle; they do not need to be minimal.
Each list item must be one complete stdin input.
Generate inputs only, not outputs.

Problem:
{problem["question"]}
{repair}{output_format}"""


def _normalize_cases(raw: dict[str, Any], categories: tuple[str, ...]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for category in categories:
        values = raw.get(category)
        if not isinstance(values, list):
            raise TypeError(f"{category} generator did not return a list")
        cases = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            case = value if value.endswith("\n") else value + "\n"
            if not case.strip() or case in seen:
                continue
            seen.add(case)
            cases.append(case)
        if not cases:
            raise ValueError(f"{category} generator produced no usable inputs")
        normalized[category] = cases
    return normalized


def run_test_generator(
    code: str,
    timeout: float,
    category_methods: dict[str, str] | None = None,
) -> GeneratorResult:
    methods = category_methods or CATEGORY_METHODS
    with tempfile.TemporaryDirectory(prefix="inputgen-") as directory:
        source = Path(directory) / "generator.py"
        source.write_text(code, encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--sandbox",
                str(source),
                json.dumps(methods),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
            },
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            output, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return GeneratorResult("timeout", {}, f"generator exceeded {timeout}s")
        if process.returncode != 0:
            return GeneratorResult("error", {}, error[-2000:])
        try:
            return GeneratorResult("ok", _normalize_cases(json.loads(output), tuple(methods)))
        except (ValueError, TypeError) as exc:
            return GeneratorResult("error", {}, str(exc))


def label_cases(
    cases: dict[str, list[str]],
    direct_code: str,
    brute_code: str,
    timeout: float,
    recovery_code: str | None = None,
    expected_to_verify: dict[str, list[dict[str, str]]] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    from runner import run_solution

    categories = tuple(cases)
    labeled = {category: [] for category in categories}
    stats = {
        category: {
            "generated": len(cases[category]),
            "agreed": 0,
            "no_consensus": 0,
            "recovery_rejected": 0,
            "recovery_resolved": 0,
        }
        for category in categories
    }
    codes = [direct_code, brute_code] + ([recovery_code] if recovery_code else [])
    expected_lookup = (
        {
            category: {case["input"]: case["expected"] for case in expected_to_verify.get(category, [])}
            for category in categories
        }
        if expected_to_verify
        else {category: {} for category in categories}
    )
    for category, inputs in cases.items():
        for stdin in inputs:
            outputs: dict[tuple[str, ...], list[str]] = {}
            results = []
            for code in codes:
                result = run_solution(code, stdin, timeout)
                results.append(result)
                if result.status == "ok":
                    outputs.setdefault(tuple(result.output.split()), []).append(result.output)
            if stdin in expected_lookup[category]:
                expected = expected_lookup[category][stdin]
                recovery = results[-1]
                consensus = (
                    expected
                    if recovery.status == "ok" and tuple(recovery.output.split()) == tuple(expected.split())
                    else None
                )
                if consensus is None:
                    stats[category]["recovery_rejected"] += 1
            else:
                consensus = next((values[0] for values in outputs.values() if len(values) >= 2), None)
                if consensus is not None and recovery_code:
                    stats[category]["recovery_resolved"] += 1
            if consensus is None:
                stats[category]["no_consensus"] += 1
            else:
                labeled[category].append({"input": stdin, "expected": consensus})
                stats[category]["agreed"] += 1
    return labeled, stats


def validate_constraint_case_count(cases: dict[str, list[str]]) -> None:
    count = sum(map(len, cases.values()))
    if count < MIN_CONSTRAINT_CASES:
        raise ValueError(f"only {count} inputs passed ProblemConstraint; need at least {MIN_CONSTRAINT_CASES}")


def unresolved_inputs(
    cases: dict[str, list[str]],
    labeled: dict[str, list[dict[str, str]]],
) -> list[str]:
    unresolved = []
    seen = set()
    for category, inputs in cases.items():
        resolved = {case["input"] for case in labeled[category]}
        for stdin in inputs:
            if stdin not in resolved and stdin not in seen:
                seen.add(stdin)
                unresolved.append(stdin)
    return unresolved


def run_sandbox() -> None:
    source = Path(sys.argv[2])
    category_methods = json.loads(sys.argv[3])
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)

    namespace: dict[str, Any] = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exec(  # noqa: S102 - model-generated code runs in this resource-limited child process
            compile(source.read_text(encoding="utf-8"), "<generator>", "exec"), namespace
        )
        generator = namespace["TestGenerator"]()
        result = {category: getattr(generator, method)() for category, method in category_methods.items()}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--sandbox":
    run_sandbox()
