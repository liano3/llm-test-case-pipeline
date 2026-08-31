"""Generate and execute statement-derived input constraints."""

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


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    accepted: list[bool]
    error: str = ""


def build_constraint_prompt(
    problem: dict[str, Any],
    previous_code: str = "",
    error: str = "",
    *,
    code_only: bool = False,
) -> str:
    repair = ""
    if error:
        repair = f"""

The previous constraint failed validation:
{error[-2000:]}

Previous constraint:
{previous_code[-16000:]}

Fix it and return a complete constraint.
"""
    output_format = ""
    if code_only:
        output_format = """

Return only one complete Python code block. Do not include explanations, analysis, or text outside the code block.
"""
    return f"""Implement a self-contained Python ProblemConstraint class for this ACM problem.

The class must have this method:
- validate(self, raw_input: str) -> bool: return whether the complete stdin input satisfies the statement.

Validate input format, value ranges, relationships, and structural requirements. Treat every condition stated as
"guaranteed" as mandatory input validity, including global feasibility, consistency, distinctness, connectivity,
and constructibility requirements. Return False on parse errors or any violated guarantee.

Problem:
{problem["question"]}
{repair}{output_format}"""


def run_constraint(code: str, inputs: list[str], timeout: float) -> ConstraintResult:
    with tempfile.TemporaryDirectory(prefix="constraint-") as directory:
        source = Path(directory) / "constraint.py"
        source.write_text(code, encoding="utf-8")
        input_path = Path(directory) / "inputs.json"
        input_path.write_text(json.dumps(inputs, ensure_ascii=False), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--sandbox", str(source), str(input_path)],
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
            return ConstraintResult("timeout", [], f"constraint exceeded {timeout}s")
        if process.returncode != 0:
            return ConstraintResult("error", [], error[-2000:])
        try:
            accepted = json.loads(output)
            if not isinstance(accepted, list) or len(accepted) != len(inputs):
                raise ValueError("constraint returned an invalid result list")
            return ConstraintResult("ok", [value is True for value in accepted])
        except (ValueError, TypeError) as exc:
            return ConstraintResult("error", [], str(exc))


def validate_constraint(problem: dict[str, Any], code: str, timeout: float) -> int:
    from problem_statements import extract_examples

    examples = extract_examples(problem["question"])
    result = run_constraint(code, [example["input"] for example in examples], timeout)
    if result.status != "ok":
        raise ValueError(f"constraint {result.status}: {result.error}")
    rejected = [index for index, accepted in enumerate(result.accepted, 1) if not accepted]
    if rejected:
        raise ValueError(f"constraint rejected statement examples: {rejected}")
    return len(examples)


def filter_cases(
    code: str,
    cases: dict[str, list[str]],
    timeout: float,
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    flattened = [(category, value) for category, values in cases.items() for value in values]
    result = run_constraint(code, [value for _, value in flattened], timeout)
    if result.status != "ok":
        raise ValueError(f"constraint {result.status}: {result.error}")
    filtered = {category: [] for category in cases}
    for (category, value), accepted in zip(flattened, result.accepted, strict=True):
        if accepted:
            filtered[category].append(value)
    stats = {category: {"generated": len(cases[category]), "accepted": len(filtered[category])} for category in cases}
    return filtered, stats


def run_sandbox() -> None:
    source = Path(sys.argv[2])
    inputs = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)

    namespace: dict[str, Any] = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exec(  # noqa: S102 - model-generated code runs in this resource-limited child process
            compile(source.read_text(encoding="utf-8"), "<constraint>", "exec"), namespace
        )
        constraint = namespace["ProblemConstraint"]()
        result = []
        for raw_input in inputs:
            try:
                result.append(constraint.validate(raw_input) is True)
            except Exception:  # noqa: BLE001 - invalid model-generated validators reject the input
                result.append(False)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--sandbox":
    run_sandbox()
