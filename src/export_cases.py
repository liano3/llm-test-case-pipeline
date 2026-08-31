"""Export selected cases without embedding candidate programs or private labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from problem_statements import extract_examples

DEFAULT_MAX_CASE_BYTES = 64 * 1024


def load_report(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(report["problem_id"]): report for report in payload["problems"]}


def within_limit(case: dict[str, str], limit: int) -> bool:
    return max(len(case["input"].encode()), len(case["expected"].encode())) <= limit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export generated cases as JSONL without copying candidate solutions."
    )
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--consensus-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-case-bytes", type=int, default=DEFAULT_MAX_CASE_BYTES)
    args = parser.parse_args()
    if min(args.max_cases, args.max_case_bytes) < 1:
        parser.error("case limits must be positive")

    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    reports = load_report(args.consensus_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = {"cases": 0, "fallback": 0, "missing": 0}
    with args.output.open("w", encoding="utf-8") as output:
        for problem in problems:
            problem_id = str(problem["id"])
            report = reports.get(problem_id)
            if report is None:
                totals["missing"] += 1
            cases = (report or {}).get("test_cases", [])[: args.max_cases]
            cases = [case for case in cases if within_limit(case, args.max_case_bytes)]
            if not cases:
                cases = [
                    case
                    for case in extract_examples(problem["question"])
                    if within_limit(case, args.max_case_bytes)
                ][: args.max_cases]
                totals["fallback"] += bool(cases)
            totals["cases"] += len(cases)
            output.write(
                json.dumps({"problem_id": problem["id"], "test_cases": cases}, ensure_ascii=False) + "\n"
            )
    print(
        f"problems={len(problems)} cases={totals['cases']} "
        f"fallback={totals['fallback']} missing_reports={totals['missing']}"
    )


if __name__ == "__main__":
    main()
