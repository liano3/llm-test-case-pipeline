"""Derive expected outputs and selected cases from a cached execution matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from consensus_core import score_programs, unanimous_core_output, unique_mode
from jsonl_io import load_latest_by_problem

REPORT_SCHEMA_VERSION = 1
DEFAULT_MAX_CASE_BYTES = 64 * 1024


def unpack_execution(record: dict[str, Any]) -> list[list[dict[str, Any]]]:
    cases = record.get("cases")
    programs = record.get("programs")
    compact_results = record.get("results")
    outputs = record.get("outputs")
    if not all(isinstance(value, list) for value in (cases, programs, compact_results)):
        raise ValueError("cached matrix metadata is invalid")
    if len(compact_results) != len(cases) or not isinstance(outputs, dict):
        raise ValueError("cached case count or output table is invalid")

    results = []
    for compact_case in compact_results:
        if not isinstance(compact_case, list) or len(compact_case) != len(programs):
            raise ValueError("cached program count is invalid")
        case_results = []
        for compact_result in compact_case:
            if not isinstance(compact_result, list) or len(compact_result) != 2:
                raise ValueError("cached result is invalid")
            status, key = compact_result
            if status == "ok" and (not isinstance(key, str) or key not in outputs):
                raise ValueError("cached successful output is missing")
            case_results.append(
                {
                    "status": status,
                    "output_key": key,
                    "output": outputs[key] if status == "ok" else "",
                }
            )
        results.append(case_results)
    return results


def within_case_limit(case: dict[str, Any], max_case_bytes: int) -> bool:
    return (
        max(
            len(case["input"].encode()),
            len(case["expected"].encode()),
        )
        <= max_case_bytes
    )


def select_cases(
    cases: list[dict[str, Any]],
    max_cases: int,
    max_case_bytes: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], str, int]:
    oversized = sum(not within_case_limit(case, max_case_bytes) for case in cases)
    selectable = {index: case for index, case in enumerate(cases) if within_case_limit(case, max_case_bytes)}
    eligible = {
        index: case
        for index, case in selectable.items()
        if case["killed_candidates"] and case["candidate_supporters"] > 0
    }
    selected_indices = []
    covered: set[int] = set()
    while eligible and len(selected_indices) < max_cases:
        best = max(
            eligible,
            key=lambda index: (
                len(set(eligible[index]["killed_candidates"]) - covered),
                eligible[index]["candidate_supporters"],
                -len(eligible[index]["input"].encode()),
                -index,
            ),
        )
        newly_covered = set(eligible[best]["killed_candidates"]) - covered
        if not newly_covered:
            break
        selected_indices.append(best)
        covered.update(newly_covered)
        del eligible[best]

    selection = "coverage"
    if not selected_indices and selectable:
        supported = [index for index, case in selectable.items() if case["candidate_supporters"] > 0]
        pool = supported or list(selectable)
        best = max(
            pool,
            key=lambda index: (
                selectable[index]["candidate_supporters"],
                -len(selectable[index]["input"].encode()),
                -index,
            ),
        )
        selected_indices = [best]
        selection = "highest_confidence"

    selected = [{"input": cases[index]["input"], "expected": cases[index]["expected"]} for index in selected_indices]
    details = [
        {
            "category": cases[index]["category"],
            "index": cases[index]["index"],
            "candidate_supporters": cases[index]["candidate_supporters"],
            "killed_candidates": cases[index]["killed_candidates"],
        }
        for index in selected_indices
    ]
    return selected, details, selection, oversized


def analyze_top2(
    record: dict[str, Any],
    accepted: list[bool],
    max_cases: int,
    max_case_bytes: int,
) -> dict[str, Any]:
    if len(accepted) != len(record["cases"]):
        raise ValueError("validation and execution case counts differ")
    raw_input_count = len(record["cases"])
    record = {
        **record,
        "cases": [case for case, valid in zip(record["cases"], accepted, strict=True) if valid],
        "results": [result for result, valid in zip(record["results"], accepted, strict=True) if valid],
    }
    cases = record["cases"]
    programs = record["programs"]
    results = unpack_execution(record)
    outputs_by_case = [[result["output_key"] for result in case_results] for case_results in results]
    preliminary_modes = [unique_mode(outputs) for outputs in outputs_by_case]
    scores = score_programs(outputs_by_case, preliminary_modes)
    core_indices = [int(score["program_index"]) for score in scores if score["is_core"]]

    labeled_cases = []
    discarded = 0
    for case, case_results, outputs in zip(cases, results, outputs_by_case, strict=True):
        final_mode = unanimous_core_output(outputs, core_indices)
        if final_mode is None:
            discarded += 1
            continue
        candidate_outputs = outputs[: record["candidate_count"]]
        labeled_cases.append(
            {
                **case,
                "expected": case_results[core_indices[0]]["output"],
                "candidate_supporters": sum(output == final_mode for output in candidate_outputs),
                "killed_candidates": [index for index, output in enumerate(candidate_outputs) if output != final_mode],
            }
        )

    selected, selected_details, selection, oversized = select_cases(labeled_cases, max_cases, max_case_bytes)
    program_reports = [
        {
            "kind": programs[index]["kind"],
            "index": programs[index]["index"],
            "matches": score["matches"],
            "evaluated_cases": score["evaluated_cases"],
            "match_ratio": score["match_ratio"],
            "is_core": score["is_core"],
        }
        for index, score in enumerate(scores)
    ]
    return {
        "problem_id": record["problem_id"],
        "status": "evaluated",
        "execution_signature": record["execution_signature"],
        "input_count": raw_input_count,
        "validated_input_count": len(cases),
        "constraint_rejected_inputs": raw_input_count - len(cases),
        "program_count": len(programs),
        "candidate_count": record["candidate_count"],
        "preliminary_mode_cases": sum(mode is not None for mode in preliminary_modes),
        "final_mode_cases": len(labeled_cases),
        "core_program_count": len(core_indices),
        "discarded_unresolved_cases": discarded,
        "oversized_labeled_cases": oversized,
        "selection": selection if selected else "none",
        "test_cases": selected,
        "selected_details": selected_details,
        "programs": program_reports,
    }


def empty_report(problem_id: str, status: str) -> dict[str, Any]:
    return {"problem_id": problem_id, "status": status, "test_cases": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-case-bytes", type=int, default=DEFAULT_MAX_CASE_BYTES)
    args = parser.parse_args()
    if min(args.max_cases, args.max_case_bytes) < 1:
        parser.error("case limits must be positive")

    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    problem_ids = {str(problem["id"]) for problem in problems}
    executions = load_latest_by_problem(args.executions, problem_ids)
    validations = load_latest_by_problem(args.validation, problem_ids)
    reports = []
    for problem in problems:
        problem_id = str(problem["id"])
        record = executions.get(problem_id)
        if not record:
            reports.append(empty_report(problem_id, "missing_execution"))
        elif record.get("status") != "executed":
            reports.append(empty_report(problem_id, str(record.get("status", "invalid_execution"))))
        else:
            validation = validations.get(problem_id)
            if not validation:
                reports.append(empty_report(problem_id, "missing_validation"))
            elif validation.get("status") != "validated":
                reports.append(empty_report(problem_id, str(validation.get("status", "invalid_validation"))))
            elif validation.get("execution_signature") != record.get("execution_signature"):
                reports.append(empty_report(problem_id, "stale_validation"))
            elif not any(validation.get("accepted", [])):
                reports.append(empty_report(problem_id, "no_valid_inputs"))
            else:
                reports.append(analyze_top2(record, validation["accepted"], args.max_cases, args.max_case_bytes))

    statuses = Counter(report["status"] for report in reports)
    summary = {
        "problems": len(reports),
        "statuses": dict(sorted(statuses.items())),
        "nonempty_problems": sum(bool(report["test_cases"]) for report in reports),
        "selected_cases": sum(len(report["test_cases"]) for report in reports),
        "core_programs": sum(report.get("core_program_count", 0) for report in reports),
        "preliminary_mode_cases": sum(report.get("preliminary_mode_cases", 0) for report in reports),
        "final_mode_cases": sum(report.get("final_mode_cases", 0) for report in reports),
        "discarded_unresolved_cases": sum(report.get("discarded_unresolved_cases", 0) for report in reports),
        "oversized_labeled_cases": sum(report.get("oversized_labeled_cases", 0) for report in reports),
        "constraint_rejected_inputs": sum(report.get("constraint_rejected_inputs", 0) for report in reports),
    }
    payload = {
        "config": {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "strategy": "top2",
            "max_cases": args.max_cases,
            "max_case_bytes": args.max_case_bytes,
        },
        "summary": summary,
        "problems": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
