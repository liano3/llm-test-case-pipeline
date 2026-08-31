"""Validate cached execution inputs with an independent ProblemConstraint checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from constraint_core import run_constraint
from jsonl_io import append_jsonl, load_latest_by_problem

VALIDATION_SCHEMA_VERSION = 1


def validation_signature(execution: dict[str, Any], constraint_code: str) -> str:
    payload = {
        "schema": VALIDATION_SCHEMA_VERSION,
        "execution_signature": execution["execution_signature"],
        "constraint_code": constraint_code,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_problem(
    problem_id: str,
    execution: dict[str, Any] | None,
    constraint: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    if not execution:
        return {"problem_id": problem_id, "status": "missing_execution"}
    if execution.get("status") == "no_inputs":
        return {
            "problem_id": problem_id,
            "status": "no_inputs",
            "execution_signature": execution["execution_signature"],
            "accepted": [],
        }
    if execution.get("status") != "executed":
        return {"problem_id": problem_id, "status": "invalid_execution"}
    if not constraint or constraint.get("status") != "ok" or not constraint.get("code"):
        return {"problem_id": problem_id, "status": "missing_constraint"}

    signature = validation_signature(execution, constraint["code"])
    result = run_constraint(constraint["code"], [case["input"] for case in execution["cases"]], timeout)
    if result.status != "ok":
        return {
            "problem_id": problem_id,
            "status": f"constraint_{result.status}",
            "execution_signature": execution["execution_signature"],
            "validation_signature": signature,
            "error": result.error,
        }
    return {
        "problem_id": problem_id,
        "status": "validated",
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "execution_signature": execution["execution_signature"],
        "validation_signature": signature,
        "case_count": len(result.accepted),
        "accepted_count": sum(result.accepted),
        "accepted": result.accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    problem_ids = [str(problem["id"]) for problem in problems]
    executions = load_latest_by_problem(args.executions, set(problem_ids))
    constraints = load_latest_by_problem(args.constraints, set(problem_ids))
    existing = load_latest_by_problem(args.checkpoint, set(problem_ids))
    pending = []
    for problem_id in problem_ids:
        execution = executions.get(problem_id)
        constraint = constraints.get(problem_id)
        record = existing.get(problem_id)
        reusable = False
        if execution and execution.get("status") == "no_inputs" and record:
            reusable = record.get("status") == "no_inputs"
            reusable &= record.get("execution_signature") == execution.get("execution_signature")
        elif execution and constraint and constraint.get("code") and record:
            reusable = record.get("validation_signature") == validation_signature(execution, constraint["code"])
            reusable &= record.get("status") == "validated"
        if not reusable:
            pending.append(problem_id)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = len(problem_ids) - len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(validate_problem, pid, executions.get(pid), constraints.get(pid), args.timeout): pid
            for pid in pending
        }
        for future in as_completed(futures):
            append_jsonl(args.checkpoint, future.result())
            completed += 1
            if completed % 100 == 0 or completed == len(problem_ids):
                print(f"validated {completed}/{len(problem_ids)}", flush=True)


if __name__ == "__main__":
    main()
