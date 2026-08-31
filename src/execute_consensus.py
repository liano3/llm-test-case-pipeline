"""Execute every program on every constraint-valid input and cache the matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from consensus_core import output_key
from jsonl_io import append_jsonl, load_latest_by_problem
from reference_core import load_checkpoint
from runner import run_solution

EXECUTION_SCHEMA_VERSION = 1


def execute(code: str, stdin: str, timeout: float) -> dict[str, Any]:
    result = run_solution(code, stdin, timeout)
    return {
        "status": result.status,
        "output": result.output if result.status == "ok" else "",
        "output_key": output_key(result.output) if result.status == "ok" else None,
    }


def constraint_cases(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []
    stages = record.get("stages")
    filtered = stages.get("constraint_filter", {}).get("cases") if isinstance(stages, dict) else None
    if not isinstance(filtered, dict):
        return []
    cases = []
    seen = set()
    for category, inputs in filtered.items():
        if not isinstance(inputs, list):
            continue
        for index, stdin in enumerate(inputs):
            if isinstance(stdin, str) and stdin not in seen:
                seen.add(stdin)
                cases.append({"category": category, "index": index, "input": stdin})
    return cases


def choose_input_record(records: list[dict[str, Any] | None]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    choices = [(record, constraint_cases(record)) for record in records if record]
    if not choices:
        return None, []
    return max(choices, key=lambda item: len(item[1]))


def build_programs(
    problem: dict[str, Any],
    problem_id: str,
    record: dict[str, Any] | None,
    references: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    programs = [{"kind": "candidate", "index": index, "code": code} for index, code in enumerate(problem["solutions"])]
    for profile in ("direct", "brute"):
        reference = references.get((problem_id, profile))
        if reference and reference.get("status") == "ok":
            programs.append({"kind": profile, "index": None, "code": reference["code"]})
    recovery_code = record.get("recovery_reference_code") if record else None
    if isinstance(recovery_code, str) and recovery_code.strip():
        programs.append({"kind": "recovery", "index": None, "code": recovery_code})
    return programs


def execution_signature(
    problem_id: str,
    cases: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    timeout: float,
) -> str:
    payload = {
        "execution_schema": EXECUTION_SCHEMA_VERSION,
        "problem_id": problem_id,
        "inputs": [case["input"] for case in cases],
        "programs": [program["code"] for program in programs],
        "timeout": timeout,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def pack_execution(data: dict[str, Any], status: str = "executed") -> dict[str, Any]:
    outputs: dict[str, str] = {}
    compact_results = []
    for case_results in data.get("results", []):
        compact_case = []
        for result in case_results:
            key = result["output_key"]
            if key is not None:
                outputs.setdefault(key, result["output"])
            compact_case.append([result["status"], key])
        compact_results.append(compact_case)
    return {
        "problem_id": data["problem_id"],
        "status": status,
        "execution_schema_version": EXECUTION_SCHEMA_VERSION,
        "execution_signature": data["execution_signature"],
        "candidate_count": len(data["problem"]["solutions"]),
        "cases": data["cases"],
        "programs": [{"kind": program["kind"], "index": program["index"]} for program in data["programs"]],
        "results": compact_results,
        "outputs": outputs,
    }


def evaluate_pending(
    pending: list[dict[str, Any]],
    workers: int,
    timeout: float,
    on_complete: Callable[[dict[str, Any]], None],
) -> None:
    """Run a bounded global job queue and return each complete result matrix."""
    pending_iter = iter(pending)
    ready: deque[dict[str, Any]] = deque()

    def activate_next() -> bool:
        try:
            data = next(pending_iter)
        except StopIteration:
            return False
        program_count = len(data["programs"])
        data["results"] = [[None] * program_count for _ in data["cases"]]
        data["next_job"] = 0
        data["total_jobs"] = len(data["cases"]) * program_count
        data["remaining_jobs"] = data["total_jobs"]
        ready.append(data)
        return True

    for _ in range(min(len(pending), workers * 2)):
        activate_next()

    futures: dict[Future[dict[str, Any]], tuple[dict[str, Any], int, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def fill_workers() -> None:
            while ready and len(futures) < workers:
                data = ready.popleft()
                program_count = len(data["programs"])
                case_index, program_index = divmod(data["next_job"], program_count)
                data["next_job"] += 1
                future = pool.submit(
                    execute,
                    data["programs"][program_index]["code"],
                    data["cases"][case_index]["input"],
                    timeout,
                )
                futures[future] = (data, case_index, program_index)
                if data["next_job"] < data["total_jobs"]:
                    ready.append(data)

        fill_workers()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                data, case_index, program_index = futures.pop(future)
                data["results"][case_index][program_index] = future.result()
                data["remaining_jobs"] -= 1
                if data["remaining_jobs"] != 0:
                    continue
                on_complete(data)
                activate_next()
            fill_workers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        parser.error("offset must be non-negative and limit must be positive")

    all_problems = json.loads(args.problems.read_text(encoding="utf-8"))
    end = None if args.limit is None else args.offset + args.limit
    problems = all_problems[args.offset : end]
    problem_ids = {str(problem["id"]) for problem in problems}
    input_sources = [load_latest_by_problem(path, problem_ids) for path in args.inputs]
    references = load_checkpoint(args.references)
    cached = load_latest_by_problem(args.checkpoint, problem_ids)
    records_by_id: dict[str, dict[str, Any]] = {}
    pending = []

    for problem in problems:
        problem_id = str(problem["id"])
        record, cases = choose_input_record([source.get(problem_id) for source in input_sources])
        programs = build_programs(problem, problem_id, record, references)
        digest = execution_signature(problem_id, cases, programs, args.timeout)
        data = {
            "problem_id": problem_id,
            "problem": problem,
            "cases": cases,
            "programs": programs,
            "execution_signature": digest,
        }
        existing = cached.get(problem_id)
        if (
            existing
            and existing.get("execution_schema_version") == EXECUTION_SCHEMA_VERSION
            and existing.get("execution_signature") == digest
        ):
            records_by_id[problem_id] = existing
            continue
        if not cases or not programs:
            packed = pack_execution(data, "no_inputs" if not cases else "no_programs")
            records_by_id[problem_id] = packed
            append_jsonl(args.checkpoint, packed)
            continue
        pending.append(data)

    print(
        f"resumed={len(records_by_id)} pending={len(pending)} total={len(problems)}",
        file=sys.stderr,
        flush=True,
    )
    completed = 0

    def save_execution(data: dict[str, Any]) -> None:
        nonlocal completed
        packed = pack_execution(data)
        records_by_id[data["problem_id"]] = packed
        append_jsonl(args.checkpoint, packed)
        completed += 1
        if completed % 10 == 0 or completed == len(pending):
            print(
                f"executed {len(records_by_id)}/{len(problems)} (new {completed}/{len(pending)})",
                file=sys.stderr,
                flush=True,
            )

    evaluate_pending(pending, args.workers, args.timeout, save_execution)
    statuses = Counter(record["status"] for record in records_by_id.values())
    print(json.dumps({"problems": len(records_by_id), "statuses": dict(sorted(statuses.items()))}, indent=2))


if __name__ == "__main__":
    main()
