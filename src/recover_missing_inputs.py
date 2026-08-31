"""Recover missing test inputs with candidate/reference output consensus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from openai import AsyncOpenAI, OpenAIError

from consensus_core import classify_vote, output_key
from constraint_core import filter_cases
from input_core import (
    build_input_prompt,
    run_test_generator,
    validate_constraint_case_count,
)
from jsonl_io import append_jsonl, load_latest_by_problem
from reference_core import extract_code, load_checkpoint
from runner import run_solution

T = TypeVar("T")
MIN_TRUSTED_CASES = 10


async def label_by_consensus(
    cases: dict[str, list[str]],
    programs: list[tuple[str, str]],
    run_program: Callable[[str, str, float], Awaitable[Any]],
    timeout: float,
    retry_timeout: float,
    min_supporters: int,
    min_support_ratio: float,
    min_margin: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    async def execute(stdin: str, code: str) -> tuple[Any, bool]:
        result = await run_program(code, stdin, timeout)
        retried = result.status == "timeout" and retry_timeout > timeout
        if retried:
            result = await run_program(code, stdin, retry_timeout)
        return result, retried

    labeled = {category: [] for category in cases}
    category_stats: dict[str, dict[str, Any]] = {}
    case_reports = []
    for category, inputs in cases.items():
        reasons: Counter[str] = Counter()
        retries = 0
        results = await asyncio.gather(*(execute(stdin, code) for stdin in inputs for _, code in programs))
        result_index = 0
        for case_index, stdin in enumerate(inputs):
            keys = []
            outputs: dict[str, str] = {}
            statuses: Counter[str] = Counter()
            for _ in programs:
                result, retried = results[result_index]
                result_index += 1
                retries += retried
                statuses[result.status] += 1
                if result.status == "ok":
                    key = output_key(result.output)
                    keys.append(key)
                    outputs.setdefault(key, result.output)
                else:
                    keys.append(None)
            decision = classify_vote(
                keys,
                min_supporters=min_supporters,
                min_support_ratio=min_support_ratio,
                min_margin=min_margin,
            )
            reasons[decision.reason] += 1
            if decision.trusted and decision.mode in outputs:
                labeled[category].append({"input": stdin, "expected": outputs[decision.mode]})
            case_reports.append(
                {
                    "category": category,
                    "case_index": case_index,
                    **decision.to_dict(),
                    "statuses": dict(statuses),
                }
            )
        category_stats[category] = {
            "generated": len(inputs),
            "trusted": len(labeled[category]),
            "timeout_retries": retries,
            "reasons": dict(reasons),
        }
    return labeled, {"categories": category_stats, "cases": case_reports}


def rejected_feedback(
    generated: dict[str, list[str]],
    filtered: dict[str, list[str]],
) -> str:
    rejected = []
    for category, inputs in generated.items():
        accepted = set(filtered[category])
        rejected.extend(stdin for stdin in inputs if stdin not in accepted)
    examples = "\n\n".join(repr(stdin[:600]) for stdin in rejected[:3])
    accepted_count = sum(map(len, filtered.values()))
    generated_count = sum(map(len, generated.values()))
    return (
        f"Only {accepted_count}/{generated_count} generated inputs passed ProblemConstraint; need at least 20. "
        "Regenerate inputs that strictly follow the statement format and guarantees. "
        f"Rejected examples:\n{examples}"
    )


async def recover(args: argparse.Namespace) -> None:
    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    end = None if args.limit is None else args.offset + args.limit
    problems = problems[args.offset : end]
    base_inputs = load_latest_by_problem(args.base_inputs)
    constraints = load_latest_by_problem(args.constraints)
    references = load_checkpoint(args.references)
    latest = load_latest_by_problem(args.output)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    model_semaphore = asyncio.Semaphore(args.max_concurrency)
    execution_semaphore = asyncio.Semaphore(args.execution_concurrency)
    checkpoint_lock = asyncio.Lock()
    trace_lock = asyncio.Lock()

    async def run_blocking(function: Callable[..., T], *values: Any) -> T:
        async with execution_semaphore:
            return await asyncio.to_thread(function, *values)

    async def run_program(code: str, stdin: str, timeout: float) -> Any:
        return await run_blocking(run_solution, code, stdin, timeout)

    async def checkpoint(record: dict[str, Any]) -> None:
        async with checkpoint_lock:
            append_jsonl(args.output, record)
            latest[str(record["problem_id"])] = record

    async def trace(record: dict[str, Any]) -> None:
        if args.trace_output:
            async with trace_lock:
                append_jsonl(args.trace_output, record)

    async def one(problem: dict[str, Any]) -> None:
        problem_id = str(problem["id"])
        base = base_inputs.get(problem_id)
        if base and base.get("status") == "ok":
            return
        existing = latest.get(problem_id)
        if existing and existing.get("status") == "ok":
            print(f"{problem_id}: keeping recovery checkpoint", flush=True)
            return

        constraint = constraints.get(problem_id)
        if not constraint or constraint.get("status") != "ok":
            await checkpoint(
                {
                    "problem_id": problem_id,
                    "status": "blocked",
                    "error": "missing successful ProblemConstraint",
                    "source_status": base.get("status") if base else "missing",
                    "model": args.model,
                }
            )
            return

        reference_profiles = []
        programs = [(f"candidate:{index}", code) for index, code in enumerate(problem["solutions"])]
        for profile in ("direct", "brute"):
            reference = references.get((problem_id, profile))
            if reference and reference.get("status") == "ok":
                reference_profiles.append(profile)
                programs.append((profile, reference["code"]))
        if len(programs) < args.min_supporters:
            await checkpoint(
                {
                    "problem_id": problem_id,
                    "status": "blocked",
                    "error": "too few candidate/reference programs for consensus",
                    "source_status": base.get("status") if base else "missing",
                    "model": args.model,
                }
            )
            return

        prior = existing if existing and existing.get("status") == "failed" else base or {}
        previous_code = prior.get("failed_code") or ""
        error = prior.get("error", "")
        last_stages: dict[str, Any] = {}
        last_stats = None
        for attempt in range(1, args.retries + 2):
            stages: dict[str, Any] = {}
            last_stages = stages
            prompt_code = previous_code
            prompt_error = error
            if attempt >= 3:
                prompt_code = ""
                prompt_error = (
                    f"{error}\nDiscard the previous implementation and write a new, simple generator "
                    "from the problem statement."
                )
            prompt = build_input_prompt(problem, prompt_code, prompt_error, code_only=attempt >= 3)
            max_tokens = args.max_tokens if attempt == 1 else args.retry_max_tokens
            response_text = ""
            code = ""
            finish_reason = None
            try:
                async with model_semaphore:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature,
                        top_p=0.95,
                        max_tokens=max_tokens,
                        timeout=args.request_timeout,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                choice = response.choices[0]
                response_text = choice.message.content or ""
                finish_reason = choice.finish_reason
                code = extract_code(response_text, require_fenced=finish_reason == "length")
                generated = await run_blocking(run_test_generator, code, args.generator_timeout)
                if generated.status != "ok":
                    raise ValueError(f"generator {generated.status}: {generated.error}")
                stages["generation"] = {"cases": generated.cases}

                filtered, constraint_stats = await run_blocking(
                    filter_cases,
                    constraint["code"],
                    generated.cases,
                    args.constraint_timeout,
                )
                stages["constraint_filter"] = {"cases": filtered, "stats": constraint_stats}
                try:
                    validate_constraint_case_count(filtered)
                except ValueError:
                    raise ValueError(rejected_feedback(generated.cases, filtered)) from None

                labeled, consensus_stats = await label_by_consensus(
                    filtered,
                    programs,
                    run_program,
                    args.program_timeout,
                    args.retry_timeout,
                    args.min_supporters,
                    args.min_support_ratio,
                    args.min_margin,
                )
                trusted_count = sum(map(len, labeled.values()))
                last_stats = consensus_stats
                stages["candidate_consensus"] = {"cases": labeled, "stats": consensus_stats}
                if trusted_count < MIN_TRUSTED_CASES:
                    reasons = {category: stats["reasons"] for category, stats in consensus_stats["categories"].items()}
                    raise ValueError(
                        f"only {trusted_count} inputs reached trusted output consensus; need at least "
                        f"{MIN_TRUSTED_CASES}; vote outcomes={reasons}"
                    )

                record = {
                    "problem_id": problem_id,
                    "status": "ok",
                    "attempt": attempt,
                    "source_status": base.get("status") if base else "missing",
                    "generator_code": code,
                    "test_cases": labeled,
                    "stages": stages,
                    "stats": consensus_stats,
                    "candidate_count": len(problem["solutions"]),
                    "reference_profiles": reference_profiles,
                    "labeling": "candidate_consensus",
                    "model": args.model,
                }
                await trace(
                    {
                        "problem_id": problem_id,
                        "attempt": attempt,
                        "max_tokens": max_tokens,
                        "prompt": prompt,
                        "response": response_text,
                        "finish_reason": finish_reason,
                        "outcome": "ok",
                        "stages": stages,
                    }
                )
                await checkpoint(record)
                print(
                    f"{problem_id}: recovered {trusted_count} inputs with {len(programs)} voting programs",
                    flush=True,
                )
                return
            except (OpenAIError, OSError, ValueError, TypeError, SyntaxError, IndexError) as exc:
                error = str(exc)
            await trace(
                {
                    "problem_id": problem_id,
                    "attempt": attempt,
                    "max_tokens": max_tokens,
                    "prompt": prompt,
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "outcome": "failed",
                    "error": error,
                    "stages": stages,
                }
            )
            previous_code = code or previous_code
            print(f"{problem_id}: recovery attempt {attempt} failed: {error}", flush=True)

        await checkpoint(
            {
                "problem_id": problem_id,
                "status": "failed",
                "attempt": args.retries + 1,
                "source_status": base.get("status") if base else "missing",
                "error": error,
                "failed_code": previous_code or None,
                "stages": last_stages,
                "stats": last_stats,
                "candidate_count": len(problem["solutions"]),
                "reference_profiles": reference_profiles,
                "labeling": "candidate_consensus",
                "model": args.model,
            }
        )

    await asyncio.gather(*(one(problem) for problem in problems))
    recovered = sum(record.get("status") == "ok" for record in latest.values())
    print(f"recovered={recovered}/{sum(base_inputs.get(str(p['id']), {}).get('status') != 'ok' for p in problems)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--base-inputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--execution-concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retry-max-tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--generator-timeout", type=float, default=10.0)
    parser.add_argument("--constraint-timeout", type=float, default=5.0)
    parser.add_argument("--program-timeout", type=float, default=5.0)
    parser.add_argument("--retry-timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-supporters", type=int, default=3)
    parser.add_argument("--min-support-ratio", type=float, default=0.70)
    parser.add_argument("--min-margin", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if min(args.max_concurrency, args.execution_concurrency) < 1:
        parser.error("concurrency values must be positive")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if min(args.max_tokens, args.retry_max_tokens) < 1:
        parser.error("token limits must be positive")
    if args.retry_max_tokens < args.max_tokens:
        parser.error("--retry-max-tokens must be at least --max-tokens")
    if (
        min(
            args.request_timeout,
            args.generator_timeout,
            args.constraint_timeout,
            args.program_timeout,
            args.retry_timeout,
        )
        <= 0
    ):
        parser.error("timeouts must be positive")
    if args.retry_timeout < args.program_timeout:
        parser.error("--retry-timeout must be at least --program-timeout")
    if args.min_supporters < 1 or args.min_margin < 1:
        parser.error("vote thresholds must be positive")
    if not 0 < args.min_support_ratio <= 1:
        parser.error("--min-support-ratio must be in (0, 1]")
    asyncio.run(recover(args))


if __name__ == "__main__":
    main()
