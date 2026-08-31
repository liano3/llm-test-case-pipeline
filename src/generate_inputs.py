"""Generate base and corner inputs and label them by reference agreement."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from constraint_core import filter_cases
from input_core import (
    RECOVERY_EXPECTED_THRESHOLD,
    build_input_prompt,
    label_cases,
    run_test_generator,
    unresolved_inputs,
    validate_constraint_case_count,
)
from jsonl_io import append_jsonl, load_latest_by_problem
from reference_core import (
    build_recovery_prompt,
    extract_code,
    load_checkpoint,
    validate_reference,
)


async def generate(args: argparse.Namespace) -> None:
    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    end = None if args.limit is None else args.offset + args.limit
    problems = problems[args.offset : end]
    references = load_checkpoint(args.references)
    constraints = load_latest_by_problem(args.constraints)
    latest = load_latest_by_problem(args.output)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    checkpoint_lock = asyncio.Lock()
    trace_lock = asyncio.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)

    async def append(path: Path, record: dict[str, Any], lock: asyncio.Lock) -> None:
        async with lock:
            append_jsonl(path, record)

    async def checkpoint(record: dict[str, Any]) -> None:
        await append(args.output, record, checkpoint_lock)
        latest[str(record["problem_id"])] = record

    async def trace(record: dict[str, Any]) -> None:
        if args.trace_output:
            await append(args.trace_output, record, trace_lock)

    async def one(problem: dict[str, Any]) -> None:
        problem_id = str(problem["id"])
        existing = latest.get(problem_id)
        if existing and existing.get("status") == "ok":
            print(f"{problem_id}: keeping checkpoint", flush=True)
            return

        direct = references.get((problem_id, "direct"))
        brute = references.get((problem_id, "brute"))
        constraint = constraints.get(problem_id)
        missing = []
        if not direct or direct.get("status") != "ok":
            missing.append("direct reference")
        if not brute or brute.get("status") != "ok":
            missing.append("brute reference")
        if not constraint or constraint.get("status") != "ok":
            missing.append("ProblemConstraint")
        if missing:
            error = "missing dependencies: " + ", ".join(missing)
            await checkpoint(
                {
                    "problem_id": problem_id,
                    "status": "blocked",
                    "error": error,
                    "model": args.model,
                }
            )
            print(f"{problem_id}: {error}", flush=True)
            return

        previous_code = (existing.get("failed_code") or "") if existing else ""
        error = existing.get("error", "") if existing and existing.get("status") == "failed" else ""
        last_stats = None
        last_constraint_stats = None
        last_stages: dict[str, Any] = {}
        for attempt in range(1, args.retries + 2):
            last_stats = None
            last_constraint_stats = None
            stages: dict[str, Any] = {}
            last_stages = stages
            prompt = build_input_prompt(
                problem,
                previous_code,
                error,
                code_only=attempt >= 3,
            )
            max_tokens = args.max_tokens if attempt == 1 else args.retry_max_tokens
            response_text = ""
            code = ""
            finish_reason = None
            try:
                async with semaphore:
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
                generated = await asyncio.to_thread(run_test_generator, code, args.generator_timeout)
                if generated.status != "ok":
                    raise ValueError(f"generator {generated.status}: {generated.error}")
                stages["generation"] = {
                    "cases": generated.cases,
                }
                filtered, constraint_stats = await asyncio.to_thread(
                    filter_cases,
                    constraint["code"],
                    generated.cases,
                    args.constraint_timeout,
                )
                last_constraint_stats = constraint_stats
                stages["constraint_filter"] = {
                    "cases": filtered,
                    "stats": constraint_stats,
                }
                validate_constraint_case_count(filtered)
                labeled, stats = await asyncio.to_thread(
                    label_cases,
                    filtered,
                    direct["code"],
                    brute["code"],
                    args.reference_timeout,
                )
                last_stats = stats
                stages["direct_brute_label"] = {
                    "cases": labeled,
                    "stats": stats,
                }
                recovery_code = None
                if sum(map(len, labeled.values())) < RECOVERY_EXPECTED_THRESHOLD:
                    disputed_inputs = unresolved_inputs(filtered, labeled)
                    recovery_prompt = build_recovery_prompt(problem, disputed_inputs)
                    async with semaphore:
                        recovery_response = await client.chat.completions.create(
                            model=args.model,
                            messages=[{"role": "user", "content": recovery_prompt}],
                            temperature=args.temperature,
                            top_p=0.95,
                            max_tokens=args.recovery_max_tokens,
                            timeout=args.request_timeout,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    recovery_choice = recovery_response.choices[0]
                    recovery_text = recovery_choice.message.content or ""
                    recovery_code = extract_code(
                        recovery_text,
                        require_fenced=recovery_choice.finish_reason == "length",
                    )
                    recovery_error, sample_count, _ = await asyncio.to_thread(
                        validate_reference,
                        problem,
                        recovery_code,
                        args.reference_timeout,
                    )
                    await trace(
                        {
                            "kind": "recovery_reference",
                            "problem_id": problem_id,
                            "attempt": attempt,
                            "max_tokens": args.recovery_max_tokens,
                            "prompt": recovery_prompt,
                            "response": recovery_text,
                            "finish_reason": recovery_choice.finish_reason,
                            "sample_count": sample_count,
                            "error": recovery_error,
                        }
                    )
                    if recovery_error:
                        raise ValueError(f"recovery reference failed validation: {recovery_error}")
                    labeled, stats = await asyncio.to_thread(
                        label_cases,
                        filtered,
                        direct["code"],
                        brute["code"],
                        args.reference_timeout,
                        recovery_code,
                        labeled,
                    )
                    last_stats = stats
                    stages["recovery_label"] = {
                        "cases": labeled,
                        "stats": stats,
                    }
                record = {
                    "problem_id": problem_id,
                    "status": "ok",
                    "attempt": attempt,
                    "generator_code": code,
                    "recovery_reference_code": recovery_code,
                    "test_cases": labeled,
                    "stages": stages,
                    "constraint_stats": constraint_stats,
                    "stats": stats,
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
                        "constraint_stats": constraint_stats,
                        "stats": stats,
                        "stages": stages,
                    }
                )
                await checkpoint(record)
                counts = "/".join(str(len(labeled[name])) for name in ("base", "corner"))
                print(f"{problem_id}: ok with base/corner={counts}", flush=True)
                return
            except (OpenAIError, OSError, ValueError, SyntaxError, IndexError) as exc:
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
                    "constraint_stats": last_constraint_stats,
                    "stats": last_stats,
                    "stages": stages,
                }
            )
            previous_code = code or previous_code
            print(f"{problem_id}: attempt {attempt} failed: {error}", flush=True)

        await checkpoint(
            {
                "problem_id": problem_id,
                "status": "failed",
                "attempt": args.retries + 1,
                "error": error,
                "failed_code": previous_code or None,
                "constraint_stats": last_constraint_stats,
                "stats": last_stats,
                "stages": last_stages,
                "model": args.model,
            }
        )

    await asyncio.gather(*(one(problem) for problem in problems))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retry-max-tokens", type=int, default=6144)
    parser.add_argument("--recovery-max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--generator-timeout", type=float, default=10.0)
    parser.add_argument("--constraint-timeout", type=float, default=5.0)
    parser.add_argument("--reference-timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if min(args.max_tokens, args.retry_max_tokens, args.recovery_max_tokens) < 1:
        parser.error("token limits must be positive")
    if (
        min(
            args.request_timeout,
            args.generator_timeout,
            args.constraint_timeout,
            args.reference_timeout,
        )
        <= 0
    ):
        parser.error("timeouts must be positive")
    if args.retry_max_tokens < args.max_tokens:
        parser.error("--retry-max-tokens must be at least --max-tokens")
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
