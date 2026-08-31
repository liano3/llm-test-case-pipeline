"""Generate and validate direct and brute-force reference programs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from jsonl_io import append_jsonl
from reference_core import (
    PROFILES,
    Profile,
    build_prompt,
    extract_code,
    load_checkpoint,
    validate_reference,
)


async def generate(args: argparse.Namespace) -> None:
    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    end = None if args.limit is None else args.offset + args.limit
    problems = problems[args.offset : end]
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    checkpoint_lock = asyncio.Lock()
    trace_lock = asyncio.Lock()
    latest = load_checkpoint(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)

    async def checkpoint(record: dict[str, Any]) -> None:
        async with checkpoint_lock:
            append_jsonl(args.output, record)
            latest[str(record["problem_id"]), record["profile"]] = record

    async def trace(record: dict[str, Any]) -> None:
        if args.trace_output:
            async with trace_lock:
                append_jsonl(args.trace_output, record)

    async def validate(problem: dict[str, Any], code: str) -> tuple[str | None, int]:
        error, sample_count, _ = await asyncio.to_thread(
            validate_reference,
            problem,
            code,
            args.sample_timeout,
        )
        return error, sample_count

    async def one(problem: dict[str, Any], profile: Profile) -> None:
        problem_id = str(problem["id"])
        existing = latest.get((problem_id, profile.name))
        if existing and existing.get("status") == "ok":
            print(f"{problem_id}/{profile.name}: keeping checkpoint", flush=True)
            return

        previous_code = (existing.get("failed_code") or "") if existing else ""
        error = existing.get("error", "") if existing else ""
        truncated_response_length = 0
        truncated_response_tail = ""
        if previous_code:
            error, sample_count = await validate(problem, previous_code)
            if error is None:
                await checkpoint(
                    {
                        "problem_id": problem_id,
                        "profile": profile.name,
                        "status": "ok",
                        "attempt": existing.get("attempt", 0),
                        "sample_count": sample_count,
                        "code": previous_code,
                        "model": existing.get("model", args.model),
                        "thinking": existing.get("thinking", args.thinking),
                        "revalidated": True,
                    }
                )
                print(f"{problem_id}/{profile.name}: recovered checkpoint", flush=True)
                return

        for attempt in range(1, args.retries + 2):
            prompt = build_prompt(problem, profile, previous_code, error, code_only=attempt >= 3)
            request_max_tokens = args.max_tokens if attempt == 1 else args.retry_max_tokens
            code = ""
            raw_response = ""
            trace_response = ""
            finish_reason = None
            recovered_from_truncation = False
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature,
                        top_p=0.95,
                        max_tokens=request_max_tokens,
                        timeout=args.request_timeout,
                        extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking}},
                    )
                choice = response.choices[0]
                raw_response = choice.message.content or ""
                trace_response = raw_response
                finish_reason = choice.finish_reason
                if finish_reason == "length":
                    truncated_response_length = len(raw_response)
                    truncated_response_tail = raw_response[-1000:]
                    try:
                        code = extract_code(raw_response, require_fenced=True)
                        recovered_from_truncation = True
                    except (ValueError, SyntaxError):
                        raw_response = ""
                        raise ValueError(
                            "model response reached max_tokens and was truncated "
                            f"(characters={truncated_response_length})"
                        ) from None
                else:
                    code = extract_code(raw_response)
                error, sample_count = await validate(problem, code)
                if error is None:
                    await trace(
                        {
                            "kind": "generation",
                            "problem_id": problem_id,
                            "profile": profile.name,
                            "attempt": attempt,
                            "max_tokens": request_max_tokens,
                            "prompt": prompt,
                            "response": trace_response,
                            "finish_reason": finish_reason,
                            "outcome": "ok",
                            "code_length": len(code),
                            "recovered_from_truncation": recovered_from_truncation,
                        }
                    )
                    await checkpoint(
                        {
                            "problem_id": problem_id,
                            "profile": profile.name,
                            "status": "ok",
                            "attempt": attempt,
                            "sample_count": sample_count,
                            "code": code,
                            "model": args.model,
                            "thinking": args.thinking,
                            "recovered_from_truncation": recovered_from_truncation,
                        }
                    )
                    print(f"{problem_id}/{profile.name}: ok on attempt {attempt}", flush=True)
                    return
            except (OpenAIError, OSError, ValueError, SyntaxError, IndexError) as exc:
                error = str(exc)
            await trace(
                {
                    "kind": "generation",
                    "problem_id": problem_id,
                    "profile": profile.name,
                    "attempt": attempt,
                    "max_tokens": request_max_tokens,
                    "prompt": prompt,
                    "response": trace_response,
                    "finish_reason": finish_reason,
                    "outcome": "failed",
                    "error": error,
                    "code_length": len(code),
                    "recovered_from_truncation": recovered_from_truncation,
                }
            )
            previous_code = code or raw_response or previous_code
            print(f"{problem_id}/{profile.name}: attempt {attempt} failed: {error}", flush=True)

        await checkpoint(
            {
                "problem_id": problem_id,
                "profile": profile.name,
                "status": "failed",
                "attempt": args.retries + 1,
                "error": error,
                "failed_code": previous_code or None,
                "model": args.model,
                "thinking": args.thinking,
                "truncated_response_length": truncated_response_length or None,
                "truncated_response_tail": truncated_response_tail or None,
            }
        )

    await asyncio.gather(*(one(problem, profile) for problem in problems for profile in PROFILES))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--retry-max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--sample-timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        parser.error("offset must be non-negative and limit must be positive")
    if min(args.max_concurrency, args.max_tokens, args.retry_max_tokens) < 1:
        parser.error("concurrency and token limits must be positive")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if min(args.request_timeout, args.sample_timeout) <= 0:
        parser.error("timeouts must be positive")
    if args.retry_max_tokens < args.max_tokens:
        parser.error("--retry-max-tokens must be at least --max-tokens")
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
