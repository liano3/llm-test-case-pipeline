"""Shared JSONL helpers for checkpoints, traces, and exported cases."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, *, tolerate_malformed: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record is not a JSON object")
            except (json.JSONDecodeError, TypeError) as exc:
                if not tolerate_malformed:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
                warnings.warn(
                    f"ignoring malformed checkpoint record at {path}:{line_number}: {exc}",
                    stacklevel=2,
                )
                continue
            records.append(record)
    return records


def load_latest_by_problem(
    path: Path,
    selected_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    latest = {}
    for record in read_jsonl(path, tolerate_malformed=True):
        if "problem_id" not in record:
            continue
        problem_id = str(record["problem_id"])
        if selected_ids is None or problem_id in selected_ids:
            latest[problem_id] = record
    return latest


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    with path.open("ab+") as output:
        output.seek(0, 2)
        if output.tell():
            output.seek(-1, 2)
            if output.read(1) != b"\n":
                output.seek(0, 2)
                output.write(b"\n")
        output.seek(0, 2)
        output.write(payload)
        output.flush()
