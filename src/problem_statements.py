"""Parse examples and output requirements from problem statements."""

from __future__ import annotations

import html
import re


def _heading(line: str) -> str:
    heading = line.strip().strip("#").strip().strip("-").strip()
    heading = re.sub(r"\s*\[[^]]+]\s*$", "", heading)
    return heading.strip().strip("-").strip().casefold()


def extract_examples(statement: str) -> list[dict[str, str]]:
    """Extract input/output pairs from the statement's example section."""
    example_headings = {"example", "examples", "sample", "samples", "пример", "примеры"}
    input_headings = {"input", "входные данные"}
    output_headings = {"output", "выходные данные"}
    end_headings = {
        "note",
        "notes",
        "explanation",
        "explanations",
        "примечание",
        "примечания",
    }
    lines = statement.splitlines()
    example_index = next(
        (index for index, line in enumerate(lines) if _heading(line) in example_headings),
        None,
    )
    if example_index is None:
        return []

    examples = []
    cursor = example_index + 1
    while cursor < len(lines):
        input_index = next(
            (index for index in range(cursor, len(lines)) if _heading(lines[index]) in input_headings),
            None,
        )
        if input_index is None:
            break
        output_index = next(
            (
                index
                for index in range(input_index + 1, len(lines))
                if _heading(lines[index]) in input_headings | output_headings
            ),
            None,
        )
        if output_index is None or _heading(lines[output_index]) not in output_headings:
            break
        end = next(
            (
                index
                for index in range(output_index + 1, len(lines))
                if _heading(lines[index]) in input_headings or _heading(lines[index]) in end_headings
            ),
            len(lines),
        )
        stdin = html.unescape("\n".join(lines[input_index + 1 : output_index]).strip()) + "\n"
        expected = html.unescape("\n".join(lines[output_index + 1 : end]).strip())
        if stdin.strip() and expected:
            examples.append({"input": stdin, "expected": expected})
        cursor = end
    return examples
