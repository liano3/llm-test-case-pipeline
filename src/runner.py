"""Run generated Python programs with resource limits.

This limits accidental resource use but is not a security boundary. Execute
untrusted programs inside a container or VM with network and filesystem isolation.
"""

from __future__ import annotations

import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    status: str
    output: str
    error: str = ""


def check_equal(actual: str, expected: str) -> bool:
    return actual.split() == expected.split()


def run_solution(code: str, stdin: str, timeout: float) -> Result:
    with tempfile.TemporaryDirectory(prefix="casegen-") as directory:
        source = Path(directory) / "solution.py"
        source.write_text(code, encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--sandbox",
                str(max(1, math.ceil(timeout))),
                str(source),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            output, error = process.communicate(stdin, timeout=timeout + 0.5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return Result("timeout", "", f"exceeded {timeout}s")
        status = "ok" if process.returncode == 0 else "error"
        return Result(status, output.strip(), error[-2000:])


def run_sandbox() -> None:
    seconds = int(sys.argv[2])
    source = sys.argv[3]
    resource.setrlimit(resource.RLIMIT_CPU, (seconds + 1, seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
    os.execv(sys.executable, [sys.executable, "-I", source])


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--sandbox":
    run_sandbox()
