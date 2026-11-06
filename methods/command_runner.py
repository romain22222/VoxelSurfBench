from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Sequence
from typing import Any, IO


CommandPart = str | os.PathLike[str]
Stream = int | IO[Any] | None


def format_command(command: Sequence[CommandPart]) -> str:
    parts = [os.fspath(part) for part in command]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def run_command(
    executable: CommandPart,
    arguments: Sequence[CommandPart] = (),
    *,
    cwd: CommandPart | None = None,
    check: bool = True,
    echo: bool = True,
    capture_output: bool = False,
    text: bool = True,
    stdout: Stream = None,
    stderr: Stream = None,
) -> subprocess.CompletedProcess[Any]:
    command = [os.fspath(executable), *(os.fspath(argument) for argument in arguments)]
    if echo:
        print(f"+ {format_command(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=os.fspath(cwd) if cwd is not None else None,
        check=check,
        capture_output=capture_output,
        text=text,
        stdout=stdout,
        stderr=stderr,
    )
