#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def running_inside_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def maybe_reexec_with_project_venv() -> None:
    if os.environ.get("UNMARK_NO_AUTO_VENV") == "1":
        return
    if running_inside_venv() or not VENV_PYTHON.exists():
        return
    if Path(sys.executable).absolute() == VENV_PYTHON.absolute():
        return
    os.environ["UNMARK_USING_PROJECT_VENV"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


maybe_reexec_with_project_venv()

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unmark.cli import main


if __name__ == "__main__":
    main()
