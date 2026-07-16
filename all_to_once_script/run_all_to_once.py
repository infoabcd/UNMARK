#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys

from common import ROOT, env_path, load_env


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    load_env()
    run([sys.executable, "all_to_once_script/import_inputs.py"])
    input_csv = env_path("UNMARK_ALL_TO_ONCE_WORK_DIR", "all_to_once_script/work") / "input.csv"
    run([sys.executable, "run_unmark.py", "run", "--input-csv", str(input_csv)])
    run([sys.executable, "all_to_once_script/export_outputs.py"])


if __name__ == "__main__":
    main()
