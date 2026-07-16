#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from common import ROOT, env_path, load_env, positive_int_env


SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unmark.input_processing import normalize_uploaded_files


def collect_files(import_dir: Path) -> list[tuple[str, Path]]:
    files = []
    for path in sorted(import_dir.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            try:
                name = path.relative_to(import_dir).as_posix()
            except ValueError:
                name = path.name
            files.append((name, path))
    return files


def main() -> None:
    load_env()
    import_dir = env_path("UNMARK_ALL_TO_ONCE_IMPORT_DIR", "all_to_once_script/import")
    work_dir = env_path("UNMARK_ALL_TO_ONCE_WORK_DIR", "all_to_once_script/work")
    if not import_dir.exists():
        raise SystemExit(f"找不到匯入目錄：{import_dir}")
    files = collect_files(import_dir)
    if not files:
        raise SystemExit(f"匯入目錄未有檔案：{import_dir}")

    if work_dir.exists():
        for child in work_dir.iterdir():
            if child.name != ".gitkeep":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    work_dir.mkdir(parents=True, exist_ok=True)
    max_total_bytes = positive_int_env("UNMARK_MAX_EMBED_MB", 2000) * 1024 * 1024
    input_csv = normalize_uploaded_files(work_dir, files, max_total_bytes)
    print(f"已匯入：{len(files)} 個檔案")
    print(f"已建立標準輸入 CSV：{input_csv}")
    print(f"下一步可執行：python3 run_unmark.py run --input-csv {input_csv}")


if __name__ == "__main__":
    main()
