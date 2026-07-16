#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

from common import env_path, load_env


def latest_job(jobs_dir: Path) -> Path:
    if not jobs_dir.is_dir():
        raise SystemExit(f"找不到 Jobs 目錄：{jobs_dir}")
    jobs = [path for path in jobs_dir.iterdir() if path.is_dir()]
    if not jobs:
        raise SystemExit(f"找不到 job：{jobs_dir}")
    return sorted(jobs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def export_job(job_dir: Path, export_root: Path) -> Path:
    target = export_root / job_dir.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for name in ("output.csv", "remote_result.json", "job_state.json"):
        src = job_dir / name
        if src.exists():
            shutil.copy2(src, target / name)

    output_images = job_dir / "output_images"
    if output_images.exists():
        shutil.copytree(output_images, target / "output_images")
    return target


def main() -> None:
    load_env()
    jobs_dir = env_path("UNMARK_JOBS_DIR", "jobs")
    export_root = env_path("UNMARK_ALL_TO_ONCE_EXPORT_DIR", "all_to_once_script/export")
    job_id = os.environ.get("UNMARK_ALL_TO_ONCE_JOB_ID", "").strip()
    job_dir = (jobs_dir / job_id).resolve() if job_id else latest_job(jobs_dir)
    if not job_dir.is_dir():
        raise SystemExit(f"找不到 job 目錄：{job_dir}")
    target = export_job(job_dir, export_root)
    print(f"已匯出 job：{job_dir.name}")
    print(f"匯出位置：{target}")


if __name__ == "__main__":
    main()
