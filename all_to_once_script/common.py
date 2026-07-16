from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def env_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name) or default).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    try:
        value = int(raw_value) if raw_value else default
    except ValueError as exc:
        raise SystemExit(f"{name} 必須是整數。") from exc
    if value < 1:
        raise SystemExit(f"{name} 必須大於 0。")
    return value
