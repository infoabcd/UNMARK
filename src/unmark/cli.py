#!/usr/bin/env python3
"""
Vast.ai GPU 批次去水印（多模型）。

讀取 input.csv（original_image 欄位）或 input_images/*，
在遠端 GPU 實例執行，下載結果到 jobs/<job-id>/output_images/，
並寫入 jobs/<job-id>/output.csv。

模型預設值請見 model_presets.json 或 README「模型對照」。
"""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import os
import secrets
import shlex
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

try:
    import requests
except ModuleNotFoundError:
    requests = None

REMOTE_HTTP = requests.Session() if requests is not None else None
if REMOTE_HTTP is not None:
    # Vast 的臨時 HTTP 端點應直接連線，避免 macOS 系統代理令圖片傳輸卡住。
    REMOTE_HTTP.trust_env = False

# ======================== 設定 ========================
# 模型快捷鍵：9B（預設）| 4B | KONTEXT | KWR。詳見 model_presets.json / README。
MODEL = "9B"

# 推理步數；預設值見 model_presets.json。較高數值通常較慢，但不保證品質較好。
STEPS = 8
SEED = 42
GPU_NAME = "H200"
DISK_GB = 80
MAX_SIDE = 1024
MAX_DOWNLOAD_MB = 100
PROMPT = ""
DEFAULT_PROMPT_PRESET = "萬能提示詞"
PROMPT_PRESET = DEFAULT_PROMPT_PRESET
PROMPT_MAX_CHARS = 20_000
INSTANCE_LABEL = "unmark-batch"
MAX_EMBED_MB = 2000  # 每個批次本地圖片的總上傳上限（MB）
BATCH_SIZE = 100
CONCURRENCY = 12  # 輸入下載及準備工作的並行數；GPU 推理固定逐張執行
DESTROY_ON_SUCCESS = True
JOB_TIMEOUT_S = 7200
BOOTSTRAP_TIMEOUT_S = 900
INSTANCE_TTL_S = 10800
REMOTE_RESULT_GRACE_S = 900
REMOTE_WORK_DIR = "/workspace/unmark"
VAST_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DIFFUSERS_COMMIT = "273f337137e329eb0268058389f17608e5f5b633"
TRANSFORMERS_COMMIT = "e52d0fd6fa9eb874f7c2da048198276b04c919b9"
# ==================================================================

ROOT = Path(__file__).resolve().parents[2]
MODEL_PRESETS_PATH = ROOT / "model_presets.json"
PROMPTS_DIR = ROOT / "Prompts"
INPUT_CSV = ROOT / "input.csv"
INPUT_BASE_DIR = ROOT
INPUT_IMAGES_DIR = ROOT / "input_images"
JOBS_DIR = ROOT / "jobs"
JOB_DIR = ROOT
OUTPUT_IMAGES_DIR = JOB_DIR / "output_images"
OUTPUT_CSV = JOB_DIR / "output.csv"
REMOTE_RESULT_JSON = JOB_DIR / "remote_result.json"
REMOTE_WORKER = ROOT / "remote_worker.py"
VAST_API = "https://console.vast.ai/api/v0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VENV_PYTHON = ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def load_env_file(path: Path = Path(".env")) -> None:
    env_path = ROOT / path
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(f"UNMARK_{name}", os.environ.get(name, default)).strip()


class PromptPresetError(ValueError):
    pass


def load_prompt_presets(prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    if not prompts_dir.is_dir():
        raise PromptPresetError(f"找不到提示詞目錄：{prompts_dir}")
    presets: dict[str, str] = {}
    for path in sorted(prompts_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        name = path.stem.strip()
        try:
            content = path.read_text(encoding="utf-8-sig").strip()
        except UnicodeError as exc:
            raise PromptPresetError(f"提示詞檔案不是有效 UTF-8：{path}") from exc
        if not name or not content:
            raise PromptPresetError(f"提示詞檔案不可留空：{path}")
        if len(content) > PROMPT_MAX_CHARS:
            raise PromptPresetError(f"提示詞檔案超過 {PROMPT_MAX_CHARS:,} 個字元：{path}")
        presets[name] = content
    if not presets:
        raise PromptPresetError(f"提示詞目錄內沒有可用的 .md 檔案：{prompts_dir}")
    if DEFAULT_PROMPT_PRESET in presets:
        presets = {
            DEFAULT_PROMPT_PRESET: presets[DEFAULT_PROMPT_PRESET],
            **{name: content for name, content in presets.items() if name != DEFAULT_PROMPT_PRESET},
        }
    return presets


def resolve_prompt_preset(name: str) -> tuple[str, str]:
    try:
        presets = load_prompt_presets()
    except PromptPresetError as exc:
        raise SystemExit(str(exc)) from exc
    selected = name.strip() or PROMPT_PRESET
    if selected not in presets:
        available = "、".join(presets)
        raise SystemExit(f"找不到提示詞「{selected}」。可選：{available}")
    return selected, presets[selected]


def env_has_value(name: str) -> bool:
    return bool(
        os.environ.get(f"UNMARK_{name}", "").strip()
        or os.environ.get(name, "").strip()
    )


def env_int(name: str, default: int) -> int:
    raw = env_str(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"環境變數 {name} 必須是整數，目前值：{raw!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    raw = env_str(name, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"環境變數 {name} 必須是 true 或 false，目前值：{raw!r}")


def arg_or_env(args: argparse.Namespace, attr: str, env_name: str, default):
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if isinstance(default, bool):
        return env_bool(env_name, default)
    if isinstance(default, int):
        return env_int(env_name, default)
    return env_str(env_name, str(default))


def apply_runtime_config(args: argparse.Namespace) -> None:
    """套用 .env、環境變數及 CLI 參數到本次執行。"""
    load_env_file()
    global MODEL, STEPS, SEED, GPU_NAME, DISK_GB, MAX_SIDE, MAX_DOWNLOAD_MB, PROMPT, PROMPT_PRESET
    global INSTANCE_LABEL, MAX_EMBED_MB
    global BATCH_SIZE, CONCURRENCY, DESTROY_ON_SUCCESS
    global JOB_TIMEOUT_S, BOOTSTRAP_TIMEOUT_S, INSTANCE_TTL_S, REMOTE_RESULT_GRACE_S
    global REMOTE_WORK_DIR, VAST_IMAGE, VAST_API
    global INPUT_CSV, INPUT_BASE_DIR, INPUT_IMAGES_DIR, JOBS_DIR

    MODEL = str(arg_or_env(args, "model", "MODEL", MODEL)).upper()
    MODEL, model_cfg = resolve_model(MODEL)
    default_steps = int(model_cfg.get("default_steps", STEPS))
    if getattr(args, "steps", None) is not None or env_has_value("STEPS"):
        STEPS = int(arg_or_env(args, "steps", "STEPS", default_steps))
    else:
        STEPS = default_steps
    SEED = int(arg_or_env(args, "seed", "SEED", SEED))
    GPU_NAME = str(arg_or_env(args, "gpu_name", "GPU_NAME", GPU_NAME))
    DISK_GB = int(arg_or_env(args, "disk_gb", "DISK_GB", DISK_GB))
    MAX_SIDE = int(arg_or_env(args, "max_side", "MAX_SIDE", MAX_SIDE))
    MAX_DOWNLOAD_MB = int(arg_or_env(args, "max_download_mb", "MAX_DOWNLOAD_MB", MAX_DOWNLOAD_MB))
    PROMPT_PRESET = str(arg_or_env(args, "prompt_preset", "PROMPT_PRESET", PROMPT_PRESET))
    PROMPT_PRESET, PROMPT = resolve_prompt_preset(PROMPT_PRESET)
    INSTANCE_LABEL = str(arg_or_env(args, "instance_label", "INSTANCE_LABEL", INSTANCE_LABEL))
    MAX_EMBED_MB = int(arg_or_env(args, "max_embed_mb", "MAX_EMBED_MB", MAX_EMBED_MB))
    BATCH_SIZE = int(arg_or_env(args, "batch_size", "BATCH_SIZE", BATCH_SIZE))
    CONCURRENCY = int(arg_or_env(args, "concurrency", "CONCURRENCY", CONCURRENCY))
    DESTROY_ON_SUCCESS = bool(arg_or_env(args, "destroy_on_success", "DESTROY_ON_SUCCESS", DESTROY_ON_SUCCESS))
    JOB_TIMEOUT_S = int(arg_or_env(args, "job_timeout", "JOB_TIMEOUT_S", JOB_TIMEOUT_S))
    BOOTSTRAP_TIMEOUT_S = int(arg_or_env(args, "bootstrap_timeout", "BOOTSTRAP_TIMEOUT_S", BOOTSTRAP_TIMEOUT_S))
    INSTANCE_TTL_S = int(arg_or_env(args, "instance_ttl", "INSTANCE_TTL_S", INSTANCE_TTL_S))
    REMOTE_RESULT_GRACE_S = int(
        arg_or_env(args, "remote_result_grace", "REMOTE_RESULT_GRACE_S", REMOTE_RESULT_GRACE_S)
    )
    REMOTE_WORK_DIR = str(arg_or_env(args, "remote_work_dir", "REMOTE_WORK_DIR", REMOTE_WORK_DIR))
    VAST_IMAGE = str(arg_or_env(args, "vast_image", "VAST_IMAGE", VAST_IMAGE))
    VAST_API = str(arg_or_env(args, "vast_api", "VAST_API", VAST_API))

    positive_values = {
        "STEPS": STEPS,
        "DISK_GB": DISK_GB,
        "MAX_SIDE": MAX_SIDE,
        "MAX_DOWNLOAD_MB": MAX_DOWNLOAD_MB,
        "MAX_EMBED_MB": MAX_EMBED_MB,
        "BATCH_SIZE": BATCH_SIZE,
        "CONCURRENCY": CONCURRENCY,
        "JOB_TIMEOUT_S": JOB_TIMEOUT_S,
        "BOOTSTRAP_TIMEOUT_S": BOOTSTRAP_TIMEOUT_S,
        "INSTANCE_TTL_S": INSTANCE_TTL_S,
        "REMOTE_RESULT_GRACE_S": REMOTE_RESULT_GRACE_S,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise SystemExit(f"以下設定必須大於 0：{', '.join(invalid)}")
    if SEED < 0:
        raise SystemExit("SEED 不可小於 0")

    input_csv = arg_or_env(args, "input_csv", "INPUT_CSV", str(INPUT_CSV))
    input_images_dir = arg_or_env(args, "input_images_dir", "INPUT_IMAGES_DIR", str(INPUT_IMAGES_DIR))
    jobs_dir = arg_or_env(args, "jobs_dir", "JOBS_DIR", str(JOBS_DIR))
    INPUT_CSV = Path(input_csv).expanduser()
    if not INPUT_CSV.is_absolute():
        INPUT_CSV = (ROOT / INPUT_CSV).resolve()
    INPUT_BASE_DIR = INPUT_CSV.parent if INPUT_CSV.exists() else ROOT
    INPUT_IMAGES_DIR = Path(input_images_dir).expanduser()
    if not INPUT_IMAGES_DIR.is_absolute():
        INPUT_IMAGES_DIR = (ROOT / INPUT_IMAGES_DIR).resolve()
    JOBS_DIR = Path(jobs_dir).expanduser()
    if not JOBS_DIR.is_absolute():
        JOBS_DIR = (ROOT / JOBS_DIR).resolve()


def runtime_config_snapshot() -> dict:
    return {
        "model": MODEL,
        "steps": STEPS,
        "seed": SEED,
        "gpu_name": GPU_NAME,
        "disk_gb": DISK_GB,
        "max_side": MAX_SIDE,
        "max_download_mb": MAX_DOWNLOAD_MB,
        "prompt": PROMPT,
        "prompt_preset": PROMPT_PRESET,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "instance_label": INSTANCE_LABEL,
        "max_embed_mb": MAX_EMBED_MB,
        "batch_size": BATCH_SIZE,
        "concurrency": CONCURRENCY,
        "destroy_on_success": DESTROY_ON_SUCCESS,
        "destroy_on_error": True,
        "job_timeout_s": JOB_TIMEOUT_S,
        "bootstrap_timeout_s": BOOTSTRAP_TIMEOUT_S,
        "instance_ttl_s": INSTANCE_TTL_S,
        "remote_result_grace_s": REMOTE_RESULT_GRACE_S,
        "remote_work_dir": REMOTE_WORK_DIR,
        "vast_image": VAST_IMAGE,
        "input_csv": str(INPUT_CSV),
        "input_images_dir": str(INPUT_IMAGES_DIR),
        "jobs_dir": str(JOBS_DIR),
    }


def new_job_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def prepare_job_dir(args: argparse.Namespace, resume: bool = False) -> Path:
    raw_job_dir = getattr(args, "job_dir", None)
    raw_job_id = getattr(args, "job_id", None)
    if raw_job_dir:
        job_dir = Path(raw_job_dir).expanduser()
        if not job_dir.is_absolute():
            job_dir = (ROOT / job_dir).resolve()
    else:
        job_id = raw_job_id or new_job_id()
        job_dir = JOBS_DIR / job_id
    if resume and not job_dir.exists():
        raise SystemExit(f"找不到 job 目錄：{job_dir}")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "output_images").mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)
    return job_dir


def apply_job_paths(job_dir: Path) -> None:
    global JOB_DIR, OUTPUT_IMAGES_DIR, OUTPUT_CSV, REMOTE_RESULT_JSON
    JOB_DIR = job_dir
    OUTPUT_IMAGES_DIR = JOB_DIR / "output_images"
    OUTPUT_CSV = JOB_DIR / "output.csv"
    REMOTE_RESULT_JSON = JOB_DIR / "remote_result.json"


def read_job_state(job_dir: Path) -> dict:
    path = job_dir / "job_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: dict | list) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_remote_session(job_dir: Path, remote_api_token: str) -> None:
    path = job_dir / ".remote_session.json"
    atomic_write_json(path, {"remote_api_token": remote_api_token})
    path.chmod(0o600)


def read_remote_session(job_dir: Path) -> str:
    path = job_dir / ".remote_session.json"
    if not path.is_file():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("remote_api_token", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def write_job_state(job_dir: Path, updates: dict) -> dict:
    state = read_job_state(job_dir)
    state.update(updates)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_json(job_dir / "job_state.json", state)
    return state


def append_job_event(job_dir: Path, event: str, payload: dict | None = None) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "payload": payload or {},
    }
    with (job_dir / "events.jsonl").open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def snapshot_input(job_dir: Path) -> None:
    if INPUT_CSV.exists():
        target = job_dir / "input.csv"
        if INPUT_CSV.resolve() != target.resolve():
            shutil.copy2(INPUT_CSV, target)
    atomic_write_json(job_dir / "runtime_config.json", runtime_config_snapshot())


def write_input_manifest(job_dir: Path, items: list[dict]) -> None:
    manifest_path = job_dir / "input_manifest.jsonl"
    tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as manifest_file:
        for item in items:
            manifest_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_path, manifest_path)


def load_job_input_items(job_dir: Path) -> list[dict]:
    manifest_path = job_dir / "input_manifest.jsonl"
    if not manifest_path.exists():
        return []
    items = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if line.strip():
                items.append(json.loads(line))
    return items


def snapshot_local_inputs(job_dir: Path, items: list[dict]) -> list[dict]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(dependency_install_hint("Pillow")) from exc
    input_dir = job_dir / "input_files"
    for item in items:
        item.setdefault(
            "input_key",
            f"{int(item['index']):06d}:{hashlib.sha1(str(item['original_ref']).encode('utf-8')).hexdigest()[:12]}",
        )
        if item.get("source_type") != "local":
            continue
        source = Path(str(item.get("local_path_resolved", "")))
        if not source.is_file():
            raise SystemExit(f"job 原圖不存在，未有建立 Vast 實例：{source}")
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / f"{int(item['index']):04d}_{source.name}"
        if source.resolve() != target.resolve() and not target.exists():
            shutil.copy2(source, target)
        try:
            with Image.open(target) as image:
                image.verify()
        except Exception as exc:
            raise SystemExit(f"圖片無法讀取，未有建立 Vast 實例：{source}（{exc}）") from exc
        item["filename"] = target.name
        item["local_path_resolved"] = str(target.resolve())
        item["size_bytes"] = target.stat().st_size
        item["sha256"] = sha256_file(target)
    return items


def validate_output_names(items: list[dict]) -> None:
    seen: dict[str, str] = {}
    for item in items:
        document_id = str(item.get("document_id", "")).strip()
        if not document_id:
            continue
        normalized = "".join(char if char.isalnum() or char in "-_" else "_" for char in document_id)
        normalized = normalized.strip("._")[:80]
        if not normalized:
            raise SystemExit(f"文件名 {document_id!r} 沒有可用字元，未有建立 Vast 實例。")
        if normalized in seen:
            raise SystemExit(
                f"文件名重複：{document_id!r} 與 {seen[normalized]!r} 會產生相同輸出檔名。"
                "請先改成不同名稱；未有建立 Vast 實例。"
            )
        seen[normalized] = document_id


def existing_completed_count() -> int:
    return sum(1 for info in load_completed_jobs().values() if info.get("status") == "ok")


class JobTimeoutError(TimeoutError):
    def __init__(self, summary: dict | None = None):
        self.summary = mark_timeout_partial(summary or {})
        super().__init__("等待遠端任務完成逾時")


class InstanceCreateUncertainError(RuntimeError):
    """建立請求可能已到達 Vast，但本地未收到完整回覆。"""


def mark_timeout_partial(summary: dict) -> dict:
    payload = dict(summary)
    payload["status"] = "timeout_partial"
    return payload


def vast_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def vast_request_with_retries(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    **kwargs,
):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Vast API 連線失敗（已重試 {attempts} 次）：{last_error}") from last_error


def require_requests() -> None:
    if requests is None:
        raise SystemExit(dependency_install_hint("requests"))


def dependency_install_hint(package_name: str) -> str:
    if VENV_PYTHON.exists():
        install_cmd = f"{VENV_PYTHON} -m pip install -r requirements.txt"
        run_cmd = f"{VENV_PYTHON} run_unmark.py"
    else:
        install_cmd = f"{sys.executable} -m pip install -r requirements.txt"
        run_cmd = f"{sys.executable} run_unmark.py"
    return (
        f"缺少本地依賴 {package_name}。\n"
        f"目前使用的 Python：{sys.executable}\n"
        f"請在項目根目錄執行：{install_cmd}\n"
        f"之後用同一個 Python 啟動：{run_cmd}"
    )


def load_model_presets() -> dict:
    return json.loads(MODEL_PRESETS_PATH.read_text(encoding="utf-8"))


def resolve_model(model_key: str) -> tuple[str, dict]:
    presets = load_model_presets()
    key = model_key.strip().upper()
    if key not in presets:
        available = ", ".join(sorted(presets))
        raise SystemExit(f"未知 MODEL={model_key!r}，可選: {available}")
    return key, presets[key]


def print_model_catalog() -> None:
    presets = load_model_presets()
    print("可用模型對照：")
    for key, cfg in presets.items():
        print(f"  {key:7s}  {cfg['label']}")
        print(f"       底座模型：{cfg.get('base_hf_url') or cfg['hf_url']}")
        if cfg.get("adapter_hf_url"):
            print(f"       LoRA：{cfg['adapter_hf_url']}")
        print(f"       建議 steps={cfg['default_steps']}  guidance={cfg['guidance_scale']}")


def print_prompt_catalog() -> None:
    print("可用提示詞：")
    try:
        presets = load_prompt_presets()
    except PromptPresetError as exc:
        raise SystemExit(str(exc)) from exc
    for name in presets:
        suffix = "（預設）" if name == DEFAULT_PROMPT_PRESET else ""
        print(f"  {name}{suffix}")


def collect_inputs() -> list[dict]:
    if INPUT_CSV.exists():
        with INPUT_CSV.open(encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if not reader.fieldnames:
                raise SystemExit("input.csv 欄位解析失敗，不應為空")

            image_col = None
            for col in ("original_image", "original_url", "image", "url"):
                if col in reader.fieldnames:
                    image_col = col
                    break

            if not image_col:
                raise SystemExit("input.csv 必須包含 original_image 或 original_url 欄位")

            items = []
            for index, row in enumerate(reader, start=1):
                image_reference = (row.get(image_col) or "").strip()
                if not image_reference:
                    continue
                document_id = (row.get("id") or row.get("document_id") or "").strip()
                items.append(build_item(index, image_reference, document_id=document_id))
            if items:
                print(f"從 {INPUT_CSV.name} 讀取 {len(items)} 條")
                return items

    if INPUT_IMAGES_DIR.exists():
        files = sorted(
            path
            for path in INPUT_IMAGES_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            items = []
            for index, path in enumerate(files, start=1):
                try:
                    relative_path = path.relative_to(INPUT_BASE_DIR).as_posix()
                except ValueError:
                    relative_path = path.name
                items.append(build_item(index, relative_path, local_path=path))
            print(f"從 input_images/ 讀取 {len(items)} 張圖片")
            return items

    raise SystemExit("找不到輸入：請提供 input.csv（含 original_image 欄位）或 input_images/ 下的圖片")


def build_item(
    index: int,
    original_ref: str,
    local_path: Path | None = None,
    document_id: str = "",
) -> dict:
    input_key = f"{index:06d}:{hashlib.sha1(original_ref.encode('utf-8')).hexdigest()[:12]}"
    if local_path is None:
        candidate = INPUT_BASE_DIR / original_ref
        if candidate.exists() and candidate.is_file():
            local_path = candidate
        else:
            fallback = ROOT / original_ref
            if fallback.exists() and fallback.is_file():
                local_path = fallback

    if local_path is not None and local_path.exists():
        item = {
            "index": index,
            "original_ref": original_ref.replace("\\", "/"),
            "source_type": "local",
            "filename": local_path.name,
            "local_path_resolved": str(local_path.resolve()),
            "size_bytes": local_path.stat().st_size,
            "input_key": input_key,
        }
        if document_id:
            item["document_id"] = document_id
        return item

    if original_ref.startswith(("http://", "https://")):
        item = {
            "index": index,
            "original_ref": original_ref,
            "source_type": "url",
            "input_key": input_key,
        }
        if document_id:
            item["document_id"] = document_id
        return item

    raise SystemExit(f"無法解析輸入：{original_ref}（不是有效 URL，本地檔案也不存在）")


def build_manifest(items: list[dict], steps: int, model_key: str, model_cfg: dict) -> dict:
    embed_bytes = sum(
        int(item.get("size_bytes", 0))
        for item in items
        if item.get("source_type") == "local"
    )
    limit = MAX_EMBED_MB * 1024 * 1024
    if embed_bytes > limit:
        mb = embed_bytes / 1024 / 1024
        raise SystemExit(
            f"本地圖片總大小 {mb:.1f} MB 超過 MAX_EMBED_MB={MAX_EMBED_MB}。"
            "請減少圖片數量或增加 MAX_EMBED_MB 限制。"
        )
    clean_items = []
    for item in items:
        payload = {k: v for k, v in item.items() if k != "local_path_resolved"}
        clean_items.append(payload)
    prompt_preset = PROMPT_PRESET
    prompt = PROMPT
    if not prompt:
        prompt_preset, prompt = resolve_prompt_preset(prompt_preset)
    return {
        "model": model_key,
        "model_config": model_cfg,
        "steps": steps,
        "seed": SEED,
        "max_side": MAX_SIDE,
        "max_download_bytes": MAX_DOWNLOAD_MB * 1024 * 1024,
        "prompt": prompt,
        "prompt_preset": prompt_preset,
        "items": clean_items,
        "concurrency": CONCURRENCY,
    }


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [items[start : start + size] for start in range(0, len(items), size)]


def search_gpu_offer(api_key: str, gpu_name: str) -> tuple[int, float]:
    variants = [
        gpu_name,
        f"NVIDIA {gpu_name}",
        f"NVIDIA-{gpu_name}",
        f"{gpu_name}_SXM",
        f"NVIDIA {gpu_name} SXM"
    ]
    # 移除重複項並保持順序
    variants = list(dict.fromkeys(variants))

    payload = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "gpu_name": {"in": variants},
        "num_gpus": {"eq": 1},
        "direct_port_count": {"gte": 1},
        "disk_space": {"gte": DISK_GB},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
        "limit": 5,
    }
    resp = vast_request_with_retries(
        "POST",
        f"{VAST_API}/bundles/",
        headers=vast_headers(api_key),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    offers = resp.json().get("offers", [])
    if not offers:
        raise RuntimeError(f"找不到可租用的 {gpu_name} 實例 (搜尋變體: {variants})")
    offer = offers[0]
    return int(offer["id"]), float(offer["dph_total"])


def build_bootstrap_server() -> str:
    return (ROOT / "src" / "unmark" / "remote_bootstrap.py").read_text(encoding="utf-8")


def build_onstart(hf_token: str, remote_api_token: str) -> str:
    work_dir = REMOTE_WORK_DIR
    quoted_work_dir = shlex.quote(work_dir)
    bootstrap_b64 = base64.b64encode(
        gzip.compress(build_bootstrap_server().encode("utf-8"), compresslevel=9)
    ).decode("ascii")
    return f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_TOKEN={shlex.quote(hf_token)}
export HUGGING_FACE_HUB_TOKEN={shlex.quote(hf_token)}
export REMOTE_API_TOKEN={shlex.quote(remote_api_token)}
export MAX_UPLOAD_BYTES={MAX_EMBED_MB * 1024 * 1024}
export INSTANCE_TTL_S={INSTANCE_TTL_S}
export REMOTE_RESULT_GRACE_S={REMOTE_RESULT_GRACE_S}
export REMOTE_START_TTL_S={BOOTSTRAP_TIMEOUT_S + 300}
export WORK_DIR={quoted_work_dir}
mkdir -p {quoted_work_dir}/output_images
mkdir -p {quoted_work_dir}/raw_images
cd {quoted_work_dir}
echo {shlex.quote(bootstrap_b64)} | base64 -d | gzip -d > {quoted_work_dir}/bootstrap_server.py
python3 {quoted_work_dir}/bootstrap_server.py > {quoted_work_dir}/bootstrap.log 2>&1 &

(
  echo "========== 開始安裝 Python 依賴環境 =========="
  if pip install -q --upgrade pip && \
     pip install -q git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_COMMIT} git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_COMMIT} accelerate safetensors pillow requests tqdm sentencepiece protobuf peft; then
    touch {quoted_work_dir}/.installed
    echo "========== 依賴環境安裝完成 =========="
  else
    touch {quoted_work_dir}/.install_failed
    echo "========== 依賴環境安裝失敗 =========="
  fi
) > {quoted_work_dir}/install.log 2>&1 &
"""


def create_instance(
    api_key: str,
    offer_id: int,
    onstart: str,
    hf_token: str,
    remote_api_token: str,
    label: str,
    disk_gb: int,
) -> int:
    payload = {
        "image": VAST_IMAGE,
        "disk": disk_gb,
        "label": label,
        "runtype": "ssh",
        "ssh": True,
        "direct": True,
        "onstart": onstart,
        "env": {
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
            "REMOTE_API_TOKEN": remote_api_token,
            "MAX_UPLOAD_BYTES": str(MAX_EMBED_MB * 1024 * 1024),
            "INSTANCE_TTL_S": str(INSTANCE_TTL_S),
            "REMOTE_RESULT_GRACE_S": str(REMOTE_RESULT_GRACE_S),
            "REMOTE_START_TTL_S": str(BOOTSTRAP_TIMEOUT_S + 300),
            "-p 8080:8080": "1",
        },
    }
    try:
        resp = requests.put(
            f"{VAST_API}/asks/{offer_id}/",
            headers=vast_headers(api_key),
            json=payload,
            timeout=60,
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise InstanceCreateUncertainError(f"建立實例時連線中斷：{exc}") from exc
    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text[:1000]}
    if not resp.ok:
        raise RuntimeError(f"建立實例被 Vast 拒絕（HTTP {resp.status_code}）：{data}")
    if not data.get("success"):
        raise RuntimeError(f"建立實例失敗: {data}")
    return int(data["new_contract"])


def get_instance(api_key: str, instance_id: int) -> dict:
    resp = vast_request_with_retries(
        "GET",
        f"{VAST_API}/instances/{instance_id}/",
        headers=vast_headers(api_key),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("instances", {})


def set_instance_state(api_key: str, instance_id: int, state: str) -> bool:
    if state not in {"running", "stopped"}:
        raise ValueError(f"不支援的 Vast 實例狀態：{state}")
    resp = vast_request_with_retries(
        "PUT",
        f"{VAST_API}/instances/{instance_id}/",
        headers=vast_headers(api_key),
        json={"state": state},
        timeout=60,
    )
    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text[:1000]}
    if resp.ok and data.get("success"):
        return False
    queued = (
        resp.ok
        and data.get("error") == "resources_unavailable"
        and "queued" in str(data.get("msg", "")).lower()
    )
    if queued:
        return True
    if not resp.ok or not data.get("success"):
        raise RuntimeError(f"Vast 未能把實例 {instance_id} 設為 {state}：HTTP {resp.status_code} {data}")
    return False


def find_instance_ids_by_label(api_key: str, label: str) -> list[int]:
    list_api = VAST_API[:-3] + "/v1" if VAST_API.endswith("/v0") else VAST_API
    resp = vast_request_with_retries(
        "GET",
        f"{list_api}/instances/",
        headers=vast_headers(api_key),
        params={
            "limit": 25,
            "select_filters": json.dumps({"label": {"eq": label}}),
            "select_cols": json.dumps(["id", "label", "actual_status"]),
        },
        timeout=60,
    )
    resp.raise_for_status()
    instances = resp.json().get("instances", [])
    if isinstance(instances, dict):
        instances = [instances]
    return [
        int(instance["id"])
        for instance in instances
        if str(instance.get("label", "")) == label and instance.get("id") is not None
    ]


def recover_uncertain_instance(api_key: str, label: str, job_dir: Path) -> bool:
    """清理不確定的建立結果；只有連續確認不存在時才回傳 True。"""
    last_error = ""
    successful_queries = 0
    for attempt in range(1, 13):
        if attempt > 1:
            time.sleep(5)
        try:
            instance_ids = find_instance_ids_by_label(api_key, label)
            successful_queries += 1
            for instance_id in instance_ids:
                write_job_state(job_dir, {"instance_id": instance_id, "status": "stopping_after_create_error"})
                destroy_instance_best_effort(api_key, instance_id, job_dir)
            if instance_ids:
                return False
        except Exception as exc:
            last_error = str(exc)
    if successful_queries:
        (job_dir / ".remote_session.json").unlink(missing_ok=True)
        write_job_state(job_dir, {"instance_destroyed": True, "instance_not_found": True})
        append_job_event(
            job_dir,
            "uncertain_instance_not_found",
            {"label": label, "successful_queries": successful_queries},
        )
        return True
    append_job_event(job_dir, "uncertain_instance_recovery_failed", {"label": label, "error": last_error})
    return False


def wait_running(
    api_key: str,
    instance_id: int,
    timeout_s: int = 900,
    transient_exited_s: int = 0,
) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        inst = get_instance(api_key, instance_id)
        status = inst.get("actual_status")
        print(f"實例狀態: {status}")
        if status == "running":
            return inst
        if status == "exited" and time.time() - start < transient_exited_s:
            time.sleep(5)
            continue
        if status in {"exited", "offline", "unknown"}:
            raise RuntimeError(f"實例啟動失敗: {status}")
        time.sleep(15)
    raise TimeoutError("等待實例 running 逾時")


def http_base(inst: dict) -> str:
    public_ip = inst.get("public_ipaddr")
    host_port = None
    for key, mappings in (inst.get("ports") or {}).items():
        if "8080" in key and mappings:
            host_port = mappings[0].get("HostPort")
            break
    if not public_ip or not host_port:
        raise RuntimeError(f"無法取得 HTTP 連接埠: {inst}")
    return f"http://{public_ip}:{host_port}"


def remote_headers(remote_api_token: str) -> dict[str, str]:
    return {"X-UNMARK-Token": remote_api_token}


def wait_bootstrap_ready(url_base: str, remote_api_token: str, timeout_s: int = 900) -> None:
    print("等待遠端接收服務及環境依賴就緒...")
    start = time.time()
    last_status = None
    while time.time() - start < timeout_s:
        try:
            resp = REMOTE_HTTP.get(
                f"{url_base}/health",
                headers=remote_headers(remote_api_token),
                timeout=10,
            )
            if resp.status_code == 200:
                status = resp.json().get("status")
                if status == "ready":
                    print("遠端環境依賴安裝完畢，服務就緒！")
                    return
                if status == "error":
                    raise RuntimeError("遠端 Python 依賴安裝失敗")
                if status == "installing":
                    if last_status != "installing":
                        print("遠端正在安裝 Python 環境與 Diffusers 依賴 (請稍候，背景進行中)...")
                        last_status = "installing"
                    try:
                        log_resp = REMOTE_HTTP.get(
                            f"{url_base}/install.log",
                            headers=remote_headers(remote_api_token),
                            timeout=5,
                        )
                        if log_resp.status_code == 200:
                            lines = log_resp.text.strip().split("\n")
                            if lines:
                                print(f"  [安裝進度] {lines[-1]}", end="\r")
                    except Exception:
                        pass
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("等待遠端接收服務或環境安裝就緒逾時")


def upload_file(
    url_base: str,
    remote_path: str,
    source: bytes | Path,
    remote_api_token: str,
    attempts: int = 3,
) -> dict:
    if isinstance(source, Path):
        size_bytes = source.stat().st_size
        expected_sha256 = sha256_file(source)
    else:
        size_bytes = len(source)
        expected_sha256 = hashlib.sha256(source).hexdigest()
    headers = remote_headers(remote_api_token) | {
        "Content-Length": str(size_bytes),
        "X-Content-SHA256": expected_sha256,
    }
    url = f"{url_base}/{quote(remote_path, safe='/')}"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if isinstance(source, Path):
                with source.open("rb") as body:
                    resp = REMOTE_HTTP.put(url, data=body, headers=headers, timeout=(30, 300))
            else:
                resp = REMOTE_HTTP.put(url, data=source, headers=headers, timeout=(30, 300))
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") != "uploaded":
                raise RuntimeError(f"遠端回覆異常：{payload}")
            if int(payload.get("size_bytes", -1)) != size_bytes:
                raise RuntimeError("遠端檔案大小不符")
            if str(payload.get("sha256", "")).lower() != expected_sha256:
                raise RuntimeError("遠端檔案 SHA-256 不符")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"上傳 {remote_path} 失敗（已重試 {attempts} 次）：{last_error}") from last_error


def upload_payload_and_start(
    url_base: str,
    manifest: dict,
    batch_items: list[dict],
    remote_api_token: str,
) -> None:
    print("上傳 remote_worker.py")
    upload_file(url_base, "remote_worker.py", REMOTE_WORKER, remote_api_token)

    # 上傳工作並行進行；每個檔案均會核對大小及 SHA-256。
    local_items = [item for item in batch_items if item.get("source_type") == "local" and "local_path_resolved" in item]
    if local_items:
        print(f"正在並行上傳及核對 {len(local_items)} 張本地原圖...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def upload_single_image(item: dict) -> None:
            path = Path(item["local_path_resolved"])
            filename = item["filename"]
            upload_file(url_base, f"raw_images/{filename}", path, remote_api_token)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(upload_single_image, item) for item in local_items]
            for future in as_completed(futures):
                future.result()
        print("所有本地圖片已上傳並通過完整性核對。")

    manifest_json = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    print("上傳 manifest.json")
    upload_file(url_base, "manifest.json", manifest_json, remote_api_token)
    print("啟動遠端任務")
    resp = REMOTE_HTTP.post(
        f"{url_base}/start",
        headers=remote_headers(remote_api_token),
        timeout=60,
    )
    if not resp.ok:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"遠端拒絕啟動任務（HTTP {resp.status_code}）：{detail}")
    payload = resp.json()
    if payload.get("status") != "started":
        raise RuntimeError(f"遠端任務啟動失敗: {payload}")


def completed_info(row: dict, local_rel: str) -> dict:
    return {
        "output_image": local_rel,
        "status": "ok",
        "error": "",
        "seconds": row.get("seconds", ""),
        "model": MODEL,
        "steps": STEPS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_size_bytes": row.get("output_size_bytes", ""),
        "output_sha256": row.get("output_sha256", ""),
    }


def wait_job_done(
    url_base: str,
    completed_map: dict[str, dict],
    remote_api_token: str,
    timeout_s: int = 7200,
) -> dict:
    print("等待遠端任務完成；每張完成後會立即下載及存檔...")
    start = time.time()
    last_status = {}
    downloaded_set = set() # 記錄本次運作中已下載的遠端路徑

    while time.time() - start < timeout_s:
        try:
            resp = REMOTE_HTTP.get(
                f"{url_base}/job_status.json",
                headers=remote_headers(remote_api_token),
                timeout=10,
            )
            if resp.status_code == 200:
                status = resp.json()
                last_status = status
                state = status.get("status")

                rows = status.get("rows", [])
                new_downloads = False
                for row in rows:
                    original = row["original_image"]
                    input_key = row.get("input_key") or original
                    remote_out = row.get("output_image", "")
                    status_str = row.get("status", "")

                    if status_str == "ok" and remote_out and remote_out not in downloaded_set:
                        filename = Path(remote_out).name
                        local_rel = f"output_images/{filename}"
                        local_path = JOB_DIR / local_rel
                        try:
                            download_file(
                                url_base,
                                remote_out,
                                local_path,
                                remote_api_token,
                                expected_size=row.get("output_size_bytes"),
                                expected_sha256=row.get("output_sha256"),
                            )
                            downloaded_set.add(remote_out)
                            completed_map[input_key] = completed_info(row, local_rel)
                            completed_map[input_key]["original_image"] = original
                            print(f"  [邊跑邊下] 已即時下載: {local_rel} (耗時 {row.get('seconds', '?')} 秒)")
                            new_downloads = True
                        except Exception as exc:
                            print(f"  [下載失敗] 圖片 {remote_out} 下載失敗: {exc}")
                    elif status_str == "error" and input_key not in completed_map:
                        completed_map[input_key] = {
                            "original_image": original,
                            "output_image": "",
                            "status": "error",
                            "error": row.get("error", ""),
                            "seconds": row.get("seconds", ""),
                            "model": MODEL,
                            "steps": STEPS,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        print(f"  [遠端錯誤] 圖片 {original} 處理出錯：{row.get('error', '')}")
                        new_downloads = True

                # 若有新下載或出錯，立即即時覆寫 output.csv 存檔！
                if new_downloads:
                    write_output_csv_map(completed_map)

                if state in {"done", "partial", "error"}:
                    return status
                elif state == "running":
                    print(f"  遠端進度：{status.get('done', 0)}/{status.get('total', '?')} 張已完成 (成功: {status.get('ok', 0)} 張)")
        except requests.RequestException:
            pass
        time.sleep(10)
    raise JobTimeoutError(last_status)


def download_file(
    url_base: str,
    remote_rel: str,
    local_path: Path,
    remote_api_token: str,
    expected_size=None,
    expected_sha256=None,
    attempts: int = 3,
) -> dict:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{url_base}/{quote(remote_rel, safe='/')}"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        temp_path = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.part")
        try:
            with REMOTE_HTTP.get(
                url,
                headers=remote_headers(remote_api_token),
                stream=True,
                timeout=(30, 120),
            ) as resp:
                resp.raise_for_status()
                header_size = int(resp.headers.get("Content-Length", "-1"))
                header_sha256 = resp.headers.get("X-Content-SHA256", "").lower()
                digest = hashlib.sha256()
                actual_size = 0
                with temp_path.open("wb") as output:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        digest.update(chunk)
                        actual_size += len(chunk)
            actual_sha256 = digest.hexdigest()
            sizes = [int(value) for value in (header_size, expected_size) if value not in (None, "", -1, "-1")]
            if any(actual_size != value for value in sizes):
                raise RuntimeError(f"下載大小不符：收到 {actual_size} bytes，預期 {sizes}")
            hashes = [str(value).lower() for value in (header_sha256, expected_sha256) if value]
            if any(actual_sha256 != value for value in hashes):
                raise RuntimeError("下載檔案 SHA-256 不符")
            os.replace(temp_path, local_path)
            return {"size_bytes": actual_size, "sha256": actual_sha256}
        except Exception as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"下載 {remote_rel} 失敗（已重試 {attempts} 次）：{last_error}") from last_error


def load_completed_jobs() -> dict[str, dict]:
    completed = {}
    if OUTPUT_CSV.exists():
        try:
            with OUTPUT_CSV.open(encoding="utf-8-sig", newline="") as output_file:
                reader = csv.DictReader(output_file)
                if reader.fieldnames and "original_image" in reader.fieldnames:
                    for row in reader:
                        original_image = (row.get("original_image") or "").strip()
                        input_key = (row.get("input_key") or original_image).strip()
                        output_image = (row.get("output_image") or "").strip()
                        status = (row.get("status") or "").strip()
                        if original_image and (output_image or status == "error"):
                            output_path = JOB_DIR / output_image if output_image else None
                            valid_output = bool(output_path and output_path.is_file())
                            expected_size = (row.get("output_size_bytes") or "").strip()
                            expected_sha256 = (row.get("output_sha256") or "").strip().lower()
                            if valid_output and expected_size:
                                valid_output = output_path.stat().st_size == int(expected_size)
                            if valid_output and expected_sha256:
                                valid_output = sha256_file(output_path) == expected_sha256
                            # 錯誤列會保留診斷，但續跑時只略過已驗證的成功檔案。
                            if status == "error" or valid_output:
                                completed[input_key] = {
                                    "original_image": original_image,
                                    "output_image": output_image,
                                    "status": status or "ok",
                                    "error": row.get("error", ""),
                                    "seconds": row.get("seconds", ""),
                                    "model": row.get("model", ""),
                                    "steps": row.get("steps", ""),
                                    "timestamp": row.get("timestamp", ""),
                                    "output_size_bytes": expected_size,
                                    "output_sha256": expected_sha256,
                                }
        except Exception as exc:
            print(f"載入既有 output.csv 出錯（將忽略並重新處理）：{exc}")
    return completed


def write_output_csv_map(completed_map: dict[str, dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_CSV.with_name(f".{OUTPUT_CSV.name}.tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=[
            "input_key", "original_image", "output_image", "status", "error", "seconds", "model", "steps", "timestamp",
            "output_size_bytes", "output_sha256",
        ])
        writer.writeheader()
        for input_key, info in sorted(completed_map.items()):
            writer.writerow({
                "input_key": input_key,
                "original_image": info.get("original_image", input_key),
                "output_image": info.get("output_image", ""),
                "status": info.get("status", "ok"),
                "error": info.get("error", ""),
                "seconds": info.get("seconds", ""),
                "model": info.get("model", MODEL),
                "steps": info.get("steps", STEPS),
                "timestamp": info.get("timestamp", ""),
                "output_size_bytes": info.get("output_size_bytes", ""),
                "output_sha256": info.get("output_sha256", ""),
            })
    os.replace(tmp_path, OUTPUT_CSV)
    print(f"已即時同步存檔至 {OUTPUT_CSV}")


def download_results_stream(
    url_base: str,
    summary: dict,
    completed_map: dict[str, dict],
    remote_api_token: str,
) -> None:
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    new_downloads = False
    for row in summary.get("rows", []):
        original = row["original_image"]
        input_key = row.get("input_key") or original
        remote_out = row.get("output_image", "")
        status_str = row.get("status", "")

        if status_str == "ok" and remote_out:
            filename = Path(remote_out).name
            local_rel = f"output_images/{filename}"
            local_path = JOB_DIR / local_rel
            if input_key not in completed_map or not local_path.exists():
                try:
                    download_file(
                        url_base,
                        remote_out,
                        local_path,
                        remote_api_token,
                        expected_size=row.get("output_size_bytes"),
                        expected_sha256=row.get("output_sha256"),
                    )
                    completed_map[input_key] = completed_info(row, local_rel)
                    completed_map[input_key]["original_image"] = original
                    new_downloads = True
                except Exception as exc:
                    print(f"  [下載失敗] 補載圖片 {remote_out} 失敗: {exc}")
        elif status_str == "error" and input_key not in completed_map:
            completed_map[input_key] = {
                "original_image": original,
                "output_image": "",
                "status": "error",
                "error": row.get("error", ""),
                "seconds": row.get("seconds", ""),
                "model": MODEL,
                "steps": STEPS,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            new_downloads = True

    if new_downloads:
        write_output_csv_map(completed_map)


def fetch_remote_summary(url_base: str, remote_api_token: str) -> dict | None:
    for name in ("job_status.json", "result.json"):
        try:
            resp = REMOTE_HTTP.get(
                f"{url_base}/{name}",
                headers=remote_headers(remote_api_token),
                timeout=20,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            continue
    return None


def final_reconcile_before_shutdown(
    job_dir: Path,
    url_base: str | None,
    completed_map: dict[str, dict],
    remote_api_token: str,
    reason: str,
) -> None:
    if not url_base:
        append_job_event(job_dir, "final_reconcile_skipped", {"reason": "missing_url_base", "trigger": reason})
        return
    print("結束前嘗試補下載遠端已完成圖片...")
    append_job_event(job_dir, "final_reconcile_started", {"reason": reason, "url_base": url_base})
    summary = fetch_remote_summary(url_base, remote_api_token)
    if not summary:
        print("  未能讀取遠端任務狀態，會繼續停機以避免持續計費。")
        append_job_event(job_dir, "final_reconcile_failed", {"reason": "missing_remote_summary"})
        return
    try:
        download_results_stream(url_base, summary, completed_map, remote_api_token)
        write_remote_result(summary)
        ok = sum(1 for info in completed_map.values() if info.get("status") == "ok")
        write_job_state(job_dir, {"ok": ok, "last_final_reconcile_reason": reason})
        append_job_event(job_dir, "final_reconcile_done", {"ok": ok, "remote_status": summary.get("status")})
    except Exception as exc:
        print(f"  補下載失敗：{exc}")
        append_job_event(job_dir, "final_reconcile_failed", {"error": str(exc)})


def write_remote_result(summary: dict) -> None:
    atomic_write_json(REMOTE_RESULT_JSON, summary)


def destroy_instance(api_key: str, instance_id: int) -> None:
    resp = vast_request_with_retries(
        "DELETE",
        f"{VAST_API}/instances/{instance_id}/",
        headers=vast_headers(api_key),
        timeout=60,
    )
    if resp.status_code == 404:
        print(f"實例 {instance_id} 已不存在。")
        return
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError:
        payload = {"raw": resp.text}
    print("銷毀實例：", payload)


def destroy_instance_best_effort(
    api_key: str,
    instance_id: int,
    job_dir: Path,
    attempts: int = 3,
) -> bool:
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            destroy_instance(api_key, instance_id)
            (job_dir / ".remote_session.json").unlink(missing_ok=True)
            write_job_state(job_dir, {"instance_destroyed": True})
            append_job_event(job_dir, "instance_destroyed", {"instance_id": instance_id, "attempt": attempt})
            return True
        except Exception as exc:
            last_error = str(exc)
            append_job_event(
                job_dir,
                "instance_destroy_failed",
                {"instance_id": instance_id, "attempt": attempt, "error": last_error},
            )
            if attempt < attempts:
                print(f"銷毀實例失敗，將重試 {attempt + 1}/{attempts}：{last_error}")
                time.sleep(5)
    write_job_state(job_dir, {"instance_destroyed": False, "destroy_error": last_error})
    print(f"銷毀實例失敗，請立即手動檢查 Vast 實例 {instance_id}：{last_error}")
    return False


def resolve_config(secret_overrides: dict[str, str] | None = None) -> tuple[str, str]:
    overrides = secret_overrides or {}
    api_key = (
        os.environ.get("VAST_API_KEY", "")
        or overrides.get("vast_api_key", "")
    ).strip()
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        or overrides.get("hf_token", "")
    ).strip()
    if not api_key:
        raise SystemExit("請在 .env 填寫 VAST_API_KEY，或設定環境變數 VAST_API_KEY")
    if not hf_token:
        raise SystemExit("請在 .env 填寫 HF_TOKEN，或設定環境變數 HF_TOKEN")
    return api_key, hf_token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UNMARK：可恢復的 Vast GPU 批次去水印工具"
    )
    parser.add_argument("--list-models", "-m", action="store_true", help="列出模型快捷鍵")
    parser.add_argument("--list-prompts", action="store_true", help="列出 Prompts 目錄內的提示詞")
    subparsers = parser.add_subparsers(dest="command")

    def add_common_runtime_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="模型快捷鍵：9B / 4B / KONTEXT / KWR")
        p.add_argument("--steps", type=int, help="推理步數")
        p.add_argument("--seed", type=int, help="隨機種子；相同輸入及設定可重現結果")
        p.add_argument("--gpu-name", help="Vast GPU 型號，例如 H200")
        p.add_argument("--disk-gb", type=int, help="Vast 實例磁碟大小")
        p.add_argument("--max-side", type=int, help="推理時最長邊縮放上限")
        p.add_argument("--max-download-mb", type=int, help="每張網址圖片的下載大小上限（MB）")
        p.add_argument("--prompt-preset", help="Prompts 目錄內的提示詞檔名，不包括 .md")
        p.add_argument("--batch-size", type=int, help="每個遠端批次最多圖片數")
        p.add_argument("--concurrency", type=int, help="遠端下載及準備輸入圖片的並行數")
        p.add_argument("--input-csv", help="CSV 輸入檔")
        p.add_argument("--input-images-dir", help="本地圖片資料夾")
        p.add_argument("--jobs-dir", help="jobs 根資料夾")
        p.add_argument("--job-dir", help="指定 job 資料夾")
        p.add_argument("--job-id", help="指定新 job id")
        p.add_argument("--instance-label", help="Vast 實例 label")
        p.add_argument("--max-embed-mb", type=int, help="本地圖片上傳總大小上限")
        p.add_argument("--job-timeout", type=int, help="單批次等待逾時秒數")
        p.add_argument("--bootstrap-timeout", type=int, help="等待遠端環境就緒逾時秒數")
        p.add_argument("--instance-ttl", type=int, help="Vast 實例最長存活秒數；逾時後由實例自行銷毀")
        p.add_argument("--remote-result-grace", type=int, help="遠端任務結束後等候本地下載的秒數")
        p.add_argument("--remote-work-dir", help="遠端工作資料夾")
        p.add_argument("--vast-image", help="Vast Docker image")
        p.add_argument("--vast-api", help="Vast API endpoint")
        p.add_argument("--destroy-on-success", dest="destroy_on_success", action="store_true", default=None, help="成功後自動銷毀實例")
        p.add_argument("--no-destroy-on-success", dest="destroy_on_success", action="store_false", help="成功後保留實例")

    run_parser = subparsers.add_parser("run", help="腳本模式：建立臨時 API GPU 機並處理圖片")
    add_common_runtime_args(run_parser)

    plan_parser = subparsers.add_parser("plan", help="Dry-run：只檢查輸入與批次規劃，不建立 Vast 實例")
    add_common_runtime_args(plan_parser)

    resume_parser = subparsers.add_parser("resume", help="用既有 job 目錄恢復未完成圖片")
    add_common_runtime_args(resume_parser)

    status_parser = subparsers.add_parser("status", help="讀取 job 目錄狀態")
    status_parser.add_argument("--job-dir", required=True, help="job 資料夾")
    status_parser.add_argument("--jobs-dir", help="jobs 根資料夾")

    reconcile_parser = subparsers.add_parser("reconcile", help="只連回遠端臨時 API 機，補下載已完成結果")
    reconcile_parser.add_argument("--job-dir", required=True, help="job 資料夾")
    reconcile_parser.add_argument("--jobs-dir", help="jobs 根資料夾")

    destroy_parser = subparsers.add_parser("destroy", help="手動銷毀 job 記錄中的 Vast 實例")
    destroy_parser.add_argument("--job-dir", required=True, help="job 資料夾")
    destroy_parser.add_argument("--jobs-dir", help="jobs 根資料夾")

    fallback_parser = subparsers.add_parser("fallback-queue", help="由失敗或人工標記結果生成補處理 CSV")
    fallback_parser.add_argument("--job-dir", required=True, help="job 資料夾")
    fallback_parser.add_argument("--jobs-dir", help="jobs 根資料夾")
    fallback_parser.add_argument("--marked-file", help="人工標記清單；每行一個 original_image 或 output_image")
    fallback_parser.add_argument("--output-csv", help="輸出 fallback CSV，預設為 job 內 fallback_input.csv")
    fallback_parser.add_argument("--no-errors", action="store_true", help="不自動加入 status=error 的項目")

    benchmark_parser = subparsers.add_parser(
        "benchmark-models",
        help="在同一台 Vast 實例依次評測多個模型及 steps",
    )
    add_common_runtime_args(benchmark_parser)
    benchmark_parser.add_argument("--models", default="9B,4B,KONTEXT,KWR", help="逗號分隔的模型快捷鍵")
    benchmark_parser.add_argument("--steps-list", default="8,16,32", help="逗號分隔的推理步數")
    benchmark_parser.add_argument(
        "--configurations",
        help="精確評測組別，例如 9B:8,9B:16,KWR:32；設定後不使用 models × steps-list",
    )
    benchmark_parser.add_argument(
        "--reuse-job-dir",
        help="重用既有 benchmark job 所記錄的 stopped/running Vast 實例及模型快取",
    )
    benchmark_parser.add_argument(
        "--output-root",
        default="output_images/model_tests",
        help="評測圖片輸出根目錄",
    )
    benchmark_parser.add_argument("--report-dir", default="Docs/Models", help="評測報告目錄")
    benchmark_parser.add_argument("--dry-run", action="store_true", help="只列出評測規劃，不建立實例")
    benchmark_parser.add_argument("--overwrite", action="store_true", help="允許覆寫既有評測輸出檔案")
    benchmark_parser.add_argument("--destroy-after", action="store_true", help="評測完成後銷毀實例（預設）")
    benchmark_parser.add_argument(
        "--keep-instance",
        dest="destroy_after",
        action="store_false",
        help="評測完成後保留實例及儲存空間；仍會繼續計費",
    )
    benchmark_parser.set_defaults(
        input_images_dir="input_images",
        disk_gb=200,
        instance_label="unmark-model-benchmark",
        job_timeout=14400,
        instance_ttl=86400,
        remote_result_grace=86400,
        destroy_on_success=False,
        destroy_after=True,
    )

    web_parser = subparsers.add_parser("serve-local", help="本地 Web 狀態頁")
    web_parser.add_argument("--jobs-dir", help="jobs 根資料夾")
    web_parser.add_argument("--host", default="127.0.0.1", help="監聽 host")
    web_parser.add_argument("--port", type=int, default=8787, help="監聽 port")
    web_parser.add_argument("--settings-json", help="LocalWeb 設定 JSON；不會保存 Vast / Hugging Face / 本地 API Key")

    return parser


def run_vast_job(
    args: argparse.Namespace,
    resume: bool = False,
    secret_overrides: dict[str, str] | None = None,
) -> None:
    apply_runtime_config(args)
    require_requests()
    api_key, hf_token = resolve_config(secret_overrides)
    model_key, model_cfg = resolve_model(MODEL)

    job_dir = prepare_job_dir(args, resume=resume)
    apply_job_paths(job_dir)
    if resume and (job_dir / "input.csv").exists() and not getattr(args, "input_csv", None):
        global INPUT_CSV, INPUT_BASE_DIR
        INPUT_CSV = job_dir / "input.csv"
        INPUT_BASE_DIR = INPUT_CSV.parent
    snapshot_input(job_dir)
    write_job_state(
        job_dir,
        {
            "job_id": job_dir.name,
            "status": "initializing",
            "job_dir": str(job_dir),
            "config": runtime_config_snapshot(),
        },
    )
    append_job_event(job_dir, "job_started", {"resume": resume})

    saved_items = load_job_input_items(job_dir) if resume else []
    if saved_items:
        all_items = saved_items
        print(f"從 job manifest 讀取 {len(all_items)} 條")
    else:
        all_items = collect_inputs()
    validate_output_names(all_items)
    all_items = snapshot_local_inputs(job_dir, all_items)
    total_input_count = len(all_items)
    write_input_manifest(job_dir, all_items)
    write_job_state(job_dir, {"total_input": total_input_count, "input_manifest": str(job_dir / "input_manifest.jsonl")})

    # 斷點續跑邏輯：載入已有紀錄，過濾掉已成功處理且實體圖片完好的項目
    completed_map = load_completed_jobs()
    if completed_map:
        completed_ok = sum(1 for info in completed_map.values() if info.get("status") == "ok")
        filtered_items = []
        for item in all_items:
            item_key = item.get("input_key") or item["original_ref"]
            if item_key in completed_map and completed_map[item_key].get("status") == "ok":
                continue
            filtered_items.append(item)
        print(f"【斷點續跑】已下載完成 {completed_ok} 張，剩餘 {len(filtered_items)}/{len(all_items)} 張圖片待處理。")
        all_items = filtered_items
    else:
        print(f"【全新開始】共計 {len(all_items)} 張圖片待處理。")

    if not all_items:
        print("所有圖片已處理完成，無需啟動遠端任務。")
        write_job_state(job_dir, {"status": "done", "ok": existing_completed_count()})
        return

    batches = chunked(all_items, BATCH_SIZE)

    offer_id, dph = search_gpu_offer(api_key, GPU_NAME)
    lora_note = f" + {model_cfg['lora_id']}" if model_cfg.get("lora_id") else ""
    print(
        f"模型 {model_key} ({model_cfg['label']}{lora_note}) | "
        f"{GPU_NAME} offer_id={offer_id} ${dph:.4f}/h | steps={STEPS} | "
        f"待處理圖片數={len(all_items)} | 批次數={len(batches)} | 每批最多={BATCH_SIZE}"
    )

    batch_summaries = []
    total_ok = sum(1 for info in completed_map.values() if info.get("status") == "ok")
    remote_api_token = secrets.token_urlsafe(32)
    write_remote_session(job_dir, remote_api_token)
    onstart = build_onstart(hf_token, remote_api_token)
    instance_run_label = f"{INSTANCE_LABEL}-{job_dir.name[-17:]}"[:64]

    start_run_time = time.time()
    write_job_state(job_dir, {"status": "creating_instance", "instance_label": instance_run_label})
    try:
        instance_id = create_instance(
            api_key,
            offer_id,
            onstart,
            hf_token,
            remote_api_token,
            instance_run_label,
            DISK_GB,
        )
    except InstanceCreateUncertainError as exc:
        write_job_state(job_dir, {"status": "instance_create_uncertain", "error": str(exc)})
        append_job_event(job_dir, "instance_create_uncertain", {"label": instance_run_label, "error": str(exc)})
        recover_uncertain_instance(api_key, instance_run_label, job_dir)
        raise
    except Exception as exc:
        (job_dir / ".remote_session.json").unlink(missing_ok=True)
        write_job_state(job_dir, {"status": "instance_create_failed", "error": str(exc)})
        append_job_event(job_dir, "instance_create_failed", {"label": instance_run_label, "error": str(exc)})
        raise
    url_base = None
    failure_reason = ""
    print(f"實例 ID: {instance_id}")
    write_job_state(
        job_dir,
        {
            "status": "instance_created",
            "instance_id": instance_id,
            "dph": dph,
            "offer_id": offer_id,
        },
    )
    append_job_event(job_dir, "instance_created", {"instance_id": instance_id, "offer_id": offer_id, "dph": dph})

    success = False
    had_timeout = False
    terminal_status = "error"
    try:
        inst = wait_running(api_key, instance_id)
        url_base = http_base(inst)
        write_job_state(job_dir, {"status": "bootstrap_waiting", "url_base": url_base})
        wait_bootstrap_ready(url_base, remote_api_token, timeout_s=BOOTSTRAP_TIMEOUT_S)
        write_job_state(job_dir, {"status": "running", "url_base": url_base})

        for batch_number, batch_items in enumerate(batches, start=1):
            print(f"開始批次 {batch_number}/{len(batches)}：圖片數={len(batch_items)}")
            write_job_state(
                job_dir,
                {
                    "status": "running",
                    "current_batch": batch_number,
                    "batches_total": len(batches),
                    "current_batch_size": len(batch_items),
                },
            )
            manifest = build_manifest(batch_items, STEPS, model_key, model_cfg)
            upload_payload_and_start(url_base, manifest, batch_items, remote_api_token)
            try:
                summary = wait_job_done(
                    url_base,
                    completed_map,
                    remote_api_token,
                    timeout_s=JOB_TIMEOUT_S,
                )
            except JobTimeoutError as exc:
                summary = exc.summary
                had_timeout = True
                print(
                    f"批次 {batch_number}/{len(batches)} 逾時：將嘗試下載已完成圖片"
                )
                append_job_event(job_dir, "batch_timeout", {"batch": batch_number, "summary": summary})
            # 增量補載缺失的結果並寫入 output.csv
            download_results_stream(url_base, summary, completed_map, remote_api_token)
            total_ok = sum(1 for info in completed_map.values() if info.get("status") == "ok")
            batch_summary = dict(summary)
            batch_summary["batch"] = batch_number
            batch_summary["batch_size"] = len(batch_items)
            batch_summaries.append(batch_summary)
            remote_status = str(summary.get("status", "error"))
            if had_timeout:
                combined_status = "timeout_partial"
            elif remote_status == "error":
                combined_status = "error" if total_ok == 0 else "partial"
            elif batch_number < len(batches):
                combined_status = "running"
            elif total_ok == total_input_count and all(
                batch_summary.get("status") == "done" for batch_summary in batch_summaries
            ):
                combined_status = "done"
            else:
                combined_status = "partial" if total_ok else "error"
            terminal_status = combined_status
            combined_summary = {
                "status": combined_status,
                "model_key": model_key,
                "total": total_input_count,
                "ok": total_ok,
                "batch_size": BATCH_SIZE,
                "batches_total": len(batches),
                "batches_done": batch_number,
                "batch_summaries": batch_summaries,
            }
            write_remote_result(combined_summary)
            write_job_state(
                job_dir,
                {
                    "status": combined_status,
                    "ok": total_ok,
                    "batches_done": batch_number,
                    "batches_total": len(batches),
                    "remote_result": str(REMOTE_RESULT_JSON),
                    "output_csv": str(OUTPUT_CSV),
                },
            )
            print(f"批次 {batch_number}/{len(batches)} 完成：累積成功 {total_ok} 張圖片")
            if remote_status == "error":
                failure_reason = str(summary.get("error") or "remote_inference_error")
                append_job_event(
                    job_dir,
                    "batch_error",
                    {"batch": batch_number, "error": failure_reason, "ok": total_ok},
                )
                print(f"遠端推理失敗，停止後續批次：{failure_reason}")
                break
            if had_timeout:
                print("批次逾時，已停止派發後續批次；稍後會先補下載已完成結果，再停掉實例。")
                break
        success = (
            not had_timeout
            and total_ok == total_input_count
            and len(batch_summaries) == len(batches)
            and all(batch_summary.get("status") == "done" for batch_summary in batch_summaries)
        )
        if success:
            terminal_status = "done"
        elif terminal_status == "running":
            terminal_status = "partial" if total_ok else "error"
    except Exception as exc:
        failure_reason = type(exc).__name__
        terminal_status = "error"
        write_job_state(job_dir, {"status": "error", "error": str(exc)})
        append_job_event(job_dir, "job_error", {"error": str(exc)})
        raise
    finally:
        if not success:
            final_reconcile_before_shutdown(
                job_dir,
                url_base,
                completed_map,
                remote_api_token,
                failure_reason or ("timeout" if had_timeout else "not_success"),
            )
            total_ok = sum(1 for info in completed_map.values() if info.get("status") == "ok")
            write_job_state(job_dir, {"status": "stopping_after_error_or_timeout"})
        should_destroy = (success and DESTROY_ON_SUCCESS) or (not success)
        if should_destroy:
            destroy_instance_best_effort(api_key, instance_id, job_dir)
        else:
            print(
                "已保留 Vast 實例。"
                f"如要手動銷毀，請執行：python3 run_unmark.py destroy --job-dir {job_dir}"
            )
            write_job_state(job_dir, {"instance_destroyed": False})
        elapsed = time.time() - start_run_time
        cost = elapsed / 3600 * dph
        write_job_state(
            job_dir,
            {
                "status": terminal_status,
                "ok": total_ok,
                "elapsed_seconds": round(elapsed, 3),
                "estimated_cost_usd": round(cost, 6),
            },
        )

        print("\n" + "=" * 50)
        print("任務執行統計報告 (Vast.ai)")
        print(f"Job 目錄：{job_dir}")
        print(f"總共耗時：{elapsed:.1f} 秒 (約 {elapsed/60:.2f} 分鐘)")
        print(f"預估花費：${cost:.6f} 美金 (單價 ${dph:.4f}/小時)")
        print(f"成功處理：{total_ok} 張圖片")
        print("=" * 50 + "\n")


def plan_job(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    model_key, model_cfg = resolve_model(MODEL)
    items = collect_inputs()
    batches = chunked(items, BATCH_SIZE)
    local_count = sum(1 for item in items if item.get("source_type") == "local")
    url_count = sum(1 for item in items if item.get("source_type") == "url")
    local_bytes = sum(int(item.get("size_bytes", 0)) for item in items)
    print("Dry-run 規劃：")
    print(f"  模型：{model_key} ({model_cfg['label']})")
    print(f"  steps：{STEPS}")
    print(f"  seed：{SEED}")
    print(f"  提示詞：{PROMPT_PRESET}")
    print(f"  GPU：{GPU_NAME}")
    print(f"  輸入總數：{len(items)}")
    print(f"  URL：{url_count}")
    print(f"  本地圖片：{local_count} ({local_bytes / 1024 / 1024:.1f} MB)")
    print(f"  每批最多：{BATCH_SIZE}")
    print(f"  批次數：{len(batches)}")
    print(f"  jobs 目錄：{JOBS_DIR}")
    print("  不會建立 Vast 實例，不會上傳圖片。")


def read_output_stats(job_dir: Path) -> dict:
    output_csv = job_dir / "output.csv"
    stats = {"rows": 0, "ok": 0, "error": 0, "missing_output": 0, "downloaded_files": 0}
    if output_csv.exists():
        with output_csv.open(encoding="utf-8-sig", newline="") as output_file:
            for row in csv.DictReader(output_file):
                stats["rows"] += 1
                status = row.get("status", "ok")
                if status == "error":
                    stats["error"] += 1
                else:
                    output_ref = (row.get("output_image") or "").strip()
                    output_path = (job_dir / output_ref).resolve() if output_ref else None
                    valid = bool(
                        output_path
                        and output_path.is_relative_to(job_dir.resolve())
                        and output_path.is_file()
                    )
                    expected_size = (row.get("output_size_bytes") or "").strip()
                    expected_sha256 = (row.get("output_sha256") or "").strip().lower()
                    if valid and expected_size:
                        valid = output_path.stat().st_size == int(expected_size)
                    if valid and expected_sha256:
                        valid = sha256_file(output_path) == expected_sha256
                    if valid:
                        stats["ok"] += 1
                    else:
                        stats["error"] += 1
                        stats["missing_output"] += 1
    output_dir = job_dir / "output_images"
    if output_dir.exists():
        stats["downloaded_files"] = sum(
            1 for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return stats


def status_job(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    job_dir = prepare_job_dir(args, resume=True)
    state = read_job_state(job_dir)
    stats = read_output_stats(job_dir)
    print(f"Job：{job_dir.name}")
    print(f"路徑：{job_dir}")
    print(f"狀態：{state.get('status', 'unknown')}")
    print(f"實例 ID：{state.get('instance_id', '')}")
    print(f"遠端 API：{state.get('url_base', '')}")
    print(f"已下載成功：{stats['ok']}")
    print(f"失敗：{stats['error']}")
    print(f"輸出檔案數：{stats['downloaded_files']}")
    print(f"output.csv：{job_dir / 'output.csv'}")


def reconcile_job(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    require_requests()
    job_dir = prepare_job_dir(args, resume=True)
    apply_job_paths(job_dir)
    state = read_job_state(job_dir)
    url_base = state.get("url_base")
    if not url_base:
        raise SystemExit("job_state.json 沒有 url_base，無法連回遠端臨時 API 機")
    remote_api_token = read_remote_session(job_dir)
    if not remote_api_token:
        raise SystemExit("找不到此 job 的遠端臨時連線憑證，無法安全連回實例")
    completed_map = load_completed_jobs()
    print(f"連回遠端：{url_base}")
    try:
        resp = REMOTE_HTTP.get(
            f"{url_base}/job_status.json",
            headers=remote_headers(remote_api_token),
            timeout=20,
        )
        resp.raise_for_status()
        summary = resp.json()
    except requests.RequestException as exc:
        raise SystemExit(f"讀取遠端 job_status.json 失敗：{exc}") from exc
    download_results_stream(url_base, summary, completed_map, remote_api_token)
    write_remote_result(summary)
    ok = sum(1 for info in completed_map.values() if info.get("status") == "ok")
    write_job_state(job_dir, {"status": summary.get("status", "reconciled"), "ok": ok})
    print(f"對帳完成，已下載成功結果：{ok} 張")


def destroy_recorded_instance(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    require_requests()
    api_key, _ = resolve_config()
    job_dir = prepare_job_dir(args, resume=True)
    state = read_job_state(job_dir)
    instance_id = state.get("instance_id")
    if not instance_id:
        raise SystemExit("job_state.json 沒有 instance_id")
    destroy_instance_best_effort(api_key, int(instance_id), job_dir)


def load_input_manifest(job_dir: Path) -> dict[str, dict]:
    manifest_path = job_dir / "input_manifest.jsonl"
    items = {}
    if not manifest_path.exists():
        return items
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            original = item.get("original_ref")
            if original:
                items[original] = item
    return items


def build_fallback_queue(args: argparse.Namespace) -> None:
    apply_runtime_config(args)
    job_dir = prepare_job_dir(args, resume=True)
    manifest = load_input_manifest(job_dir)
    output_csv = job_dir / "output.csv"
    if not output_csv.exists():
        raise SystemExit(f"找不到 output.csv：{output_csv}")

    marked = set()
    if args.marked_file:
        marked_path = Path(args.marked_file).expanduser()
        if not marked_path.is_absolute():
            marked_path = (ROOT / marked_path).resolve()
        if not marked_path.exists():
            raise SystemExit(f"找不到人工標記清單：{marked_path}")
        marked = {line.strip() for line in marked_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    selected: dict[str, dict] = {}
    with output_csv.open(encoding="utf-8-sig", newline="") as output_file:
        for row in csv.DictReader(output_file):
            original = (row.get("original_image") or "").strip()
            output_image = (row.get("output_image") or "").strip()
            status = (row.get("status") or "ok").strip()
            if not original:
                continue
            include = False
            if status == "error" and not args.no_errors:
                include = True
            if original in marked or output_image in marked:
                include = True
            if include:
                selected[original] = manifest.get(original, {"original_ref": original})

    out_path = Path(args.output_csv or (job_dir / "fallback_input.csv")).expanduser()
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as fallback_file:
        writer = csv.DictWriter(fallback_file, fieldnames=["id", "original_image"])
        writer.writeheader()
        for original, item in sorted(selected.items()):
            writer.writerow({
                "id": item.get("document_id") or item.get("index") or "",
                "original_image": original,
            })
    os.replace(tmp_path, out_path)
    append_job_event(job_dir, "fallback_queue_created", {"path": str(out_path), "count": len(selected)})
    print(f"已建立 fallback CSV：{out_path}")
    print(f"項目數：{len(selected)}")
    print(f"建議用 KWR 補處理：python3 run_unmark.py run --input-csv {out_path} --model KWR")


def serve_local(args: argparse.Namespace) -> None:
    from .local_web import serve_local as serve_local_web

    serve_local_web(args, sys.modules[__name__])


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print_model_catalog()
        return
    if args.list_prompts:
        print_prompt_catalog()
        return
    if not args.command:
        args.command = "run"
    if args.command == "run":
        run_vast_job(args, resume=False)
    elif args.command == "plan":
        plan_job(args)
    elif args.command == "resume":
        run_vast_job(args, resume=True)
    elif args.command == "status":
        status_job(args)
    elif args.command == "reconcile":
        reconcile_job(args)
    elif args.command == "destroy":
        destroy_recorded_instance(args)
    elif args.command == "fallback-queue":
        build_fallback_queue(args)
    elif args.command == "benchmark-models":
        from .benchmark import run_benchmark

        run_benchmark(args)
    elif args.command == "serve-local":
        serve_local(args)
    else:
        parser.error(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
