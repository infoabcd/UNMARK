from __future__ import annotations

import errno
import hashlib
import html
import http.client
import json
import os
import secrets
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

from .input_processing import IMAGE_EXTENSIONS, normalize_json_payload, normalize_uploaded_files

DEFAULT_SETTINGS = {
    "model": "9B",
    "steps": "",
    "seed": 42,
    "max_side": 1024,
    "max_download_mb": 100,
    "prompt_preset": "萬能提示詞",
    "gpu_name": "H200",
    "disk_gb": 80,
    "instance_label": "unmark-batch",
    "vast_image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    "remote_work_dir": "/workspace/unmark",
    "batch_size": 100,
    "concurrency": 12,
    "max_embed_mb": 2000,
    "job_timeout": 7200,
    "bootstrap_timeout": 900,
    "instance_ttl": 10800,
    "remote_result_grace": 900,
    "destroy_on_success": True,
    "jobs_dir": "jobs",
    "localweb_uploads_dir": "localweb_uploads",
    "api_auth_required": True,
}
SETTINGS_ENV_MAP = {
    "model": "UNMARK_MODEL",
    "steps": "UNMARK_STEPS",
    "seed": "UNMARK_SEED",
    "max_side": "UNMARK_MAX_SIDE",
    "max_download_mb": "UNMARK_MAX_DOWNLOAD_MB",
    "prompt_preset": "UNMARK_PROMPT_PRESET",
    "gpu_name": "UNMARK_GPU_NAME",
    "disk_gb": "UNMARK_DISK_GB",
    "instance_label": "UNMARK_INSTANCE_LABEL",
    "vast_image": "UNMARK_VAST_IMAGE",
    "remote_work_dir": "UNMARK_REMOTE_WORK_DIR",
    "batch_size": "UNMARK_BATCH_SIZE",
    "concurrency": "UNMARK_CONCURRENCY",
    "max_embed_mb": "UNMARK_MAX_EMBED_MB",
    "job_timeout": "UNMARK_JOB_TIMEOUT_S",
    "bootstrap_timeout": "UNMARK_BOOTSTRAP_TIMEOUT_S",
    "instance_ttl": "UNMARK_INSTANCE_TTL_S",
    "remote_result_grace": "UNMARK_REMOTE_RESULT_GRACE_S",
    "destroy_on_success": "UNMARK_DESTROY_ON_SUCCESS",
    "jobs_dir": "UNMARK_JOBS_DIR",
    "localweb_uploads_dir": "UNMARK_LOCALWEB_UPLOADS_DIR",
    "api_auth_required": "UNMARK_LOCAL_API_AUTH_REQUIRED",
}
SECRET_ENV_NAMES = {
    "vast_api_key": "VAST_API_KEY",
    "hf_token": "HF_TOKEN",
}
@dataclass
class LocalWebState:
    root: Path
    cli: object
    settings_path: Path
    settings: dict = field(default_factory=dict)
    temp_secrets: dict = field(default_factory=dict)
    temp_api_keys: list[dict] = field(default_factory=list)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)
    job_lock: threading.Lock = field(default_factory=threading.Lock)
    active_thread: threading.Thread | None = None
    active_job_id: str = ""
    flash_messages: dict[str, tuple[str, str]] = field(default_factory=dict)
    flash_lock: threading.Lock = field(default_factory=threading.Lock)


def text_response(handler: BaseHTTPRequestHandler, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
    payload = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("X-UNMARK-LocalWeb", "1")
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict | list) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("X-UNMARK-LocalWeb", "1")
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def redirect_response(handler: BaseHTTPRequestHandler, location: str, status: int = 303) -> None:
    handler.send_response(status)
    handler.send_header("X-UNMARK-LocalWeb", "1")
    handler.send_header("Location", location)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


def file_response(handler: BaseHTTPRequestHandler, path: Path, download_name: str) -> None:
    handler.send_response(200)
    handler.send_header("X-UNMARK-LocalWeb", "1")
    handler.send_header("Content-Type", "application/zip")
    handler.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
    handler.send_header("Content-Length", str(path.stat().st_size))
    handler.end_headers()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            handler.wfile.write(chunk)


def parse_bool(value, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"布林設定只可使用 true 或 false，目前值：{value!r}")


def parse_int(value, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def env_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def env_file_values(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    values = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_source_label(state: LocalWebState, env_name: str) -> str:
    current = env_value(env_name)
    env_file = env_file_values(state.root)
    if current:
        if env_file.get(env_name) == current:
            return f".env：{env_name}"
        return f"環境變數：{env_name}"
    return ""


def api_key_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def api_key_preview(value: str) -> str:
    stripped = value.strip()
    if len(stripped) <= 8:
        return "短 Key"
    return f"{stripped[:4]}...{stripped[-4:]}"


def new_api_key_record(
    plain_key: str,
    note: str,
    source: str,
    enabled: bool = True,
    persistent: bool = False,
) -> dict:
    return {
        "id": secrets.token_hex(8),
        "note": note.strip() or "未命名 API Key",
        "fingerprint": api_key_fingerprint(plain_key.strip()),
        "preview": api_key_preview(plain_key),
        "source": source,
        "enabled": enabled,
        "persistent": persistent,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_local_api_key() -> str:
    return "unmark_" + secrets.token_urlsafe(32)


def load_settings(settings_path: Path) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if settings_path.exists():
        try:
            settings_path.chmod(0o600)
        except OSError:
            pass
        try:
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"設定檔 JSON 格式錯誤：{settings_path}") from exc
        if not isinstance(saved, dict):
            raise ValueError(f"設定檔內容必須是 JSON object：{settings_path}")
        for key in DEFAULT_SETTINGS:
            if key in saved:
                settings[key] = saved[key]

    for key, env_name in SETTINGS_ENV_MAP.items():
        value = env_value(env_name)
        if value == "":
            continue
        default = DEFAULT_SETTINGS[key]
        if isinstance(default, bool):
            settings[key] = parse_bool(value, default)
        elif isinstance(default, int):
            settings[key] = parse_int(value, default)
        else:
            settings[key] = value

    return validate_settings(settings)


def save_settings(settings_path: Path, settings: dict) -> None:
    persisted = {key: settings.get(key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_name(f".{settings_path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, settings_path)
        settings_path.chmod(0o600)
    finally:
        tmp.unlink(missing_ok=True)


def queue_flash(state: LocalWebState, message: str, level: str = "notice") -> str:
    token = secrets.token_urlsafe(12)
    with state.flash_lock:
        if len(state.flash_messages) >= 20:
            state.flash_messages.pop(next(iter(state.flash_messages)))
        state.flash_messages[token] = (message, level)
    return token


def pop_flash(state: LocalWebState, token: str) -> tuple[str, str]:
    if not token:
        return "", ""
    with state.flash_lock:
        return state.flash_messages.pop(token, ("", ""))


def secret_env_source(state: LocalWebState, name: str) -> str:
    env_name = SECRET_ENV_NAMES[name]
    source = env_source_label(state, env_name)
    if source:
        return source
    if name == "hf_token" and env_value("HUGGING_FACE_HUB_TOKEN"):
        return env_source_label(state, "HUGGING_FACE_HUB_TOKEN")
    return ""


def secret_env_value(name: str) -> str:
    env_name = SECRET_ENV_NAMES[name]
    value = env_value(env_name)
    if value:
        return value
    if name == "hf_token":
        return env_value("HUGGING_FACE_HUB_TOKEN")
    return ""


def secret_source(state: LocalWebState, name: str) -> str:
    source = secret_env_source(state, name)
    if source:
        return source
    if state.temp_secrets.get(name):
        return "LocalWeb 記憶體，重啟後消失"
    return "未設定"


def get_secret(state: LocalWebState, name: str) -> str:
    return secret_env_value(name) or state.temp_secrets.get(name, "")


def configured_api_keys(state: LocalWebState) -> list[dict]:
    records = []
    env_key = env_value("UNMARK_LOCAL_API_KEY")
    if env_key:
        fingerprint = api_key_fingerprint(env_key)
        records.append(
            {
                "id": f"env_{fingerprint[:16]}",
                "note": env_source_label(state, "UNMARK_LOCAL_API_KEY"),
                "fingerprint": fingerprint,
                "preview": api_key_preview(env_key),
                "source": "env",
                "enabled": True,
                "persistent": True,
                "created_at": "",
            }
        )
    for record in state.temp_api_keys:
        item = dict(record)
        item.setdefault("source", "memory")
        item.setdefault("persistent", False)
        records.append(item)
    return records


def api_key_matches(state: LocalWebState, supplied: str) -> bool:
    if not supplied:
        return False
    supplied_hash = api_key_fingerprint(supplied.strip())
    for record in configured_api_keys(state):
        if not record.get("enabled", True):
            continue
        if secrets.compare_digest(supplied_hash, str(record.get("fingerprint", ""))):
            return True
    return False


def require_api_key(handler: BaseHTTPRequestHandler, state: LocalWebState) -> bool:
    if not state.settings.get("api_auth_required", True):
        return True
    if not configured_api_keys(state):
        json_response(
            handler,
            403,
            {
                "error": "local_api_key_missing",
                "message": "API 保護已啟用，但未設定本地 API Key。請到設定頁新增 API Key，或在 .env 設定 UNMARK_LOCAL_API_KEY。",
            },
        )
        return False
    supplied = (
        handler.headers.get("X-API-Key", "")
        or handler.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    if not api_key_matches(state, supplied):
        json_response(handler, 401, {"error": "unauthorized", "message": "API Key 不正確。"})
        return False
    return True


def page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-HK">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --background: #f6f7f9;
      --foreground: #111827;
      --muted: #667085;
      --muted-bg: #f2f4f7;
      --card: #ffffff;
      --border: #d9dee7;
      --input: #c7ceda;
      --primary: #101828;
      --primary-foreground: #ffffff;
      --ring: #175cd3;
      --danger: #b42318;
      --warning: #b54708;
      --ok: #027a48;
      --sidebar: #ffffff;
      --sidebar-muted: #98a2b3;
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--background); }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--foreground);
      background: var(--background);
      letter-spacing: 0;
      font-size: 14px;
    }}
    a {{ color: inherit; }}
    .app-shell {{ display: grid; grid-template-columns: 224px minmax(0, 1fr); min-height: 100vh; }}
    aside {{
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 18px 14px;
      background: var(--sidebar);
      color: var(--foreground);
      border-right: 1px solid var(--border);
    }}
    .brand {{ display: flex; align-items: center; gap: 10px; margin: 2px 4px 22px; }}
    .brand-title {{ font-weight: 760; font-size: 15px; line-height: 1.15; }}
    .brand-subtitle {{ color: var(--sidebar-muted); font-size: 12px; margin-top: 2px; }}
    nav {{ display: grid; gap: 6px; }}
    nav a {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 10px;
      border-radius: 8px;
      color: #344054;
      text-decoration: none;
      font-size: 14px;
      transition: background .15s ease, color .15s ease;
    }}
    nav a:hover {{ background: #f2f4f7; color: #101828; }}
    nav a span {{ width: 18px; text-align: center; color: var(--sidebar-muted); }}
    main {{ width: 100%; min-width: 0; padding: 24px 28px 40px; max-width: 1440px; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      margin: 0 0 16px;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: .02em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }}
    h1 {{ font-size: 24px; line-height: 1.2; margin: 0; font-weight: 760; }}
    h2 {{ font-size: 17px; line-height: 1.25; margin: 22px 0 10px; font-weight: 720; }}
    h3 {{ font-size: 15px; line-height: 1.25; margin: 16px 0 8px; font-weight: 700; }}
    p, li {{ line-height: 1.6; }}
    .muted {{ color: var(--muted); font-size: 13px; margin-top: 5px; }}
    .panel {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin: 12px 0;
      box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }}
    .warn {{ border-color: #fbbf24; background: #fffbeb; }}
    .ok {{ color: var(--ok); }}
    .danger {{ color: var(--danger); }}
    .notice {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      margin: 0 0 12px;
      font-size: 13px;
      font-weight: 650;
      background: #fff;
    }}
    .notice.success {{ color: var(--ok); background: #ecfdf3; border-color: #abefc6; }}
    .notice.error {{ color: var(--danger); background: #fef3f2; border-color: #fecdca; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 24px;
      padding: 0 9px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #fff;
      color: #334155;
      font-size: 12px;
      font-weight: 650;
      white-space: nowrap;
    }}
    .badge-ok {{ color: #166534; background: #f0fdf4; border-color: #bbf7d0; }}
    .badge-warn {{ color: #92400e; background: #fffbeb; border-color: #fde68a; }}
    .badge-danger {{ color: #991b1b; background: #fef2f2; border-color: #fecaca; }}
    .badge-muted {{ color: #475569; background: #f8fafc; border-color: #e2e8f0; }}
    table {{
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      margin: 12px 0;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(15, 23, 42, .03);
    }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 12px; text-align: left; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ background: #f9fafb; color: #475467; font-size: 12px; font-weight: 720; }}
    td {{ font-size: 13px; }}
    label {{ display: block; font-weight: 650; margin: 13px 0 5px; font-size: 13px; }}
    input, select, textarea {{
      width: 100%;
      max-width: 760px;
      height: 38px;
      padding: 8px 10px;
      border: 1px solid var(--input);
      border-radius: 8px;
      background: #fff;
      font: inherit;
      font-size: 14px;
      outline: none;
      transition: border-color .15s ease, box-shadow .15s ease;
    }}
    input:focus, select:focus, textarea:focus {{
      border-color: var(--ring);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, .14);
    }}
    input[type="file"] {{ height: auto; padding: 12px; background: #f8fafc; border-style: dashed; }}
    input[type="checkbox"] {{ width: auto; height: auto; margin-right: 7px; }}
    textarea {{ min-height: 108px; height: auto; resize: vertical; }}
    button, .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 38px;
      border: 0;
      background: var(--primary);
      color: var(--primary-foreground);
      padding: 9px 14px;
      border-radius: 8px;
      font: inherit;
      font-size: 14px;
      font-weight: 650;
      text-decoration: none;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(15, 23, 42, .12);
    }}
    .button.secondary {{ background: #fff; color: #0f172a; border: 1px solid var(--border); }}
    code, pre {{ background: #f2f4f7; border-radius: 6px; }}
    code {{ padding: 2px 5px; font-size: .92em; }}
    pre {{ padding: 14px; overflow: auto; border: 1px solid var(--border); }}
    .table-wrap {{ width: 100%; overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; background: #fff; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
    .table-wrap table {{ margin: 0; border: 0; border-radius: 0; box-shadow: none; }}
    .jobs-table {{ min-width: 760px; table-layout: fixed; }}
    .jobs-table th:nth-child(1) {{ width: 270px; }}
    .jobs-table th:nth-child(2) {{ width: 150px; }}
    .jobs-table th:nth-child(3), .jobs-table th:nth-child(4) {{ width: 72px; }}
    .jobs-table th:nth-child(5) {{ width: auto; }}
    .jobs-table a {{ font-weight: 650; text-decoration-thickness: 1px; text-underline-offset: 3px; }}
    .path-code {{
      display: inline-block;
      max-width: min(520px, 42vw);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      vertical-align: middle;
    }}
    .path-block {{ display: block; white-space: normal; overflow-wrap: anywhere; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin: 12px 0; }}
    .metric {{ background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
    .metric-label {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    .metric-value {{ margin-top: 4px; font-size: 20px; font-weight: 780; }}
    .section-card {{ background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
    .form-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 10px 18px; }}
    .form-grid .full {{ grid-column: 1 / -1; }}
    @media (max-width: 820px) {{
      .app-shell {{ grid-template-columns: 1fr; }}
      aside {{ position: relative; height: auto; }}
      nav {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      nav a {{ justify-content: center; padding-inline: 6px; }}
      main {{ padding: 18px; }}
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .form-grid {{ grid-template-columns: 1fr; }}
      table:not(.jobs-table) {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <aside>
      <div class="brand">
        <div>
          <div class="brand-title">UNMARK</div>
          <div class="brand-subtitle">批次去水印控制台</div>
        </div>
      </div>
      <nav>
        <a href="/"> 任務</a>
        <a href="/new"> 新增任務</a>
        <a href="/settings"> 設定</a>
        <a href="/api-docs"> API</a>
      </nav>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1>{html.escape(title)}</h1>
        </div>
        <a class="button secondary" href="/new">新增任務</a>
      </div>
      {body}
    </main>
  </div>
</body>
</html>"""


def settings_value(state: LocalWebState, key: str) -> str:
    return html.escape(str(state.settings.get(key, DEFAULT_SETTINGS.get(key, ""))))


SETTING_HELP = {
    "vast_api_key": "Vast API Key。用來建立、查詢、銷毀 Vast 實例。",
    "hf_token": "Hugging Face Token。用來下載模型。",
    "local_api_key": "保護本地 API，避免未獲授權的程式建立付費任務。",
    "model": "模型選擇：9B / 4B / KONTEXT / KWR。",
    "steps": "留空時使用 model_presets.json 內該模型的 default_steps。數值越高，推理時間及費用通常越高，但不保證品質一定較好。",
    "seed": "固定推理的隨機種子，讓相同模型、圖片及 steps 可重現同一結果。一般保留 42。",
    "max_side": "圖片推理時最長邊上限，單位像素。1024 即超過 1024 的圖會等比縮小再處理。",
    "max_download_mb": "每張網址圖片最多可下載的大小，單位 MB；超過上限會標記失敗，不會塞滿記憶體或磁碟。",
    "prompt_preset": "選項來自 Prompts 目錄內的 .md 檔案；檔名就是選項名稱。加入新的 .md 後重新載入此頁即可選用。",
    "gpu_name": "搜尋 Vast 報價時指定的 GPU 型號；預設為 H200。",
    "disk_gb": "Vast 實例的磁碟容量，供模型、依賴、暫存輸入及輸出使用。容量越大，儲存費通常越高。",
    "instance_label": "顯示在 Vast 控制台的實例名稱，亦用於建立結果不確定時尋回實例。",
    "vast_image": "建立 Vast 實例使用的 Docker Image，須包含相容的 Python、PyTorch 及 CUDA 環境。",
    "remote_work_dir": "Vast 實例內存放 worker、manifest、日誌及輸出圖片的工作目錄。",
    "input_csv": "命令列模式的預設輸入 CSV 路徑。",
    "input_images_dir": "命令列模式的本地圖片輸入資料夾。",
    "jobs_dir": "每個任務的狀態、日誌及已下載圖片均存放在此目錄。",
    "localweb_uploads_dir": "LocalWeb 接收圖片、ZIP、CSV 或 TXT 時使用的本地暫存目錄。",
    "all_to_once": "all_to_once_script 使用的匯入、工作及匯出目錄。",
    "batch_size": "每個遠端批次最多處理的圖片數量；已完成圖片仍會逐張下載。",
    "concurrency": "遠端下載及準備輸入圖片的並行數。GPU 推理固定逐張執行，避免共用 pipeline 導致結果損壞。",
    "max_embed_mb": "一個批次內所有本地圖片的上傳大小上限，單位 MB。",
    "job_timeout": "單個遠端批次等待完成的最長時間，單位秒；逾時後會補下載已完成圖片並停機。",
    "bootstrap_timeout": "等待遠端環境安裝及 API 就緒的最長時間，單位秒。",
    "instance_ttl": "實例最長存活秒數。即使本地程式斷線，遠端超時後亦會嘗試自行銷毀，避免無限計費。預設 10800 秒。",
    "remote_result_grace": "遠端任務結束後預留給本地下載的時間。若本地程式消失，超過此時間會由遠端嘗試自行銷毀實例。預設 900 秒。",
    "destroy_on_success": "成功後銷毀 Vast 實例；出錯 / 逾時後固定先補下載，再銷毀實例。",
    "api_auth_required": "建議保持開啟。API Key 可在上方新增多個；設定頁生成的 key 重啟後會消失，若要長期使用，請把生成後的一次性明文放入 .env 的 UNMARK_LOCAL_API_KEY。",
}


def field_help(key: str) -> str:
    return f'<p class="muted">{html.escape(SETTING_HELP[key])}</p>'


def prompt_preset_options(state: LocalWebState, selected: str | None = None) -> str:
    current = str(selected if selected is not None else state.settings.get("prompt_preset", "萬能提示詞"))
    try:
        presets = state.cli.load_prompt_presets()
    except ValueError as exc:
        return f'<option selected disabled>無法載入提示詞：{html.escape(str(exc))}</option>'
    missing = ""
    if current not in presets:
        missing = f'<option selected disabled>找不到：{html.escape(current)}</option>'
    return missing + "".join(
        f'<option value="{html.escape(name)}" {"selected" if name == current else ""}>{html.escape(name)}</option>'
        for name in presets
    )


def status_badge(status: str) -> str:
    value = (status or "unknown").strip()
    lowered = value.lower()
    if lowered in {"done", "reconciled"} or value in {"啟用"}:
        cls = "badge-ok"
    elif lowered in {"error", "localweb_error", "timeout_partial", "download_incomplete"}:
        cls = "badge-danger"
    elif lowered in {
        "running", "bootstrap_waiting", "instance_created", "creating_instance", "initializing",
        "stopping_after_error_or_timeout", "partial",
    }:
        cls = "badge-warn"
    elif value in {"停用"}:
        cls = "badge-muted"
    else:
        cls = "badge-muted"
    labels = {
        "done": "完成",
        "reconciled": "已對帳",
        "error": "失敗",
        "localweb_error": "本地執行失敗",
        "partial": "部分完成",
        "timeout_partial": "逾時，部分完成",
        "download_incomplete": "下載不完整",
        "running": "處理中",
        "bootstrap_waiting": "準備遠端環境",
        "instance_created": "實例已建立",
        "creating_instance": "正在建立實例",
        "initializing": "初始化",
        "stopping_after_error_or_timeout": "補下載並停機中",
        "unknown": "未知",
    }
    label = labels.get(lowered, value)
    return f'<span class="badge {cls}">{html.escape(label)}</span>'


def effective_job_status(job_state: dict, stats: dict) -> str:
    status = str(job_state.get("status", "unknown"))
    if status == "done" and stats.get("error", 0):
        return "partial" if stats.get("ok", 0) else "error"
    if status == "done" and stats.get("downloaded_files", 0) < stats.get("ok", 0):
        return "download_incomplete"
    return status


def render_settings_page(state: LocalWebState, message: str = "", message_level: str = "notice") -> str:
    models = ["9B", "4B", "KONTEXT", "KWR"]
    model_options = "".join(
        f'<option value="{model_name}" '
        f'{"selected" if str(state.settings.get("model", "9B")).upper() == model_name else ""}>'
        f'{model_name}</option>'
        for model_name in models
    )
    notice_class = "notice error" if message_level == "error" else "notice success"
    msg_html = f'<div class="{notice_class}">{html.escape(message)}</div>' if message else ""
    api_rows = []
    for record in configured_api_keys(state):
        record_id = html.escape(str(record.get("id", "")))
        enabled = bool(record.get("enabled", True))
        source = str(record.get("source", ""))
        if source == "env":
            source_label = record.get("note", "環境變數")
        elif source == "memory":
            source_label = "LocalWeb 記憶體"
        else:
            source_label = "LocalWeb 記憶體"
        action_controls = ""
        if source != "env":
            action_controls = (
                f'<button type="submit" name="api_key_action" value="toggle:{record_id}" class="button secondary">'
                f'{"停用" if enabled else "啟用"}</button> '
                f'<button type="submit" name="api_key_action" value="delete:{record_id}" class="button secondary">刪除</button>'
            )
        else:
            action_controls = '<span class="muted">請在 .env 修改</span>'
        api_rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('note', '未命名 API Key')))}</td>"
            f"<td><code>{html.escape(str(record.get('preview', '已保存')))}</code></td>"
            f"<td>{html.escape(source_label)}</td>"
            f"<td>{status_badge('啟用') if enabled else status_badge('停用')}</td>"
            f"<td>{action_controls}</td>"
            "</tr>"
        )
    body = f"""
{msg_html}
<div class="panel warn">
  <p><strong>密鑰說明：</strong>本頁輸入的 Vast / Hugging Face 密鑰只存在目前程序的記憶體，不會寫入 JSON。狀態欄會標示來源；長期使用請設定 <code>.env</code> 內的 <code>VAST_API_KEY</code> 及 <code>HF_TOKEN</code>。</p>
</div>
<form method="post" action="/settings">
  <h2>密鑰狀態</h2>
  <table>
    <tr><th>項目</th><th>狀態</th><th>臨時填寫</th></tr>
    <tr><td>Vast API Key{field_help("vast_api_key")}</td><td>{html.escape(secret_source(state, "vast_api_key"))}</td><td><input type="password" name="vast_api_key" placeholder="留空即不改動"></td></tr>
    <tr><td>Hugging Face Token{field_help("hf_token")}</td><td>{html.escape(secret_source(state, "hf_token"))}</td><td><input type="password" name="hf_token" placeholder="留空即不改動"></td></tr>
  </table>


  <h2>去水印的提示詞</h2>
  <select name="prompt_preset">{prompt_preset_options(state)}</select>
  {field_help("prompt_preset")}

  <h2>本地 API Key 管理</h2>
  <div class="panel">
    <p>填寫用途備註後，系統會隨機生成 API Key。明文只顯示一次；本頁生成的 Key 會在重啟時消失，長期使用請把明文放入 <code>.env</code> 的 <code>UNMARK_LOCAL_API_KEY</code>。</p>
    {field_help("local_api_key")}
    <div class="form-grid">
      <div>
        <label>新 API Key 備註</label>
        <input name="new_api_key_note" placeholder="例如：Zapier、內部後台、測試機">
      </div>
    </div>
  </div>
  <table>
    <tr><th>備註</th><th>Key 預覽</th><th>來源</th><th>狀態</th><th>操作</th></tr>
    {''.join(api_rows) or '<tr><td colspan="5">未設定 API Key</td></tr>'}
  </table>

  <h2>模型與成本</h2>
  <label>模型</label>
  <select name="model">{model_options}</select>
  {field_help("model")}

  <label>Steps</label>
  <input name="steps" value="{settings_value(state, "steps")}" placeholder="留空即使用 model_presets.json 的預設值">
  {field_help("steps")}

  <label>Seed</label>
  <input name="seed" type="number" min="0" value="{settings_value(state, "seed")}">
  {field_help("seed")}

  <label>最長邊像素</label>
  <input name="max_side" value="{settings_value(state, "max_side")}">
  {field_help("max_side")}

  <label>每張網址圖片下載上限 MB</label>
  <input name="max_download_mb" value="{settings_value(state, "max_download_mb")}">
  {field_help("max_download_mb")}

  <label>GPU 型號</label>
  <input name="gpu_name" value="{settings_value(state, "gpu_name")}">
  {field_help("gpu_name")}

  <label>每批圖片數</label>
  <input name="batch_size" value="{settings_value(state, "batch_size")}">
  {field_help("batch_size")}

  <label>輸入下載並行數</label>
  <input name="concurrency" value="{settings_value(state, "concurrency")}">
  {field_help("concurrency")}

  <h2>Vast 臨時機</h2>
  <label>磁碟大小 GB</label>
  <input name="disk_gb" value="{settings_value(state, "disk_gb")}">
  {field_help("disk_gb")}
  <label>實例標籤</label>
  <input name="instance_label" value="{settings_value(state, "instance_label")}">
  {field_help("instance_label")}
  <label>Docker image</label>
  <input name="vast_image" value="{settings_value(state, "vast_image")}">
  {field_help("vast_image")}
  <label>遠端工作目錄</label>
  <input name="remote_work_dir" value="{settings_value(state, "remote_work_dir")}">
  {field_help("remote_work_dir")}

  <h2>本地目錄與逾時</h2>
  <label>Jobs 目錄</label>
  <input name="jobs_dir" value="{settings_value(state, "jobs_dir")}">
  {field_help("jobs_dir")}
  <label>LocalWeb 上傳暫存目錄</label>
  <input name="localweb_uploads_dir" value="{settings_value(state, "localweb_uploads_dir")}">
  {field_help("localweb_uploads_dir")}
  <label>本地圖片上傳大小上限 MB</label>
  <input name="max_embed_mb" value="{settings_value(state, "max_embed_mb")}">
  {field_help("max_embed_mb")}
  <label>單批次逾時秒數</label>
  <input name="job_timeout" value="{settings_value(state, "job_timeout")}">
  {field_help("job_timeout")}
  <label>等待遠端就緒秒數</label>
  <input name="bootstrap_timeout" value="{settings_value(state, "bootstrap_timeout")}">
  {field_help("bootstrap_timeout")}
  <label>實例最長存活秒數</label>
  <input name="instance_ttl" value="{settings_value(state, "instance_ttl")}">
  {field_help("instance_ttl")}
  <label>任務完成後下載寬限秒數</label>
  <input name="remote_result_grace" value="{settings_value(state, "remote_result_grace")}">
  {field_help("remote_result_grace")}

  <h2>API 保護</h2>
  <label><input type="checkbox" name="api_auth_required" value="1" {"checked" if state.settings.get("api_auth_required", True) else ""}> API 需要 API Key</label>
  {field_help("api_auth_required")}
  <label><input type="checkbox" name="destroy_on_success" value="1" {"checked" if state.settings.get("destroy_on_success", True) else ""}> 成功後銷毀 Vast 實例</label>
  {field_help("destroy_on_success")}

  <p><button type="submit">保存設定</button></p>
</form>
"""
    return page_shell("設定", body)


def validate_settings(settings: dict) -> dict:
    candidate = dict(settings)
    model = str(candidate.get("model", "9B")).upper()
    if model not in {"9B", "4B", "KONTEXT", "KWR"}:
        raise ValueError("模型只可選擇 9B、4B、KONTEXT 或 KWR。")
    candidate["model"] = model

    steps = str(candidate.get("steps", "")).strip()
    if steps:
        try:
            if int(steps) < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Steps 必須是大於 0 的整數，或留空使用模型預設值。") from exc
        candidate["steps"] = str(int(steps))

    try:
        seed = int(candidate.get("seed", DEFAULT_SETTINGS["seed"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Seed 必須是大於或等於 0 的整數。") from exc
    if seed < 0:
        raise ValueError("Seed 必須是大於或等於 0 的整數。")
    candidate["seed"] = seed

    positive_fields = {
        "max_side": "最長邊像素",
        "max_download_mb": "每張網址圖片下載上限",
        "disk_gb": "磁碟大小",
        "batch_size": "每批圖片數",
        "concurrency": "輸入下載並行數",
        "max_embed_mb": "本地圖片上傳大小上限",
        "job_timeout": "單批次逾時秒數",
        "bootstrap_timeout": "等待遠端就緒秒數",
        "instance_ttl": "實例最長存活秒數",
        "remote_result_grace": "任務完成後下載寬限秒數",
    }
    for key, label in positive_fields.items():
        try:
            value = int(candidate.get(key, DEFAULT_SETTINGS[key]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}必須是整數。") from exc
        if value < 1:
            raise ValueError(f"{label}必須大於 0。")
        candidate[key] = value

    prompt_preset = str(candidate.get("prompt_preset", "萬能提示詞")).strip()
    if not prompt_preset or Path(prompt_preset).name != prompt_preset:
        raise ValueError("提示詞名稱無效。")
    candidate["prompt_preset"] = prompt_preset
    return candidate


def update_settings_from_form(state: LocalWebState, form: dict[str, str]) -> str:
    candidate = dict(state.settings)
    string_fields = [
        "model", "steps", "prompt_preset", "gpu_name", "instance_label", "vast_image",
        "remote_work_dir", "jobs_dir", "localweb_uploads_dir",
    ]
    int_fields = [
        "seed", "max_side", "max_download_mb", "disk_gb", "batch_size", "concurrency", "max_embed_mb", "job_timeout",
        "bootstrap_timeout", "instance_ttl", "remote_result_grace",
    ]
    for key in string_fields:
        if key in form:
            candidate[key] = form[key].strip()
    for key in int_fields:
        if key in form:
            candidate[key] = form[key].strip()
    candidate["destroy_on_success"] = form.get("destroy_on_success") == "1"
    candidate["api_auth_required"] = form.get("api_auth_required") == "1"
    state.settings = validate_settings(candidate)

    action = form.get("api_key_action", "").strip()
    if action:
        api_key_action(state, action)
    for secret_key in SECRET_ENV_NAMES:
        value = form.get(secret_key, "").strip()
        if value:
            state.temp_secrets[secret_key] = value
    generated_key = ""
    note = form.get("new_api_key_note", "").strip()
    if note:
        new_key = generate_local_api_key()
        record = new_api_key_record(
            new_key,
            note,
            "memory",
            enabled=True,
            persistent=False,
        )
        state.temp_api_keys.append(record)
        generated_key = new_key
    return generated_key


def api_key_action(state: LocalWebState, action: str) -> None:
    if ":" not in action:
        return
    command, record_id = action.split(":", 1)
    record_id = record_id.strip()
    if not record_id:
        return

    if command == "delete":
        state.temp_api_keys = [
            record for record in state.temp_api_keys
            if str(record.get("id", "")) != record_id
        ]
        return

    if command == "toggle":
        for record in state.temp_api_keys:
            if str(record.get("id", "")) == record_id:
                record["enabled"] = not bool(record.get("enabled", True))
                return


def read_urlencoded(handler: BaseHTTPRequestHandler, max_body_bytes: int = 256 * 1024) -> dict[str, str]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Content-Length 無效。") from exc
    if length < 0 or length > max_body_bytes:
        raise ValueError("設定表單內容過大。")
    body = handler.rfile.read(length).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def read_json_body(handler: BaseHTTPRequestHandler, max_body_bytes: int = 4 * 1024 * 1024):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Content-Length 無效。") from exc
    if length < 1 or length > max_body_bytes:
        raise ValueError(f"JSON 內容大小必須介乎 1 byte 至 {max_body_bytes} bytes。")
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def parse_header_params(value: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in value.split(";")]
    main = parts[0].lower() if parts else ""
    params = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        params[key.strip().lower()] = raw.strip().strip('"')
    return main, params


def cleanup_temp_uploads(files: list[tuple[str, bytes | Path]]) -> None:
    for _, content in files:
        if isinstance(content, Path):
            content.unlink(missing_ok=True)


def read_multipart(
    handler: BaseHTTPRequestHandler,
    max_body_bytes: int,
) -> tuple[dict[str, str], list[tuple[str, Path]]]:
    try:
        from python_multipart.multipart import create_form_parser
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "缺少本地依賴 python-multipart。請使用本項目的 venv 執行："
            "venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    content_type, params = parse_header_params(handler.headers.get("Content-Type", ""))
    if content_type != "multipart/form-data" or not params.get("boundary"):
        raise ValueError("Content-Type 必須是 multipart/form-data。")
    try:
        length = int(handler.headers.get("Content-Length", ""))
    except ValueError as exc:
        raise ValueError("Content-Length 無效。") from exc
    if length < 1 or length > max_body_bytes:
        raise ValueError(f"上傳內容超過限制：最多 {max_body_bytes // 1024 // 1024} MB。")

    fields: dict[str, str] = {}
    files: list[tuple[str, Path]] = []
    upload_handles = []

    def on_field(field) -> None:
        if len(fields) >= 100:
            raise ValueError("表單欄位過多。")
        name = (field.field_name or b"").decode("utf-8", errors="replace")
        value = field.value or b""
        if len(value) > 64 * 1024:
            raise ValueError(f"表單欄位 {name!r} 過大。")
        fields[name] = value.decode("utf-8", errors="replace")

    def on_file(upload) -> None:
        if len(files) >= 1000:
            raise ValueError("一次上傳的檔案數量超過 1000。")
        filename = (upload.file_name or b"upload.bin").decode("utf-8", errors="replace")
        upload_handles.append(upload)
        suffix = Path(filename).suffix[:16]
        file_descriptor, temp_name = tempfile.mkstemp(prefix="unmark_upload_", suffix=suffix)
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        try:
            upload.file_object.seek(0)
            with temp_path.open("wb") as output:
                shutil.copyfileobj(upload.file_object, output, length=1024 * 1024)
            files.append((filename, temp_path))
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    headers = {
        "Content-Type": handler.headers.get("Content-Type", "").encode("latin-1"),
        "Content-Length": str(length).encode("ascii"),
    }
    parser = create_form_parser(
        headers,
        on_field,
        on_file,
        config={
            "MAX_BODY_SIZE": max_body_bytes,
            "MAX_MEMORY_FILE_SIZE": 1024 * 1024,
            "MAX_HEADER_COUNT": 32,
            "MAX_HEADER_SIZE": 8192,
        },
    )
    remaining = length
    try:
        while remaining:
            chunk = handler.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"上傳中斷，尚欠 {remaining} bytes。")
            parser.write(chunk)
            remaining -= len(chunk)
        parser.finalize()
        return fields, files
    except Exception:
        cleanup_temp_uploads(files)
        raise
    finally:
        parser.close()
        for upload in upload_handles:
            upload.close()


def args_from_settings(
    state: LocalWebState,
    input_csv: Path,
    job_id: str,
    overrides: dict | None = None,
    settings_snapshot: dict | None = None,
) -> SimpleNamespace:
    settings = {
        **DEFAULT_SETTINGS,
        **dict(settings_snapshot if settings_snapshot is not None else state.settings),
    }
    overrides = overrides or {}
    for key, value in overrides.items():
        if value not in (None, ""):
            settings[key] = value
    settings = validate_settings(settings)
    steps_raw = settings["steps"]
    steps = int(steps_raw) if str(steps_raw).strip() else None
    return SimpleNamespace(
        command="run",
        model=str(settings["model"]).upper(),
        steps=steps,
        seed=settings["seed"],
        gpu_name=str(settings["gpu_name"]),
        disk_gb=settings["disk_gb"],
        max_side=settings["max_side"],
        max_download_mb=settings["max_download_mb"],
        prompt_preset=str(settings["prompt_preset"]),
        batch_size=settings["batch_size"],
        concurrency=settings["concurrency"],
        input_csv=str(input_csv),
        input_images_dir=None,
        jobs_dir=str((state.root / str(settings["jobs_dir"])).resolve()),
        job_dir=None,
        job_id=job_id,
        instance_label=str(settings["instance_label"]),
        max_embed_mb=settings["max_embed_mb"],
        job_timeout=settings["job_timeout"],
        bootstrap_timeout=settings["bootstrap_timeout"],
        instance_ttl=settings["instance_ttl"],
        remote_result_grace=settings["remote_result_grace"],
        remote_work_dir=str(settings["remote_work_dir"]),
        vast_image=str(settings["vast_image"]),
        vast_api=None,
        destroy_on_success=bool(settings["destroy_on_success"]),
        destroy_on_error=True,
    )


def assert_can_start_job(state: LocalWebState) -> None:
    if getattr(state.cli, "requests", None) is None:
        hint = getattr(state.cli, "dependency_install_hint", lambda name: f"缺少本地依賴 {name}。")
        raise RuntimeError(hint("requests"))
    if state.job_lock.locked():
        raise RuntimeError("已有任務正在執行。請等它完成後再開新任務。")
    if not get_secret(state, "vast_api_key"):
        raise RuntimeError("未設定 Vast API Key。請到設定頁填寫，或在 .env 設定 VAST_API_KEY。")
    if not get_secret(state, "hf_token"):
        raise RuntimeError("未設定 Hugging Face Token。請到設定頁填寫，或在 .env 設定 HF_TOKEN。")


def launch_job(
    state: LocalWebState,
    input_csv: Path,
    overrides: dict | None = None,
    cleanup_dir: Path | None = None,
) -> str:
    assert_can_start_job(state)
    with state.settings_lock:
        settings_snapshot = validate_settings(dict(state.settings))
        secret_snapshot = {name: get_secret(state, name) for name in SECRET_ENV_NAMES}
    job_id = time.strftime("localweb_%Y%m%d_%H%M%S_") + secrets.token_hex(4)
    args = args_from_settings(
        state,
        input_csv,
        job_id,
        overrides,
        settings_snapshot=settings_snapshot,
    )
    if not state.job_lock.acquire(blocking=False):
        raise RuntimeError("已有任務正在執行。請等它完成後再開新任務。")
    state.active_job_id = job_id

    def worker() -> None:
        try:
            state.cli.run_vast_job(args, resume=False, secret_overrides=secret_snapshot)
        except BaseException as exc:
            job_dir = Path(args.jobs_dir) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            state.cli.write_job_state(job_dir, {"status": "localweb_error", "error": str(exc)})
            state.cli.append_job_event(job_dir, "localweb_error", {"error": str(exc)})
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            state.active_job_id = ""
            state.job_lock.release()

    thread = threading.Thread(target=worker, name=f"localweb-job-{job_id}", daemon=False)
    state.active_thread = thread
    try:
        thread.start()
    except BaseException:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        state.active_job_id = ""
        state.job_lock.release()
        raise
    return job_id


def render_jobs_page(state: LocalWebState) -> str:
    jobs_dir = (state.root / str(state.settings.get("jobs_dir", "jobs"))).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    job_count = 0
    total_ok = 0
    total_error = 0
    for job_dir in sorted(jobs_dir.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        state_payload = state.cli.read_job_state(job_dir)
        stats = state.cli.read_output_stats(job_dir)
        job_count += 1
        total_ok += int(stats["ok"])
        total_error += int(stats["error"])
        full_path = str(job_dir)
        short_path = f"jobs/{job_dir.name}"
        rows.append(
            "<tr>"
            f"<td><a href=\"/job/{html.escape(job_dir.name)}\">{html.escape(job_dir.name)}</a></td>"
            f"<td>{status_badge(effective_job_status(state_payload, stats))}</td>"
            f"<td>{stats['ok']}</td>"
            f"<td>{stats['error']}</td>"
            f"<td><code class=\"path-code\" title=\"{html.escape(full_path)}\">{html.escape(short_path)}</code></td>"
            "</tr>"
        )
    active = f"<p class=\"ok\">正在執行：{html.escape(state.active_job_id)}</p>" if state.active_job_id else ""
    body = f"""
{active}
<div class="metric-grid">
  <div class="metric"><div class="metric-label">任務數</div><div class="metric-value">{job_count}</div></div>
  <div class="metric"><div class="metric-label">已下載成功</div><div class="metric-value">{total_ok}</div></div>
  <div class="metric"><div class="metric-label">失敗</div><div class="metric-value">{total_error}</div></div>
  <div class="metric"><div class="metric-label">目前任務</div><div class="metric-value">{'1' if state.active_job_id else '0'}</div></div>
</div>
<div class="section-card">
<h2>任務列表</h2>
<div class="table-wrap">
<table class="jobs-table">
  <thead><tr><th>任務 ID</th><th>狀態</th><th>成功</th><th>失敗</th><th>路徑</th></tr></thead>
  <tbody>{''.join(rows) or '<tr><td colspan="5">未有 job</td></tr>'}</tbody>
</table>
</div>
</div>
"""
    return page_shell("任務", body)


def render_new_job_page(state: LocalWebState, message: str = "") -> str:
    model = html.escape(str(state.settings.get("model", "9B")).upper())
    msg_html = f'<div class="panel warn">{html.escape(message)}</div>' if message else ""
    model_options = [
        ("9B", "9B｜價錢與效果的平衡"),
        ("4B", "4B｜低成本測試"),
        ("KONTEXT", "KONTEXT｜無 LoRA 對照"),
        ("KWR", "KWR｜較慢，適合頑固水印補處理"),
    ]
    body = f"""
{msg_html}
<div class="panel">
  <h2>可以上傳甚麼？</h2>
  <p><strong>圖片：</strong>支援 jpg、jpeg、png、webp、bmp，可以一次選多張。</p>
  <p><strong>ZIP：</strong>把一批圖片壓成 zip 上傳，系統只會讀裏面的圖片。</p>
  <p><strong>CSV：</strong>標題列可用 <code>文件名</code> 和 <code>要去水印的link</code>。文件名會用作輸出檔名；link 是圖片網址。</p>
  <p><strong>TXT：</strong>一行一個圖片 link，不能指定文件名，系統會自動產生名字。</p>
  <p class="muted">雲端 GPU 需要準備環境及載入模型；少量圖片通常不划算，本項目較適合大量批次。</p>
</div>
<form method="post" action="/new" enctype="multipart/form-data">
  <label>選擇檔案</label>
  <input type="file" name="input_files" multiple>
  <p class="muted">如果是 CSV，最簡單格式如下：</p>
  <pre>文件名,要去水印的link
客廳_001,https://example.com/photo1.jpg
睡房_002,https://example.com/photo2.jpg</pre>

  <label>模型</label>
  <select name="model">
    {"".join(f'<option value="{value}" {"selected" if value == model else ""}>{label}</option>' for value, label in model_options)}
  </select>

  <label>Steps</label>
  <input name="steps" placeholder="留空即使用模型預設值">
  <p class="muted">不懂就留空。留空時使用 model_presets.json 的預設步數；步數越高通常越慢。</p>

  <label>Seed</label>
  <input name="seed" type="number" min="0" value="{settings_value(state, "seed")}">

  <label>提示詞</label>
  <select name="prompt_preset">{prompt_preset_options(state)}</select>
  {field_help("prompt_preset")}

  <button type="submit">建立並執行任務</button>
</form>
"""
    return page_shell("新增任務", body)


def render_job_page(state: LocalWebState, job_name: str) -> str:
    jobs_dir = (state.root / str(state.settings.get("jobs_dir", "jobs"))).resolve()
    job_dir = (jobs_dir / job_name).resolve()
    if not job_dir.is_relative_to(jobs_dir) or not job_dir.is_dir():
        return page_shell("找不到 Job", "<div class=\"panel warn\">找不到這個 Job。</div>")
    job_state = state.cli.read_job_state(job_dir)
    stats = state.cli.read_output_stats(job_dir)
    effective_status = effective_job_status(job_state, stats)
    error_notice = ""
    if stats["error"]:
        if stats["downloaded_files"]:
            message = f"{stats['error']} 張處理失敗；已下載 {stats['downloaded_files']} 張成功圖片。"
        else:
            message = f"處理失敗：{stats['error']} 張；沒有產生輸出圖片。"
        error_notice = f'<div class="notice error">{html.escape(message)}</div>'
    download_action = ""
    if stats["ok"]:
        download_action = f'<a class="button" href="/job/{html.escape(job_name)}/download.zip">下載全部結果</a>'
    events = []
    events_path = job_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines()[-40:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    event_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(event.get('time', '')))}</td>"
        f"<td>{html.escape(str(event.get('event', '')))}</td>"
        f"<td><code>{html.escape(json.dumps(event.get('payload', {}), ensure_ascii=False))}</code></td>"
        "</tr>"
        for event in events
    )
    body = f"""
{error_notice}
<p><a href="/">返回任務</a> {download_action}</p>
<table>
  <tr><th>狀態</th><td>{status_badge(effective_status)}</td></tr>
  <tr><th>實例 ID</th><td>{html.escape(str(job_state.get('instance_id', '')))}</td></tr>
  <tr><th>遠端 API</th><td>{html.escape(str(job_state.get('url_base', '')))}</td></tr>
  <tr><th>已下載成功</th><td>{stats['ok']}</td></tr>
  <tr><th>失敗</th><td>{stats['error']}</td></tr>
  <tr><th>輸出檔案數</th><td>{stats['downloaded_files']}</td></tr>
  <tr><th>預估費用（USD）</th><td>{html.escape(str(job_state.get('estimated_cost_usd', '')))}</td></tr>
  <tr><th>圖片輸出目錄</th><td><code class="path-block">{html.escape(str(job_dir / 'output_images'))}</code></td></tr>
  <tr><th>路徑</th><td><code class="path-block">{html.escape(str(job_dir))}</code></td></tr>
</table>
<h2>任務狀態</h2>
<pre>{html.escape(json.dumps(job_state, indent=2, ensure_ascii=False))}</pre>
<h2>最近事件</h2>
<table><thead><tr><th>時間</th><th>事件</th><th>資料</th></tr></thead><tbody>{event_rows or '<tr><td colspan="3">未有事件</td></tr>'}</tbody></table>
"""
    return page_shell(job_name, body)


def build_job_output_zip(job_dir: Path) -> Path:
    output_dir = job_dir / "output_images"
    output_files = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if output_dir.is_dir() else []
    if not output_files:
        raise FileNotFoundError("這個任務沒有可下載的輸出圖片。")
    zip_path = job_dir / f"UNMARK_{job_dir.name}_results.zip"
    source_files = output_files + [
        path for path in (job_dir / "output.csv", job_dir / "job_state.json") if path.is_file()
    ]
    latest_source_mtime = max(path.stat().st_mtime_ns for path in source_files)
    if zip_path.is_file() and zip_path.stat().st_mtime_ns >= latest_source_mtime:
        return zip_path
    temp_path = zip_path.with_name(f".{zip_path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in output_files:
                archive.write(path, f"output_images/{path.name}")
            for name in ("output.csv", "job_state.json"):
                path = job_dir / name
                if path.is_file():
                    archive.write(path, name)
        os.replace(temp_path, zip_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return zip_path


def render_api_docs() -> str:
    body = """
<div class="panel">
  <p>API 用來給其他程式呼叫。預設需要 API Key，請在設定頁的「本地 API Key 管理」新增，或在 <code>.env</code> 設定 <code>UNMARK_LOCAL_API_KEY</code>。</p>
  <p>呼叫時使用 header：<code>X-API-Key: 你的本地APIKey</code>。</p>
</div>
<h2>健康檢查</h2>
<pre>curl http://127.0.0.1:8787/api/health \\
  -H "X-API-Key: 你的本地APIKey"</pre>
<h2>建立任務：JSON links</h2>
<pre>curl -X POST http://127.0.0.1:8787/api/jobs \\
  -H "X-API-Key: 你的本地APIKey" \\
  -H "Content-Type: application/json" \\
  -d '{"items":[{"filename":"客廳_001","link":"https://example.com/photo.jpg"}],"prompt_preset":"萬能提示詞"}'</pre>
<h2>建立任務：上傳 CSV / TXT / ZIP / 圖片</h2>
<pre>curl -X POST http://127.0.0.1:8787/api/jobs \\
  -H "X-API-Key: 你的本地APIKey" \\
  -F "input_files=@links.csv" \\
  -F "model=9B" \\
  -F "prompt_preset=萬能提示詞"</pre>
<h2>CSV 格式</h2>
<pre>文件名,要去水印的link
客廳_001,https://example.com/photo1.jpg
睡房_002,https://example.com/photo2.jpg</pre>
<p><strong>TXT：</strong>只支援一行一個 link，不支援指定文件名。</p>
"""
    return page_shell("API", body)


def make_handler(state: LocalWebState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            path = unquote(parsed_url.path)
            query = parse_qs(parsed_url.query)
            if path == "/":
                text_response(self, 200, render_jobs_page(state))
                return
            if path == "/settings":
                message, level = pop_flash(state, (query.get("notice") or [""])[-1])
                text_response(self, 200, render_settings_page(state, message, level or "notice"))
                return
            if path == "/new":
                message, _ = pop_flash(state, (query.get("error") or [""])[-1])
                text_response(self, 200, render_new_job_page(state, message))
                return
            if path == "/api-docs":
                text_response(self, 200, render_api_docs())
                return
            if path.startswith("/job/") and path.endswith("/download.zip"):
                job_name = path.removeprefix("/job/").removesuffix("/download.zip").strip("/")
                jobs_dir = (state.root / str(state.settings.get("jobs_dir", "jobs"))).resolve()
                job_dir = (jobs_dir / job_name).resolve()
                if not job_dir.is_relative_to(jobs_dir) or not job_dir.is_dir():
                    self.send_error(404)
                    return
                try:
                    zip_path = build_job_output_zip(job_dir)
                except FileNotFoundError:
                    self.send_error(404)
                    return
                file_response(self, zip_path, zip_path.name)
                return
            if path.startswith("/job/"):
                text_response(self, 200, render_job_page(state, path.removeprefix("/job/").strip("/")))
                return
            if path == "/api/health":
                if not require_api_key(self, state):
                    return
                json_response(self, 200, {"status": "ok", "active_job_id": state.active_job_id})
                return
            if path == "/api/jobs":
                if not require_api_key(self, state):
                    return
                jobs_dir = (state.root / str(state.settings.get("jobs_dir", "jobs"))).resolve()
                payload = []
                if jobs_dir.exists():
                    for job_dir in sorted(jobs_dir.iterdir(), reverse=True):
                        if job_dir.is_dir():
                            payload.append({"job_id": job_dir.name, "state": state.cli.read_job_state(job_dir)})
                json_response(self, 200, payload)
                return
            if path.startswith("/api/jobs/"):
                if not require_api_key(self, state):
                    return
                job_id = path.removeprefix("/api/jobs/").strip("/")
                jobs_dir = (state.root / str(state.settings.get("jobs_dir", "jobs"))).resolve()
                job_dir = (jobs_dir / job_id).resolve()
                if not job_dir.is_relative_to(jobs_dir) or not job_dir.is_dir():
                    json_response(self, 404, {"error": "not_found"})
                    return
                json_response(self, 200, {"job_id": job_id, "state": state.cli.read_job_state(job_dir), "stats": state.cli.read_output_stats(job_dir)})
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/settings":
                    form = read_urlencoded(self)
                    with state.settings_lock:
                        generated_key = update_settings_from_form(state, form)
                        save_settings(state.settings_path, state.settings)
                    message = "設定已保存。Vast / Hugging Face 臨時密鑰只在記憶體保存；設定頁生成的本地 API Key 亦只存在記憶體。"
                    if generated_key:
                        message += f" 新 API Key 只顯示一次：{generated_key}"
                    notice = queue_flash(state, message)
                    redirect_response(self, f"/settings?notice={notice}")
                    return
                if path == "/new":
                    limit = (int(state.settings.get("max_embed_mb", 2000)) + 16) * 1024 * 1024
                    fields, files = read_multipart(self, limit)
                    try:
                        job_id = handle_job_submission(state, fields, files)
                    finally:
                        cleanup_temp_uploads(files)
                    redirect_response(self, f"/job/{job_id}")
                    return
                if path == "/api/settings":
                    if not require_api_key(self, state):
                        return
                    payload = read_json_body(self, max_body_bytes=256 * 1024)
                    if not isinstance(payload, dict):
                        raise ValueError("settings payload must be an object")
                    with state.settings_lock:
                        candidate = dict(state.settings)
                        for key in DEFAULT_SETTINGS:
                            if key in payload:
                                candidate[key] = payload[key]
                        state.settings = validate_settings(candidate)
                        save_settings(state.settings_path, state.settings)
                    json_response(self, 200, {"status": "saved"})
                    return
                if path == "/api/jobs":
                    if not require_api_key(self, state):
                        return
                    content_type = self.headers.get("Content-Type", "")
                    if content_type.startswith("multipart/form-data"):
                        limit = (int(state.settings.get("max_embed_mb", 2000)) + 16) * 1024 * 1024
                        fields, files = read_multipart(self, limit)
                        try:
                            job_id = handle_job_submission(state, fields, files)
                        finally:
                            cleanup_temp_uploads(files)
                    else:
                        payload = read_json_body(self)
                        job_id = handle_json_submission(state, payload)
                    json_response(self, 202, {"status": "accepted", "job_id": job_id, "job_url": f"/job/{job_id}"})
                    return
            except (ValueError, RuntimeError) as exc:
                if path.startswith("/api/"):
                    status = 409 if isinstance(exc, RuntimeError) else 400
                    json_response(self, status, {"error": "request_rejected", "message": str(exc)})
                elif path == "/settings":
                    notice = queue_flash(state, str(exc), "error")
                    redirect_response(self, f"/settings?notice={notice}")
                else:
                    notice = queue_flash(state, str(exc), "error")
                    redirect_response(self, f"/new?error={notice}")
                return
            except Exception:
                traceback.print_exc()
                if path.startswith("/api/"):
                    json_response(self, 500, {"error": "internal_error", "message": "伺服器處理請求時發生錯誤。"})
                elif path == "/settings":
                    notice = queue_flash(state, "保存設定時發生錯誤，詳情請查看終端輸出。", "error")
                    redirect_response(self, f"/settings?notice={notice}")
                else:
                    notice = queue_flash(state, "建立任務時發生錯誤，詳情請查看終端輸出。", "error")
                    redirect_response(self, f"/new?error={notice}")
                return
            self.send_error(404)

    return Handler


def handle_job_submission(
    state: LocalWebState,
    fields: dict[str, str],
    files: list[tuple[str, bytes | Path]],
) -> str:
    assert_can_start_job(state)
    if not files:
        raise ValueError("請先選擇圖片、ZIP、CSV 或 TXT。")
    upload_root = (state.root / str(state.settings.get("localweb_uploads_dir", "localweb_uploads"))).resolve()
    upload_dir = upload_root / (time.strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(4))
    max_total_bytes = int(state.settings.get("max_embed_mb", 2000)) * 1024 * 1024
    try:
        input_csv = normalize_uploaded_files(upload_dir, files, max_total_bytes)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    overrides = {
        "model": fields.get("model", ""),
        "steps": fields.get("steps", ""),
        "seed": fields.get("seed", ""),
        "prompt_preset": fields.get("prompt_preset", ""),
    }
    try:
        return launch_job(state, input_csv, overrides, cleanup_dir=upload_dir)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise


def handle_json_submission(state: LocalWebState, payload: dict) -> str:
    assert_can_start_job(state)
    upload_root = (state.root / str(state.settings.get("localweb_uploads_dir", "localweb_uploads"))).resolve()
    upload_dir = upload_root / (time.strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(4))
    try:
        input_csv = normalize_json_payload(upload_dir, payload)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    overrides = {
        "model": payload.get("model", ""),
        "steps": payload.get("steps", ""),
        "seed": payload.get("seed", ""),
        "prompt_preset": payload.get("prompt_preset", ""),
    }
    try:
        return launch_job(state, input_csv, overrides, cleanup_dir=upload_dir)
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise


def is_running_unmark_localweb(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        connection = http.client.HTTPConnection(probe_host, port, timeout=1.5)
        connection.request("GET", "/")
        response = connection.getresponse()
        marker = response.getheader("X-UNMARK-LocalWeb", "")
        body = response.read(65536) if not marker else b""
        connection.close()
        return marker == "1" or b'<div class="brand-title">UNMARK</div>' in body
    except (OSError, http.client.HTTPException):
        return False


def serve_local(args, cli_module) -> None:
    root = cli_module.ROOT
    cli_module.load_env_file()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("LocalWeb 設定頁沒有登入功能，只可監聽 127.0.0.1、localhost 或 ::1。")
    settings_path = Path(getattr(args, "settings_json", "") or (root / "localweb_settings.json")).expanduser()
    if not settings_path.is_absolute():
        settings_path = (root / settings_path).resolve()
    settings = load_settings(settings_path)
    if getattr(args, "jobs_dir", None):
        settings["jobs_dir"] = args.jobs_dir
    state = LocalWebState(root=root, cli=cli_module, settings_path=settings_path, settings=settings)
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        url = f"http://{url_host}:{args.port}"
        if is_running_unmark_localweb(args.host, args.port):
            print(f"UNMARK LocalWeb 已在運行：{url}")
            return
        raise SystemExit(
            f"無法啟動：{args.host}:{args.port} 已被其他程式使用。\n"
            f"請關閉佔用該連接埠的程式，或改用：python3 run_unmark.py serve-local --port {args.port + 1}"
        ) from None
    print(f"LocalWeb 已啟動：http://{args.host}:{args.port}")
    print("Vast / Hugging Face 臨時密鑰只會存在記憶體；設定頁生成的本地 API Key 亦只存在記憶體。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLocalWeb 已停止")
