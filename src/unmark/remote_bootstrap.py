#!/usr/bin/env python3
"""Vast 實例內的檔案傳輸及任務控制服務。"""
from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WORK_DIR = Path(os.environ.get("WORK_DIR", "/workspace/unmark")).resolve()
REMOTE_API_TOKEN = os.environ.get("REMOTE_API_TOKEN", "")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", "2147483648"))
INSTANCE_TTL_S = int(os.environ.get("INSTANCE_TTL_S", "10800"))
REMOTE_RESULT_GRACE_S = int(os.environ.get("REMOTE_RESULT_GRACE_S", "900"))
REMOTE_START_TTL_S = int(os.environ.get("REMOTE_START_TTL_S", "1200"))
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", "8080"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "output_images").mkdir(parents=True, exist_ok=True)
(WORK_DIR / "raw_images").mkdir(parents=True, exist_ok=True)
PRESERVE_FILE = WORK_DIR / ".preserve_instance"
current_process: subprocess.Popen | None = None
last_controller_activity = time.monotonic()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def inside_work_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(WORK_DIR)
        return True
    except ValueError:
        return False


def destroy_own_instance(reason: str) -> None:
    api_key = os.environ.get("CONTAINER_API_KEY", "")
    instance_id = os.environ.get("CONTAINER_ID", "")
    if not api_key or not instance_id:
        return
    log_path = WORK_DIR / "watchdog.log"
    last_error = ""
    for attempt in range(1, 4):
        try:
            request = Request(
                f"https://console.vast.ai/api/v0/instances/{instance_id}/",
                method="DELETE",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urlopen(request, timeout=30) as response:
                response.read()
            log_path.write_text(f"已因「{reason}」銷毀實例\n", encoding="utf-8")
            return
        except HTTPError as exc:
            if exc.code == 404:
                log_path.write_text(f"實例已不存在（{reason}）\n", encoding="utf-8")
                return
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        if attempt < 3:
            time.sleep(5)
    log_path.write_text(f"自動銷毀失敗（已重試 3 次）：{last_error}\n", encoding="utf-8")


def hard_ttl_watchdog() -> None:
    if INSTANCE_TTL_S > 0:
        time.sleep(INSTANCE_TTL_S)
        if not PRESERVE_FILE.exists():
            destroy_own_instance("超過實例最長存活時間")


def start_ttl_watchdog() -> None:
    if REMOTE_START_TTL_S > 0:
        time.sleep(REMOTE_START_TTL_S)
        if current_process is None and not PRESERVE_FILE.exists():
            destroy_own_instance("本地端未有啟動首個任務")


def completed_job_watchdog(process: subprocess.Popen) -> None:
    global current_process
    return_code = process.wait()
    status_path = WORK_DIR / "job_status.json"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    if payload.get("status") not in {"done", "partial", "error"}:
        payload.update(
            {
                "status": "error",
                "error": f"遠端工作程序意外結束（exit code {return_code}）",
                "ok": int(payload.get("ok", 0)),
                "rows": payload.get("rows", []),
            }
        )
        atomic_write_json(status_path, payload)
        atomic_write_json(WORK_DIR / "result.json", payload)
    if REMOTE_RESULT_GRACE_S <= 0:
        return
    while current_process is process:
        idle_seconds = time.monotonic() - last_controller_activity
        remaining = REMOTE_RESULT_GRACE_S - idle_seconds
        if remaining <= 0:
            if not PRESERVE_FILE.exists():
                destroy_own_instance("任務完成後本地端未有及時銷毀")
            return
        time.sleep(min(5, remaining))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        global last_controller_activity
        supplied = self.headers.get("X-UNMARK-Token", "")
        if REMOTE_API_TOKEN and hmac.compare_digest(supplied, REMOTE_API_TOKEN):
            last_controller_activity = time.monotonic()
            return True
        self._send(401, {"status": "unauthorized"})
        return False

    def do_GET(self) -> None:
        if not self._authorized():
            return
        request_path = unquote(urlparse(self.path).path)
        if request_path == "/health":
            is_installed = (WORK_DIR / ".installed").exists()
            install_failed = (WORK_DIR / ".install_failed").exists()
            status = "error" if install_failed else ("ready" if is_installed else "installing")
            self._send(200, {"status": status})
            return

        relative_path = request_path.lstrip("/")
        allowed = {"job_status.json", "result.json", "run.log", "install.log", "watchdog.log"}
        if relative_path not in allowed and not relative_path.startswith("output_images/"):
            self.send_error(404)
            return
        target = (WORK_DIR / relative_path).resolve()
        if not inside_work_dir(target) or not target.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("X-Content-SHA256", sha256_file(target))
        self.end_headers()
        with target.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                self.wfile.write(chunk)

    def do_PUT(self) -> None:
        if not self._authorized():
            return
        request_path = unquote(urlparse(self.path).path)
        targets = {
            "/remote_worker.py": WORK_DIR / "remote_worker.py",
            "/manifest.json": WORK_DIR / "manifest.json",
        }
        target = targets.get(request_path)
        if target is None:
            if not request_path.startswith("/raw_images/"):
                self.send_error(404)
                return
            filename = os.path.basename(request_path)
            if not filename or filename in {".", ".."}:
                self._send(400, {"status": "invalid_filename"})
                return
            target = WORK_DIR / "raw_images" / filename
        if not inside_work_dir(target):
            self._send(400, {"status": "invalid_path"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_UPLOAD_BYTES:
            self._send(413, {"status": "invalid_size", "size_bytes": length})
            return

        expected_sha256 = self.headers.get("X-Content-SHA256", "").lower()
        temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        digest = hashlib.sha256()
        remaining = length
        try:
            with temp_path.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"上傳中斷，尚欠 {remaining} bytes")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual_sha256 = digest.hexdigest()
            if expected_sha256 and not hmac.compare_digest(expected_sha256, actual_sha256):
                raise ValueError("SHA-256 不符")
            os.replace(temp_path, target)
            self._send(
                200,
                {
                    "status": "uploaded",
                    "path": str(target),
                    "size_bytes": length,
                    "sha256": actual_sha256,
                },
            )
        except ValueError as exc:
            temp_path.unlink(missing_ok=True)
            self._send(400, {"status": "upload_failed", "error": str(exc)})
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            self._send(500, {"status": "upload_failed", "error": str(exc)})

    def do_POST(self) -> None:
        global current_process
        if not self._authorized():
            return
        request_path = unquote(urlparse(self.path).path)
        if request_path == "/preserve":
            PRESERVE_FILE.touch(mode=0o600, exist_ok=True)
            self._send(200, {"status": "preserved"})
            return
        if request_path != "/start":
            self.send_error(404)
            return
        if current_process is not None and current_process.poll() is None:
            self._send(409, {"status": "busy"})
            return
        worker_path = WORK_DIR / "remote_worker.py"
        manifest_path = WORK_DIR / "manifest.json"
        if not worker_path.is_file() or not manifest_path.is_file():
            self._send(400, {"status": "missing_payload"})
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._send(400, {"status": "invalid_manifest", "error": str(exc)})
            return

        invalid_inputs = []
        for item in manifest.get("items", []):
            if item.get("source_type") != "local":
                continue
            target = WORK_DIR / "raw_images" / os.path.basename(str(item.get("filename", "")))
            expected_size = int(item.get("size_bytes", -1))
            expected_sha256 = str(item.get("sha256", "")).lower()
            if not target.is_file():
                invalid_inputs.append({"filename": target.name, "error": "missing"})
                continue
            if expected_size >= 0 and target.stat().st_size != expected_size:
                invalid_inputs.append({"filename": target.name, "error": "size_mismatch"})
                continue
            if expected_sha256 and not hmac.compare_digest(sha256_file(target), expected_sha256):
                invalid_inputs.append({"filename": target.name, "error": "sha256_mismatch"})
        if invalid_inputs:
            self._send(400, {"status": "invalid_inputs", "items": invalid_inputs})
            return

        for name in ("job_status.json", "result.json", "run.log"):
            path = WORK_DIR / name
            if path.exists():
                path.unlink()
        script = WORK_DIR / "run_remote.sh"
        script.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\ncd "$WORK_DIR"\n'
            'python3 "$WORK_DIR/remote_worker.py" > "$WORK_DIR/run.log" 2>&1\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
        current_process = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(WORK_DIR),
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=completed_job_watchdog, args=(current_process,), daemon=True).start()
        self._send(200, {"status": "started"})


def main() -> None:
    threading.Thread(target=hard_ttl_watchdog, daemon=True).start()
    threading.Thread(target=start_ttl_watchdog, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", REMOTE_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
