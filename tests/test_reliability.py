from __future__ import annotations

import hashlib
import gzip
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unmark import benchmark, cli, input_processing, local_web  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_bootstrap_module(work_dir: Path):
    with patch.dict(
        os.environ,
        {
            "WORK_DIR": str(work_dir),
            "REMOTE_API_TOKEN": "test",
            "INSTANCE_TTL_S": "0",
            "REMOTE_RESULT_GRACE_S": "1",
            "REMOTE_START_TTL_S": "0",
        },
    ):
        spec = importlib.util.spec_from_file_location(
            f"unmark_test_bootstrap_{time.time_ns()}",
            ROOT / "src" / "unmark" / "remote_bootstrap.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class BootstrapServer:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.token = "test-remote-token"
        self.port = free_port()
        env = os.environ.copy()
        env.update(
            {
                "WORK_DIR": str(work_dir),
                "REMOTE_API_TOKEN": self.token,
                "REMOTE_PORT": str(self.port),
                "INSTANCE_TTL_S": "0",
                "REMOTE_RESULT_GRACE_S": "0",
            }
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / ".installed").touch()
        self.process = subprocess.Popen(
            [sys.executable, str(ROOT / "src" / "unmark" / "remote_bootstrap.py")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                response = cli.requests.get(
                    f"{self.url}/health",
                    headers=cli.remote_headers(self.token),
                    timeout=0.2,
                )
                if response.status_code == 200:
                    return
            except cli.requests.RequestException:
                time.sleep(0.05)
        self.process.wait(timeout=1)
        raise RuntimeError("測試遠端服務啟動失敗")

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)


class ReliabilityTests(unittest.TestCase):
    def test_vast_queued_state_change_is_waited_instead_of_rejected(self) -> None:
        response = SimpleNamespace(
            ok=True,
            status_code=200,
            text="",
            json=lambda: {
                "success": False,
                "error": "resources_unavailable",
                "msg": "Required resources are currently unavailable, state change queued.",
            },
        )
        with patch.object(cli, "vast_request_with_retries", return_value=response):
            self.assertTrue(cli.set_instance_state("test-key", 123, "running"))

    def test_vast_onstart_stays_below_argument_limit_and_restores_bootstrap(self) -> None:
        self.assertFalse(cli.REMOTE_HTTP.trust_env)
        onstart = cli.build_onstart("hf_test", "remote-test")
        self.assertLess(len(onstart.encode("utf-8")), 16_384)
        encoded = onstart.split("echo ", 1)[1].split(" | base64 -d", 1)[0]
        restored = gzip.decompress(__import__("base64").b64decode(encoded)).decode("utf-8")
        self.assertEqual(restored, cli.build_bootstrap_server())

    def test_local_images_are_snapshotted_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            for index, name in enumerate(("one.png", "two.jpg", "three.jpeg", "four.webp"), start=1):
                Image.new("RGB", (32, 24), (index * 30, 60, 90)).save(source_dir / name)
            image_paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in cli.IMAGE_EXTENSIONS)
            items = [cli.build_item(index, str(path), path) for index, path in enumerate(image_paths, start=1)]
            job_dir = root / "job"
            job_dir.mkdir()
            snapshotted = cli.snapshot_local_inputs(job_dir, items)
            manifest = cli.build_manifest(snapshotted, 8, "9B", {"pipeline": "flux2_klein"})
            self.assertEqual(len(manifest["items"]), len(image_paths))
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["prompt_preset"], "萬能提示詞")
            self.assertIn("If an element is uncertain", manifest["prompt"])
            for item in manifest["items"]:
                snapshot = job_dir / "input_files" / item["filename"]
                self.assertTrue(snapshot.is_file())
                self.assertEqual(item["size_bytes"], snapshot.stat().st_size)
                self.assertEqual(item["sha256"], cli.sha256_file(snapshot))
                self.assertNotIn("local_path_resolved", item)

    def test_remote_transfer_checks_auth_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server = BootstrapServer(Path(temp_dir))
            try:
                unauthorized = cli.requests.get(f"{server.url}/health", timeout=2)
                self.assertEqual(unauthorized.status_code, 401)

                source = b"UNMARK-transfer-test" * 2000
                payload = cli.upload_file(server.url, "raw_images/test.bin", source, server.token)
                self.assertEqual(payload["size_bytes"], len(source))
                self.assertEqual(payload["sha256"], hashlib.sha256(source).hexdigest())

                remote_output = Path(temp_dir) / "output_images" / "result.bin"
                remote_output.parent.mkdir(parents=True, exist_ok=True)
                remote_output.write_bytes(source)
                local_output = Path(temp_dir) / "downloaded" / "result.bin"
                cli.download_file(
                    server.url,
                    "output_images/result.bin",
                    local_output,
                    server.token,
                    expected_size=len(source),
                    expected_sha256=hashlib.sha256(source).hexdigest(),
                )
                self.assertEqual(local_output.read_bytes(), source)

                bad_output = Path(temp_dir) / "downloaded" / "bad.bin"
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    cli.download_file(
                        server.url,
                        "output_images/result.bin",
                        bad_output,
                        server.token,
                        expected_sha256="0" * 64,
                        attempts=1,
                    )
                self.assertFalse(bad_output.exists())
                self.assertEqual(list(bad_output.parent.glob("*.part")), [])
            finally:
                server.close()

    def test_remote_preserve_requires_auth_and_sets_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            server = BootstrapServer(work_dir)
            try:
                unauthorized = cli.requests.post(f"{server.url}/preserve", timeout=2)
                self.assertEqual(unauthorized.status_code, 401)
                response = cli.requests.post(
                    f"{server.url}/preserve",
                    headers=cli.remote_headers(server.token),
                    timeout=2,
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "preserved")
                self.assertTrue((work_dir / ".preserve_instance").is_file())
            finally:
                server.close()

    def test_remote_refuses_to_start_when_an_input_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server = BootstrapServer(Path(temp_dir))
            try:
                cli.upload_file(server.url, "remote_worker.py", b"print('not run')\n", server.token)
                manifest = {
                    "items": [
                        {
                            "source_type": "local",
                            "filename": "missing.png",
                            "size_bytes": 10,
                            "sha256": "0" * 64,
                        }
                    ]
                }
                cli.upload_file(
                    server.url,
                    "manifest.json",
                    json.dumps(manifest).encode(),
                    server.token,
                )
                response = cli.requests.post(
                    f"{server.url}/start",
                    headers=cli.remote_headers(server.token),
                    timeout=2,
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["status"], "invalid_inputs")
            finally:
                server.close()

    def test_remote_reports_an_unexpected_worker_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server = BootstrapServer(Path(temp_dir))
            try:
                cli.upload_file(server.url, "remote_worker.py", b"raise SystemExit(7)\n", server.token)
                cli.upload_file(server.url, "manifest.json", b'{"items": []}', server.token)
                response = cli.requests.post(
                    f"{server.url}/start",
                    headers=cli.remote_headers(server.token),
                    timeout=2,
                )
                self.assertEqual(response.status_code, 200)
                deadline = time.time() + 3
                status = {}
                while time.time() < deadline:
                    status_response = cli.requests.get(
                        f"{server.url}/job_status.json",
                        headers=cli.remote_headers(server.token),
                        timeout=2,
                    )
                    if status_response.status_code == 200:
                        status = status_response.json()
                        if status.get("status") == "error":
                            break
                    time.sleep(0.05)
                self.assertEqual(status.get("status"), "error")
                self.assertIn("exit code 7", status.get("error", ""))
            finally:
                server.close()

    def test_old_batch_watchdog_cannot_stop_a_newer_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bootstrap = load_bootstrap_module(Path(temp_dir))

            class FinishedProcess:
                @staticmethod
                def wait():
                    return 0

            old_process = FinishedProcess()
            bootstrap.current_process = old_process
            bootstrap.last_controller_activity = 100.0

            def switch_to_new_batch(_seconds):
                bootstrap.current_process = object()

            with patch.object(bootstrap, "destroy_own_instance") as destroy, patch.object(
                bootstrap.time,
                "monotonic",
                return_value=100.0,
            ), patch.object(bootstrap.time, "sleep", side_effect=switch_to_new_batch):
                bootstrap.completed_job_watchdog(old_process)
            destroy.assert_not_called()

    def test_uncertain_instance_recovery_removes_stale_session_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir)
            (job_dir / ".remote_session.json").write_text('{"remote_api_token":"test"}', encoding="utf-8")
            with patch.object(cli, "find_instance_ids_by_label", return_value=[]) as find, patch.object(
                cli.time,
                "sleep",
                return_value=None,
            ):
                confirmed_absent = cli.recover_uncertain_instance("vast", "missing-label", job_dir)
            self.assertEqual(find.call_count, 12)
            self.assertTrue(confirmed_absent)
            self.assertFalse((job_dir / ".remote_session.json").exists())
            state = cli.read_job_state(job_dir)
            self.assertTrue(state["instance_not_found"])
            self.assertTrue(state["instance_destroyed"])

    def test_benchmark_configuration_names_and_reports(self) -> None:
        self.assertEqual(benchmark.configuration_name("9B", 8), "9b_8steps")
        self.assertEqual(benchmark.configuration_name("KONTEXT", 32), "kontext_32steps")
        self.assertEqual(benchmark.parse_steps("8,16,32,16"), [8, 16, 32])
        self.assertEqual(
            benchmark.parse_configurations(
                "9B:8,9B:16,KWR:32,9B:8",
                cli.load_model_presets(),
            ),
            [("9B", 8), ("9B", 16), ("KWR", 32)],
        )
        args = cli.build_parser().parse_args(["benchmark-models"])
        self.assertEqual(args.disk_gb, 200)
        self.assertEqual(args.report_dir, "Docs/Models")
        self.assertEqual(args.input_images_dir, "input_images")
        self.assertEqual(args.output_root, "output_images/model_tests")
        self.assertTrue(args.destroy_after)
        self.assertIsNone(args.seed)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "results" / "9b_8steps"
            records = [
                benchmark.configuration_record(
                    "9B",
                    8,
                    output_dir,
                    {
                        "status": "done",
                        "ok": 1,
                        "failed": 0,
                        "not_run": 0,
                        "pipeline_load_seconds": 2.5,
                        "rows": [
                            {
                                "original_image": "one.png",
                                "output_image": "output_images/one.png",
                                "status": "ok",
                                "seconds": 1.25,
                                "error": "",
                            }
                        ],
                    },
                    4.0,
                    3.6,
                )
            ]
            markdown, summary_csv, image_csv = benchmark.write_reports(
                root / "Models",
                "model_benchmark_test",
                {
                    "status": "done",
                    "gpu_name": "H200",
                    "instance_id": 123,
                    "dph": 3.6,
                    "input_count": 1,
                    "seed": 42,
                    "total_elapsed_seconds": 4.0,
                    "estimated_total_cost_usd": 0.004,
                    "instance_kept": True,
                },
                records,
            )
            self.assertTrue(markdown.is_file())
            self.assertTrue(summary_csv.is_file())
            self.assertTrue(image_csv.is_file())
            report = markdown.read_text(encoding="utf-8")
            summary = summary_csv.read_text(encoding="utf-8-sig")
            self.assertIn("9b_8steps", report)
            self.assertIn("萬能提示詞", report)
            self.assertIn("$0.004000", report)
            self.assertNotIn(str(root), report)
            self.assertNotIn(str(root), summary)

    def test_prompt_presets_are_loaded_from_markdown_files(self) -> None:
        presets = cli.load_prompt_presets()
        self.assertEqual(
            set(presets),
            {"萬能提示詞", "樓盤Logo", "二維碼", "亂塗亂畫", "線稿粗黑筆", "密集文字水印"},
        )
        self.assertIn("user-interface controls", presets["萬能提示詞"])
        self.assertNotIn("room geometry", presets["萬能提示詞"])
        self.assertEqual(next(iter(presets)), "萬能提示詞")
        self.assertTrue(cli.build_parser().parse_args(["--list-prompts"]).list_prompts)

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "商品Logo.md"
            custom_path.write_text("Remove only the overlaid product logo.", encoding="utf-8")
            self.assertEqual(
                cli.load_prompt_presets(Path(temp_dir)),
                {"商品Logo": "Remove only the overlaid product logo."},
            )

    def test_parallel_upload_propagates_a_single_file_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "one.png"
            second = Path(temp_dir) / "two.png"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            items = [
                {"source_type": "local", "local_path_resolved": str(first), "filename": first.name},
                {"source_type": "local", "local_path_resolved": str(second), "filename": second.name},
            ]

            def fake_upload(_url, remote_path, _source, _token, attempts=3):
                if remote_path.endswith("two.png"):
                    raise RuntimeError("模擬斷線")
                return {"status": "uploaded"}

            with patch.object(cli, "upload_file", side_effect=fake_upload), patch.object(cli.REMOTE_HTTP, "post") as post:
                with self.assertRaisesRegex(RuntimeError, "模擬斷線"):
                    cli.upload_payload_and_start("http://unused", {"items": []}, items, "token")
                post.assert_not_called()

    def test_localweb_multipart_spools_large_file_to_disk(self) -> None:
        image_buffer = io.BytesIO()
        Image.new("RGB", (48, 32), "white").save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        source_name = "watermarking-result.png"
        request = cli.requests.Request(
            "POST",
            "http://127.0.0.1/new",
            data={"model": "9B"},
            files={"input_files": (source_name, image_bytes, "image/png")},
        ).prepare()
        handler = SimpleNamespace(headers=request.headers, rfile=io.BytesIO(request.body))
        fields, files = local_web.read_multipart(handler, 10 * 1024 * 1024)
        try:
            self.assertEqual(fields["model"], "9B")
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0][0], source_name)
            self.assertEqual(files[0][1].read_bytes(), image_bytes)
        finally:
            local_web.cleanup_temp_uploads(files)

    def test_zip_extraction_honours_uncompressed_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "large.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("large.png", b"0" * (2 * 1024 * 1024))
            with self.assertRaisesRegex(ValueError, "超過設定上限"):
                input_processing.safe_extract_zip(zip_path, root / "images", 1024 * 1024)

    def test_multiple_zips_share_one_uncompressed_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archives = []
            for index in range(2):
                zip_path = root / f"input_{index}.zip"
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(f"image_{index}.png", b"0" * (700 * 1024))
                archives.append((zip_path.name, zip_path))
            with self.assertRaisesRegex(ValueError, "超過設定上限"):
                input_processing.normalize_uploaded_files(root / "upload", archives, 1024 * 1024)

    def test_all_to_once_import_uses_configured_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            import_dir = root / "import"
            work_dir = root / "work"
            import_dir.mkdir()
            Image.new("RGB", (32, 24), "white").save(import_dir / "sample.png")
            env = {
                **os.environ,
                "UNMARK_ALL_TO_ONCE_IMPORT_DIR": str(import_dir),
                "UNMARK_ALL_TO_ONCE_WORK_DIR": str(work_dir),
                "UNMARK_MAX_EMBED_MB": "1",
            }
            result = subprocess.run(
                [sys.executable, str(ROOT / "all_to_once_script" / "import_inputs.py")],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("images/sample.png", (work_dir / "input.csv").read_text(encoding="utf-8-sig"))

    def test_invalid_settings_are_rejected_before_launch(self) -> None:
        settings = dict(local_web.DEFAULT_SETTINGS)
        settings["batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "每批圖片數必須大於 0"):
            local_web.validate_settings(settings)

    def test_flash_message_is_read_only_once(self) -> None:
        state = local_web.LocalWebState(ROOT, cli, ROOT / "unused-settings.json")
        token = local_web.queue_flash(state, "只顯示一次")
        self.assertEqual(local_web.pop_flash(state, token), ("只顯示一次", "notice"))
        self.assertEqual(local_web.pop_flash(state, token), ("", ""))

    def test_settings_json_excludes_secrets_and_uses_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            settings = dict(local_web.DEFAULT_SETTINGS)
            settings.update({"vast_api_key": "must-not-save", "hf_token": "must-not-save"})
            local_web.save_settings(settings_path, settings)
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertNotIn("vast_api_key", saved)
            self.assertNotIn("hf_token", saved)
            self.assertEqual(settings_path.stat().st_mode & 0o777, 0o600)

    def test_invalid_settings_json_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings_path = root / "settings.json"
            settings_path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON 格式錯誤"):
                local_web.load_settings(settings_path)

    def test_invalid_boolean_environment_value_is_rejected(self) -> None:
        with patch.dict(os.environ, {"UNMARK_DESTROY_ON_SUCCESS": "sometimes"}):
            with self.assertRaisesRegex(SystemExit, "必須是 true 或 false"):
                cli.env_bool("DESTROY_ON_SUCCESS", True)
            with self.assertRaisesRegex(ValueError, "只可使用 true 或 false"):
                local_web.load_settings(Path("missing-settings.json"))

    def test_api_rejects_unauthenticated_multipart_before_reading_body(self) -> None:
        class UnreadableBody:
            def read(self, _size=-1):
                raise AssertionError("驗證 API Key 前不應讀取 request body")

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"UNMARK_LOCAL_API_KEY": ""},
        ):
            root = Path(temp_dir)
            state = local_web.LocalWebState(root, cli, root / "settings.json")
            state.settings = dict(local_web.DEFAULT_SETTINGS)
            handler_type = local_web.make_handler(state)
            handler = object.__new__(handler_type)
            handler.path = "/api/jobs"
            handler.headers = {
                "Content-Type": "multipart/form-data; boundary=test",
                "Content-Length": str(1024 * 1024),
            }
            handler.rfile = UnreadableBody()
            handler.wfile = io.BytesIO()
            statuses = []
            handler.send_response = lambda status: statuses.append(status)
            handler.send_header = lambda *_args: None
            handler.end_headers = lambda: None
            handler.do_POST()
            self.assertEqual(statuses, [403])

    def test_localweb_rejects_invalid_job_override_before_starting_thread(self) -> None:
        class FakeCli:
            requests = object()

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"VAST_API_KEY": "", "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""},
        ):
            root = Path(temp_dir)
            input_csv = root / "input.csv"
            input_csv.write_text("original_image\nhttps://example.com/a.jpg\n", encoding="utf-8")
            state = local_web.LocalWebState(root, FakeCli(), root / "settings.json")
            state.settings = dict(local_web.DEFAULT_SETTINGS)
            state.temp_secrets = {"vast_api_key": "vast", "hf_token": "hf"}
            with self.assertRaisesRegex(ValueError, "模型只可選擇"):
                local_web.launch_job(state, input_csv, {"model": "invalid"})
            self.assertFalse(state.job_lock.locked())
            self.assertIsNone(state.active_thread)

    def test_localweb_only_listens_on_loopback(self) -> None:
        cli_stub = SimpleNamespace(ROOT=ROOT, load_env_file=lambda: None)
        args = SimpleNamespace(host="0.0.0.0")
        with self.assertRaisesRegex(SystemExit, "只可監聽"):
            local_web.serve_local(args, cli_stub)

    def test_multiple_memory_api_keys_can_be_managed_independently(self) -> None:
        with patch.dict(os.environ, {"UNMARK_LOCAL_API_KEY": ""}):
            state = local_web.LocalWebState(ROOT, cli, ROOT / "unused-settings.json")
            first = local_web.generate_local_api_key()
            second = local_web.generate_local_api_key()
            state.temp_api_keys = [
                local_web.new_api_key_record(first, "第一個用途", "memory"),
                local_web.new_api_key_record(second, "第二個用途", "memory", enabled=False),
            ]
            self.assertTrue(local_web.api_key_matches(state, first))
            self.assertFalse(local_web.api_key_matches(state, second))
            local_web.api_key_action(state, f"toggle:{state.temp_api_keys[1]['id']}")
            self.assertTrue(local_web.api_key_matches(state, second))

    def test_environment_api_key_record_is_stable(self) -> None:
        with patch.dict(os.environ, {"UNMARK_LOCAL_API_KEY": "unmark_stable_test_key"}):
            state = local_web.LocalWebState(ROOT, cli, ROOT / "unused-settings.json")
            first = local_web.configured_api_keys(state)[0]
            second = local_web.configured_api_keys(state)[0]
            self.assertEqual(first["id"], second["id"])
            self.assertTrue(first["persistent"])

    def test_duplicate_urls_keep_separate_completion_records(self) -> None:
        original_job_dir = cli.JOB_DIR
        original_output_csv = cli.OUTPUT_CSV
        original_output_images_dir = cli.OUTPUT_IMAGES_DIR
        original_remote_result_json = cli.REMOTE_RESULT_JSON
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                job_dir = Path(temp_dir)
                output_dir = job_dir / "output_images"
                output_dir.mkdir()
                first = output_dir / "first.png"
                second = output_dir / "second.png"
                first.write_bytes(b"first")
                second.write_bytes(b"second")
                cli.apply_job_paths(job_dir)
                records = {
                    "000001:aaa": {
                        "original_image": "https://example.com/same.jpg",
                        "output_image": "output_images/first.png",
                        "status": "ok",
                        "output_size_bytes": first.stat().st_size,
                        "output_sha256": cli.sha256_file(first),
                    },
                    "000002:bbb": {
                        "original_image": "https://example.com/same.jpg",
                        "output_image": "output_images/second.png",
                        "status": "ok",
                        "output_size_bytes": second.stat().st_size,
                        "output_sha256": cli.sha256_file(second),
                    },
                }
                cli.write_output_csv_map(records)
                loaded = cli.load_completed_jobs()
                self.assertEqual(set(loaded), set(records))
        finally:
            cli.JOB_DIR = original_job_dir
            cli.OUTPUT_CSV = original_output_csv
            cli.OUTPUT_IMAGES_DIR = original_output_images_dir
            cli.REMOTE_RESULT_JSON = original_remote_result_json

    def test_job_download_zip_contains_outputs_but_not_session_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir)
            output_dir = job_dir / "output_images"
            output_dir.mkdir()
            (output_dir / "one.png").write_bytes(b"image")
            (job_dir / "output.csv").write_text("status\nok\n", encoding="utf-8")
            (job_dir / "job_state.json").write_text('{"status":"done"}', encoding="utf-8")
            (job_dir / ".remote_session.json").write_text('{"remote_api_token":"secret"}', encoding="utf-8")
            zip_path = local_web.build_job_output_zip(job_dir)
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
            self.assertIn("output_images/one.png", names)
            self.assertIn("output.csv", names)
            self.assertNotIn(".remote_session.json", names)

    def test_localweb_allows_only_one_paid_job_at_a_time(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class FakeCli:
            requests = object()
            captured = {}

            @staticmethod
            def run_vast_job(args, resume=False, secret_overrides=None):
                FakeCli.captured = {
                    "model": args.model,
                    "secrets": dict(secret_overrides or {}),
                }
                started.set()
                release.wait(timeout=3)

            @staticmethod
            def write_job_state(_job_dir, _updates):
                return None

            @staticmethod
            def append_job_event(_job_dir, _event, _payload):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_csv = root / "input.csv"
            input_csv.write_text("original_image\nhttps://example.com/a.jpg\n", encoding="utf-8")
            with patch.dict(os.environ, {"VAST_API_KEY": "", "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""}):
                state = local_web.LocalWebState(root, FakeCli(), root / "settings.json")
                state.settings = dict(local_web.DEFAULT_SETTINGS)
                state.temp_secrets = {"vast_api_key": "vast", "hf_token": "hf"}
                local_web.launch_job(state, input_csv)
                self.assertTrue(started.wait(timeout=1))
                state.settings["model"] = "4B"
                state.temp_secrets["vast_api_key"] = "changed"
                with self.assertRaisesRegex(RuntimeError, "已有任務"):
                    local_web.launch_job(state, input_csv)
                self.assertEqual(FakeCli.captured["model"], "9B")
                self.assertEqual(FakeCli.captured["secrets"], {"vast_api_key": "vast", "hf_token": "hf"})
                self.assertEqual(os.environ["VAST_API_KEY"], "")
                self.assertEqual(os.environ["HF_TOKEN"], "")
                release.set()
                state.active_thread.join(timeout=2)
                self.assertFalse(state.job_lock.locked())


if __name__ == "__main__":
    unittest.main()
