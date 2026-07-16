from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import cli


TERMINAL_STATUSES = {"done", "partial", "error"}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cli.ROOT / path
    return path.resolve()


def portable_output_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(cli.ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def parse_csv_values(raw: str, *, uppercase: bool = False) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if uppercase:
        values = [value.upper() for value in values]
    return list(dict.fromkeys(values))


def parse_steps(raw: str) -> list[int]:
    values = []
    for part in parse_csv_values(raw):
        try:
            value = int(part)
        except ValueError as exc:
            raise SystemExit(f"無效 steps：{part!r}") from exc
        if value < 1:
            raise SystemExit("steps 必須大於 0")
        values.append(value)
    if not values:
        raise SystemExit("至少要提供一個 steps")
    return values


def parse_configurations(raw: str, presets: dict) -> list[tuple[str, int]]:
    configurations: list[tuple[str, int]] = []
    for part in parse_csv_values(raw):
        model_raw, separator, steps_raw = part.partition(":")
        model = model_raw.strip().upper()
        if not separator or not model or not steps_raw.strip():
            raise SystemExit(f"無效評測組別：{part!r}；格式應為 MODEL:STEPS")
        if model not in presets:
            raise SystemExit(f"未知模型：{model}")
        try:
            steps = int(steps_raw)
        except ValueError as exc:
            raise SystemExit(f"無效 steps：{steps_raw!r}") from exc
        if steps < 1:
            raise SystemExit("steps 必須大於 0")
        configuration = (model, steps)
        if configuration not in configurations:
            configurations.append(configuration)
    if not configurations:
        raise SystemExit("至少要提供一個評測組別")
    return configurations


def configuration_name(model: str, steps: int) -> str:
    return f"{model.lower()}_{steps}steps"


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def valid_local_output(path: Path, row: dict) -> bool:
    if not path.is_file():
        return False
    expected_size = row.get("output_size_bytes")
    expected_hash = str(row.get("output_sha256", "")).lower()
    if expected_size not in (None, "") and path.stat().st_size != int(expected_size):
        return False
    if expected_hash and cli.sha256_file(path) != expected_hash:
        return False
    return True


def download_available_rows(
    url_base: str,
    token: str,
    summary: dict,
    output_dir: Path,
    downloaded: set[str],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for row in summary.get("rows", []):
        if row.get("status") != "ok" or not row.get("output_image"):
            continue
        input_key = str(row.get("input_key") or row.get("original_image"))
        remote_path = str(row["output_image"])
        local_path = output_dir / Path(remote_path).name
        if input_key in downloaded and valid_local_output(local_path, row):
            continue
        try:
            if not valid_local_output(local_path, row):
                cli.download_file(
                    url_base,
                    remote_path,
                    local_path,
                    token,
                    expected_size=row.get("output_size_bytes"),
                    expected_sha256=row.get("output_sha256"),
                )
            downloaded.add(input_key)
            print(f"  已下載：{output_dir.name}/{local_path.name}")
        except Exception as exc:
            errors.append(f"{remote_path}: {exc}")
    return errors


def write_configuration_files(output_dir: Path, model: str, steps: int, summary: dict) -> None:
    rows = []
    for row in summary.get("rows", []):
        item = dict(row)
        item["model"] = model
        item["steps"] = steps
        if item.get("output_image"):
            item["output_image"] = Path(str(item["output_image"])).name
        rows.append(item)
    fieldnames = [
        "input_key",
        "original_image",
        "output_image",
        "status",
        "error",
        "seconds",
        "model",
        "steps",
        "output_size_bytes",
        "output_sha256",
    ]
    atomic_write_csv(output_dir / "output.csv", fieldnames, rows)
    cli.atomic_write_json(output_dir / "metrics.json", summary)


def start_remote_configuration(url_base: str, token: str, manifest: dict) -> None:
    cli.upload_file(
        url_base,
        "manifest.json",
        json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
        token,
    )
    for attempt in range(1, 31):
        response = cli.REMOTE_HTTP.post(
            f"{url_base}/start",
            headers=cli.remote_headers(token),
            timeout=60,
        )
        if response.status_code == 409 and response.json().get("status") == "busy":
            if attempt == 30:
                raise RuntimeError("上一組遠端評測在 60 秒後仍未退出")
            time.sleep(2)
            continue
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"遠端拒絕啟動評測（HTTP {response.status_code}）：{detail}")
        payload = response.json()
        if payload.get("status") != "started":
            raise RuntimeError(f"遠端評測啟動回覆異常：{payload}")
        return


def preserve_remote_instance(url_base: str, token: str) -> None:
    response = cli.REMOTE_HTTP.post(
        f"{url_base}/preserve",
        headers=cli.remote_headers(token),
        timeout=30,
    )
    if not response.ok or response.json().get("status") != "preserved":
        raise RuntimeError(f"未能取消實例自動停機保護：HTTP {response.status_code} {response.text[:500]}")


def wait_configuration(
    url_base: str,
    token: str,
    output_dir: Path,
    timeout_s: int,
) -> dict:
    started = time.time()
    downloaded: set[str] = set()
    latest: dict = {}
    last_progress = None
    while time.time() - started < timeout_s:
        try:
            response = cli.REMOTE_HTTP.get(
                f"{url_base}/job_status.json",
                headers=cli.remote_headers(token),
                timeout=20,
            )
            if response.status_code == 200:
                latest = response.json()
                download_available_rows(url_base, token, latest, output_dir, downloaded)
                done_count = latest.get("done")
                if done_count is None:
                    done_count = len(latest.get("rows", []))
                progress = (latest.get("status"), done_count, latest.get("ok"))
                if progress != last_progress:
                    print(
                        f"  遠端狀態：{progress[0]}，"
                        f"完成 {progress[1] or 0}/{latest.get('total', '?')}，成功 {progress[2] or 0}"
                    )
                    last_progress = progress
                if str(latest.get("status")) in TERMINAL_STATUSES:
                    break
        except cli.requests.RequestException:
            pass
        time.sleep(5)
    else:
        raise TimeoutError(f"評測超過 {timeout_s} 秒仍未完成")

    missing = []
    last_errors = []
    for attempt in range(1, 4):
        last_errors = download_available_rows(url_base, token, latest, output_dir, downloaded)
        missing = []
        for row in latest.get("rows", []):
            if row.get("status") != "ok" or not row.get("output_image"):
                continue
            local_path = output_dir / Path(str(row["output_image"])).name
            if not valid_local_output(local_path, row):
                missing.append(local_path.name)
        if not missing:
            break
        if attempt < 3:
            time.sleep(attempt * 2)
    if missing:
        raise RuntimeError(
            f"遠端已有成功結果，但仍有 {len(missing)} 張未能完整下載：{missing}；錯誤：{last_errors}"
        )
    return latest


def upload_inputs_once(url_base: str, token: str, items: list[dict]) -> None:
    print("上傳遠端評測程式...")
    cli.upload_file(url_base, "remote_worker.py", cli.REMOTE_WORKER, token)
    local_items = [item for item in items if item.get("source_type") == "local"]

    def upload_one(item: dict) -> None:
        cli.upload_file(
            url_base,
            f"raw_images/{item['filename']}",
            Path(item["local_path_resolved"]),
            token,
        )

    print(f"上傳並核對 {len(local_items)} 張原圖；整次評測只上傳一次...")
    with ThreadPoolExecutor(max_workers=min(8, len(local_items))) as executor:
        futures = [executor.submit(upload_one, item) for item in local_items]
        for future in as_completed(futures):
            future.result()
    print("原圖已全部上傳並通過 SHA-256 核對。")


def configuration_record(
    model: str,
    steps: int,
    output_dir: Path,
    summary: dict,
    elapsed_seconds: float,
    dph: float,
) -> dict:
    rows = summary.get("rows", [])
    successful_times = [float(row.get("seconds", 0)) for row in rows if row.get("status") == "ok"]
    return {
        "configuration": configuration_name(model, steps),
        "model": model,
        "steps": steps,
        "status": summary.get("status", "error"),
        "ok": int(summary.get("ok", 0)),
        "failed": int(summary.get("failed", 0)),
        "not_run": int(summary.get("not_run", 0)),
        "pipeline_load_seconds": float(summary.get("pipeline_load_seconds", 0) or 0),
        "image_seconds_total": round(sum(successful_times), 4),
        "image_seconds_average": round(sum(successful_times) / len(successful_times), 4) if successful_times else 0,
        "wall_seconds": round(elapsed_seconds, 4),
        "estimated_cost_usd": round(elapsed_seconds / 3600 * dph, 6),
        "output_dir": portable_output_path(output_dir),
        "rows": rows,
        "error": summary.get("error", ""),
    }


def write_reports(
    report_dir: Path,
    job_id: str,
    benchmark: dict,
    records: list[dict],
) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / f"{job_id}.md"
    summary_csv = report_dir / f"{job_id}_summary.csv"
    image_csv = report_dir / f"{job_id}_images.csv"

    summary_fields = [
        "configuration",
        "model",
        "steps",
        "status",
        "ok",
        "failed",
        "not_run",
        "pipeline_load_seconds",
        "image_seconds_total",
        "image_seconds_average",
        "wall_seconds",
        "estimated_cost_usd",
        "output_dir",
        "error",
    ]
    atomic_write_csv(summary_csv, summary_fields, records)

    image_rows = []
    for record in records:
        for row in record.get("rows", []):
            image_rows.append(
                {
                    "configuration": record["configuration"],
                    "model": record["model"],
                    "steps": record["steps"],
                    "original_image": row.get("original_image", ""),
                    "output_image": Path(str(row.get("output_image", ""))).name if row.get("output_image") else "",
                    "status": row.get("status", ""),
                    "seconds": row.get("seconds", ""),
                    "error": row.get("error", ""),
                }
            )
    atomic_write_csv(
        image_csv,
        ["configuration", "model", "steps", "original_image", "output_image", "status", "seconds", "error"],
        image_rows,
    )

    lines = [
        f"# 模型評測：{job_id}",
        "",
        "## 測試資料",
        "",
        f"- 狀態：`{benchmark.get('status', 'running')}`",
        f"- GPU：`{benchmark.get('gpu_name', '')}`",
        f"- Vast 實例：`{benchmark.get('instance_id', '')}`",
        f"- 重用既有實例：`{'是' if benchmark.get('reused_instance') else '否'}`",
        f"- 實際單價：`${benchmark.get('dph', 0):.4f}` / 小時",
        f"- 建立前報價：`${benchmark.get('quoted_dph', benchmark.get('dph', 0)):.4f}` / 小時",
        f"- 測試圖片：`{benchmark.get('input_count', 0)}` 張",
        f"- Seed：`{benchmark.get('seed', 42)}`",
        f"- 提示詞：`{benchmark.get('prompt_preset', '萬能提示詞')}`",
        f"- 開始時間：`{benchmark.get('started_at', '')}`",
        f"- 報告時間：`{benchmark.get('reported_at', '')}`",
        f"- 累計實例時間：`{benchmark.get('total_elapsed_seconds', 0):.1f}` 秒",
        f"- 截至報告時間預估費用：`${benchmark.get('estimated_total_cost_usd', 0):.6f}`",
        f"- 實例是否保留：`{'是' if benchmark.get('instance_kept') else '否'}`",
        "",
        f"每組 `wall_seconds` 包括模型載入、{benchmark.get('input_count', 0)} 張推理及結果下載；`image_seconds_average` 只計成功圖片的逐張清洗時間。",
        "費用是按實例單價及本次端到端時間估算，不包括另行結算的儲存及頻寬費。實例保留期間仍會繼續計費。",
        "",
        "## 組別摘要",
        "",
        "| 輸出目錄 | 狀態 | 成功 | 失敗 | 模型載入秒數 | 單張平均秒數 | 組別總秒數 | 組別預估費用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| `{record['output_dir']}` | {record['status']} | {record['ok']} | "
            f"{record['failed']} | {record['pipeline_load_seconds']:.4f} | "
            f"{record['image_seconds_average']:.4f} | {record['wall_seconds']:.4f} | "
            f"${record['estimated_cost_usd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 每張圖片耗時",
            "",
            "| 組別 | 圖片 | 狀態 | 秒數 | 錯誤 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in image_rows:
        error = str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")
        image_name = Path(str(row.get("original_image", ""))).name
        lines.append(
            f"| {row['configuration']} | {image_name} | {row['status']} | {row['seconds']} | {error} |"
        )
    lines.extend(
        [
            "",
            f"完整組別數據：[摘要 CSV]({summary_csv.name})",
            "",
            f"完整逐圖數據：[逐圖 CSV]({image_csv.name})",
            "",
        ]
    )
    cli.atomic_write_text(markdown_path, "\n".join(lines))
    return markdown_path, summary_csv, image_csv


def run_benchmark(args) -> None:
    cli.apply_runtime_config(args)
    presets = cli.load_model_presets()
    if args.configurations:
        configurations = parse_configurations(args.configurations, presets)
    else:
        models_requested = parse_csv_values(args.models, uppercase=True)
        unknown_models = [model for model in models_requested if model not in presets]
        if unknown_models:
            raise SystemExit(f"未知模型：{', '.join(unknown_models)}")
        steps_requested = parse_steps(args.steps_list)
        configurations = [(model, steps) for model in models_requested for steps in steps_requested]
    models = list(dict.fromkeys(model for model, _ in configurations))
    steps_list = list(dict.fromkeys(steps for _, steps in configurations))

    input_dir = resolve_project_path(args.input_images_dir or "input_images")
    source_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in cli.IMAGE_EXTENSIONS
    ) if input_dir.is_dir() else []
    if not source_paths:
        raise SystemExit(f"找不到測試圖片：{input_dir}")

    output_root = resolve_project_path(args.output_root)
    report_dir = resolve_project_path(args.report_dir)
    planned = [configuration_name(model, steps) for model, steps in configurations]
    occupied = [
        output_root / name
        for name in planned
        if (output_root / name).exists() and any((output_root / name).iterdir())
    ]
    if occupied and not args.overwrite:
        joined = "\n".join(str(path) for path in occupied)
        raise SystemExit(f"以下輸出目錄已有內容；如確定重跑，請加 --overwrite：\n{joined}")
    if args.overwrite:
        for path in occupied:
            shutil.rmtree(path)

    raw_items = [
        cli.build_item(index, path.name, path, document_id=path.stem)
        for index, path in enumerate(source_paths, start=1)
    ]
    cli.validate_output_names(raw_items)
    if args.dry_run:
        print("模型評測 Dry-run：")
        print(f"  輸入目錄：{input_dir}")
        print(f"  圖片數：{len(raw_items)}")
        print(f"  GPU：{cli.GPU_NAME}")
        print(f"  提示詞：{cli.PROMPT_PRESET}")
        print(f"  seed：{cli.SEED}")
        print(f"  組別數：{len(planned)}")
        for name in planned:
            print(f"  - {output_root / name}")
        print("  不會建立 Vast 實例。")
        return

    cli.require_requests()
    api_key, hf_token = cli.resolve_config()
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    job_id = args.job_id or time.strftime("model_benchmark_%Y%m%d_%H%M%S")
    job_dir = (cli.JOBS_DIR / job_id).resolve()
    if job_dir.exists():
        raise SystemExit(f"Benchmark job 已存在：{job_dir}")
    job_dir.mkdir(parents=True)
    cli.apply_job_paths(job_dir)

    items = cli.snapshot_local_inputs(job_dir, raw_items)
    cli.write_input_manifest(job_dir, items)
    cli.atomic_write_json(job_dir / "runtime_config.json", cli.runtime_config_snapshot())

    benchmark = {
        "job_id": job_id,
        "status": "initializing",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_name": cli.GPU_NAME,
        "input_dir": str(input_dir),
        "input_count": len(items),
        "seed": cli.SEED,
        "prompt_preset": cli.PROMPT_PRESET,
        "prompt_sha256": hashlib.sha256(cli.PROMPT.encode("utf-8")).hexdigest(),
        "models": models,
        "steps": steps_list,
        "configurations": [
            {"model": model, "steps": steps, "name": configuration_name(model, steps)}
            for model, steps in configurations
        ],
        "planned_configurations": planned,
        "completed_configurations": [],
        "instance_kept": False,
    }
    records: list[dict] = []
    cli.atomic_write_json(job_dir / "benchmark_state.json", benchmark)
    cli.write_job_state(job_dir, {"job_id": job_id, "status": "benchmark_initializing"})
    cli.append_job_event(job_dir, "benchmark_started", {"configurations": planned, "input_count": len(items)})

    reuse_job_dir: Path | None = None
    reused_instance = False
    if args.reuse_job_dir:
        reuse_job_dir = resolve_project_path(args.reuse_job_dir)
        if not reuse_job_dir.is_dir():
            raise SystemExit(f"找不到要重用的 benchmark job：{reuse_job_dir}")
        remote_token = cli.read_remote_session(reuse_job_dir)
        source_state = cli.read_job_state(reuse_job_dir)
        if not remote_token:
            raise SystemExit(f"既有 benchmark job 缺少遠端連線憑證：{reuse_job_dir}")
        if not source_state.get("instance_id"):
            raise SystemExit(f"既有 benchmark job 沒有 Vast 實例 ID：{reuse_job_dir}")
    else:
        remote_token = secrets.token_urlsafe(32)
        source_state = {}
    cli.write_remote_session(job_dir, remote_token)
    instance_label = f"{cli.INSTANCE_LABEL}-{job_id[-24:]}"[:64]
    instance_id = None
    offer_id = None
    dph = 0.0
    billing_started = None
    try:
        if reuse_job_dir is not None:
            instance_id = int(source_state["instance_id"])
            existing = cli.get_instance(api_key, instance_id)
            if not existing:
                raise RuntimeError(f"Vast 找不到要重用的實例：{instance_id}")
            reused_instance = True
            offer_id = source_state.get("offer_id")
            dph = float(existing.get("dph_total") or source_state.get("dph") or 0)
            instance_label = str(existing.get("label") or source_state.get("instance_label") or instance_label)
            billing_started = time.time()
            if existing.get("actual_status") != "running":
                queued = cli.set_instance_state(api_key, instance_id, "running")
                if queued:
                    cli.append_job_event(
                        job_dir,
                        "benchmark_instance_start_queued",
                        {"instance_id": instance_id, "previous_status": existing.get("actual_status")},
                    )
            benchmark.update({"reused_instance": True, "reuse_job_dir": str(reuse_job_dir)})
            cli.append_job_event(
                job_dir,
                "benchmark_instance_reuse_requested",
                {"instance_id": instance_id, "source_job_dir": str(reuse_job_dir)},
            )
        else:
            onstart = cli.build_onstart(hf_token, remote_token)
            for attempt in range(1, 4):
                offer_id, dph = cli.search_gpu_offer(api_key, cli.GPU_NAME)
                try:
                    instance_id = cli.create_instance(
                        api_key,
                        offer_id,
                        onstart,
                        hf_token,
                        remote_token,
                        instance_label,
                        cli.DISK_GB,
                    )
                    break
                except cli.InstanceCreateUncertainError as exc:
                    cli.write_job_state(job_dir, {"status": "instance_create_uncertain", "error": str(exc)})
                    confirmed_absent = cli.recover_uncertain_instance(api_key, instance_label, job_dir)
                    if confirmed_absent and attempt < 3:
                        cli.append_job_event(
                            job_dir,
                            "benchmark_create_retry_after_confirmed_absent",
                            {"attempt": attempt, "error": str(exc)},
                        )
                        continue
                    raise
                except RuntimeError as exc:
                    if "建立實例被 Vast 拒絕" not in str(exc) or attempt == 3:
                        raise
                    cli.append_job_event(
                        job_dir,
                        "benchmark_offer_rejected",
                        {"attempt": attempt, "offer_id": offer_id, "error": str(exc)},
                    )
                    time.sleep(2)
    except BaseException as exc:
        if instance_id is None:
            (job_dir / ".remote_session.json").unlink(missing_ok=True)
            cli.write_job_state(job_dir, {"status": "benchmark_instance_create_failed", "error": str(exc)})
            cli.append_job_event(job_dir, "benchmark_instance_create_failed", {"error": str(exc)})
        elif reused_instance:
            cli.write_job_state(job_dir, {"instance_id": instance_id, "status": "benchmark_instance_reuse_failed"})
            destroyed = cli.destroy_instance_best_effort(api_key, instance_id, job_dir)
            if destroyed and reuse_job_dir is not None:
                (reuse_job_dir / ".remote_session.json").unlink(missing_ok=True)
                cli.write_job_state(
                    reuse_job_dir,
                    {
                        "instance_destroyed": True,
                        "instance_destroyed_by_job": job_id,
                        "instance_kept_for_benchmark": False,
                    },
                )
        raise
    if instance_id is None:
        (job_dir / ".remote_session.json").unlink(missing_ok=True)
        raise RuntimeError("未能建立 Vast 實例")
    if billing_started is None:
        billing_started = time.time()

    benchmark.update({"instance_id": instance_id, "offer_id": offer_id, "dph": dph, "quoted_dph": dph})
    cli.write_job_state(
        job_dir,
        {
            "status": "benchmark_instance_reused" if reused_instance else "benchmark_instance_created",
            "instance_id": instance_id,
            "offer_id": offer_id,
            "dph": dph,
            "instance_label": instance_label,
        },
    )
    event_name = "instance_reused" if reused_instance else "instance_created"
    cli.append_job_event(job_dir, event_name, {"instance_id": instance_id, "offer_id": offer_id, "dph": dph})
    action = "重用" if reused_instance else "建立"
    print(f"Benchmark 已{action}實例 ID：{instance_id}，單價 ${dph:.4f}/小時")

    url_base = None
    current_output_dir: Path | None = None
    current_summary: dict = {}
    current_model = ""
    current_steps = 0
    current_config_started = 0.0
    benchmark_completed = False
    try:
        instance = cli.wait_running(
            api_key,
            instance_id,
            timeout_s=cli.BOOTSTRAP_TIMEOUT_S,
            transient_exited_s=120 if reused_instance else 0,
        )
        actual_dph = float(instance.get("dph_total") or dph)
        if actual_dph != dph:
            cli.append_job_event(
                job_dir,
                "benchmark_actual_price_updated",
                {"quoted_dph": dph, "actual_dph": actual_dph},
            )
            dph = actual_dph
            benchmark["dph"] = dph
        url_base = cli.http_base(instance)
        cli.write_job_state(
            job_dir,
            {"status": "benchmark_bootstrap_waiting", "url_base": url_base, "dph": dph},
        )
        cli.wait_bootstrap_ready(url_base, remote_token, timeout_s=cli.BOOTSTRAP_TIMEOUT_S)
        upload_inputs_once(url_base, remote_token, items)

        for model, steps in configurations:
            model_config = presets[model]
            name = configuration_name(model, steps)
            current_output_dir = output_root / name
            current_output_dir.mkdir(parents=True, exist_ok=True)
            current_model = model
            current_steps = steps
            manifest = cli.build_manifest(items, steps, model, model_config)
            current_config_started = time.time()
            print(f"\n開始評測 {name}（{len(items)} 張）")
            cli.write_job_state(
                job_dir,
                {
                    "status": "benchmark_running",
                    "current_configuration": name,
                    "completed_configurations": [record["configuration"] for record in records],
                },
            )
            cli.append_job_event(job_dir, "benchmark_configuration_started", {"configuration": name})
            start_remote_configuration(url_base, remote_token, manifest)
            current_summary = wait_configuration(
                url_base,
                remote_token,
                current_output_dir,
                timeout_s=cli.JOB_TIMEOUT_S,
            )
            elapsed = time.time() - current_config_started
            write_configuration_files(current_output_dir, model, steps, current_summary)
            record = configuration_record(model, steps, current_output_dir, current_summary, elapsed, dph)
            records.append(record)
            benchmark.update(
                {
                    "status": "running",
                    "reported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "completed_configurations": [item["configuration"] for item in records],
                    "total_elapsed_seconds": round(time.time() - billing_started, 3),
                    "estimated_total_cost_usd": round((time.time() - billing_started) / 3600 * dph, 6),
                }
            )
            write_reports(report_dir, job_id, benchmark, records)
            cli.atomic_write_json(job_dir / "benchmark_state.json", benchmark)
            cli.append_job_event(
                job_dir,
                "benchmark_configuration_finished",
                {"configuration": name, "status": record["status"], "ok": record["ok"], "failed": record["failed"]},
            )
            print(
                f"完成 {name}：成功 {record['ok']}，失敗 {record['failed']}，"
                f"單張平均 {record['image_seconds_average']:.4f} 秒，組別費用約 ${record['estimated_cost_usd']:.6f}"
            )
            current_summary = {}
            current_output_dir = None
            current_model = ""
            current_steps = 0
            current_config_started = 0.0

        benchmark_completed = True
        final_status = "done" if all(record["status"] == "done" for record in records) else "partial"
        if not args.destroy_after:
            preserve_remote_instance(url_base, remote_token)
        benchmark.update(
            {
                "status": final_status,
                "reported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_elapsed_seconds": round(time.time() - billing_started, 3),
                "estimated_total_cost_usd": round((time.time() - billing_started) / 3600 * dph, 6),
                "instance_kept": not args.destroy_after,
            }
        )
        markdown_path, summary_csv, image_csv = write_reports(report_dir, job_id, benchmark, records)
        cli.atomic_write_json(job_dir / "benchmark_state.json", benchmark)
        cli.write_job_state(
            job_dir,
            {
                "status": "benchmark_done",
                "completed_configurations": [record["configuration"] for record in records],
                "benchmark_report": str(markdown_path),
                "benchmark_summary_csv": str(summary_csv),
                "benchmark_image_csv": str(image_csv),
                "estimated_cost_usd": benchmark["estimated_total_cost_usd"],
            },
        )
    except BaseException as exc:
        benchmark.update(
            {
                "status": "error",
                "error": str(exc),
                "reported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_elapsed_seconds": round(time.time() - billing_started, 3) if billing_started else 0,
                "estimated_total_cost_usd": (
                    round((time.time() - billing_started) / 3600 * dph, 6) if billing_started else 0
                ),
                "instance_kept": False,
            }
        )
        if url_base and current_output_dir is not None:
            try:
                summary = cli.fetch_remote_summary(url_base, remote_token)
                if summary:
                    download_available_rows(url_base, remote_token, summary, current_output_dir, set())
                    current_summary = summary
            except Exception as reconcile_error:
                cli.append_job_event(job_dir, "benchmark_final_reconcile_failed", {"error": str(reconcile_error)})
        if current_summary and current_output_dir is not None:
            write_configuration_files(current_output_dir, current_model, current_steps, current_summary)
            current_name = configuration_name(current_model, current_steps)
            if all(record["configuration"] != current_name for record in records):
                records.append(
                    configuration_record(
                        current_model,
                        current_steps,
                        current_output_dir,
                        current_summary,
                        time.time() - current_config_started,
                        dph,
                    )
                )
        write_reports(report_dir, job_id, benchmark, records)
        cli.atomic_write_json(job_dir / "benchmark_state.json", benchmark)
        cli.write_job_state(job_dir, {"status": "benchmark_error", "error": str(exc)})
        cli.append_job_event(job_dir, "benchmark_error", {"error": str(exc)})
        raise
    finally:
        if benchmark_completed and not args.destroy_after:
            cli.write_job_state(job_dir, {"instance_destroyed": False, "instance_kept_for_benchmark": True})
            cli.append_job_event(job_dir, "benchmark_instance_kept", {"instance_id": instance_id})
            print(f"評測完成；按要求保留實例 {instance_id}。實例仍以 ${dph:.4f}/小時計費。")
        else:
            destroyed = cli.destroy_instance_best_effort(api_key, instance_id, job_dir)
            if destroyed and reuse_job_dir is not None:
                (reuse_job_dir / ".remote_session.json").unlink(missing_ok=True)
                cli.write_job_state(
                    reuse_job_dir,
                    {
                        "instance_destroyed": True,
                        "instance_destroyed_by_job": job_id,
                        "instance_kept_for_benchmark": False,
                    },
                )
                cli.append_job_event(
                    reuse_job_dir,
                    "reused_instance_destroyed",
                    {"instance_id": instance_id, "benchmark_job_id": job_id},
                )
            if benchmark_completed:
                benchmark["instance_kept"] = False
                benchmark["reported_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                write_reports(report_dir, job_id, benchmark, records)
