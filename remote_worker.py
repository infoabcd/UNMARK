#!/usr/bin/env python3
"""在 Vast GPU 實例上執行批次去水印（多模型）。"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_for_model(image: Image.Image, max_side: int) -> Image.Image:
    scale = min(1.0, max_side / max(image.size))
    if scale >= 1:
        return image
    width = round(image.width * scale)
    height = round(image.height * scale)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def rounded_dims(image: Image.Image) -> tuple[int, int]:
    width = max(64, round(image.width / 16) * 16)
    height = max(64, round(image.height / 16) * 16)
    return width, height


def restore_original_size(image: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    if image.size == original_size:
        return image
    return image.resize(original_size, Image.Resampling.LANCZOS)


def safe_name_part(value: str, max_length: int = 80) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value.strip()
    )
    safe = safe.strip("._")
    return safe[:max_length]


def output_name(index: int, original_ref: str, document_id: str = "") -> str:
    # 取得原始圖片的副檔名（預設為 .jpg）
    suffix = Path(urlparse(original_ref).path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"

    id_part = safe_name_part(document_id)
    if id_part:
        return f"{id_part}{suffix}"

    digest = hashlib.sha1(original_ref.encode("utf-8")).hexdigest()[:8]
    stem = Path(original_ref).stem
    safe_stem = safe_name_part(stem, max_length=40)
    return f"{index:04d}_{safe_stem}_{digest}{suffix}"


def load_source_image(item: dict, raw_dir: Path, max_download_bytes: int) -> Path:
    if item["source_type"] == "url":
        url = item["original_ref"]
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        path = raw_dir / f"{item['index']:04d}{suffix}"
        if not path.exists():
            parsed = urlparse(url)
            referer = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else ""
            temp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
            try:
                with requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": referer},
                    stream=True,
                    timeout=(20, 60),
                ) as response:
                    response.raise_for_status()
                    declared_size = int(response.headers.get("Content-Length", "-1"))
                    if declared_size > max_download_bytes:
                        raise RuntimeError(
                            f"網址圖片超過下載上限：{declared_size} bytes > {max_download_bytes} bytes"
                        )
                    received = 0
                    with temp_path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            if received > max_download_bytes:
                                raise RuntimeError(
                                    f"網址圖片超過下載上限：已收到 {received} bytes"
                                )
                            output.write(chunk)
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)
        return path

    filename = item["filename"]
    path = raw_dir / filename
    if path.exists():
        expected_size = item.get("size_bytes")
        expected_sha256 = str(item.get("sha256", "")).lower()
        if expected_size not in (None, "") and path.stat().st_size != int(expected_size):
            raise RuntimeError(f"本地上傳圖片大小不符：{path}")
        if expected_sha256 and sha256_file(path) != expected_sha256:
            raise RuntimeError(f"本地上傳圖片 SHA-256 不符：{path}")
        return path

    if "data_b64" in item and item["data_b64"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(item["data_b64"]))
        return path

    raise RuntimeError(f"本地上傳圖片不存在於遠端路徑：{path}")


def atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_status(work_dir: Path, payload: dict) -> None:
    atomic_write_json(work_dir / "job_status.json", payload)


def build_running_status(
    model_key: str,
    total: int,
    done: int,
    ok_count: int,
    load_seconds: float,
    rows: list[dict],
) -> dict:
    return {
        "status": "running",
        "model": model_key,
        "total": total,
        "done": done,
        "ok": ok_count,
        "pipeline_load_seconds": round(load_seconds, 4),
        "rows": rows,
    }


def load_pipeline(model_cfg: dict, work_dir: Path, token: str):
    cache_dir = str(work_dir / "hf_cache")
    pipeline_type = model_cfg["pipeline"]
    model_id = model_cfg["model_id"]

    if pipeline_type == "flux2_klein":
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token,
            cache_dir=cache_dir,
        )
        pipe.to("cuda")
        return pipe, "flux2_klein"

    if pipeline_type == "kontext":
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token,
            cache_dir=cache_dir,
        )
        lora_id = model_cfg.get("lora_id")
        if lora_id:
            pipe.load_lora_weights(
                lora_id,
                weight_name=model_cfg.get("lora_weight"),
                adapter_name="watermark_remover",
                token=token,
            )
            pipe.set_adapters(["watermark_remover"], adapter_weights=[1.0])
        pipe.to("cuda")
        return pipe, "kontext"

    raise ValueError(f"不支援的 pipeline：{pipeline_type}")


def run_inference(
    pipe,
    pipeline_kind: str,
    model_cfg: dict,
    image: Image.Image,
    steps: int,
    max_side: int,
    seed: int,
    prompt: str,
) -> Image.Image:
    guidance = float(model_cfg.get("guidance_scale", 1.0))
    original_size = image.size
    model_image = resize_for_model(image, max_side)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if pipeline_kind == "flux2_klein":
        width, height = rounded_dims(model_image)
        result = pipe(
            image=model_image,
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        return restore_original_size(result, original_size)

    result = pipe(
        image=model_image,
        prompt=prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        width=model_image.width,
        height=model_image.height,
        generator=generator,
    ).images[0]
    return restore_original_size(result, original_size)


def main() -> None:
    work_dir = Path(os.environ.get("WORK_DIR", "/workspace/unmark"))
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))

    model_key = manifest.get("model", "9B")
    model_cfg = manifest["model_config"]
    steps = int(manifest.get("steps", model_cfg.get("default_steps", 8)))
    seed = int(manifest.get("seed", 42))
    if seed < 0:
        raise ValueError("seed 不可小於 0")
    max_side = int(manifest.get("max_side", 1024))
    items = manifest["items"]
    concurrency = int(manifest.get("concurrency", 12))
    max_download_bytes = int(manifest.get("max_download_bytes", 100 * 1024 * 1024))
    prompt = str(manifest.get("prompt", "")).strip()
    prompt_preset = str(manifest.get("prompt_preset", "")).strip()
    if not prompt:
        raise ValueError("manifest 缺少提示詞內容")

    raw_dir = work_dir / "raw_images"
    out_dir = work_dir / "output_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(work_dir / "hf_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(work_dir / "hf_cache" / "hub"))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("必須設定 HF_TOKEN")

    write_status(
        work_dir,
        {
            "status": "loading_model",
            "model": model_key,
            "seed": seed,
            "prompt_preset": prompt_preset,
            "total": len(items),
            "done": 0,
        },
    )

    load_start = time.time()
    pipe, pipeline_kind = load_pipeline(model_cfg, work_dir, token)
    load_seconds = time.time() - load_start

    rows: list[dict] = []
    ok_count = 0

    def append_row(
        item: dict,
        status: str,
        output_relative_path: str,
        seconds: float,
        error: str = "",
        output_size_bytes: int | str = "",
        output_sha256: str = "",
    ) -> None:
        nonlocal rows
        index = int(item["index"])
        rows.append(
            {
                "original_image": item["original_ref"],
                "input_key": item.get("input_key", item["original_ref"]),
                "output_image": output_relative_path,
                "status": status,
                "seconds": round(seconds, 4),
                "error": error,
                "output_size_bytes": output_size_bytes,
                "output_sha256": output_sha256,
                "index": index,
            }
        )
        rows.sort(key=lambda row: row["index"])
        write_status(
            work_dir,
            build_running_status(
                model_key=model_key,
                total=len(items),
                done=len(rows),
                ok_count=ok_count,
                load_seconds=load_seconds,
                rows=rows,
            ),
        )

    def prepare_source(item: dict) -> tuple[dict, Path | None, str]:
        try:
            raw_path = load_source_image(item, raw_dir, max_download_bytes)
            return item, raw_path, ""
        except Exception as exc:
            return item, None, str(exc)

    prepare_workers = max(1, min(concurrency, len(items)))
    print(f"準備 {len(items)} 張輸入圖片，下載並行數：{prepare_workers}")
    with ThreadPoolExecutor(max_workers=prepare_workers) as executor:
        prepared_items = list(executor.map(prepare_source, items))

    print("開始逐張 GPU 推理...")
    fatal_error = ""
    for item, raw_path, input_error in prepared_items:
        started_at = time.time()
        if input_error:
            append_row(item, "error", "", time.time() - started_at, f"讀取輸入失敗：{input_error}")
            continue

        original_ref = item["original_ref"]
        output_filename = output_name(int(item["index"]), original_ref, item.get("document_id", ""))
        output_relative_path = f"output_images/{output_filename}"
        try:
            with Image.open(raw_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
        except Exception as exc:
            append_row(item, "error", "", time.time() - started_at, f"圖片解碼失敗：{exc}")
            continue

        try:
            result = run_inference(pipe, pipeline_kind, model_cfg, image, steps, max_side, seed, prompt)
            output_path = out_dir / output_filename
            temp_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}{output_path.suffix}")
            try:
                result.save(temp_path)
                output_size_bytes = temp_path.stat().st_size
                output_sha256 = sha256_file(temp_path)
                os.replace(temp_path, output_path)
            finally:
                temp_path.unlink(missing_ok=True)
            ok_count += 1
            append_row(
                item,
                "ok",
                output_relative_path,
                time.time() - started_at,
                output_size_bytes=output_size_bytes,
                output_sha256=output_sha256,
            )
        except Exception as exc:
            fatal_error = f"GPU 推理失敗：{exc}"
            append_row(item, "error", "", time.time() - started_at, fatal_error)
            print(traceback.format_exc(), flush=True)
            break

    rows = sorted(rows, key=lambda row: row["index"])
    for row in rows:
        row.pop("index", None)

    with (work_dir / "output.csv").open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["original_image", "output_image"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"original_image": row["original_image"], "output_image": row["output_image"]}
            )

    if fatal_error or ok_count == 0:
        terminal_status = "error"
    elif ok_count < len(items):
        terminal_status = "partial"
    else:
        terminal_status = "done"

    summary = {
        "status": terminal_status,
        "model_key": model_key,
        "model": model_cfg.get("model_id"),
        "lora": model_cfg.get("lora_id"),
        "steps": steps,
        "seed": seed,
        "prompt_preset": prompt_preset,
        "max_side": max_side,
        "total": len(items),
        "ok": ok_count,
        "failed": sum(1 for row in rows if row["status"] == "error"),
        "not_run": len(items) - len(rows),
        "error": fatal_error,
        "pipeline_load_seconds": round(load_seconds, 4),
        "rows": rows,
    }
    write_status(work_dir, summary)
    atomic_write_json(work_dir / "result.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        work_dir = Path(os.environ.get("WORK_DIR", "/workspace/unmark"))
        failure = {
            "status": "error",
            "total": 0,
            "ok": 0,
            "failed": 0,
            "not_run": 0,
            "error": f"遠端工作程序終止：{exc}",
            "rows": [],
        }
        try:
            manifest_path = work_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                failure["total"] = len(manifest.get("items", []))
                failure["not_run"] = failure["total"]
            write_status(work_dir, failure)
            atomic_write_json(work_dir / "result.json", failure)
        finally:
            print(traceback.format_exc(), flush=True)
        raise
