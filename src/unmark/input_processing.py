from __future__ import annotations

import csv
import os
import secrets
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CSV_NAME_COLUMNS = ("文件名", "filename", "file_name", "name", "id", "document_id")
CSV_LINK_COLUMNS = ("要去水印的link", "original_image", "original_url", "link", "url", "image")


def clean_name(value: str, fallback: str) -> str:
    raw = Path(value.strip()).stem if value.strip() else fallback
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("_")
    return cleaned or fallback


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"無法建立不重複檔名：{path}")


def safe_extract_zip(zip_path: Path, image_dir: Path, max_total_bytes: int) -> list[tuple[str, str]]:
    rows = []
    extracted_bytes = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if len(members) > 10_000:
            raise ValueError("ZIP 內檔案數量超過 10,000。")
        for info in members:
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if info.file_size < 0 or extracted_bytes + info.file_size > max_total_bytes:
                raise ValueError("ZIP 解壓後的圖片總大小超過設定上限。")
            target = unique_path(image_dir / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            actual_size = target.stat().st_size
            extracted_bytes += actual_size
            if actual_size != info.file_size or extracted_bytes > max_total_bytes:
                target.unlink(missing_ok=True)
                raise ValueError("ZIP 內圖片大小異常或超過設定上限。")
            fallback = f"image_{len(rows) + 1:04d}"
            rows.append((clean_name(target.name, fallback), f"images/{target.name}"))
    return rows


def detect_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return ""


def rows_from_csv(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("CSV 沒有標題列。")
        name_column = detect_column(reader.fieldnames, CSV_NAME_COLUMNS)
        link_column = detect_column(reader.fieldnames, CSV_LINK_COLUMNS)
        if not link_column:
            raise ValueError("CSV 必須包含「要去水印的link」或 original_image / original_url / link / url 欄位。")
        rows = []
        for index, row in enumerate(reader, start=1):
            link = (row.get(link_column) or "").strip()
            if not link:
                continue
            if name_column:
                name = clean_name(row.get(name_column, ""), f"link_{index:04d}")
            else:
                name = f"link_{index:04d}_{secrets.token_hex(3)}"
            rows.append((name, link))
        return rows


def rows_from_txt(path: Path) -> list[tuple[str, str]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        link = line.strip()
        if not link or link.startswith("#"):
            continue
        rows.append((f"link_{index:04d}_{secrets.token_hex(3)}", link))
    return rows


def write_normalized_csv(input_csv: Path, rows: list[tuple[str, str]]) -> None:
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = input_csv.with_name(f".{input_csv.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=["id", "original_image"])
            writer.writeheader()
            for name, original in rows:
                writer.writerow({"id": name, "original_image": original})
        os.replace(temporary_path, input_csv)
    finally:
        temporary_path.unlink(missing_ok=True)


def normalize_uploaded_files(
    upload_dir: Path,
    files: list[tuple[str, bytes | Path]],
    max_total_bytes: int,
) -> Path:
    image_dir = upload_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    materialized_bytes = 0
    saved_inputs = upload_dir / "source_files"
    saved_inputs.mkdir(parents=True, exist_ok=True)

    for filename, data in files:
        safe_filename = Path(filename or f"upload_{secrets.token_hex(4)}").name
        suffix = Path(safe_filename).suffix.lower()
        saved = unique_path(saved_inputs / safe_filename)
        if isinstance(data, Path):
            shutil.copy2(data, saved)
        else:
            saved.write_bytes(data)

        if suffix in IMAGE_EXTENSIONS:
            target = unique_path(image_dir / safe_filename)
            shutil.copy2(saved, target)
            materialized_bytes += target.stat().st_size
            if materialized_bytes > max_total_bytes:
                raise ValueError("圖片總大小超過設定上限。")
            fallback = f"image_{len(rows) + 1:04d}"
            rows.append((clean_name(target.name, fallback), f"images/{target.name}"))
        elif suffix == ".zip":
            remaining_bytes = max_total_bytes - materialized_bytes
            if remaining_bytes < 1:
                raise ValueError("圖片總大小超過設定上限。")
            zip_rows = safe_extract_zip(saved, image_dir, remaining_bytes)
            materialized_bytes += sum((upload_dir / relative_path).stat().st_size for _, relative_path in zip_rows)
            rows.extend(zip_rows)
        elif suffix == ".csv":
            rows.extend(rows_from_csv(saved))
        elif suffix == ".txt":
            rows.extend(rows_from_txt(saved))
        else:
            raise ValueError(f"不支援的檔案類型：{safe_filename}")

    if not rows:
        raise ValueError("未找到可處理的圖片或連結。")
    input_csv = upload_dir / "input.csv"
    write_normalized_csv(input_csv, rows)
    return input_csv


def normalize_json_payload(upload_dir: Path, payload: dict) -> Path:
    rows: list[tuple[str, str]] = []
    items = payload.get("items")
    if isinstance(items, list):
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            link = str(
                item.get("link")
                or item.get("original_image")
                or item.get("original_url")
                or item.get("url")
                or ""
            ).strip()
            if not link:
                continue
            raw_name = str(item.get("filename") or item.get("name") or item.get("id") or "")
            rows.append((clean_name(raw_name, f"link_{index:04d}"), link))
    links = payload.get("links")
    if isinstance(links, list):
        for index, link in enumerate(links, start=1):
            link_text = str(link).strip()
            if link_text:
                rows.append((f"link_{index:04d}_{secrets.token_hex(3)}", link_text))
    if not rows:
        raise ValueError("JSON 需要提供 items 或 links。")
    input_csv = upload_dir / "input.csv"
    write_normalized_csv(input_csv, rows)
    return input_csv
