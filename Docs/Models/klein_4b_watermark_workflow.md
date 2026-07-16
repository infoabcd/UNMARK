# FLUX.2 Klein 4B 去水印工作流技術文檔

## 結論摘要

本次測試使用 `black-forest-labs/FLUX.2-klein-4B`，在 H100 SXM 80GB 機器上處理圖片去水印。

實測結果：

| 項目 | 數值 |
|------|-----:|
| 測試圖片數量 | 120 張成功 |
| 跳過失敗 URL | 23 個 404 |
| 總耗時 | 236.5 秒 |
| 平均端到端耗時 | 1.97 秒 / 張 |
| 平均模型推理耗時 | 1.53 秒 / 張 |
| 端到端吞吐量 | 約 1,826 張 / 小時 |
| 純推理吞吐量 | 約 2,350 張 / 小時 |

如果伺服器價格是 `$4 / 小時`：

| 方案 | 單張成本 | 對比舊 API |
|------|---------:|-----------:|
| 舊 API | `$0.075 / 張` | 基準 |
| Klein 4B 自建，按端到端計算 | `$0.00219 / 張` | 便宜約 34.3 倍 |
| Klein 4B 自建，只按推理計算 | `$0.00170 / 張` | 便宜約 44.0 倍 |

建議業務側使用端到端數字估算成本：

```text
約 1,800 張 / 小時
約 $0.0022 / 張
比舊 API 便宜約 34 倍
成本下降約 97.1%
```

## 成本計算

舊 API：

```text
$0.075 / 張
```

自建 Klein 4B：

```text
機器價格 = $4 / 小時
端到端速度 = 1.97 秒 / 張
每小時可處理 = 3600 / 1.97 = 約 1,826 張
單張成本 = 4 / 1826 = $0.00219 / 張
```

便宜倍數：

```text
0.075 / 0.00219 = 約 34.3 倍
```

節省比例：

```text
(0.075 - 0.00219) / 0.075 = 97.1%
```

## 本次測試機器規格

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA H100 SXM 80GB |
| GPU 顯存 | 80GB HBM3 |
| 機器成本 | $4 / 小時 |
| CUDA | CUDA 12.4 runtime / driver supports CUDA 13.0 |
| PyTorch | 2.6.0+cu124 |
| Diffusers | 0.38.0 |
| Python | python3 |
| 推理精度 | bfloat16 |

## 模型參數

| 參數 | 值 |
|------|----|
| 模型 | `black-forest-labs/FLUX.2-klein-4B` |
| Pipeline | `Flux2KleinPipeline` |
| Steps | `8` |
| Guidance scale | `1.0` |
| 輸出格式 | PNG |
| 原圖儲存格式 | JPG |
| 輸出尺寸 | 與原圖保持一致 |

提示詞：

```text
[photo content], remove any watermark text or logos from the image while preserving the background, texture, lighting, and overall realism. Do not crop, zoom, resize, or change the camera framing. Ensure the edited areas blend naturally with surrounding details, leaving no visible traces of watermark removal.
```

## 輸入 CSV 格式

CSV 至少需要以下欄位：

```csv
id,original_url
581377,https://example.com/image.jpg
581376,https://example.com/image2.jpg
```

實際使用的 CSV 欄位：

```csv
id,original_url,s3_url,file_size,mime_type,created_at,updated_at,thumbnail_s3_url
```

工作流只依賴：

```text
id
original_url
```

## 輸出目錄結構

每一張成功圖片單獨一個資料夾：

```text
klein_4b_120_pairs_steps8_ready/
  001_581377/
    original.jpg
    result.png
    meta.json
  002_581376/
    original.jpg
    result.png
    meta.json
  summary.json
  manifest.jsonl
  failures.jsonl
  contact_sheet_first_12.jpg
```

檔案說明：

| 檔案 | 說明 |
|------|------|
| `original.jpg` | 下載後的原圖 |
| `result.png` | Klein 4B 去水印結果 |
| `meta.json` | 單張圖片的 URL、尺寸、耗時、輸出路徑等 metadata |
| `summary.json` | 整批任務的統計結果 |
| `manifest.jsonl` | 每張成功圖片一行 JSON |
| `failures.jsonl` | 下載失敗或處理失敗的圖片紀錄 |
| `contact_sheet_first_12.jpg` | 前 12 張 before / after 預覽 |

## 安裝依賴

建議在乾淨 GPU 環境執行：

```bash
pip install -U --no-cache-dir \
  torch \
  diffusers \
  transformers \
  accelerate \
  huggingface_hub \
  pillow
```

如果在 Colab 或 Vast 上，優先使用環境已安裝的 CUDA 版 PyTorch，避免重複安裝錯誤版本。

## 核心程式碼

### 載入模型

```python
import os
import torch
from diffusers import Flux2KleinPipeline

HF_TOKEN = os.environ.get("HF_TOKEN")

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
).to("cuda")
```

### 圖片預處理

FLUX 模型較適合處理寬高為 16 的倍數的圖片。為了不改變最終尺寸，可以先 padding 到 16 的倍數，輸出後再 crop 回原始尺寸。

```python
from io import BytesIO
from PIL import Image, ImageOps


def multiple_of(value, divisor):
    return ((value + divisor - 1) // divisor) * divisor


def prepare_image(raw_bytes):
    original = Image.open(BytesIO(raw_bytes))
    original = ImageOps.exif_transpose(original).convert("RGB")
    width, height = original.size

    padded_width = multiple_of(width, 16)
    padded_height = multiple_of(height, 16)

    if (padded_width, padded_height) == (width, height):
        return original, original.copy()

    padded = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
    padded.paste(original, (0, 0))
    return original, padded
```

### 單張推理

```python
from pathlib import Path

import torch

prompt = Path("Prompts/萬能提示詞.md").read_text(encoding="utf-8").strip()


@torch.inference_mode()
def remove_watermark(pipe, model_input, original, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        image=model_input,
        prompt=prompt,
        width=model_input.width,
        height=model_input.height,
        guidance_scale=1.0,
        num_inference_steps=8,
        generator=generator,
    ).images[0]

    result = result.convert("RGB")
    result = result.crop((0, 0, original.width, original.height))
    return result
```

### 單張輸出結構

```python
folder = OUTPUT_ROOT / f"{index:03d}_{image_id}"
folder.mkdir(parents=True, exist_ok=True)

original_path = folder / "original.jpg"
result_path = folder / "result.png"
meta_path = folder / "meta.json"

original.save(original_path, quality=95)
result.save(result_path)
```

## 批處理邏輯

建議流程：

1. 讀取 CSV。
2. 過濾沒有 `original_url` 的行。
3. 逐行下載圖片。
4. 如果 URL 404 或下載失敗，寫入 `failures.jsonl`，繼續下一張。
5. 下載成功後才建立輸出資料夾。
6. 儲存 `original.jpg`。
7. 執行 Klein 4B 推理。
8. Crop 回原始尺寸。
9. 儲存 `result.png`。
10. 寫入 `meta.json` 和 `manifest.jsonl`。
11. 到達目標成功數量後停止。
12. 寫入 `summary.json`。

## 可重用腳本

本地已有完整可重用腳本：

```text
<project>/klein_4b_120_batch_vast.py
```

執行方式示例。先在 shell 設定 `HF_TOKEN`，再執行腳本：

```bash
CSV_PATH="/workspace/watermark_urls.csv" \
OUTPUT_ROOT="/workspace/klein_4b_batch_output" \
IMAGE_COUNT="120" \
NUM_INFERENCE_STEPS="8" \
GUIDANCE_SCALE="1.0" \
python3 -u klein_4b_120_batch_vast.py
```

## 質素說明

Klein 4B 的優點：

- 速度非常快。
- 成本顯著低於線上 API。
- 對明顯文字水印和 logo 去除效果不錯。
- 可保持原圖尺寸。

Klein 4B 的限制：

- 相比 `FLUX.1-Kontext-dev + watermark LoRA`，Klein 4B 更容易改變局部細節。
- 某些複雜水印區域可能會重繪背景。
- 對極小文字、半透明水印、強遮擋水印，需要繼續抽樣檢查。

生產建議：

- 用 Klein 4B 做大批量低成本處理。
- 每批抽樣生成 contact sheet 做人工 QC。
- 對失敗或質素較差的圖片，再用較慢但更穩的 Kontext 模型補處理。

## 本次實測輸出

清理後的 120 張結果目錄：

```text
~/google_colab_watermark/klein_4b_120_pairs_steps8_ready
```

預覽圖：

```text
~/google_colab_watermark/klein_4b_120_pairs_steps8_ready/contact_sheet_first_12.jpg
```
