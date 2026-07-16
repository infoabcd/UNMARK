# model_presets.json 模型預設總結

本文整理 [`model_presets.json`](../../model_presets.json) 內的模型快捷鍵、Hugging Face 來源、推理設定及適用情境。

重點先講清楚：以下不是同一個 Hugging Face 倉庫。

- `9B` 指向 `black-forest-labs/FLUX.2-klein-9B`
- `4B` 指向 `black-forest-labs/FLUX.2-klein-4B`
- `KONTEXT` 指向 `black-forest-labs/FLUX.1-Kontext-dev`
- `KWR` 是 `black-forest-labs/FLUX.1-Kontext-dev` 底座，再額外載入 `prithivMLmods/Kontext-Watermark-Remover` LoRA

## 模型一覽

| 快捷鍵 | 顯示名稱 | Pipeline | Hugging Face 底座模型 | LoRA / 適配器 | 建議 steps | Guidance | 建議 GPU | 用途 |
|--------|----------|----------|----------------------|---------------|------------|----------|----------|------|
| `9B` | FLUX.2-klein-9B | `flux2_klein` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | 無 | 8 | 1.0 | H200 | 預設模型，速度快、成本低，適合大批量去水印 |
| `4B` | FLUX.2-klein-4B | `flux2_klein` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 無 | 8 | 1.0 | H100 | 低顯存、低成本測試；對應 4B 工作流實測文檔 |
| `KONTEXT` | FLUX.1-Kontext-dev | `kontext` | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | 無 | 28 | 2.5 | H200 | Kontext 基礎圖像編輯模型，用作無 LoRA 對照 |
| `KWR` | FLUX.1-Kontext-dev + Kontext-Watermark-Remover | `kontext` | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | [prithivMLmods/Kontext-Watermark-Remover](https://huggingface.co/prithivMLmods/Kontext-Watermark-Remover) | 24 | 2.5 | H200 | Kontext 底座加專用去水印 LoRA，適合高質量補處理 |

## 模型授權

| 快捷鍵 | 模型授權 |
|---|---|
| `9B` | FLUX Non-Commercial License；須在 Hugging Face 模型頁同意條款。 |
| `4B` | Apache License 2.0。 |
| `KONTEXT` | FLUX.1 dev Non-Commercial License；須在 Hugging Face 模型頁同意條款。 |
| `KWR` | 底座沿用 FLUX.1 dev Non-Commercial License；LoRA 頁面標示為 Apache License 2.0。組合使用時仍須遵守底座模型條款。 |

本倉庫不包含或再分發模型權重。執行前應直接查看各模型頁面的最新授權及使用政策。

## 欄位說明

| 欄位 | 說明 |
|------|------|
| `label` | 顯示名稱，會在命令列輸出及匯報中使用。 |
| `description` | 模型用途摘要。 |
| `pipeline` | 遠端 worker 載入模型時使用的 pipeline 類型；目前支援 `flux2_klein` 及 `kontext`。 |
| `model_id` | Hugging Face 底座模型 ID，直接傳入 Diffusers `from_pretrained()`。 |
| `base_hf_url` | 底座模型的 Hugging Face 頁面。 |
| `lora_id` | LoRA / 適配器的 Hugging Face repo ID；沒有 LoRA 時為 `null`。 |
| `lora_weight` | LoRA 權重檔名；沒有 LoRA 時為 `null`。 |
| `adapter_hf_url` | LoRA / 適配器的 Hugging Face 頁面；沒有 LoRA 時為 `null`。 |
| `hf_url` | 主要參考頁面。純底座模型會指向 `base_hf_url`；KWR 會指向專用去水印 LoRA 頁面。 |
| `default_steps` | 建議推理步數。 |
| `guidance_scale` | 建議 guidance scale。 |
| `recommended_gpu` | 建議使用的 Vast GPU 型號。 |

## 使用建議

`9B` 是目前預設選擇，重點是快、平、適合大量圖片。`4B` 是另一個獨立的 FLUX.2 Klein 倉庫，適合低成本或低顯存測試。

`KONTEXT` 同 `KWR` 都使用 `FLUX.1-Kontext-dev` 作為底座；分別只在於 `KWR` 會額外載入 [Kontext-Watermark-Remover](https://huggingface.co/prithivMLmods/Kontext-Watermark-Remover) LoRA。

如果要跑大批量，優先用 `9B` 或 `4B`；如果某批圖片需要更精細補處理，再抽樣用 `KWR` 比較效果。

## 切換方式

在 `.env` 設定：

```env
UNMARK_MODEL=9B
UNMARK_STEPS=8
```

亦可在 LocalWeb 設定頁選擇模型及步數，或先檢視模型清單：

```bash
python3 run_unmark.py --list-models
```
