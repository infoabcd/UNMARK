# 任務、恢復與停機

## Job 目錄

```text
jobs/
  20260715_153000_ab12cd34/
    input.csv
    input_manifest.jsonl
    runtime_config.json
    job_state.json
    events.jsonl
    output.csv
    remote_result.json
    output_images/
    logs/
```

| 檔案 | 用途 |
|------|------|
| `input.csv` | 輸入快照。 |
| `input_manifest.jsonl` | 解析後的圖片清單，一行一張。 |
| `runtime_config.json` | 實際使用的設定。 |
| `job_state.json` | Vast 實例 ID、遠端 API URL、狀態、數量。 |
| `events.jsonl` | 事件紀錄。 |
| `output.csv` | 已下載或失敗的圖片對照表。 |
| `remote_result.json` | 遠端任務摘要。 |
| `output_images/` | 已下載到本地的結果圖片。 |

## 命令

### 規劃，不開機

```bash
python3 run_unmark.py plan --input-csv input.csv
```

### 執行

```bash
python3 run_unmark.py run --input-csv input.csv
```

指定 job id：

```bash
python3 run_unmark.py run --job-id batch_001 --input-csv input.csv
```

### 查看狀態

```bash
python3 run_unmark.py status --job-dir jobs/batch_001
```

### 補下載

```bash
python3 run_unmark.py reconcile --job-dir jobs/batch_001
```

### 繼續處理未完成圖片

```bash
python3 run_unmark.py resume --job-dir jobs/batch_001
```

### 建立補處理 CSV

```bash
python3 run_unmark.py fallback-queue --job-dir jobs/batch_001
python3 run_unmark.py run --input-csv jobs/batch_001/fallback_input.csv --model KWR
```

### 手動停機

```bash
python3 run_unmark.py destroy --job-dir jobs/batch_001
```

### LocalWeb

```bash
python3 run_unmark.py serve-local
```

```text
http://127.0.0.1:8787
```

## 出錯策略

預設設定：

```env
UNMARK_BATCH_SIZE=100
UNMARK_DESTROY_ON_SUCCESS=true
```

行為：

- 成功完成：銷毀 Vast 實例。
- 例外、逾時、未知錯誤：先讀取遠端狀態，補下載已完成圖片，然後銷毀 Vast 實例。
- 每批最多 100 張。
- `output.csv`、`job_state.json`、`remote_result.json` 使用原子寫檔。
