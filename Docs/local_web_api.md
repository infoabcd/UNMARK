# LocalWeb、API 與匯入匯出

## 啟動

```bash
python3 run_unmark.py serve-local
```

打開：

```text
http://127.0.0.1:8787
```

## 密鑰

| 密鑰 | 用途 | 建議保存位置 |
|------|------|--------------|
| `VAST_API_KEY` | 建立、查詢、銷毀 Vast 實例。 | `.env` |
| `HF_TOKEN` | 下載 Hugging Face 模型。 | `.env` |
| `UNMARK_LOCAL_API_KEY` | 保護 LocalWeb API，避免其他程式亂開任務。 | `.env` |

LocalWeb 的設定頁亦可以臨時填寫 Vast / Hugging Face 密鑰。臨時密鑰只保存在記憶體，重啟 LocalWeb 後會消失，不會寫入 `localweb_settings.json`。

一般設定會保存到 `localweb_settings.json`，例如模型、GPU、批次大小、逾時時間、jobs 目錄。密鑰不會保存到 JSON。

本地 API Key 有單獨管理區，可以新增多個 key、備註、啟用 / 停用、刪除。新增時只需填備註，系統會隨機生成 API Key。設定頁生成的 key 只存在今次 LocalWeb 記憶體，重啟後會消失。若要長期使用，請把生成後的一次性明文放入 `.env` 的 `UNMARK_LOCAL_API_KEY`。

新生成的 API Key 只會顯示一次。離開頁面後，同一個 LocalWeb session 只保留預覽和雜湊，不能反查明文。

## GUI 建立任務

雲端 GPU 需要準備環境及載入模型，少量圖片通常無法攤薄啟動成本。本項目較適合大量圖片批次；只處理幾張圖片時，不建議個人用戶租用 GPU。

在 LocalWeb 點擊「新增任務」，可以上傳：

| 類型 | 說明 |
|------|------|
| 圖片 | 支援 `.jpg`、`.jpeg`、`.png`、`.webp`、`.bmp`，可以一次選多張。 |
| ZIP | ZIP 內可以放多張圖片；系統只會讀圖片，並共用設定頁的解壓總大小上限。 |
| CSV | 支援指定輸出文件名及圖片 link。 |
| TXT | 一行一個圖片 link；不能指定文件名，系統會自動命名。 |

新增任務頁可選提示詞，預設為「萬能提示詞」。選項來自 `Prompts/` 內的 `.md` 檔案；加入新檔案後重新載入頁面即可使用。詳見 [提示詞選擇](prompt_presets.md)。

CSV 最簡單格式：

```csv
文件名,要去水印的link
客廳_001,https://example.com/photo1.jpg
睡房_002,https://example.com/photo2.jpg
```

TXT 格式：

```text
https://example.com/photo1.jpg
https://example.com/photo2.jpg
```

設定頁的「每張網址圖片下載上限 MB」會限制每條 link 的下載大小；即使伺服器沒有提供 `Content-Length`，實際串流下載超過上限時亦會停止。

任務有成功輸出後，可在任務詳情頁按「下載全部結果」。ZIP 只會包含輸出圖片、`output.csv` 及 `job_state.json`，不會包含遠端臨時連線憑證。

## API

API 預設需要本地 API Key。設定頁可以管理多個 key。呼叫時請在 header 加入：

```text
X-API-Key: 你的本地APIKey
```

### 健康檢查

```bash
curl http://127.0.0.1:8787/api/health \
  -H "X-API-Key: 你的本地APIKey"
```

### 建立任務：JSON

```bash
curl -X POST http://127.0.0.1:8787/api/jobs \
  -H "X-API-Key: 你的本地APIKey" \
  -H "Content-Type: application/json" \
  -d '{"items":[{"filename":"客廳_001","link":"https://example.com/photo.jpg"}],"model":"9B","prompt_preset":"樓盤Logo"}'
```

`items` 欄位：

| 欄位 | 說明 |
|------|------|
| `filename` | 輸出文件名，不需要副檔名。 |
| `link` | 要去水印的圖片網址。 |

任務層設定亦接受 `model`、`steps`、`seed` 及 `prompt_preset`。`prompt_preset` 是 `Prompts/` 內不包括 `.md` 的檔名；省略時使用設定頁的選項。

亦可用：

```json
{"links":["https://example.com/photo1.jpg","https://example.com/photo2.jpg"]}
```

這種寫法會自動生成文件名。

### 建立任務：上傳檔案

```bash
curl -X POST http://127.0.0.1:8787/api/jobs \
  -H "X-API-Key: 你的本地APIKey" \
  -F "input_files=@links.csv" \
  -F "model=9B" \
  -F "prompt_preset=萬能提示詞"
```

可以上傳圖片、ZIP、CSV 或 TXT。

### 查看任務

```bash
curl http://127.0.0.1:8787/api/jobs \
  -H "X-API-Key: 你的本地APIKey"

curl http://127.0.0.1:8787/api/jobs/<job-id> \
  -H "X-API-Key: 你的本地APIKey"
```

## all_to_once_script

給熟悉命令列的用戶使用。

目錄由 `.env` 約定：

```env
UNMARK_ALL_TO_ONCE_IMPORT_DIR=all_to_once_script/import
UNMARK_ALL_TO_ONCE_WORK_DIR=all_to_once_script/work
UNMARK_ALL_TO_ONCE_EXPORT_DIR=all_to_once_script/export
```

### 匯入

把圖片、ZIP、CSV 或 TXT 放入 `all_to_once_script/import/`，然後執行：

```bash
python3 all_to_once_script/import_inputs.py
```

腳本會建立：

```text
all_to_once_script/work/input.csv
```

### 執行

```bash
python3 run_unmark.py run --input-csv all_to_once_script/work/input.csv
```

### 匯出

```bash
python3 all_to_once_script/export_outputs.py
```

結果會複製到：

```text
all_to_once_script/export/<job-id>/
```

### 完整執行

```bash
python3 all_to_once_script/run_all_to_once.py
```

這個命令會匯入檔案、開 Vast GPU 機、完成後匯出結果。Vast 會按機器運行時間收費；只想整理輸入或檢查設定時，不要執行這個命令。
