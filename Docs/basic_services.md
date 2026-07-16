# 基礎服務設定

UNMARK 需要兩個憑證：Hugging Face Token 用於下載模型，Vast.ai API Key 用於搜尋、建立及銷毀 GPU 實例。

## Hugging Face

1. 註冊並登入 [Hugging Face](https://huggingface.co/)。
2. 開啟 [Access Tokens](https://huggingface.co/settings/tokens)，建立一個專供 UNMARK 使用的 `fine-grained` Token。
3. 只授予所需模型倉庫的讀取權限。UNMARK 不會上傳或修改模型，因此不需要寫入權限。
4. 如模型頁面要求同意使用條款或申請存取權，須先以同一個 Hugging Face 帳戶完成；gated model 的權限屬於個別用戶。
5. 把 Token 寫入專案根目錄的 `.env`：

```env
HF_TOKEN=你的_Hugging_Face_Token
```

模型倉庫及授權資料見 [模型預設總結](Models/model_presets_summary.md)。Hugging Face 建議每個應用程式使用獨立的 fine-grained Token，詳見官方 [User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens) 文件。

## Vast.ai

1. 註冊並登入 [Vast.ai Console](https://cloud.vast.ai/)，完成電郵驗證及儲值。
2. 開啟 [API Keys](https://cloud.vast.ai/manage-keys/?tab=api-keys)，建立一個名為 `UNMARK` 的獨立 Key。
3. 使用 scoped key，僅授予 UNMARK 所需的權限類別：`instance_read`、`instance_write` 及 `misc`。
4. 不要授予 `billing_write`、`user_write`、`machine_write` 或團隊管理權限。
5. 把 Key 寫入專案根目錄的 `.env`：

```env
VAST_API_KEY=你的_Vast_API_Key
```

Vast 的權限介面及分類可能調整；建立 Key 前可核對官方 [API Key](https://docs.vast.ai/guides/reference/api-keys) 及 [Permissions](https://docs.vast.ai/cli/permissions) 文件。

## 密鑰安全

- `.env` 已加入 `.gitignore`，不要改名後提交，也不要把密鑰貼到 Issue、截圖、日誌或命令列參數。
- 如懷疑密鑰外洩，立即在對應平台撤銷並重新建立。
- LocalWeb 設定頁臨時輸入的 Vast 及 Hugging Face 密鑰只保存在目前程序的記憶體，不會寫入設定 JSON。
- Vast 主機供應者在技術上可能接觸實例內的檔案；敏感圖片應使用可信任的資料中心及適當的額外保護。

## 收費

Vast 的運算、儲存及頻寬分開計費。運算按實例實際運行時間以秒計算，`Loading` 狀態不收 GPU 租金；模型載入、依賴安裝及檔案傳輸仍會增加整體時間或其他費用。這些固定開銷常被稱為「開機成本」，但並非固定收取一小時費用。

停止實例只會暫停 GPU 租金，儲存費仍會繼續；不再使用時必須銷毀實例。圖片數量太少時，固定開銷難以攤薄，因此本項目不建議用於個人少量處理。執行前請查看官方 [Billing](https://docs.vast.ai/guides/reference/billing) 及當前報價明細。
