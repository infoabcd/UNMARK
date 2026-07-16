# 模型資料

| 檔案 | 內容 |
|------|------|
| [model_presets_summary.md](model_presets_summary.md) | 四個模型快捷鍵、來源及預設參數。 |
| [FLUX2_9B_成本速度對比匯報.pdf](FLUX2_9B_成本速度對比匯報.pdf) | 既有 FLUX.2 Klein 9B 成本與速度測試。 |
| [klein_4b_watermark_workflow.md](klein_4b_watermark_workflow.md) | 既有 FLUX.2 Klein 4B 工作流測試。 |
| [model_benchmark_20260716_universal_v1.md](model_benchmark_20260716_universal_v1.md) | H200 實測：9B、KWR、4B 共七組耗時、費用及目測結論。 |
| [model_benchmark_20260716_universal_v1_summary.csv](model_benchmark_20260716_universal_v1_summary.csv) | 七組測試摘要，可供試算表分析。 |
| [model_benchmark_20260716_universal_v1_images.csv](model_benchmark_20260716_universal_v1_images.csv) | 49 張輸出的逐圖耗時。 |
| [prompt_tests_20260716.md](prompt_tests_20260716.md) | 9B 8 steps 專用提示詞實測、採用結果、失敗案例及實際成本。 |

## 統一評測

先把測試圖片放入 `input_images/`。以下命令會在同一台 Vast 實例依次測試指定組別：

```bash
python3 run_unmark.py benchmark-models \
  --configurations 9B:8,9B:16,9B:32,KWR:8,KWR:16,KWR:32,4B:16
```

如不設定 `--configurations`，程式才會按照 `--models` × `--steps-list` 產生完整交叉組合。四個模型的 Hugging Face 檔案合計超過 130 GiB；評測命令預設使用 200 GB 磁碟，Vast 搜尋亦會排除磁碟不足的報價。

輸出圖片放在 `output_images/model_tests/<模型>_<步數>steps/`，例如 `output_images/model_tests/9b_8steps/`。摘要、逐圖耗時及預估費用會寫入本目錄的 Markdown 及 CSV 報告。

評測報告會記錄提示詞檔名、提示詞 SHA-256 及 seed。模型、steps、提示詞或 seed 不一致的結果不可直接比較。

評測成功、失敗或逾時後預設會補下載可用結果並銷毀實例。只有確定需要立即繼續測試時才可加入 `--keep-instance`；停止但未銷毀的 Vast 實例仍會收儲存費。

要重用仍存在的 benchmark 實例及模型快取：

```bash
python3 run_unmark.py benchmark-models \
  --reuse-job-dir jobs/<舊-benchmark-job-id> \
  --configurations 9B:8,KWR:8
```

重用評測完成後亦會預設銷毀該實例。

只檢查測試矩陣而不建立實例：

```bash
python3 run_unmark.py benchmark-models --dry-run
```
