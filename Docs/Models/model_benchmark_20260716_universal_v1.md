# 模型評測：model_benchmark_20260716_universal_v1

## 測試資料

- 狀態：`done`
- GPU：`H200`
- Vast 實例：`44998942`
- 重用既有實例：`是`
- 實際單價：`$4.1338` / 小時
- 建立前報價：`$4.1338` / 小時
- 測試圖片：`7` 張
- Seed：`42`
- 提示詞：`萬能提示詞`
- 開始時間：`2026-07-16 13:02:25`
- 報告時間：`2026-07-16 13:13:38`
- 累計實例時間：`670.1` 秒
- 截至報告時間預估費用：`$0.769468`
- 實例是否保留：`否`

每組 `wall_seconds` 包括模型載入、7 張推理及結果下載；`image_seconds_average` 只計成功圖片的逐張清洗時間。
費用是按實例單價及本次端到端時間估算，不包括另行結算的儲存及頻寬費。實例保留期間仍會繼續計費。

## 組別摘要

| 輸出目錄 | 狀態 | 成功 | 失敗 | 模型載入秒數 | 單張平均秒數 | 組別總秒數 | 組別預估費用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `9b_8steps` | done | 7 | 0 | 14.6490 | 3.3690 | 55.6541 | $0.063906 |
| `9b_16steps` | done | 7 | 0 | 13.7108 | 5.2092 | 63.3085 | $0.072695 |
| `9b_32steps` | done | 7 | 0 | 14.0144 | 8.9568 | 127.2626 | $0.146132 |
| `kwr_8steps` | done | 7 | 0 | 15.5513 | 5.9252 | 83.2326 | $0.095574 |
| `kwr_16steps` | done | 7 | 0 | 15.8826 | 10.0140 | 95.5555 | $0.109723 |
| `kwr_32steps` | done | 7 | 0 | 15.1038 | 18.3919 | 161.1482 | $0.185042 |
| `4b_16steps` | done | 7 | 0 | 9.0205 | 3.5864 | 58.7212 | $0.067428 |

七組合共產生 49 張成功圖片，整次評測預估運算費為 `$0.769468`，即本次測試平均約 `$0.015703` / 張。這個平均數包括七次模型切換及載入，不能當作只跑一個模型的大批量單張報價。

若只按逐圖推理時間線性推算 1,000 張圖片，9B 8 steps 約 `$3.869`、9B 16 steps 約 `$5.982`、9B 32 steps 約 `$10.285`、KWR 8 steps 約 `$6.804`、KWR 16 steps 約 `$11.499`、KWR 32 steps 約 `$21.119`、4B 16 steps 約 `$4.118`。以上只屬同一台 H200、同類尺寸圖片的理論推算，未計模型載入、建立環境、上下載、儲存及頻寬費。

## 抽樣目測

本節是對 7 張測試圖的人工目測，不是有乾淨原圖作基準的客觀分數。

| 場景 | 目測結果 |
|---|---|
| 鋪滿 `sample` 文字的花田相片 | 全部組別都清走文字。9B 三組最接近原構圖；KWR 明顯重畫花田視角；4B 亦有局部重畫。 |
| PicMonkey 山景相片 | 9B 及 4B 清走中央文字及大部分 stock 水印，但右下角 `Remove watermark` 仍在。KWR 清得較完整，但移除外框並重畫整張相片。 |
| 手繪圖上的粗黑亂畫 | **七組全部不合格。**9B 三組都留下由左上至右下的粗線；KWR 雖清走黑線，卻刪除或重畫人物手臂及原畫線條；4B 亦改動手臂和線稿。這張不能列作成功案例。 |
| 軟件介面亂畫 | 9B 及 4B 清走亂畫但連正常標題也刪除；KWR 對介面排版及文字改動較大。 |
| QR code 桌面截圖 | 全部組別都能清走 QR code。9B 及 4B 會誤刪左上角正常圖示；KWR 對桌面文字、視窗邊界及構圖改動較多。 |
| 樓宇標誌相片 | 9B 把目標標誌連同真實外牆標誌一併刪除；KWR 保留真實外牆標誌，較適合配合「樓盤 Logo」提示詞再測。 |

## 建議

- 日常批次先用 **9B 8 steps**。在本次樣本中，16 及 32 steps 未帶來穩定可見的改善，但逐圖平均耗時分別增至約 1.55 倍及 2.66 倍。
- 對 9B 未清乾淨的圖片，先改用對應提示詞；仍失敗才以 **KWR 8 steps** 補處理。KWR 32 steps 成本最高，而且本次目測沒有足以抵銷重畫風險的穩定收益。
- **4B 16 steps** 的速度及理論成本接近 9B 8 steps，可作低成本候選；但它在手繪圖上同樣改動手臂和線稿，不能當作該案例的解法。
- 「萬能提示詞」只適合首輪分流，不能保證分辨所有正常文字、Logo、圖示及水印。涉及樓盤標誌、QR code 或亂畫時，應改用對應提示詞並抽查首批結果。

表格內的 `ok` 只代表 worker 成功產生及下載檔案，不代表人工目測合格。`ari_artdrawing` 的七個輸出雖然技術狀態全部是 `ok`，品質判定仍是全部失敗。

## 每張圖片耗時

| 組別 | 圖片 | 狀態 | 秒數 | 錯誤 |
|---|---|---:|---:|---|
| 9b_8steps | ImageDoodles.png | ok | 3.1399 |  |
| 9b_8steps | Watermarked_Image_in_PicMonkey.jpg | ok | 2.2919 |  |
| 9b_8steps | ari.artdrawing.jpg | ok | 2.5852 |  |
| 9b_8steps | cos-centre_01.png | ok | 6.9972 |  |
| 9b_8steps | cos-centre_02.png | ok | 5.0738 |  |
| 9b_8steps | tiled_text_watermark.jpeg | ok | 1.0228 |  |
| 9b_8steps | watermarking-result.png | ok | 2.4724 |  |
| 9b_16steps | ImageDoodles.png | ok | 4.7234 |  |
| 9b_16steps | Watermarked_Image_in_PicMonkey.jpg | ok | 4.4076 |  |
| 9b_16steps | ari.artdrawing.jpg | ok | 4.9423 |  |
| 9b_16steps | cos-centre_01.png | ok | 9.3845 |  |
| 9b_16steps | cos-centre_02.png | ok | 6.8231 |  |
| 9b_16steps | tiled_text_watermark.jpeg | ok | 1.9565 |  |
| 9b_16steps | watermarking-result.png | ok | 4.2268 |  |
| 9b_32steps | ImageDoodles.png | ok | 8.4566 |  |
| 9b_32steps | Watermarked_Image_in_PicMonkey.jpg | ok | 8.5852 |  |
| 9b_32steps | ari.artdrawing.jpg | ok | 9.6563 |  |
| 9b_32steps | cos-centre_01.png | ok | 14.1748 |  |
| 9b_32steps | cos-centre_02.png | ok | 10.3236 |  |
| 9b_32steps | tiled_text_watermark.jpeg | ok | 3.8062 |  |
| 9b_32steps | watermarking-result.png | ok | 7.6948 |  |
| kwr_8steps | ImageDoodles.png | ok | 6.0167 |  |
| kwr_8steps | Watermarked_Image_in_PicMonkey.jpg | ok | 4.4493 |  |
| kwr_8steps | ari.artdrawing.jpg | ok | 4.4983 |  |
| kwr_8steps | cos-centre_01.png | ok | 8.5349 |  |
| kwr_8steps | cos-centre_02.png | ok | 8.0603 |  |
| kwr_8steps | tiled_text_watermark.jpeg | ok | 4.5537 |  |
| kwr_8steps | watermarking-result.png | ok | 5.3634 |  |
| kwr_16steps | ImageDoodles.png | ok | 9.8594 |  |
| kwr_16steps | Watermarked_Image_in_PicMonkey.jpg | ok | 8.5822 |  |
| kwr_16steps | ari.artdrawing.jpg | ok | 8.8564 |  |
| kwr_16steps | cos-centre_01.png | ok | 12.8799 |  |
| kwr_16steps | cos-centre_02.png | ok | 11.649 |  |
| kwr_16steps | tiled_text_watermark.jpeg | ok | 8.7627 |  |
| kwr_16steps | watermarking-result.png | ok | 9.5082 |  |
| kwr_32steps | ImageDoodles.png | ok | 18.0368 |  |
| kwr_32steps | Watermarked_Image_in_PicMonkey.jpg | ok | 16.8686 |  |
| kwr_32steps | ari.artdrawing.jpg | ok | 17.0332 |  |
| kwr_32steps | cos-centre_01.png | ok | 21.2725 |  |
| kwr_32steps | cos-centre_02.png | ok | 19.9713 |  |
| kwr_32steps | tiled_text_watermark.jpeg | ok | 17.3247 |  |
| kwr_32steps | watermarking-result.png | ok | 18.2359 |  |
| 4b_16steps | ImageDoodles.png | ok | 3.1393 |  |
| 4b_16steps | Watermarked_Image_in_PicMonkey.jpg | ok | 2.3638 |  |
| 4b_16steps | ari.artdrawing.jpg | ok | 2.6956 |  |
| 4b_16steps | cos-centre_01.png | ok | 6.7659 |  |
| 4b_16steps | cos-centre_02.png | ok | 6.4635 |  |
| 4b_16steps | tiled_text_watermark.jpeg | ok | 1.0777 |  |
| 4b_16steps | watermarking-result.png | ok | 2.5987 |  |

完整組別數據：[摘要 CSV](model_benchmark_20260716_universal_v1_summary.csv)

完整逐圖數據：[逐圖 CSV](model_benchmark_20260716_universal_v1_images.csv)
