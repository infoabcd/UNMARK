# 提示詞實測：2026-07-16

## 測試資料

- 模型：9B，8 steps，seed 42
- GPU：H200
- Vast 實例：`45065283`
- 實際單價：`$3.736491` / 小時
- 實例存活：3,008 秒（50 分 08 秒）
- 預估運算費：`$3.122046`
- 狀態：已銷毀

費用按實例由建立至銷毀的完整時間估算，已包括檢查圖片及修改提示詞期間；未計另行結算的儲存及頻寬費。其後建立第二台實例時，Vast 因帳戶餘額不足而拒絕請求，沒有建立實例或產生 GPU 費用。

## 最終結果

| 提示詞 | 圖片 | 推理時間 | 目測 | 結果 |
|---|---|---:|---|---|
| `線稿粗黑筆` | `ari.artdrawing.jpg` | 3.4136 秒 | 粗黑筆已清走，人物雙臂、鉛筆線稿及右側實體筆保留 | [圖片](../Examples/results/線稿粗黑筆/9b_8steps/ari_artdrawing.jpg) |
| `亂塗亂畫` | `ImageDoodles.png` | 3.2065 秒 | 塗鴉已清走，時間、標題、按鈕、色帶及主圖保留 | [圖片](../Examples/results/亂塗亂畫/9b_8steps/ImageDoodles.png) |
| `二維碼` | `watermarking-result.png` | 3.3035 秒 | QR code 已清走，左上角 Recycle Bin、視窗及工作列保留 | [圖片](../Examples/results/二維碼/9b_8steps/watermarking-result.png) |
| `樓盤Logo` | `cos-centre_02.png` | 5.9678 秒 | 半透明印記已清走，門牌、玻璃導覽字、店舖及人物保留 | [圖片](../Examples/results/樓盤Logo/9b_8steps/cos-centre_02.png) |

## 未能採用

- `cos-centre_01.png`：9B 在不同提示詞版本中分別誤刪真實招牌、生成假地址／亂碼，或大幅刪除建築內容。只靠文字提示未能得到可靠結果，代表性輸出放在 [失敗案例](../Examples/failures/README.md)。
- `Watermarked_Image_in_PicMonkey.jpg`：準備測試「密集文字水印」時，Vast 因餘額不足拒絕建立實例，因此沒有新結果；既有萬能提示詞結果仍會漏掉右下角標籤。

worker 的 `ok` 只代表檔案成功生成及下載；本文件的「目測」才是本輪是否採用的判定。
