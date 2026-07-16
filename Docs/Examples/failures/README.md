# 已知失敗案例

遠端 worker 顯示 `ok`，只代表成功產生及下載圖片，不代表去水印品質合格。以下結果已排除在展示案例之外。

## 線稿上的粗黑筆

`9B`、8 steps 配合萬能提示詞時，粗黑筆未有完整清除。改用 `線稿粗黑筆` 提示詞後的採用結果放在 [results/線稿粗黑筆](../results/線稿粗黑筆/9b_8steps/ari_artdrawing.jpg)。

- [原圖](../source_images/ari_artdrawing.jpg)
- [未採用結果](ari_artdrawing_universal_prompt_failed.jpg)

## 樓盤相片的真假標誌

模型移除水印時亦改動了真實外牆標誌，屬誤刪，不能採用。

- [原圖](../source_images/property_logo.jpg)
- [未採用結果](property_logo_false_positive.png)
