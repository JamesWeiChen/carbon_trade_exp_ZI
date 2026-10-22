# 納入回購與轉售後的結果比較

**結果有明顯變化。** 真人的買方平均相關係數由負轉正，賣方平均相關係數往零移動。價格動態沒有改變。這是觀察值納入規則改變造成的結果，不能單獨視為新的行為模型檢定。

本分支只把真人交易改為與 ZI 一樣：每次買進與賣出都計入，包括回購、轉售和零邊際價值單位。價值／機會成本仍取當下持有量對應的邊際單位；不是當初買進的價格。兩張碳權的平均價值假設、ZI 機制與種子、價格統計、至少四包交易門檻、回合1–6、包內單位相同秩次，以及繪圖方法都保留。

## 直接看圖

- [逐筆交易：左舊右新](before_after_package_standardized.png)
- [逐張碳權：左舊右新](before_after_unit_standardized.png)
- [最終逐筆交易標準化圖](../figure2_zi_simultaneous_two_sided_resale/figure_2abcd_package_standardized_price_change.png)
- [最終逐筆交易原始價差圖](../figure2_zi_simultaneous_two_sided_resale/figure_2abcd_package_raw_price_change.png)
- [最終逐張碳權標準化圖](../figure2_zi_simultaneous_two_sided_resale/figure_2abcd_unit_standardized_price_change.png)
- [最終逐張碳權原始價差圖](../figure2_zi_simultaneous_two_sided_resale/figure_2abcd_unit_raw_price_change.png)

## 變化多大

48場真人市場、705包交易、988張碳權的基礎樣本不變。買方有效單位由 **793增至988（增加195）**，賣方由 **803增至988（增加185）**。

下表是圖中共同完整樣本的市場相關係數平均，也就是 panel d 的 CTAT 紅色菱形座標。

| 指標 | 舊圖47場 | 新圖48場 | 固定原47場的新值 |
|---|---:|---:|---:|
| 逐包買方 rho | -0.105908 | 0.114844 | 0.106315 |
| 逐包賣方 rho | 0.192656 | 0.023560 | 0.021031 |
| 逐張買方 rho | -0.103836 | 0.090969 | 0.082691 |
| 逐張賣方 rho | 0.167715 | 0.021242 | 0.018941 |

固定同一批47場後，變化仍存在，因此不是單純新增一場造成。逐包買方41/47個原有限rho改變，另1場由未定義變為有限；逐包賣方41/48場改變。逐張買方41/47場改變，另1場變為有限；逐張賣方44/48場改變。沒有有限rho變為未定義。

Panels a、b、d 的共同完整樣本由47場變48場：新增 `mz6ar9u6` 第4回合，沒有退出。早期組仍24場，晚期由23增至24場。`rho_change_summary.csv` 是各指標本身可用樣本的比較；`plotted_mean_comparison.csv` 另外使用圖的三指標共同完整樣本，兩者分母不可混用。

## 哪些沒有變

- 交易價格、數量、順序、持有量、單位ordinal及原始價值表不變。
- 各市場的所有價格rho、相鄰價差對、零價差、panel c OLS與顯示篩選不變。
- 四張圖 panel c 的右上區域含標籤與色條，前後像素完全一致。原價格迴歸CSV也維持原始位元組。
- Panel d 的價格座標平均會因共同完整樣本新增一場而稍移動：逐包 -0.440488→-0.449386，逐張 -0.292764→-0.302655。固定原47場時，價格座標完全不變。
- ZI橢圓使用同一批原有10,000場結果；MA/AN理論參考、全部繪圖函式均未改。
- 早晚期KS檢定仍沿用原方法；本次四個買賣／觀察層級的p值均大於0.05。這不是對重複受試者或相依性的另行校正。

## 怎麼確認沒有改錯

1. 五項迴歸測試通過：初始3張後「賣3→2、買2→3、買3→4」；「買3→4、賣4→3、賣3→2」；跨越初始持有量的兩張交易；兩張碳權平均；零邊際價值仍納入。
2. 另寫獨立計算，從原始成本／擾動逐項累加利潤、重建生產frontier與平均價值、逐單位更新持有量，再以平均秩次加Pearson重算Spearman。全988個單位、705包平均／總和及48場四種買賣rho都相符。
3. 舊版程式從固定提交 `b06cfb712817fae45e4e14a965a28f520a14204c` 載入，先成功重現該提交的既有真人CSV，再用同一環境重畫舊圖，與新圖公平比較。
4. AST核對確認只修改 `reconstruct_one_market`、驗證報告文字與main的檢查提示；ZI、價值函式、抽樣、種子、統計與繪圖函式未改。
5. 三個輸入CSV與七個ZI稽核／結果CSV均與基準提交逐位元組相同，SHA-256收錄在 `verification.json`。驗證全部10,000筆抽樣市場索引符合原種子，另用原模型重跑ID 1、2、3作有10秒上限的來源抽查，回傳摘要與既有結果相符；這三場沒有替換任何儲存結果。`valid_for_95_region`確認為實際boolean欄位。

**本次沒有重跑10,000場ZI。** 也未加入加速原型、修改README／runner、開PR或合併主分支。

## 重現

在 repository 根目錄，使用安裝 NumPy、pandas、SciPy、statsmodels、Matplotlib、Pillow 的Python環境：

```sh
MPLBACKEND=Agg python -m unittest discover -s analysis/tests -v
MPLBACKEND=Agg python analysis/code/compare_empirical_inclusion.py
```

比較腳本需要本地git保有基準提交。它會重建新舊圖及比較表，但不呼叫10,000場模擬入口、不替換ZI資料。環境版本與檢查結果在 `verification.json`；逐市場差異在 `market_comparison.csv`；早晚統計在 `early_late_comparison.csv`；完整樣本進出在 `plot_sample_membership.csv`。四張最終PNG及兩張並排圖已人工檢視，無截字、缺圖或圖例遮擋。
