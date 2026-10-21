
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# 路徑
CSV_PATH  = (Path(__file__).resolve().parents[2] / "analysis" / "input" / "clean_2025-08-29_trades_long.csv")
META_PATH = (Path(__file__).resolve().parents[2] / "analysis" / "input" / "clean_2025-08-29.csv")
OUT_DIR   = (Path(__file__).resolve().parents[2] / "analysis" / "output")
OUT_BASENAME = "price2_CarbonTrading"

# 讀取檔案
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

meta = pd.read_csv(META_PATH, encoding="utf-8-sig")
need_cols = ["session_code", "round", "carbon_multiplier"]
meta = meta[need_cols].copy()
meta["round"] = pd.to_numeric(meta["round"], errors="coerce").astype("Int64")
meta["carbon_multiplier"] = pd.to_numeric(meta["carbon_multiplier"], errors="coerce")
meta = meta.dropna(subset=["round", "carbon_multiplier"]).drop_duplicates(subset=["session_code", "round"])

df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
df = df.merge(meta, on=["session_code", "round"], how="left")

# 新增 price_ 欄位 = 6 * carbon_multiplier（理論價格）
df["price_"] = 6 * df["carbon_multiplier"]

# 必要欄位
time_col  = "seconds"
price_col = "price"
round_col = "round"

work = df[[time_col, price_col, round_col, "price_"]].copy()
work[time_col]   = pd.to_numeric(work[time_col], errors="coerce")
work[price_col]  = pd.to_numeric(work[price_col], errors="coerce")
work[round_col]  = pd.to_numeric(work[round_col], errors="coerce").astype("Int64")
work["price_"]   = pd.to_numeric(work["price_"], errors="coerce")
work = work.dropna(subset=[time_col, price_col, "price_"])

# 理論價格固定三種
theory_prices = sorted(work["price_"].dropna().unique().tolist())
assert set(theory_prices) == {3, 6, 9}, f"理論價格不是 {3,6,9}：{theory_prices}"

# 固定顏色與形狀對應
color_map  = {3: 'tab:orange', 6: 'tab:blue', 9: 'tab:green'}
marker_map = {3: 'o', 6: 's', 9: '^'}
fixed_order = [3, 6, 9]

# 畫單張疊圖（散點）：依理論價格著色+形狀
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(10, 6), dpi=150)

# 逐回合畫散點
for r in sorted(work[round_col].unique()):
    rr = work[work[round_col] == r]
    if rr.empty:
        continue
    tp = rr["price_"].iloc[0]  # 該回合的理論價
    y = rr[price_col] - rr["price_"]
    plt.scatter(rr[time_col], y, s=30, c=color_map[tp], marker=marker_map[tp], alpha=0.8)

# 各理論價的線性回歸線（同色）
x_line = np.linspace(0, 120, 200)
for tp in fixed_order:
    sub = work[work["price_"] == tp]
    x = sub[time_col].to_numpy()
    y = (sub[price_col] - sub["price_"]).to_numpy()
    # 一階多項式擬合：y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    y_hat = a * x_line + b
    plt.plot(x_line, y_hat, color=color_map[tp], linewidth=2)

plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("time (second)")
plt.xlim(0, 120)
plt.xticks(list(range(0, 121, 30)))

# 固定 legend 順序：3 → 6 → 9
legend_handles = [
    Line2D([0], [0], marker=marker_map[tp], linestyle='-', linewidth=2, markersize=6,
           color=color_map[tp], label=f"Theory price = {tp}")
    for tp in fixed_order
]
plt.legend(handles=legend_handles, ncol=1, fontsize=10, frameon=False, title="Categories by theory price")

plt.tight_layout()

out_path = OUT_DIR / f"{OUT_BASENAME}_fold.png"
plt.savefig(out_path)
plt.close()

print("輸出完成：", out_path)
