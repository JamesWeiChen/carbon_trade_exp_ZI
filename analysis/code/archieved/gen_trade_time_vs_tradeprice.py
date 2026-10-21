#生出Trade回合交易圖片

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import math

# 路徑設定
CSV_PATH = (Path(__file__).resolve().parents[2] / "analysis" / "input" / "clean_2025-08-29_trades_long.csv")
OUT_DIR  = (Path(__file__).resolve().parents[2] / "analysis" / "output")
OUT_BASENAME = "trade_price_CarbonTrading"

# 讀取檔案
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 必要欄位
time_col  = "seconds"
price_col = "price"
round_col = "round"

work = df[[time_col, price_col, round_col]].copy()
work[time_col]  = pd.to_numeric(work[time_col], errors="coerce")
work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
work[round_col] = pd.to_numeric(work[round_col], errors="coerce").astype("Int64")
work = work.dropna(subset=[time_col, price_col])

rounds = sorted([int(r) for r in work[round_col].dropna().unique()])

# 回合顯示格式:3x4
ROWS, COLS = 3, 4
PER_PAGE = ROWS * COLS
num_pages = math.ceil(len(rounds) / PER_PAGE)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_files = []

xticks = list(range(0, 121, 30))  # 每30秒一個刻度

for page in range(num_pages):
    start = page * PER_PAGE
    end   = min((page + 1) * PER_PAGE, len(rounds))
    page_rounds = rounds[start:end]

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 4.0, ROWS * 3.2), dpi=150)
    axes = np.array(axes).reshape(ROWS, COLS)

    page_df = work[work[round_col].isin(page_rounds)]
    y_min = np.nanmin(page_df[price_col].values) if not page_df.empty else 0
    y_max = np.nanmax(page_df[price_col].values) if not page_df.empty else 1
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1
    y_min -= pad
    y_max += pad

    for idx, r in enumerate(page_rounds):
        rr = work[work[round_col] == r]
        ax = axes[idx // COLS, idx % COLS]
        ax.scatter(rr[time_col], rr[price_col], s=15)
        ax.set_title(f"round {r} ", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(0, 120)
        ax.set_xticks(xticks)                   # X 軸標示 0,30,60,90,120
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("time（second）", fontsize=9)
        ax.set_ylabel("Trade Price", fontsize=9)

    fig.suptitle("Time vs. Trade Price (Carbon Trade Group)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(out_path)
    plt.close(fig)
    out_files.append(str(out_path))

print("輸出完成：")
for f in out_files:
    print(" -", f)
