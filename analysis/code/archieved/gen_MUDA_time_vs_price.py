#生出MUDA回合交易圖片
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import math


CSV_PATH = (Path(__file__).resolve().parents[2] / "analysis" / "input" / "clean_2025-08-29_trades_long_MUDA.csv")
OUT_DIR  = (Path(__file__).resolve().parents[2] / "analysis" / "output")
OUT_BASENAME = "trade_price_MUDA"  

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 必要欄位
time_col  = "seconds" 
price_col = "price"
round_col = "round"

work = df[[time_col, price_col, round_col]].copy()
work[time_col]  = pd.to_numeric(work[time_col], errors="coerce")
work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
work[round_col] = pd.to_numeric(work[round_col], errors="coerce").astype("Int64")


rounds = sorted([int(r) for r in work[round_col].dropna().unique()])

ROWS, COLS = 2, 2
PER_PAGE = ROWS * COLS
num_pages = math.ceil(len(rounds) / PER_PAGE)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_files = []

for page in range(num_pages):
    start = page * PER_PAGE
    end   = min((page + 1) * PER_PAGE, len(rounds))
    page_rounds = rounds[start:end]

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 4.0, ROWS * 3.2), dpi=150, sharex=True, sharey=True)
    axes = np.array(axes).reshape(ROWS, COLS)

    x_min, x_max = 0, 120

    page_df = work[work[round_col].isin(page_rounds)]
    y_min = np.nanmin(page_df[price_col].values) if not page_df.empty else 0
    y_max = np.nanmax(page_df[price_col].values) if not page_df.empty else 1
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 1
    y_min -= pad
    y_max += pad

    for idx, r in enumerate(page_rounds):
        rr = work[work[round_col] == r]
        ax = axes[idx // COLS, idx % COLS]
        ax.scatter(rr[time_col], rr[price_col], s=15)   # 一張圖只有一個 round
        ax.set_title(f"Round {r} ", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if idx // COLS == ROWS - 1:
            ax.set_xlabel("time（second）", fontsize=10)
        if idx % COLS == 0:
            ax.set_ylabel("trade_price", fontsize=10)

    fig.suptitle("Time vs. Trade Price (MUDA)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(out_path)
    plt.close(fig)
    out_files.append(str(out_path))

print("輸出完成：")
for f in out_files:
    print(" -", f)

