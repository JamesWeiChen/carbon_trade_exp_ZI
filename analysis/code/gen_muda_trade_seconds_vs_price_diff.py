import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# Part 0: 路徑設定
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.join(BASE_DIR, "..", "input", "muda_trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

ANALYSIS_DIR = os.path.dirname(BASE_DIR)
PROJECT_DIR = os.path.dirname(ANALYSIS_DIR)
DOC_INPUT_DIR = os.path.join(PROJECT_DIR, "doc", "input")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOC_INPUT_DIR, exist_ok=True)

print(f"讀取檔案: {INPUT}")
print(f"輸出資料夾: {OUTPUT_DIR}")


def copy_to_doc(path):
    doc_path = os.path.join(DOC_INPUT_DIR, os.path.basename(path))
    shutil.copy2(path, doc_path)
    print(f"[OK] 已複製到 doc/input: {doc_path}")


# ==============================================================================
# Part 1: 讀取 MUDA 交易資料
# ==============================================================================

if not os.path.exists(INPUT):
    raise FileNotFoundError(
        "找不到 muda_trades_panel.csv。請先執行 build/code/gen_muda_trade_panel.py"
    )

df = pd.read_csv(INPUT)

print(f"原始 MUDA 交易資料共有 {df.shape[0]} 筆交易。")


# ==============================================================================
# Part 2: 基本清理，保留所有 session 的 MUDA 交易
# ==============================================================================

required_cols = ["session_code", "round", "seconds", "price", "eq_price"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"缺少必要欄位: {col}。"
            f"請先確認 gen_muda_trade_panel.py 是否已正確輸出 MUDA 的 eq_price。"
        )

numeric_cols = [
    "round", "seconds", "price", "eq_price",
    "price_diff", "muda_eq_price_low", "muda_eq_price_high",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["session_code", "round", "seconds", "price", "eq_price"]).copy()

print(f"清理後共有 {df.shape[0]} 筆交易。")
print(f"包含 session 數量: {df['session_code'].nunique()}")
print(f"包含 session: {sorted(df['session_code'].astype(str).unique())}")
rounds = sorted(int(r) for r in df["round"].dropna().astype(int).unique())
print(f"包含 MUDA 回合: {rounds}")

if df.empty:
    raise ValueError("沒有足夠資料可以分析 MUDA 交易。")


# ==============================================================================
# Part 3: 建立 price_diff（強制使用 MUDA 的 eq_price）
# ==============================================================================

df["price_diff"] = df["price"] - df["eq_price"]

y_col = "price_diff"
y_label = "price_diff = price - MUDA eq_price"

plot_df = df[["session_code", "round", "seconds", y_col]].dropna().copy()

print(f"可用於畫圖的資料筆數: {plot_df.shape[0]}")

if plot_df.empty:
    raise ValueError("沒有足夠資料可以畫圖。")


# ==============================================================================
# Part 4: 全場次 pooled 5 秒 bin
# ==============================================================================

def bin_data_by_5_seconds_pooled(data):
    g = data.copy()

    if g.empty:
        return pd.DataFrame()

    g["bin"] = ((g["seconds"] // 5) * 5).astype(int)

    bin_stats = (
        g.groupby("bin")
        .agg(
            y_mean=(y_col, "mean"),
            sample_count=(y_col, "count"),
            session_count=("session_code", "nunique"),
        )
        .reset_index()
    )

    bin_stats["bin_center"] = bin_stats["bin"] + 2.5
    return bin_stats


binned_all = bin_data_by_5_seconds_pooled(plot_df)

if binned_all.empty:
    raise ValueError("5-second bin 後沒有資料可以畫圖。")

print(f"5 秒分組後共有 {binned_all.shape[0]} 個時間 bin。")


# ==============================================================================
# Part 5: 畫全場次 pooled 單一趨勢線圖
# ==============================================================================

plt.figure(figsize=(9, 6))

min_count = binned_all["sample_count"].min()
max_count = binned_all["sample_count"].max()
min_size = 20
max_size = 200

x = binned_all["bin_center"].to_numpy()
y = binned_all["y_mean"].to_numpy()
counts = binned_all["sample_count"].to_numpy()

if max_count > min_count:
    sizes = min_size + (max_size - min_size) * (counts - min_count) / (max_count - min_count)
else:
    sizes = np.full_like(counts, min_size, dtype=float)

plt.scatter(x, y, alpha=0.65, s=sizes, linewidths=0)

if len(binned_all) >= 2 and np.std(x) > 0:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = intercept + slope * xs
    plt.plot(xs, ys, linewidth=2)
    print(f"pooled 趨勢線：y = {intercept:.4f} + {slope:.4f} * x")
else:
    print("資料點不足，無法估計 pooled 趨勢線。")

plt.axhline(0, color="gray", linewidth=1, linestyle="--")

plt.xlabel("seconds (5-second bins)")
plt.ylabel(y_label)
plt.title("MUDA: seconds vs price_diff, pooled across all sessions")

plt.tight_layout()

outfile_all = os.path.join(OUTPUT_DIR, "muda_seconds_vs_price_diff_pooled_all_sessions.png")
plt.savefig(outfile_all, dpi=200)
plt.close()

print(f"[OK] 已輸出全場次 pooled 圖: {outfile_all}")
copy_to_doc(outfile_all)


# ==============================================================================
# Part 6: 輸出輔助檔案
# ==============================================================================

clean_outfile = os.path.join(OUTPUT_DIR, "muda_trades_panel_all_sessions_clean.csv")
df.to_csv(clean_outfile, index=False, encoding="utf-8-sig")
print(f"[OK] 已輸出清理後資料: {clean_outfile}")
copy_to_doc(clean_outfile)

analysis_input_outfile = os.path.join(
    BASE_DIR, "..", "input", "muda_trades_panel_all_sessions_clean.csv"
)
df.to_csv(analysis_input_outfile, index=False, encoding="utf-8-sig")
print(f"[OK] 已額外輸出到 analysis/input: {analysis_input_outfile}")

binned_outfile = os.path.join(
    OUTPUT_DIR, "muda_seconds_vs_price_diff_binned_pooled_all_sessions.csv"
)
binned_all.to_csv(binned_outfile, index=False, encoding="utf-8-sig")
print(f"[OK] 已輸出 pooled bin 資料: {binned_outfile}")
copy_to_doc(binned_outfile)

eq_cols = ["session_code", "round", "eq_price"]
optional_eq_cols = ["muda_eq_price_low", "muda_eq_price_high", "muda_eq_method"]

existing_eq_cols = eq_cols + [c for c in optional_eq_cols if c in df.columns]

eq_check = (
    df[existing_eq_cols]
    .drop_duplicates()
    .sort_values(["session_code", "round"])
    .reset_index(drop=True)
)

eq_check_outfile = os.path.join(OUTPUT_DIR, "muda_eq_price_by_session_round.csv")
eq_check.to_csv(eq_check_outfile, index=False, encoding="utf-8-sig")
print(f"[OK] 已輸出各 session-round 的均衡價格表: {eq_check_outfile}")
copy_to_doc(eq_check_outfile)


print("\nMUDA 全場次 pooled 交易分析完成！")
