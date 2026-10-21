import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import shutil

# ==============================================================================
# Part 0: 路徑設定
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.join(BASE_DIR, "..", "input", "muda_trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

# doc 跟 analysis 是平行資料夾
ANALYSIS_DIR = os.path.dirname(BASE_DIR)
PROJECT_DIR = os.path.dirname(ANALYSIS_DIR)
DOC_INPUT_DIR = os.path.join(PROJECT_DIR, "doc", "input")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOC_INPUT_DIR, exist_ok=True)

print(f"讀取檔案: {INPUT}")
print(f"輸出資料夾: {OUTPUT_DIR}")


# ==============================================================================
# Part 1: 讀取 MUDA 交易資料
# ==============================================================================

if not os.path.exists(INPUT):
    raise FileNotFoundError(
        "找不到 muda_trades_panel.csv。請先執行 build/code/gen_muda_trade_panel.py"
    )

df = pd.read_csv(INPUT)

print(f"原始 MUDA 交易資料共有 {df.shape[0]} 筆交易。")
print(f"欄位: {df.columns.tolist()}")


# ==============================================================================
# Part 2: 基本清理，只保留 MUDA 四回合
# ==============================================================================

required_cols = ["session_code", "round", "seconds", "price"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要欄位: {col}")

df["round"] = pd.to_numeric(df["round"], errors="coerce")
df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# 只保留 MUDA 1~4 回合
df = df[(df["round"] >= 1) & (df["round"] <= 4)].copy()

# 刪掉無法畫圖的資料
df = df.dropna(subset=["round", "seconds", "price"])

print(f"保留 MUDA 第 1~4 回合後，共有 {df.shape[0]} 筆交易。")
print(f"包含 session 數量: {df['session_code'].nunique()}")
print(f"包含回合: {sorted(df['round'].unique())}")


# ==============================================================================
# Part 3: 建立 price_diff
# ==============================================================================

# 如果資料有 eq_price，就用 price - eq_price
# 如果沒有 eq_price，但有 market_price，就用 price - market_price
# 如果兩個都沒有，就只畫 price，不畫 price_diff

if "eq_price" in df.columns:
    df["eq_price"] = pd.to_numeric(df["eq_price"], errors="coerce")
    df["price_diff"] = df["price"] - df["eq_price"]
    y_col = "price_diff"
    y_label = "price_diff = price - eq_price"
    print("使用 eq_price 計算 price_diff。")

elif "market_price" in df.columns:
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")
    df["price_diff"] = df["price"] - df["market_price"]
    y_col = "price_diff"
    y_label = "price_diff = price - market_price"
    print("找不到 eq_price，改用 market_price 計算 price_diff。")

else:
    y_col = "price"
    y_label = "trade price"
    print("找不到 eq_price 或 market_price，因此直接畫成交價格 price。")

plot_df = df[["session_code", "round", "seconds", y_col]].dropna().copy()


# ==============================================================================
# Part 4: 5 秒 binning
# ==============================================================================

def bin_data_by_5_seconds(data):
    """
    依照 round 和 seconds 每 5 秒分組，
    計算每個 bin 的平均價格差與樣本數。
    """
    binned_data = []

    for r in sorted(data["round"].unique()):
        g = data[data["round"] == r].copy()

        if g.empty:
            continue

        g["bin"] = ((g["seconds"] // 5) * 5).astype(int)

        bin_stats = (
            g.groupby("bin")
            .agg(
                y_mean=(y_col, "mean"),
                sample_count=(y_col, "count")
            )
            .reset_index()
        )

        bin_stats["round"] = r
        bin_stats["bin_center"] = bin_stats["bin"] + 2.5

        binned_data.append(bin_stats)

    if binned_data:
        return pd.concat(binned_data, ignore_index=True)

    return pd.DataFrame()


binned_all = bin_data_by_5_seconds(plot_df)

if binned_all.empty:
    raise ValueError("沒有足夠資料可以畫圖。")


# ==============================================================================
# Part 5: 畫 MUDA 全四回合圖
# ==============================================================================

round_colors = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
}

round_linestyles = {
    1: "solid",
    2: "dashed",
    3: "dotted",
    4: "dashdot",
}

plt.figure(figsize=(9, 6))

min_count = binned_all["sample_count"].min()
max_count = binned_all["sample_count"].max()
min_size = 20
max_size = 200

for r in [1, 2, 3, 4]:
    g = binned_all[binned_all["round"] == r]

    if g.empty:
        continue

    x = g["bin_center"].to_numpy()
    y = g["y_mean"].to_numpy()
    counts = g["sample_count"].to_numpy()

    if max_count > min_count:
        sizes = min_size + (max_size - min_size) * (counts - min_count) / (max_count - min_count)
    else:
        sizes = np.full_like(counts, min_size)

    color = round_colors.get(r, None)
    linestyle = round_linestyles.get(r, "solid")

    plt.scatter(
        x,
        y,
        alpha=0.65,
        s=sizes,
        c=[color],
        linewidths=0,
        label=f"round {int(r)}"
    )

    # 趨勢線
    if len(g) >= 2 and np.std(x) > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = a + b * xs
        plt.plot(xs, ys, c=color, linestyle=linestyle, linewidth=2)

plt.axhline(0, color="gray", linewidth=1, linestyle="--")

plt.xlabel("seconds (5-second bins)")
plt.ylabel(y_label)
plt.title("MUDA: seconds vs price difference, all 4 rounds")

legend_handles = []

for r in [1, 2, 3, 4]:
    if r in binned_all["round"].unique():
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=round_colors.get(r, None),
                linestyle=round_linestyles.get(r, "solid"),
                marker="o",
                label=f"round {r}",
                linewidth=2,
                markersize=8,
            )
        )

plt.legend(handles=legend_handles, title="MUDA round", loc="best")
plt.tight_layout()

outfile_all = os.path.join(OUTPUT_DIR, "muda_seconds_vs_price_diff_all4.png")
plt.savefig(outfile_all, dpi=200)
plt.close()

print(f"[OK] 已輸出全四回合圖: {outfile_all}")

try:
    doc_outfile_all = os.path.join(DOC_INPUT_DIR, "muda_seconds_vs_price_diff_all4.png")
    shutil.copy2(outfile_all, doc_outfile_all)
    print(f"[OK] 已複製到 doc/input: {doc_outfile_all}")
except Exception as e:
    print(f"Warning: Could not copy plot to doc/input: {e}")


# ==============================================================================
# Part 6: 各回合分開畫圖
# ==============================================================================

for r in [1, 2, 3, 4]:
    g = binned_all[binned_all["round"] == r]

    if g.empty:
        print(f"[跳過] round {r} 沒有交易資料。")
        continue

    plt.figure(figsize=(9, 6))

    x = g["bin_center"].to_numpy()
    y = g["y_mean"].to_numpy()
    counts = g["sample_count"].to_numpy()

    min_count_r = g["sample_count"].min()
    max_count_r = g["sample_count"].max()

    if max_count_r > min_count_r:
        sizes = min_size + (max_size - min_size) * (counts - min_count_r) / (max_count_r - min_count_r)
    else:
        sizes = np.full_like(counts, min_size)

    color = round_colors.get(r, None)

    plt.scatter(
        x,
        y,
        alpha=0.65,
        s=sizes,
        c=[color],
        linewidths=0
    )

    if len(g) >= 2 and np.std(x) > 0:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = a + b * xs
        plt.plot(xs, ys, c=color, linewidth=2)

    plt.axhline(0, color="gray", linewidth=1, linestyle="--")

    plt.xlabel("seconds (5-second bins)")
    plt.ylabel(y_label)
    plt.title(f"MUDA: seconds vs price difference, round {r}")

    plt.tight_layout()

    outfile_round = os.path.join(OUTPUT_DIR, f"muda_seconds_vs_price_diff_round_{r}.png")
    plt.savefig(outfile_round, dpi=200)
    plt.close()

    print(f"[OK] 已輸出 round {r} 圖: {outfile_round}")

    try:
        doc_outfile_round = os.path.join(DOC_INPUT_DIR, f"muda_seconds_vs_price_diff_round_{r}.png")
        shutil.copy2(outfile_round, doc_outfile_round)
    except Exception as e:
        print(f"Warning: Could not copy round {r} plot to doc/input: {e}")


# ==============================================================================
# Part 7: 輸出整理後的 MUDA 全四回合交易資料
# ==============================================================================

clean_outfile = os.path.join(OUTPUT_DIR, "muda_trades_panel_all4_clean.csv")
df.to_csv(clean_outfile, index=False, encoding="utf-8-sig")

analysis_input_outfile = os.path.join(BASE_DIR, "..", "input", "muda_trades_panel_all4_clean.csv")
df.to_csv(analysis_input_outfile, index=False, encoding="utf-8-sig")

print(f"[OK] 已輸出清理後資料: {clean_outfile}")
print(f"[OK] 已額外輸出到 analysis/input: {analysis_input_outfile}")

print("\nMUDA 全四回合交易分析完成！")