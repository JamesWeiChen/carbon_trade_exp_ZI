import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import shutil

# 參數與路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "input", "trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
# doc 跟 analysis 是平行資料夾
ANALYSIS_DIR = os.path.dirname(BASE_DIR)  # 從 analysis/code 到 analysis
PROJECT_DIR = os.path.dirname(ANALYSIS_DIR)  # 從 analysis 到上一層
DOC_INPUT_DIR = os.path.join(PROJECT_DIR, "doc", "input")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOC_INPUT_DIR, exist_ok=True)

# 讀資料
if not os.path.exists(INPUT):
    raise FileNotFoundError("trades_panel.csv not found in input directory.")
df = pd.read_csv(INPUT)
df["price_diff"] = df["price"] - df["eq_price"]


# 顏色: is_equal+eq_price, 形狀: eq_price, 線型: eq_price
# is_equal=0: 冷色系（淺綠、藍、深藍），is_equal=1: 暖色系（淺橘、橘、深紅）
color_map = {
    (0, 3): "#9ACD32",   # 淺綠
    (0, 6): "#008B8B",   # 藍
    (0, 9): "#006400",   # 深綠
    (1, 3): "#FD8D3C",   # 淺橘
    (1, 6): "#FF6F61",   # 橘
    (1, 9): "#B30000",   # 深紅
}
eq_price_markers = {3: "o", 6: "o", 9: "o"}
eq_price_linestyles = {3: "dotted", 6: "dashed", 9: "solid"}
levels = [3, 6, 9]

plot_df = df[["seconds", "price_diff", "eq_price", "is_equal", "round", "round_raw"]].dropna()

# 創建5秒binning函數
def bin_data_by_5_seconds(data):
    """將數據按5秒為單位進行分組，計算每個bin的平均price_diff和樣本數量"""
    binned_data = []
    
    for ie in [0, 1]:
        for lvl in levels:
            g = data[(data["is_equal"] == ie) & (data["eq_price"] == lvl)]
            if g.empty:
                continue
            
            # 創建5秒bins
            min_sec = int(g["seconds"].min() // 5) * 5
            max_sec = int(g["seconds"].max() // 5 + 1) * 5
            bins = np.arange(min_sec, max_sec + 5, 5)
            
            # 為每個數據點分配bin
            g = g.copy()
            g["bin"] = ((g["seconds"] // 5) * 5).astype(int)
            
            # 計算每個bin的統計數據
            bin_stats = g.groupby("bin").agg({
                "price_diff": "mean",
                "seconds": "count"  # 樣本數量
            }).reset_index()
            bin_stats.rename(columns={"seconds": "sample_count"}, inplace=True)
            
            # 添加分組信息
            bin_stats["is_equal"] = ie
            bin_stats["eq_price"] = lvl
            bin_stats["bin_center"] = bin_stats["bin"] + 2.5  # bin中心點
            
            binned_data.append(bin_stats)
    
    if binned_data:
        return pd.concat(binned_data, ignore_index=True)
    else:
        return pd.DataFrame()

def plot_binned_scatter(data, title, filename):
    binned = bin_data_by_5_seconds(data)
    plt.figure(figsize=(9, 6))

    if not binned.empty:
        min_count = binned["sample_count"].min()
        max_count = binned["sample_count"].max()
        min_size = 20
        max_size = 200

        for ie in [1, 0]:
            for lvl in levels:
                g = binned[(binned["is_equal"] == ie) & (binned["eq_price"] == lvl)]
                if g.empty:
                    continue

                x = g["bin_center"].to_numpy()
                y = g["price_diff"].to_numpy()
                counts = g["sample_count"].to_numpy()

                if max_count > min_count:
                    sizes = min_size + (max_size - min_size) * (counts - min_count) / (max_count - min_count)
                else:
                    sizes = np.full_like(counts, min_size)

                color = color_map[(ie, lvl)]
                marker = eq_price_markers[lvl]
                linestyle = eq_price_linestyles[lvl]
                plt.scatter(x, y, alpha=0.65, s=sizes, c=[color], marker=marker, linewidths=0)

                if len(g) >= 2 and np.std(x) > 0:
                    b, a = np.polyfit(x, y, 1)
                    xs = np.linspace(x.min(), x.max(), 100)
                    ys = a + b * xs
                    plt.plot(xs, ys, c=color, linestyle=linestyle, linewidth=2)

    plt.xlabel("seconds (5-second bins)")
    plt.ylabel("price_diff = price - eq_price")
    plt.title(title)

    legend_handles = []
    for ie in [1, 0]:
        for lvl in levels:
            if binned.empty or binned[(binned["is_equal"] == ie) & (binned["eq_price"] == lvl)].empty:
                continue
            color = color_map[(ie, lvl)]
            marker = eq_price_markers[lvl]
            linestyle = eq_price_linestyles[lvl]
            legend_handles.append(
                mlines.Line2D(
                    [], [], color=color, linestyle=linestyle, marker=marker,
                    label=f"is_equal={ie}, eq_price={lvl}",
                    linewidth=2, markersize=8
                )
            )

    if legend_handles:
        plt.legend(handles=legend_handles, title="is_equal, eq_price: color/marker/line", loc="upper right")

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(outfile, dpi=200)
    plt.close()

    try:
        shutil.copy2(outfile, os.path.join(DOC_INPUT_DIR, filename))
    except Exception as e:
        print(f"Warning: Could not copy {filename} to doc/input: {e}")


# === 全部 normalized round ===
plot_binned_scatter(
    plot_df,
    "seconds vs price_diff (5-second bins, size = sample count)",
    "trade_mkt_seconds_vs_price_diff.png",
)

# === first 6 treatment-specific rounds for each allocation method ===
# is_equal=0: grandfathering round 1-6; is_equal=1: equal round 1-6
plot_df_first6 = plot_df[plot_df["round"] <= 6].copy()
plot_binned_scatter(
    plot_df_first6,
    "seconds vs price_diff (first 6 rounds within allocation, 5-second bins)",
    "trade_mkt_seconds_vs_price_diff_first6_by_allocation.png",
)

# === last 6 actual trading rounds ===
# round_raw 13-18 captures the experienced phase for both allocation orders.
plot_df_last6 = plot_df[(plot_df["round_raw"] >= 13) & (plot_df["round_raw"] <= 18)].copy()
plot_binned_scatter(
    plot_df_last6,
    "seconds vs price_diff (last 6 actual trade rounds, 5-second bins)",
    "trade_mkt_seconds_vs_price_diff_last6_by_allocation.png",
)

print("All scatter plots generated and copied to doc/input successfully!")
