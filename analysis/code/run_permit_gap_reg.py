# %% 分析最終碳權持有量相對理論量的偏離（permit_gap）
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from reg_utils import reg_func


# ==============================================================================
# Part 0: Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INFILE = os.path.join(INPUT_DIR, "subj_lv_panel.csv")
if not os.path.exists(INFILE):
    raise FileNotFoundError("subj_lv_panel.csv not found in input directory.")

print(f"[LOAD] 讀取：{INFILE}")
df = pd.read_csv(INFILE, encoding="utf-8-sig")

# 相容舊資料檔：若尚未重新產生 panel，仍可將 permit_gap_tax 當作 permit_gap。
if "permit_gap" not in df.columns and "permit_gap_tax" in df.columns:
    df = df.rename(columns={"permit_gap_tax": "permit_gap"})
    print("[INFO] 已將舊欄位 permit_gap_tax 重新命名為 permit_gap。")

df = df.rename(columns={"round_raw": "total_round"})
df["is_equal_first"] = 1 - pd.to_numeric(df["first"], errors="coerce")
# ==============================================================================
# Part 1: Check columns, convert types, and keep carbon-trading observations
# ==============================================================================
required_cols = [
    "participant_code",
    "permit_gap",
    "is_trade",
    "is_dominant",
    "is_equal",
    "first",
]

if "round_raw" not in df.columns and "round" not in df.columns:
    required_cols.append("round_raw")
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(
        "subj_lv_panel.csv 缺少以下必要欄位：" + ", ".join(missing_cols)
    )

# 先轉成數值，再篩選 is_trade；避免 is_trade 被讀成字串時全部篩掉。
for c in [
    "permit_gap",
    "is_trade",
    "is_dominant",
    "is_equal",
    "first",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if "round_raw" in df.columns:
    df["round_raw"] = pd.to_numeric(df["round_raw"], errors="coerce")
else:
    df["round_raw"] = pd.to_numeric(df["round"], errors="coerce")
    print("[INFO] 資料沒有 round_raw，已使用 round 建立 round_raw。")

# permit_gap 只分析碳交易處理。
df = df.loc[df["is_trade"].eq(1)].copy()

if df.empty:
    raise ValueError(
        "篩選 is_trade == 1 後沒有資料。請檢查 subj_lv_panel.csv 的 is_trade 欄位。"
    )

if df["permit_gap"].notna().sum() == 0:
    raise ValueError(
        "碳交易樣本中的 permit_gap 全部是缺失值，無法執行回歸。"
    )

print(f"[CHECK] 碳交易觀察值：{len(df)}")
print(f"[CHECK] permit_gap 非缺失值：{df['permit_gap'].notna().sum()}")
print("[CHECK] permit_gap 摘要：")
print(df["permit_gap"].describe())


# ==============================================================================
# Part 2: Regression specifications
# ==============================================================================
# permit_gap = 最終碳權持有量 - 理論應持有量
#
# (1) 核心模型：廠商角色 × 分配制度
# (2) 加入一般回合趨勢
# (3) 再加入實驗先後順序，以及制度 × 先後順序
# (4) 再檢驗 dominant 與 non-dominant 是否有不同的回合趨勢
MODELS = [
    "permit_gap ~ is_dominant * is_equal",

    "permit_gap ~ is_dominant * is_equal + total_round",

    (
        "permit_gap ~ "
        "is_dominant * is_equal + "
        "total_round + "
        "is_equal_first + "
        "is_equal:is_equal_first"
    ),

    (
        "permit_gap ~ "
        "is_dominant * is_equal + "
        "total_round + "
        "is_equal_first + "
        "is_equal:is_equal_first + "
        "is_dominant:total_round"
    ),
]


# ==============================================================================
# Part 3: Convert to reg_func input format
# ==============================================================================
y_to_formulas = defaultdict(list)
for formula in MODELS:
    y = formula.split("~", 1)[0].strip()
    y_to_formulas[y].append(formula)

print(f"[CHECK] 預計執行模型數：{len(MODELS)}")
for i, formula in enumerate(MODELS, 1):
    print(f"  ({i}) {formula}")


# ==============================================================================
# Part 4: Run regressions
# ==============================================================================
# 同一受試者跨回合重複出現，因此以 participant_code 聚類標準誤。
OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, "permit_gap")

reg_func(
    df,
    y_to_formulas,
    OUTPUT_PREFIX,
    cluster_col="participant_code",
)

print("[OUTPUT] 結果檔案：")
print(OUTPUT_PREFIX + "_reg_results.txt")
print(OUTPUT_PREFIX + "_reg_results_tabulate.txt")
