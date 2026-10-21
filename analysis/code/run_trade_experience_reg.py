# %% 分析買方與賣方交易經驗對交易結果的影響
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
INPUT = os.path.join(BASE_DIR, "..", "input", "trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# Part 1: Load data
# ==============================================================================
if not os.path.exists(INPUT):
    raise FileNotFoundError("trades_panel.csv not found in input directory.")

print(f"[LOAD] 讀取：{INPUT}")
df = pd.read_csv(INPUT, encoding="utf-8-sig")


# ==============================================================================
# Part 2: Check required columns and create outcomes
# ==============================================================================
required_cols = [
    "session_code",
    "price",
    "eq_price",
    "quantity",
    "buyer_trade_experience",
    "seller_trade_experience",
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(
        "trades_panel.csv 缺少以下必要欄位："
        + ", ".join(missing_cols)
    )

# 老師交易層級回歸使用的三個 outcome
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["eq_price"] = pd.to_numeric(df["eq_price"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

df["price_diff"] = df["price"] - df["eq_price"]
df["abs_price_diff"] = df["price_diff"].abs()

# Allocation order
df["first"] = pd.to_numeric(df["first"], errors="coerce")
df["is_equal_first"] = 1 - df["first"]
df = df.rename(columns={"round_raw": "total_round"})
# ==============================================================================
# Part 3: Create clustering variable
# ==============================================================================
# 優先使用原始實驗回合 round_raw；若沒有，才退回使用 round。
round_col = "round_raw" if "round_raw" in df.columns else "round"

if round_col not in df.columns:
    raise ValueError("trades_panel.csv 必須包含 round_raw 或 round。")

df["cluster_round"] = (
    df["session_code"].astype(str)
    + "_"
    + df[round_col].astype(str)
)


# ==============================================================================
# Part 4: Construct experience variables
# ==============================================================================
# 將問卷中的交易經驗年數轉成數值。
# 無法轉換的內容保留為 NaN，由 statsmodels 在各回歸中自動排除。
df["buyer_trade_experience_years"] = pd.to_numeric(
    df["buyer_trade_experience"],
    errors="coerce",
)

df["seller_trade_experience_years"] = pd.to_numeric(
    df["seller_trade_experience"],
    errors="coerce",
)

# Dummy：年數大於 0 代表有交易經驗。
# 問卷缺失值仍保留為 NaN，不直接當成「沒有經驗」。
df["buyer_experienced"] = np.where(
    df["buyer_trade_experience_years"].isna(),
    np.nan,
    (df["buyer_trade_experience_years"] > 0).astype(int),
)

df["seller_experienced"] = np.where(
    df["seller_trade_experience_years"].isna(),
    np.nan,
    (df["seller_trade_experience_years"] > 0).astype(int),
)


# ==============================================================================
# Part 5: Regression specifications
# ==============================================================================
OUTCOMES = [
    "price_diff",
    "abs_price_diff",
    "quantity",
]

# Dummy 版與年數版皆採相同的模型加入順序：
# (1) 核心經驗變數
# (2) 加入 seconds、round_raw、is_equal、first、is_equal:first
# (3) 再加入 eq_price_low、eq_price_high
# (4) 最後加入 buyer_is_dominant、seller_is_dominant

DUMMY_MODELS = []
NUMERIC_MODELS = []

for y in OUTCOMES:

    # =========================================================
    # Dummy 版本
    # =========================================================
    # DUMMY_MODELS.extend([

    #     # (1)
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced"
    #     ),

    #     # (2)
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first + "
    #         "is_equal:is_equal_first"
    #     ),

    #     # (3)
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first + "
    #         "is_equal:is_equal_first + "
    #         "eq_price_low + "
    #         "eq_price_high"
    #     ),

    #     # (4)
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first + "
    #         "is_equal:is_equal_first + "
    #         "eq_price_low + "
    #         "eq_price_high + "
    #         "buyer_is_dominant + "
    #         "seller_is_dominant"
    #     ),
    # ])
    # DUMMY_MODELS.extend([

    #     # (1) 只放買賣方經驗的主效果
    #     # 用來觀察 buyer 與 seller 經驗各自和 price_diff 的關係
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced + "
    #         "seller_experienced"
    #     ),

    #     # (2) 加入買方經驗 × 賣方經驗
    #     # 比較加入交乘項後，buyer、seller 主效果是否改變
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced"
    #     ),

    #     # (3) 加入交易時間
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds"
    #     ),

    #     # (4) 再加入總回合
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round"
    #     ),

    #     # (5) 再加入分配制度
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal"
    #     ),

    #     # (6) 再加入制度實施順序
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first"
    #     ),

    #     # (7) 加入分配制度 × 實施順序交乘項
    #     # 可看經驗係數是否是被制度與順序的聯合效果吸收
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first + "
    #         "is_equal:is_equal_first"
    #     ),

    #     # (8) 再加入均衡價格條件
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first + "
    #         "is_equal:is_equal_first + "
    #         "eq_price_low + "
    #         "eq_price_high"
    #     ),

    #     # (9) 完整模型：再控制買賣方是否為 dominant
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal + "
    #         "is_equal_first + "
    #         "is_equal:is_equal_first + "
    #         "eq_price_low + "
    #         "eq_price_high + "
    #         "buyer_is_dominant + "
    #         "seller_is_dominant"
    #     ),
    # ])
    # DUMMY_MODELS.extend([

    #     # (1) 經驗的基本關係
    #     # 先看買方與賣方經驗各自和 price_diff 的關係
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced + "
    #         "seller_experienced"
    #     ),

    #     # (2) 買賣雙方經驗的聯合作用
    #     # 檢驗雙方都有經驗時，是否存在額外的價格效果
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced"
    #     ),

    #     # (3) 控制交易過程
    #     # seconds 與 total_round 使用中心化，
    #     # 使經驗主效果代表平均秒數、平均回合下的差異
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round"
    #     ),

    #     # (4) 加入制度與制度順序控制
    #     # is_equal * is_equal_first 會自動包含：
    #     # is_equal、is_equal_first、is_equal:is_equal_first
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal * is_equal_first"
    #     ),

    #     # (5) 核心機制模型：經驗 × 交易時間
    #     # 檢驗買賣方經驗是否改變回合內的時間趨勢
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal * is_equal_first + "
    #         "buyer_experienced:seconds + "
    #         "seller_experienced:seconds"
    #     ),

    #     # (6) 完整控制模型
    #     # 加入均衡價格條件與買賣方市場角色
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal * is_equal_first + "
    #         "buyer_experienced:seconds + "
    #         "seller_experienced:seconds + "
    #         "eq_price_low + "
    #         "eq_price_high + "
    #         "buyer_is_dominant + "
    #         "seller_is_dominant"
    #     ),

    #     # (7) 延伸模型：經驗 × 回合數
    #     # 檢驗既有交易經驗是否改變實驗內的跨回合學習
    #     (
    #         f"{y} ~ "
    #         "buyer_experienced * seller_experienced + "
    #         "seconds + "
    #         "total_round + "
    #         "is_equal * is_equal_first + "
    #         "buyer_experienced:seconds + "
    #         "seller_experienced:seconds + "
    #         "buyer_experienced:total_round + "
    #         "seller_experienced:total_round + "
    #         "eq_price_low + "
    #         "eq_price_high + "
    #         "buyer_is_dominant + "
    #         "seller_is_dominant"
    #     ),
    # ])
    DUMMY_MODELS.extend([

        # (1) 買賣方經驗主效果
        (
            f"{y} ~ "
            "buyer_experienced + "
            "seller_experienced"
        ),

        # (2) 加入買方經驗 × 賣方經驗
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced"
        ),

        # (3) 加入交易秒數
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds"
        ),

        # (4) 再加入總回合
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round"
        ),

        # (5) 加入分配制度
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal"
        ),

        # (6) 再加入制度實施順序
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first"
        ),

        # (7) 再加入制度 × 實施順序
        # 這一步用來確認買家經驗為何失去顯著性
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first"
        ),
            # (8) 在 Model (7) 加入經驗 × 秒數
    # 檢驗買賣方經驗是否改變交易時間效果
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "buyer_experienced:seconds + "
            "seller_experienced:seconds"
        ),

        # (8) 完整控制模型，但先不放經驗 × 時間
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "eq_price_low + "
            "eq_price_high + "
            "buyer_is_dominant + "
            "seller_is_dominant +"
            "buyer_experienced:seconds + "
            "seller_experienced:seconds"
        ),

        # (9) 在 Model (7) 加入經驗 × 總回合
        # 檢驗買賣方經驗是否改變跨回合趨勢
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "buyer_experienced:total_round + "
            "seller_experienced:total_round"
        ),

        # (8) 完整控制模型，但先不放經驗 × 時間
        (
            f"{y} ~ "
            "buyer_experienced * seller_experienced + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "eq_price_low + "
            "eq_price_high"
        ),
        

    ])

    # # =========================================================
    # # 年數版本
    # # =========================================================
    NUMERIC_MODELS.extend([

        # (1) 買賣方交易經驗年數主效果
        (
            f"{y} ~ "
            "buyer_trade_experience_years + "
            "seller_trade_experience_years"
        ),

        # (2) 加入買方經驗年數 × 賣方經驗年數
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years"
        ),

        # (3) 加入交易秒數
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds"
        ),

        # (4) 再加入總回合
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round"
        ),

        # (5) 加入分配制度
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal"
        ),

        # (6) 再加入制度實施順序
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first"
        ),

        # (7) 再加入制度 × 實施順序
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first"
        ),

        # (8) 在 Model (7) 加入經驗年數 × 秒數
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "buyer_trade_experience_years:seconds + "
            "seller_trade_experience_years:seconds"
        ),

        # (9) 經驗年數 × 秒數的完整控制模型
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "buyer_trade_experience_years:seconds + "
            "seller_trade_experience_years:seconds + "
            "eq_price_low + "
            "eq_price_high + "
            "buyer_is_dominant + "
            "seller_is_dominant"
        ),

        # (10) 在 Model (7) 加入經驗年數 × 總回合
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "buyer_trade_experience_years:total_round + "
            "seller_trade_experience_years:total_round"
        ),

        # (11) 經驗年數 × 總回合的完整控制模型
        (
            f"{y} ~ "
            "buyer_trade_experience_years * seller_trade_experience_years + "
            "seconds + "
            "total_round + "
            "is_equal + "
            "is_equal_first + "
            "is_equal:is_equal_first + "
            "buyer_trade_experience_years:total_round + "
            "seller_trade_experience_years:total_round + "
            "eq_price_low + "
            "eq_price_high + "
            "buyer_is_dominant + "
            "seller_is_dominant"
        ),

    ])

# ==============================================================================
# Part 6: Convert to reg_func input format
# ==============================================================================
def group_formulas_by_outcome(formulas):
    result = defaultdict(list)

    for formula in formulas:
        y = formula.split("~", 1)[0].strip()
        result[y].append(formula)

    return result


dummy_y_to_formulas = group_formulas_by_outcome(DUMMY_MODELS)
numeric_y_to_formulas = group_formulas_by_outcome(NUMERIC_MODELS)


# ==============================================================================
# Part 7: Run regressions
# ==============================================================================
# 與老師的交易市場回歸相同：
# 1. reg_func 會輸出 HC3 robust SE。
# 2. 另外以 session_code × round_raw 聚類標準誤。

# 1. 有無交易經驗 dummy
reg_func(
    df,
    dummy_y_to_formulas,
    os.path.join(OUTPUT_DIR, "trade_experience_dummy"),
    cluster_col="cluster_round",
)

# 2. 交易經驗原始數值／年數
reg_func(
    df,
    numeric_y_to_formulas,
    os.path.join(OUTPUT_DIR, "trade_experience_numeric"),
    cluster_col="cluster_round",
)
