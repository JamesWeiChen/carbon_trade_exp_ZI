# --- Minimal, clean version: run regressions and output txt/tabulate like run_trade_mkt_reg.py ---
import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from reg_utils import reg_func

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INFILE = os.path.join(INPUT_DIR, "subj_lv_panel.csv")
if not os.path.exists(INFILE):
    raise FileNotFoundError("subj_lv_panel.csv not found in input directory.")
print(f"[LOAD] 讀取：{INFILE}")

df = pd.read_csv(INFILE, encoding="utf-8-sig")
# 新增 *_diff 欄位
# df["emission_diff"] = df["emission"] - df["e_soc"]
# df["net_profit_diff"] = df["net_profit"] - df["pi_soc"]
# df["production_diff"] = df["emission"] - df["q_soc"]
df["emission_diff"] = df["emission"] - df["e_soc"]
df["net_profit_diff"] = df["net_profit"] - df["pi_soc"]
df["production_diff"] = df["production"] - df["q_soc"]

if "app" in df.columns:
    df = df.query("app != 'Stage_MUDA'").copy()
for c in ["is_tax","is_trade","is_dominant","is_equal","tax_rate","carbon_multiplier"]:
    if c not in df.columns:
        df[c] = 0

trade_mask = df["is_trade"].eq(1)
equal_first_by_session = (
    df.loc[trade_mask & df["first"].eq(1)]
    .groupby("session_code")["is_equal"]
    .max()
)
df["is_equal_first"] = df["session_code"].map(equal_first_by_session)
df["round_c"] = pd.to_numeric(df["round"], errors="coerce") - 1

missing_order = df.loc[trade_mask & df["is_equal_first"].isna(), "session_code"].unique()
if len(missing_order):
    raise ValueError(f"Missing is_equal_first assignment for sessions: {missing_order}")

##################################
# Models to run (grouped by y)
OUTCOMES = ["emission_diff", "net_profit_diff", "production_diff"]
TRADE_BASE = "is_equal * is_equal_first + is_dominant + eq_price_low + eq_price_high"
DOMINANT_EQ_PRICE_INTERACTIONS = (
    "is_dominant:eq_price_low + is_dominant:eq_price_high"
)
MODELS = []
for y in OUTCOMES:
    MODELS.extend([
        # All-sample models: trading regime relative to control.
        f"{y} ~ is_trade + is_dominant",
        f"{y} ~ is_trade + is_dominant + is_dominant:is_trade",
        # Trade-only models: allocation order, eq_price, and dominant status.
        f"{y} ~ {TRADE_BASE}",
        f"{y} ~ {TRADE_BASE} + {DOMINANT_EQ_PRICE_INTERACTIONS}",
        f"{y} ~ {TRADE_BASE} + {DOMINANT_EQ_PRICE_INTERACTIONS} + round_c",
        f"{y} ~ is_equal * is_equal_first * round_c + is_dominant + "
        f"eq_price_low + eq_price_high + {DOMINANT_EQ_PRICE_INTERACTIONS}",
    ])
###################################
from collections import defaultdict
y_to_formulas = defaultdict(list)
for fml in MODELS:
    y = fml.split("~")[0].strip()
    y_to_formulas[y].append(fml)

def _scalar(x):
    return float(np.asarray(x).ravel()[0])

def order_simple_effects(res, idx, label):
    params = res.params.index.astype(str)
    if "is_equal:is_equal_first" not in params.values:
        return None

    round_note = " at round 1" if "is_equal:round_c" in params.values else ""
    lines = []
    tt_second = res.t_test("is_equal = 0")
    lines.append(
        f"equal effect when equal is second order{round_note}: "
        f"effect={_scalar(tt_second.effect):.3f}, SE={_scalar(tt_second.sd):.3f}, "
        f"t={_scalar(tt_second.tvalue):.3f}, p={_scalar(tt_second.pvalue):.3f}"
    )
    tt_first = res.t_test("is_equal + is_equal:is_equal_first = 0")
    lines.append(
        f"equal effect when equal is first order{round_note}: "
        f"effect={_scalar(tt_first.effect):.3f}, SE={_scalar(tt_first.sd):.3f}, "
        f"t={_scalar(tt_first.tvalue):.3f}, p={_scalar(tt_first.pvalue):.3f}"
    )
    tt_order = res.t_test("is_equal:is_equal_first = 0")
    lines.append(
        f"order heterogeneity in equal effect{round_note}: "
        f"effect={_scalar(tt_order.effect):.3f}, SE={_scalar(tt_order.sd):.3f}, "
        f"t={_scalar(tt_order.tvalue):.3f}, p={_scalar(tt_order.pvalue):.3f}"
    )
    if "is_equal:is_equal_first:round_c" in params.values:
        tt_slope_second = res.t_test("is_equal:round_c = 0")
        lines.append(
            f"equal-vs-GF round slope when equal is second order: "
            f"effect={_scalar(tt_slope_second.effect):.3f}, SE={_scalar(tt_slope_second.sd):.3f}, "
            f"t={_scalar(tt_slope_second.tvalue):.3f}, p={_scalar(tt_slope_second.pvalue):.3f}"
        )
        tt_slope_first = res.t_test(
            "is_equal:round_c + is_equal:is_equal_first:round_c = 0"
        )
        lines.append(
            f"equal-vs-GF round slope when equal is first order: "
            f"effect={_scalar(tt_slope_first.effect):.3f}, SE={_scalar(tt_slope_first.sd):.3f}, "
            f"t={_scalar(tt_slope_first.tvalue):.3f}, p={_scalar(tt_slope_first.pvalue):.3f}"
        )
    return lines


# 呼叫共用回歸函數
reg_func(
    df,
    y_to_formulas,
    os.path.join(OUTPUT_DIR, "subj_lv"),
    cluster_col="participant_code",
    extra_effects_func=order_simple_effects,
)

