from pathlib import Path

import numpy as np
import pandas as pd


import sys
sys.path.append(str(Path(__file__).parent))


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "input"
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
INFILE = INPUT_DIR / "group_lv_panel.csv"
if not INFILE.exists():
    raise FileNotFoundError("group_lv_panel.csv not found in input directory.")
print(f"[LOAD] 讀取：{INFILE}")

df = pd.read_csv(INFILE, encoding="utf-8-sig")
# 新增 *_diff 欄位
# df["emission_diff"] = df["emission"] - df["E_soc"]
# df["net_profit_diff"] = df["net_profit"] - df["Pi_soc"]
# df["production_diff"] = df["emission"] - df["Q_soc"]
df["emission_diff"] = df["emission"] - df["E_soc"]
df["net_profit_diff"] = df["net_profit"] - df["Pi_soc"]
df["production_diff"] = df["production"] - df["Q_soc"]

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

OUTCOMES = ["emission_diff", "net_profit_diff", "production_diff"]
TRADE_BASE = "is_equal * is_equal_first + eq_price_low + eq_price_high"
EQ_PRICE_INTERACTIONS = "is_equal:eq_price_low + is_equal:eq_price_high"
MODELS = []
for y in OUTCOMES:
    MODELS.extend([
        # All-sample model: trading regime relative to control.
        (f"{y} ~ is_trade", "all"),
        # Trade-only models: allocation order and eq_price variation.
        (f"{y} ~ {TRADE_BASE}", "is_trade"),
        (f"{y} ~ {TRADE_BASE} + {EQ_PRICE_INTERACTIONS}", "is_trade"),
        (f"{y} ~ {TRADE_BASE} + {EQ_PRICE_INTERACTIONS} + round_c", "is_trade"),
        (f"{y} ~ is_equal * is_equal_first * round_c + "
         f"eq_price_low + eq_price_high + {EQ_PRICE_INTERACTIONS}", "is_trade"),
    ])
#####################################################
from collections import defaultdict
y_to_formulas = defaultdict(list)
for formula, sample in MODELS:
    y = formula.split("~")[0].strip()
    y_to_formulas[y].append(formula)

from reg_utils import reg_func

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

reg_func(
    df,
    y_to_formulas,
    str(OUTPUT_DIR / "group_lv"),
    extra_effects_func=order_simple_effects,
)
