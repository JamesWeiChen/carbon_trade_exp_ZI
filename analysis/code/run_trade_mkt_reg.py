# %% 分析 CarbonTrading 市場交易是否受 allocation order 影響
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
from reg_utils import reg_func

# 參數與路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "input", "trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 讀資料與衍生欄
if not os.path.exists(INPUT):
	raise FileNotFoundError("trades_panel.csv not found in input directory.")
df = pd.read_csv(INPUT)
df["price_diff"] = df["price"] - df["eq_price"]
df["abs_price_diff"] = df["price_diff"].abs()
df["last6_raw"] = df["round_raw"].between(13, 18).astype(int)
df["cluster_round"] = df["session_code"].astype(str) + "_" + df["round_raw"].astype(str)
equal_first_by_session = (
	df.loc[df["first"].eq(1)]
	.groupby("session_code")["is_equal"]
	.max()
)
df["is_equal_first"] = df["session_code"].map(equal_first_by_session)
df["round_c"] = pd.to_numeric(df["round"], errors="coerce") - 1

missing_order = df.loc[df["is_equal_first"].isna(), "session_code"].unique()
if len(missing_order):
	raise ValueError(f"Missing is_equal_first assignment for sessions: {missing_order}")

# 回歸模型設計
formulas = [
	"price_diff ~ is_equal * is_equal_first + seconds + round_c",
	"price_diff ~ is_equal * is_equal_first + seconds + round_c"
	" + eq_price_low + eq_price_high",
	"price_diff ~ is_equal * is_equal_first + seconds + round_c"
	" + eq_price_low + eq_price_high"
	" + is_equal:eq_price_low + is_equal:eq_price_high",
	"price_diff ~ is_equal * is_equal_first * round_c"
	" + seconds + eq_price_low + eq_price_high"
	" + is_equal:eq_price_low + is_equal:eq_price_high",
	# Raw last-six diagnostic kept separate from is_equal_first interactions.
	"price_diff ~ is_equal + last6_raw + is_equal:last6_raw"
	" + seconds + round_c + eq_price_low + eq_price_high",
	"price_diff ~ is_equal + last6_raw + is_equal:last6_raw"
	" + seconds + round_c + eq_price_low + eq_price_high"
	" + is_equal:eq_price_low + is_equal:eq_price_high",
	"abs_price_diff ~ is_equal * is_equal_first + seconds + round_c"
	" + eq_price_low + eq_price_high",
	"abs_price_diff ~ is_equal * is_equal_first * round_c"
	" + seconds + eq_price_low + eq_price_high",
	"abs_price_diff ~ is_equal + last6_raw + is_equal:last6_raw"
	" + seconds + round_c + eq_price_low + eq_price_high",
]

# 組成 y_to_formulas 結構
from collections import defaultdict
y_to_formulas = defaultdict(list)
for fml in formulas:
	y = fml.split("~")[0].strip()
	y_to_formulas[y].append(fml)

# Wald/simple effects callback for reg_func
def _scalar(x):
	return float(np.asarray(x).ravel()[0])

def trade_simple_effects(res, idx, label):
	lines = []
	params = res.params.index.astype(str)
	if "is_equal:is_equal_first" in params.values:
		round_note = " at round 1" if "is_equal:round_c" in params.values else ""
		equal_second = res.t_test("is_equal = 0")
		lines.append(
			f"equal effect when equal is second order{round_note}: "
			f"effect={_scalar(equal_second.effect):.3f}, SE={_scalar(equal_second.sd):.3f}, "
			f"t={_scalar(equal_second.tvalue):.3f}, p={_scalar(equal_second.pvalue):.3f}"
		)
		equal_first = res.t_test("is_equal + is_equal:is_equal_first = 0")
		lines.append(
			f"equal effect when equal is first order{round_note}: "
			f"effect={_scalar(equal_first.effect):.3f}, SE={_scalar(equal_first.sd):.3f}, "
			f"t={_scalar(equal_first.tvalue):.3f}, p={_scalar(equal_first.pvalue):.3f}"
		)
		order_heterogeneity = res.t_test("is_equal:is_equal_first = 0")
		lines.append(
			f"order heterogeneity in equal effect{round_note}: "
			f"effect={_scalar(order_heterogeneity.effect):.3f}, "
			f"SE={_scalar(order_heterogeneity.sd):.3f}, "
			f"t={_scalar(order_heterogeneity.tvalue):.3f}, "
			f"p={_scalar(order_heterogeneity.pvalue):.3f}"
		)
	if "is_equal:is_equal_first:round_c" in params.values:
		slope_second = res.t_test("is_equal:round_c = 0")
		lines.append(
			f"equal-vs-GF round slope when equal is second order: "
			f"effect={_scalar(slope_second.effect):.3f}, SE={_scalar(slope_second.sd):.3f}, "
			f"t={_scalar(slope_second.tvalue):.3f}, p={_scalar(slope_second.pvalue):.3f}"
		)
		slope_first = res.t_test(
			"is_equal:round_c + is_equal:is_equal_first:round_c = 0"
		)
		lines.append(
			f"equal-vs-GF round slope when equal is first order: "
			f"effect={_scalar(slope_first.effect):.3f}, SE={_scalar(slope_first.sd):.3f}, "
			f"t={_scalar(slope_first.tvalue):.3f}, p={_scalar(slope_first.pvalue):.3f}"
		)
	if ("is_equal:eq_price_low" in params.values) and ("is_equal:eq_price_high" in params.values):
		context = ""
		if "is_equal:is_equal_first" in params.values:
			context += " when equal is second order"
		if "is_equal:round_c" in params.values:
			context += " at round 1"
		# eq_price=6（基準）
		tt6 = res.t_test("is_equal = 0")
		lines.append(f"eq_price=6{context}: effect={_scalar(tt6.effect):.3f}, SE={_scalar(tt6.sd):.3f}, t={_scalar(tt6.tvalue):.3f}, p={_scalar(tt6.pvalue):.3f}")
		# eq_price=3
		tt3 = res.t_test("is_equal + is_equal:eq_price_low = 0")
		lines.append(f"eq_price=3{context}: effect={_scalar(tt3.effect):.3f}, SE={_scalar(tt3.sd):.3f}, t={_scalar(tt3.tvalue):.3f}, p={_scalar(tt3.pvalue):.3f}")
		# eq_price=9
		tt9 = res.t_test("is_equal + is_equal:eq_price_high = 0")
		lines.append(f"eq_price=9{context}: effect={_scalar(tt9.effect):.3f}, SE={_scalar(tt9.sd):.3f}, t={_scalar(tt9.tvalue):.3f}, p={_scalar(tt9.pvalue):.3f}")
		# Wald 檢定：異質性
		wald = res.wald_test([
			"is_equal:eq_price_low = 0",
			"is_equal:eq_price_high = 0",
		], scalar=True)
		lines.append(f"Wald heterogeneity: chi2={_scalar(wald.statistic):.3f}, df={int(wald.df_denom)}, p={_scalar(wald.pvalue):.3f}")
	if "is_equal:last6_raw" in params.values:
		early = res.t_test("is_equal = 0")
		lines.append(f"is_equal effect outside raw rounds 13-18: effect={_scalar(early.effect):.3f}, SE={_scalar(early.sd):.3f}, t={_scalar(early.tvalue):.3f}, p={_scalar(early.pvalue):.3f}")
		last6 = res.t_test("is_equal + is_equal:last6_raw = 0")
		lines.append(f"is_equal effect in raw rounds 13-18: effect={_scalar(last6.effect):.3f}, SE={_scalar(last6.sd):.3f}, t={_scalar(last6.tvalue):.3f}, p={_scalar(last6.pvalue):.3f}")
		gf_change = res.t_test("last6_raw = 0")
		lines.append(f"last6 effect when is_equal=0: effect={_scalar(gf_change.effect):.3f}, SE={_scalar(gf_change.sd):.3f}, t={_scalar(gf_change.tvalue):.3f}, p={_scalar(gf_change.pvalue):.3f}")
		eq_change = res.t_test("last6_raw + is_equal:last6_raw = 0")
		lines.append(f"last6 effect when is_equal=1: effect={_scalar(eq_change.effect):.3f}, SE={_scalar(eq_change.sd):.3f}, t={_scalar(eq_change.tvalue):.3f}, p={_scalar(eq_change.pvalue):.3f}")
	return lines if lines else None

reg_func(df, y_to_formulas, os.path.join(OUTPUT_DIR, "trade_mkt"), cluster_col="cluster_round", extra_effects_func=trade_simple_effects)
