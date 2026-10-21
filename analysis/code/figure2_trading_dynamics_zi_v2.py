from __future__ import annotations

"""
CTAT Figure 2, adapted from Lin et al. (2020): teacher-confirmed ZI version 2.
Place in analysis/code; reads analysis/input/{trades,subj_lv,group_lv}_panel.csv.
Writes four figures and audit CSV/TXT files to its version-specific output folder.
ZI model: every activation selects one participant, who simultaneously submits
one-unit bid and ask whenever physically feasible. Both sides use that person's
current holding-state marginal values; acquired permits may be resold. Self-trade
is forbidden. Only the best bid and best ask survive, same-price quotes retain
the earlier order, execution uses the pre-existing standing quote, and every
crossing side executes. If both incoming sides cross, both execute atomically
from the same pre-event state; the book is cleared after all executions in that
activation. The market stops at the induced-value
equilibrium (no distinct-person transfer can raise total induced value).
Ten thousand independent sampled markets can run through a process pool; this
parallelism changes neither random seeds nor any within-market rule.
The average-permit value convention is preserved, and true integer-production
frontier effects are audited separately (induced no-loss is not physical no-loss).
Package: one price per executed transaction.
Permit unit: repeat each price by quantity, retaining all zero price changes.
Unit value/cost ranks retain tied package order because units execute together.
Raw and standardized plots share the standardized [-4,4] display filter.
MA/AN overlays are Lin's analytical restrictions, NOT fitted CTAT equilibria.
The former greedy matching heuristic is removed: it does not solve MA or AN.
References: https://osf.io/9mfws/ (utils.R, plot.theory.comp);
Lin Supplement B.3 pp26-27; Wilson (1987); Friedman (1991).
"""

import ast
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import gaussian_kde


# ============================================================
# 0. Paths and settings
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
INPUT_DIR = ANALYSIS_DIR / "input"
ZI_ROLE_MODE = "simultaneous_two_sided_resale"
OUTPUT_DIR = ANALYSIS_DIR / "output" / f"figure2_zi_{ZI_ROLE_MODE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_PATH = INPUT_DIR / "trades_panel.csv"
SUBJ_PATH = INPUT_DIR / "subj_lv_panel.csv"
GROUP_PATH = INPUT_DIR / "group_lv_panel.csv"

MARKET_KEYS = ["session_code", "round_raw"]
MIN_TRANSACTION_PACKAGES = 4
FIGURE_ROUND_RAW_MIN = 1
FIGURE_ROUND_RAW_MAX = 6

EARLY_LABEL = "Rounds 1-3"
LATE_LABEL = "Rounds 4-6"

EARLY_COLOR = "#d9bd55"
LATE_COLOR = "#5e8fc7"
REGRESSION_COLOR = "#2f8f46"
# Lin et al.'s public R palette for Figure 2d theory overlays.
MA_COLOR = "#CCBB44"
AN_COLOR = "#228833"
ZI_COLOR = "#EE6677"
EMPIRICAL_MEAN_COLOR = "#CC3311"

# Analytical restrictions from Lin Supplement B.3 (risk neutrality/symmetry).
MA_RHO_PRICE = 0.0
AN_RHO_PRICE_PLOT_MIN = 0.0  # graphical endpoint; theory is strictly positive
AN_RHO_PRICE_PLOT_MAX = 1.0
# Requested replication count: exactly 10,000 sampled market runs, not 10,000
# complete-rho runs. There is NO quote-event or elapsed-time termination cap.
ZI_SIMULATIONS = 10_000
ZI_RANDOM_SEED = 20260901
ZI_INITIAL_CASH = 1_000.0
ZI_ASK_CEILING = 1_000.0
ZI_DUAL_CROSS_RULE = "execute_both_atomically_then_clear"
ZI_NUMERIC_TOLERANCE = 1e-9  # validation/tie tolerance, not a stopping threshold
ZI_KEEP_QUOTE_LOG = False   # execution logs are always exported
ZI_PROGRESS_EVERY_QUOTES = 100_000  # print progress ONLY; never stop here
# Process-level parallelism is across independent markets only. Set to 1 for
# sequential debugging. Deterministic per-simulation seeds make results invariant
# to worker count and scheduling order.
ZI_N_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
ZI_PARALLEL_CHUNKSIZE = 25
PRICE_CHANGE_PLOT_LIMIT = 4.0
PRICE_CHANGE_HEXBINS = 100
PRICE_CHANGE_MODES = ("standardized", "raw")


# ============================================================
# 1. Utilities and data loading
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到輸入檔：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} 缺少欄位：{', '.join(sorted(missing))}")


def parse_numeric_sequence(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        return np.array([], dtype=float)

    text = str(value)
    try:
        return np.asarray(json.loads(text), dtype=float)
    except Exception:
        return np.asarray(ast.literal_eval(text), dtype=float)


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(pair["x"], pair["y"]).statistic)


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(stats.pearsonr(pair["x"], pair["y"]).statistic)


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    critical = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - critical * se)), float(np.tanh(z + critical * se))


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = read_csv(TRADES_PATH)
    subj = read_csv(SUBJ_PATH)
    group = read_csv(GROUP_PATH)

    require_columns(
        trades,
        {
            "session_code", "round_raw", "trade_index", "seconds",
            "buyer_id", "seller_id", "price", "quantity", "eq_price",
        },
        "trades_panel.csv",
    )
    require_columns(
        subj,
        {
            "session_code", "round_raw", "app", "id_in_group", "permits",
            "current_permits", "disturbance_values",
            "marginal_cost_coefficient", "carbon_emission_per_unit",
            "max_production", "market_price", "group_cap_total",
        },
        "subj_lv_panel.csv",
    )
    require_columns(
        group,
        {
            "session_code", "round_raw", "app", "round", "first",
            "is_equal", "allocation_method",
        },
        "group_lv_panel.csv",
    )

    subj = subj.loc[subj["app"].eq("Stage_CarbonTrading")].copy()
    group = group.loc[group["app"].eq("Stage_CarbonTrading")].copy()

    for col in [
        "round_raw", "trade_index", "seconds", "buyer_id", "seller_id",
        "price", "quantity", "eq_price",
    ]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    for col in [
        "round_raw", "id_in_group", "permits", "current_permits",
        "marginal_cost_coefficient", "carbon_emission_per_unit",
        "max_production", "market_price", "group_cap_total",
    ]:
        subj[col] = pd.to_numeric(subj[col], errors="coerce")

    for col in ["round_raw", "round", "first", "is_equal"]:
        group[col] = pd.to_numeric(group[col], errors="coerce")

    trades = trades.dropna(
        subset=[
            "session_code", "round_raw", "trade_index", "buyer_id",
            "seller_id", "price", "quantity", "eq_price",
        ]
    ).copy()
    trades["quantity"] = trades["quantity"].round().astype(int)
    if (trades["quantity"] <= 0).any():
        raise ValueError("trades_panel.csv 出現 quantity <= 0。")

    # Figure 2 scope: use only round_raw 1-6 for every panel/output.
    # Panels a-b were already comparing 1-3 vs 4-6; this filter makes
    # panels c-d and all exported analysis tables use the same 1-6 range.
    round_mask_trades = trades["round_raw"].between(
        FIGURE_ROUND_RAW_MIN, FIGURE_ROUND_RAW_MAX
    )
    round_mask_subj = subj["round_raw"].between(
        FIGURE_ROUND_RAW_MIN, FIGURE_ROUND_RAW_MAX
    )
    round_mask_group = group["round_raw"].between(
        FIGURE_ROUND_RAW_MIN, FIGURE_ROUND_RAW_MAX
    )
    trades = trades.loc[round_mask_trades].copy()
    subj = subj.loc[round_mask_subj].copy()
    group = group.loc[round_mask_group].copy()

    return trades, subj, group


# ============================================================
# 2. Reconstruct permit values and seller opportunity costs
# ============================================================

def production_profit_frontier(subject_row: pd.Series, cap_total: int) -> np.ndarray:
    """Maximum production profit attainable at permit holdings 0..cap_total."""
    emissions_per_output = max(
        1,
        int(round(subject_row["carbon_emission_per_unit"])),
    )
    max_production = max(
        0,
        int(round(subject_row["max_production"])),
    )
    marginal_cost_coefficient = float(subject_row["marginal_cost_coefficient"])
    output_price = float(subject_row["market_price"])

    disturbances = parse_numeric_sequence(subject_row["disturbance_values"])
    if disturbances.size < max_production:
        raise ValueError(
            "disturbance_values 長度小於 max_production："
            f"id_in_group={subject_row['id_in_group']}，"
            f"length={disturbances.size}，max_production={max_production}"
        )

    q = np.arange(1, max_production + 1, dtype=float)
    marginal_cost = (
        marginal_cost_coefficient * q
        + disturbances[:max_production]
    )
    cumulative_profit = np.concatenate(
        [[0.0], np.cumsum(output_price - marginal_cost)]
    )

    frontier = np.zeros(cap_total + 1, dtype=float)
    for holding in range(cap_total + 1):
        feasible_max_q = min(
            max_production,
            holding // emissions_per_output,
        )
        frontier[holding] = float(
            np.max(cumulative_profit[: feasible_max_q + 1])
        )
    return frontier


def permit_marginal_schedule(
    subject_row: pd.Series,
    cap_total: int,
    frontier: np.ndarray | None = None,
) -> np.ndarray:
    """Return an averaged marginal value/cost for permit ordinals 1..cap_total.

    The physical production frontier is step-shaped when one output requires more
    than one permit.  Example with b=2: F(0)=0, F(1)=0, F(2)=20 would naively give
    marginal permit values 0 and 20.  For Figure 2's unit-ranking interpretation,
    the code instead assigns the complete bundle gain equally to both permits:
    10 and 10.  This is the requested treatment of the large-firm 'middle permit'.

    For b=1 this is exactly the ordinary one-permit marginal profit.
    """
    if frontier is None:
        frontier = production_profit_frontier(subject_row, cap_total)

    b = max(1, int(round(subject_row["carbon_emission_per_unit"])))
    max_production = max(0, int(round(subject_row["max_production"])))
    schedule = np.zeros(cap_total + 1, dtype=float)  # index = permit ordinal

    for output_unit in range(1, max_production + 1):
        block_start = (output_unit - 1) * b
        block_end = output_unit * b
        if block_end > cap_total:
            break
        block_gain = float(frontier[block_end] - frontier[block_start])
        schedule[block_start + 1 : block_end + 1] = block_gain / b

    return schedule


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else np.nan


def finite_sum(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.sum()) if array.size else 0.0


def add_price_changes(panel: pd.DataFrame, eq_price: float,
                      observation_level: str) -> pd.DataFrame:
    """Mutate one ordered market panel; return adjacent change pairs, zeros kept."""
    panel["observation_level"] = observation_level
    panel["price_observation_order"] = np.arange(1, len(panel) + 1)
    panel["price_change_raw"] = panel["price"].diff()
    panel["lagged_price_change_raw"] = panel["price_change_raw"].shift(1)
    for prefix in ("price_change", "lagged_price_change"):
        panel[f"{prefix}_standardized"] = panel[f"{prefix}_raw"] / eq_price
        panel[prefix] = panel[f"{prefix}_standardized"]
    return panel.dropna(subset=["price_change_raw", "lagged_price_change_raw"]).copy()


def reconstruct_one_market(
    trade_rows: pd.DataFrame,
    subject_rows: pd.DataFrame,
    market_meta: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct package-level and permit-unit-level transaction data.

    Supply-side eligibility:
      seller cost is retained only when the sold permit ordinal was part of the
      seller's initial endowment (ordinal <= initial permits).

    Demand-side eligibility:
      buyer value is retained only when the purchased permit ordinal was outside
      the buyer's initial endowment (ordinal > initial permits).

    Holdings are still updated through every actual trade, including excluded
    observations, so later permit ordinals are reconstructed correctly.
    """
    ordered_subjects = subject_rows.sort_values("id_in_group").copy()

    cap_values = ordered_subjects["group_cap_total"].dropna().unique()
    if len(cap_values) != 1:
        raise ValueError("同一 market-period 的 group_cap_total 不唯一。")
    cap_total = int(round(float(cap_values[0])))

    initial_holdings = {
        int(round(row["id_in_group"])): int(round(row["permits"]))
        for _, row in ordered_subjects.iterrows()
    }
    if len(initial_holdings) != len(ordered_subjects):
        raise ValueError("同一 market-period 的 id_in_group 重複。")
    holdings = initial_holdings.copy()
    final_holdings = {
        int(round(row["id_in_group"])): int(round(row["current_permits"]))
        for _, row in ordered_subjects.iterrows()
    }
    if sum(initial_holdings.values()) != cap_total:
        raise ValueError("初始 permits 加總不等於 group_cap_total。")
    if sum(final_holdings.values()) != cap_total:
        raise ValueError("交易後 current_permits 加總不等於 group_cap_total。")
    frontiers = {
        int(round(row["id_in_group"])): production_profit_frontier(row, cap_total)
        for _, row in ordered_subjects.iterrows()
    }
    marginal_schedules = {
        int(round(row["id_in_group"])): permit_marginal_schedule(
            row,
            cap_total,
            frontiers[int(round(row["id_in_group"]))],
        )
        for _, row in ordered_subjects.iterrows()
    }

    ordered_trades = trade_rows.sort_values(
        ["trade_index", "seconds"],
        kind="mergesort",
    ).copy()

    package_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for _, trade in ordered_trades.iterrows():
        buyer_id = int(round(trade["buyer_id"]))
        seller_id = int(round(trade["seller_id"]))
        quantity = int(round(trade["quantity"]))

        if buyer_id not in holdings or seller_id not in holdings:
            raise ValueError(
                f"交易者 ID 無法在 subj_lv_panel 找到：buyer={buyer_id}, "
                f"seller={seller_id}"
            )

        buyer_h_before = holdings[buyer_id]
        seller_h_before = holdings[seller_id]
        buyer_unit_values: list[float] = []
        seller_unit_costs: list[float] = []

        for unit_number in range(1, quantity + 1):
            buyer_h = holdings[buyer_id]
            seller_h = holdings[seller_id]

            if buyer_h + 1 > cap_total:
                raise ValueError(
                    f"買方持有量超過市場總量：{trade['session_code']} "
                    f"round_raw={trade['round_raw']} trade_index={trade['trade_index']}"
                )
            if seller_h <= 0:
                raise ValueError(
                    f"賣方碳權不足：{trade['session_code']} "
                    f"round_raw={trade['round_raw']} trade_index={trade['trade_index']}"
                )

            # Permit ordinal in the participant-specific endowment/value schedule.
            buyer_ordinal = buyer_h + 1
            seller_ordinal = seller_h

            buyer_value_raw = float(marginal_schedules[buyer_id][buyer_ordinal])
            seller_cost_raw = float(marginal_schedules[seller_id][seller_ordinal])

            # Requested induced-value/cost analogue:
            # - buyers: only units NOT initially held
            # - sellers: only units initially held
            buyer_eligible = buyer_ordinal > initial_holdings[buyer_id]
            seller_eligible = seller_ordinal <= initial_holdings[seller_id]

            buyer_value = buyer_value_raw if buyer_eligible else np.nan
            seller_cost = seller_cost_raw if seller_eligible else np.nan

            buyer_unit_values.append(buyer_value)
            seller_unit_costs.append(seller_cost)

            unit_rows.append(
                {
                    "session_code": trade["session_code"],
                    "round_raw": int(round(trade["round_raw"])),
                    "round": int(round(market_meta["round"])),
                    "first": int(round(market_meta["first"])),
                    "is_equal": int(round(market_meta["is_equal"])),
                    "allocation_method": market_meta["allocation_method"],
                    "trade_index": int(round(trade["trade_index"])),
                    "seconds": float(trade["seconds"]),
                    "unit_number_within_package": unit_number,
                    "package_quantity": quantity,
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "price": float(trade["price"]),
                    "eq_price": float(trade["eq_price"]),
                    "buyer_permit_value": buyer_value,
                    "seller_permit_cost": seller_cost,
                    "buyer_permit_value_raw": buyer_value_raw,
                    "seller_permit_cost_raw": seller_cost_raw,
                    "buyer_value_eligible": bool(buyer_eligible),
                    "seller_cost_eligible": bool(seller_eligible),
                    "buyer_permit_ordinal": buyer_ordinal,
                    "seller_permit_ordinal": seller_ordinal,
                    "buyer_initial_permits": initial_holdings[buyer_id],
                    "seller_initial_permits": initial_holdings[seller_id],
                    "buyer_holding_before_unit": buyer_h,
                    "seller_holding_before_unit": seller_h,
                    # Every unit in one package has the same observed transaction
                    # order. Spearman therefore treats them as tied ranks.
                    "transaction_order": int(round(trade["trade_index"])),
                }
            )

            holdings[buyer_id] += 1
            holdings[seller_id] -= 1

        package_rows.append(
            {
                "session_code": trade["session_code"],
                "round_raw": int(round(trade["round_raw"])),
                "round": int(round(market_meta["round"])),
                "first": int(round(market_meta["first"])),
                "is_equal": int(round(market_meta["is_equal"])),
                "allocation_method": market_meta["allocation_method"],
                "trade_index": int(round(trade["trade_index"])),
                "transaction_order": int(round(trade["trade_index"])),
                "seconds": float(trade["seconds"]),
                "buyer_id": buyer_id,
                "seller_id": seller_id,
                "price": float(trade["price"]),
                "quantity": quantity,
                "eq_price": float(trade["eq_price"]),
                "buyer_holding_before_package": buyer_h_before,
                "seller_holding_before_package": seller_h_before,
                # Average ONLY the eligible unit values/costs in the package.
                # A package with no eligible unit on one side receives NaN for
                # that side and is automatically omitted from that Spearman rho.
                "buyer_permit_value_avg": finite_mean(buyer_unit_values),
                "seller_permit_cost_avg": finite_mean(seller_unit_costs),
                "buyer_permit_value_total": finite_sum(buyer_unit_values),
                "seller_permit_cost_total": finite_sum(seller_unit_costs),
                "buyer_n_eligible_units": int(np.isfinite(buyer_unit_values).sum()),
                "seller_n_eligible_units": int(np.isfinite(seller_unit_costs).sum()),
            }
        )

    reconstruction_ok = all(
        holdings[player_id] == final_holding
        for player_id, final_holding in final_holdings.items()
    )

    packages = pd.DataFrame(package_rows).sort_values("trade_index").reset_index(drop=True)
    units = pd.DataFrame(unit_rows).sort_values(
        ["trade_index", "unit_number_within_package"]
    ).reset_index(drop=True)

    eq_values = packages["eq_price"].dropna().unique()
    if len(eq_values) != 1 or float(eq_values[0]) <= 0:
        raise ValueError("同一 market-period 必須有唯一且大於 0 的 eq_price。")
    market_eq_price = float(eq_values[0])
    # All units enter the price sequence, including those without eligible
    # buyer/seller values. Never remove zeros or join across market boundaries.
    package_pairs = add_price_changes(packages, market_eq_price, "package")
    unit_pairs = add_price_changes(units, market_eq_price, "permit_unit")
    price_pairs = pd.concat([package_pairs, unit_pairs], ignore_index=True)

    summary = {
        "session_code": trade_rows["session_code"].iloc[0],
        "round_raw": int(round(trade_rows["round_raw"].iloc[0])),
        "round": int(round(market_meta["round"])),
        "first": int(round(market_meta["first"])),
        "is_equal": int(round(market_meta["is_equal"])),
        "allocation_method": market_meta["allocation_method"],
        "n_transaction_packages": int(len(packages)),
        "n_permit_units": int(len(units)),
        "n_buyer_eligible_units": int(units["buyer_permit_value"].notna().sum()),
        "n_seller_eligible_units": int(units["seller_permit_cost"].notna().sum()),
        "observed_total_quantity": int(packages["quantity"].sum()),
        "reconstruction_ok": bool(reconstruction_ok),
        # One transaction package = one observation in the package version.
        "rho_buyer_package": safe_spearman(
            packages["buyer_permit_value_avg"],
            packages["transaction_order"],
        ),
        "rho_seller_package": safe_spearman(
            packages["seller_permit_cost_avg"],
            packages["transaction_order"],
        ),
        # Every eligible permit unit is included; same-package units share rank.
        "rho_buyer_unit": safe_spearman(
            units["buyer_permit_value"],
            units["transaction_order"],
        ),
        "rho_seller_unit": safe_spearman(
            units["seller_permit_cost"],
            units["transaction_order"],
        ),
        "rho_price_standardized": safe_pearson(
            package_pairs["price_change_standardized"],
            package_pairs["lagged_price_change_standardized"],
        ),
        "rho_price_raw": safe_pearson(
            package_pairs["price_change_raw"],
            package_pairs["lagged_price_change_raw"],
        ),
    }

    # Dividing both variables in one market by the same positive constant does
    # not change Pearson correlation. Keep the old name as a compatibility
    # alias and reject unexpected numerical disagreement.
    summary["rho_price"] = summary["rho_price_standardized"]
    if (
        np.isfinite(summary["rho_price_standardized"])
        and np.isfinite(summary["rho_price_raw"])
        and not np.isclose(
            summary["rho_price_standardized"],
            summary["rho_price_raw"],
            atol=1e-12,
            rtol=0,
        )
    ):
        raise RuntimeError(
            "同一市場的 raw 與 standardized price-change correlation 不一致。"
        )

    for level, pairs in [("package", package_pairs), ("unit", unit_pairs)]:
        for mode in ("raw", "standardized"):
            summary[f"rho_price_{level}_{mode}"] = safe_pearson(
                pairs[f"price_change_{mode}"], pairs[f"lagged_price_change_{mode}"]
            )
        summary[f"n_price_pairs_{level}"] = len(pairs)
        summary[f"n_zero_changes_{level}"] = int(
            (packages if level == "package" else units)["price_change_raw"].eq(0).sum()
        )

    return packages, units, price_pairs, summary


# ============================================================
# 3. Build panels
# ============================================================

def build_analysis_data(
    trades: pd.DataFrame,
    subj: pd.DataFrame,
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicate_group = group.groupby(MARKET_KEYS).size()
    duplicate_group = duplicate_group.loc[duplicate_group.gt(1)]
    if not duplicate_group.empty:
        raise ValueError(
            "group_lv_panel.csv 的 session_code x round_raw 不是唯一：\n"
            + duplicate_group.to_string()
        )

    group_index = group.set_index(MARKET_KEYS)

    package_frames: list[pd.DataFrame] = []
    unit_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    market_rows: list[dict[str, Any]] = []

    for (session_code, round_raw), trade_rows in trades.groupby(MARKET_KEYS, sort=True):
        subject_rows = subj.loc[
            subj["session_code"].eq(session_code)
            & subj["round_raw"].eq(round_raw)
        ].copy()
        if subject_rows.empty:
            raise ValueError(
                f"subj_lv_panel 缺少 market-period：{session_code}, {round_raw}"
            )

        try:
            market_meta = group_index.loc[(session_code, round_raw)]
        except KeyError as exc:
            raise ValueError(
                f"group_lv_panel 缺少 market-period：{session_code}, {round_raw}"
            ) from exc
        if isinstance(market_meta, pd.DataFrame):
            market_meta = market_meta.iloc[0]

        packages, units, price_pairs, summary = reconstruct_one_market(
            trade_rows,
            subject_rows,
            market_meta,
        )

        if summary["n_transaction_packages"] < MIN_TRANSACTION_PACKAGES:
            continue

        package_frames.append(packages)
        unit_frames.append(units)
        pair_frames.append(price_pairs)
        market_rows.append(summary)

    if not market_rows:
        raise ValueError("沒有符合至少四筆交易的 market-period。")

    market_panel = pd.DataFrame(market_rows).sort_values(MARKET_KEYS).reset_index(drop=True)
    package_panel = pd.concat(package_frames, ignore_index=True).sort_values(
        MARKET_KEYS + ["trade_index"]
    ).reset_index(drop=True)
    unit_panel = pd.concat(unit_frames, ignore_index=True).sort_values(
        MARKET_KEYS + ["trade_index", "unit_number_within_package"]
    ).reset_index(drop=True)
    price_pairs = pd.concat(pair_frames, ignore_index=True).sort_values(
        MARKET_KEYS + ["trade_index"]
    ).reset_index(drop=True)

    # Paper-style early/late split using the actual experiment sequence.
    # IMPORTANT: use round_raw, not the within-allocation `round` variable.
    market_panel["period_group"] = np.select(
        [
            market_panel["round_raw"].between(1, 3),
            market_panel["round_raw"].between(4, 6),
        ],
        [EARLY_LABEL, LATE_LABEL],
        default="Outside comparison",
    )

    return market_panel, package_panel, unit_panel, price_pairs


# ============================================================
# 4. Teacher-confirmed CTAT ZI-C v2: two-sided quotes and resale
# ============================================================

def _zi_integer(value: Any, name: str) -> int:
    x = float(value)
    if not np.isfinite(x) or not np.isclose(x, round(x), atol=1e-9, rtol=0):
        raise ValueError(f"ZI {name} 必須是有限整數；讀到 {value!r}。")
    return int(round(x))


def _zi_optimal_holdings(
    players: dict[int, dict[str, Any]], cap: int,
    frontier_key: str,
) -> tuple[dict[int, int], float, int]:
    """Maximize a supplied holding frontier subject to the fixed permit total.

    Among value ties, minimize total absolute movement from the experimental
    initial allocation. This target is an audit benchmark; the ZI mechanism is
    stopped by its pairwise induced-value equilibrium condition, not by forcing
    a particular tie-broken target allocation.
    """
    ids = sorted(players)
    profits = np.full(cap + 1, -np.inf)
    distances = np.full(cap + 1, np.iinfo(np.int64).max // 4, dtype=np.int64)
    profits[0], distances[0] = 0.0, 0
    choices: list[np.ndarray] = []
    for pid in ids:
        frontier = np.asarray(players[pid][frontier_key], dtype=float)
        initial = players[pid]["initial_holding"]
        new_profits = np.full(cap + 1, -np.inf)
        new_distances = np.full_like(distances, np.iinfo(np.int64).max // 4)
        choice = np.full(cap + 1, -1, dtype=int)
        for previous in np.flatnonzero(np.isfinite(profits)):
            hh = np.arange(cap - int(previous) + 1, dtype=int)
            totals = previous + hh
            candidate = profits[previous] + frontier[hh]
            distance = distances[previous] + np.abs(hh - initial)
            old = new_profits[totals]
            better = candidate > old + ZI_NUMERIC_TOLERANCE
            tied = np.isfinite(old) & (np.abs(candidate - old) <= ZI_NUMERIC_TOLERANCE)
            take = better | (tied & (distance < new_distances[totals]))
            selected = totals[take]
            new_profits[selected] = candidate[take]
            new_distances[selected] = distance[take]
            choice[selected] = hh[take]
        profits, distances = new_profits, new_distances
        choices.append(choice)
    target: dict[int, int] = {}
    remaining = cap
    for idx in range(len(ids) - 1, -1, -1):
        holding = int(choices[idx][remaining])
        if holding < 0:
            raise RuntimeError(f"ZI {frontier_key} 配置回溯失敗。")
        target[ids[idx]] = holding
        remaining -= holding
    if remaining != 0 or sum(target.values()) != cap:
        raise RuntimeError(f"ZI {frontier_key} 配置不符合總碳權限制。")
    return target, float(profits[cap]), int(distances[cap])


def build_zi_market_spec(subject_rows: pd.DataFrame) -> dict[str, Any]:
    """Build a fresh-market specification from CTAT experimental primitives.

    The production-frontier bundle gains are converted to average per-permit
    values by ``permit_marginal_schedule``. Sorting those values from high to low
    creates a concave induced holding frontier. At a current holding h, unit h+1
    is the participant's priority purchase and unit h is the priority sale. This
    makes both directions and resale depend on the current allocation rather than
    on fixed initial buy/sell queues. The original integer-production frontier is
    retained separately for sensitivity auditing.
    """
    rows = subject_rows.sort_values("id_in_group").copy()
    if len(rows) < 2:
        raise ValueError("ZI 市場設定至少需要兩位交易者。")
    keys = rows[MARKET_KEYS].drop_duplicates()
    if len(keys) != 1:
        raise ValueError("build_zi_market_spec 一次只能接受一個 session × round_raw。")
    key = keys.iloc[0]
    round_raw = _zi_integer(key["round_raw"], "round_raw")
    if not FIGURE_ROUND_RAW_MIN <= round_raw <= FIGURE_ROUND_RAW_MAX:
        raise ValueError("ZI 抽樣市場必須在 round_raw 1–6。")
    caps = rows["group_cap_total"].unique()
    if len(caps) != 1:
        raise ValueError("同一 ZI 市場 group_cap_total 不唯一或缺值。")
    cap = _zi_integer(caps[0], "group_cap_total")
    if cap < 0:
        raise ValueError("ZI group_cap_total 不可為負。")
    players: dict[int, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        pid = _zi_integer(row["id_in_group"], "id_in_group")
        initial = _zi_integer(row["permits"], "permits")
        b = _zi_integer(row["carbon_emission_per_unit"], "carbon_emission_per_unit")
        max_q = _zi_integer(row["max_production"], "max_production")
        if pid in players or not 0 <= initial <= cap or b < 1 or max_q < 0:
            raise ValueError(f"ZI 初始資料不合法：id={pid}, permits={initial}, b={b}。")
        frontier = production_profit_frontier(row, cap)
        raw_schedule = permit_marginal_schedule(row, cap, frontier)
        if not np.isfinite(frontier).all() or not np.isfinite(raw_schedule).all():
            raise ValueError(f"ZI 價值函數有非有限數值：id={pid}。")
        if (raw_schedule < -ZI_NUMERIC_TOLERANCE).any():
            raise ValueError(f"ZI 平均邊際價值出現負值：id={pid}。")
        raw_schedule = np.maximum(raw_schedule, 0.0)
        # Include all cap positions. Non-productive permit positions carry value 0
        # and can therefore be reallocated to participants with positive values.
        marginal_values = np.zeros(cap + 1, dtype=float)
        marginal_values[1:] = np.sort(raw_schedule[1:])[::-1]
        induced_frontier = np.concatenate(
            [np.array([0.0]), np.cumsum(marginal_values[1:])]
        )
        maximum = min(cap, b * max_q)
        players[pid] = {
            "initial_holding": initial, "max_productive_ordinal": maximum,
            "frontier": frontier, "raw_schedule": raw_schedule,
            "marginal_values": marginal_values,
            "induced_frontier": induced_frontier,
            "marginal_values_reordered": not np.array_equal(
                marginal_values[1:], raw_schedule[1:]
            ),
        }
    if sum(p["initial_holding"] for p in players.values()) != cap:
        raise ValueError(f"ZI 初始 permits 加總不等於 group_cap_total：{tuple(key)}。")
    induced_targets, induced_optimum, induced_movement = _zi_optimal_holdings(
        players, cap, "induced_frontier"
    )
    physical_targets, physical_optimum, physical_movement = _zi_optimal_holdings(
        players, cap, "frontier"
    )
    for pid, player in players.items():
        induced_delta = induced_targets[pid] - player["initial_holding"]
        physical_delta = physical_targets[pid] - player["initial_holding"]
        player["equilibrium_target_holding"] = induced_targets[pid]
        player["equilibrium_net_purchase"] = induced_delta
        player["initial_equilibrium_direction"] = (
            "net_buyer" if induced_delta > 0
            else "net_seller" if induced_delta < 0
            else "no_net_change"
        )
        player["physical_target_holding"] = physical_targets[pid]
        player["physical_net_purchase"] = physical_delta
    return {
        "session_code": str(key["session_code"]), "round_raw": round_raw,
        "players": players, "cap_total": cap, "price_ceiling": ZI_ASK_CEILING,
        "induced_optimal_total_value": induced_optimum,
        "induced_minimum_net_quantity": induced_movement / 2.0,
        "physical_optimal_production_profit": physical_optimum,
        "physical_minimum_net_quantity": physical_movement / 2.0,
        "equilibrium_allocation_source": (
            "descending_average_permit_values_DP_min_L1_tie"
        ),
        "unit_model": "current_holding_priority_units_two_sided_with_resale",
    }


def _zi_options(
    players: dict[int, dict[str, Any]],
    state: dict[int, dict[str, float | int]],
    cap: int,
    ceiling: float,
) -> tuple[
    dict[int, tuple[int, float]],
    dict[int, tuple[int, float]],
    dict[int, tuple[int, float, float]],
    dict[int, tuple[int, float, float]],
]:
    """Return economic and quote-feasible one-unit options at current holdings."""
    economic_buys: dict[int, tuple[int, float]] = {}
    economic_sells: dict[int, tuple[int, float]] = {}
    quote_buys: dict[int, tuple[int, float, float]] = {}
    quote_sells: dict[int, tuple[int, float, float]] = {}
    for pid, player in players.items():
        holding = int(state[pid]["holding"])
        cash = float(state[pid]["cash"])
        marginal_values = player["marginal_values"]
        if holding < cap:
            ordinal = holding + 1
            value = float(marginal_values[ordinal])
            if value > ZI_NUMERIC_TOLERANCE:
                economic_buys[pid] = (ordinal, value)
                upper = min(value, cash)
                if upper > 0.0:
                    quote_buys[pid] = (ordinal, value, upper)
        if holding > 0:
            ordinal = holding
            cost = float(marginal_values[ordinal])
            economic_sells[pid] = (ordinal, cost)
            strict_lower = float(np.nextafter(cost, np.inf))
            if strict_lower <= ceiling:
                quote_sells[pid] = (ordinal, cost, strict_lower)
    return economic_buys, economic_sells, quote_buys, quote_sells


def _zi_positive_pairs(
    buys: dict[int, tuple[int, float]],
    sells: dict[int, tuple[int, float]],
) -> list[tuple[int, int, float]]:
    return [
        (buyer, seller, value - cost)
        for buyer, (_, value) in buys.items()
        for seller, (_, cost) in sells.items()
        if buyer != seller and value > cost + ZI_NUMERIC_TOLERANCE
    ]


def _zi_executable_pairs(
    buys: dict[int, tuple[int, float, float]],
    sells: dict[int, tuple[int, float, float]],
) -> list[tuple[int, int, float]]:
    # A positive-width overlap is necessary because bids are strictly below their
    # upper bound and asks are strictly above the seller's marginal value.
    return [
        (buyer, seller, upper - cost)
        for buyer, (_, value, upper) in buys.items()
        for seller, (_, cost, _) in sells.items()
        if buyer != seller
        and value > cost + ZI_NUMERIC_TOLERANCE
        and upper > cost + ZI_NUMERIC_TOLERANCE
    ]


def simulate_one_zi_market(
    spec: dict[str, Any], rng: np.random.Generator,
    keep_details: bool = False,
) -> dict[str, Any]:
    """Simulate one independent v2 market without an event/time cutoff.

    Each activation chooses one currently quote-capable participant uniformly.
    That participant submits both a bid and an ask when the corresponding current
    holding-state units exist. Both incoming quotes are compared with the book as
    it stood before the activation, so they can never match each other. If both
    sides cross standing quotes, both transactions execute atomically at their
    respective pre-existing prices; only then is the entire book cleared.
    """
    players = spec["players"]
    ids = sorted(players)
    cap, ceiling = int(spec["cap_total"]), float(spec["price_ceiling"])
    if ceiling != ZI_ASK_CEILING:
        raise ValueError("ZI v2 的賣價上限固定為 1000。")
    state = {
        pid: {
            "holding": int(players[pid]["initial_holding"]),
            "cash": float(ZI_INITIAL_CASH),
            "gross_bought": 0,
            "gross_sold": 0,
        }
        for pid in ids
    }
    tokens: dict[int, list[dict[str, Any]]] = {
        pid: [
            {
                "token_id": f"{pid}:{ordinal}",
                "origin_owner": pid,
                "transfer_count": 0,
            }
            for ordinal in range(1, int(players[pid]["initial_holding"]) + 1)
        ]
        for pid in ids
    }
    initial_total_cash = ZI_INITIAL_CASH * len(ids)
    prices: list[float] = []
    values: list[float] = []
    costs: list[float] = []
    trade_log: list[dict[str, Any]] = []
    quote_log: list[dict[str, Any]] = []
    counters = {
        "n_quote_events_zi": 0, "n_quote_submissions_zi": 0,
        "n_bid_submissions_zi": 0, "n_ask_submissions_zi": 0,
        "n_cash_constrained_buy_quotes_zi": 0,
        "n_ignored_nonimproving_quotes_zi": 0,
        "n_equal_price_quotes_retaining_earlier_zi": 0,
        "n_self_crosses_not_executed_zi": 0,
        "n_dual_cross_activations_zi": 0,
        "n_resale_trades_zi": 0,
        "n_self_trades_zi": 0,
        "n_true_frontier_buyer_losses_zi": 0, "n_true_frontier_seller_losses_zi": 0,
    }
    stop_reason = ""
    final_economic_pairs: list[tuple[int, int, float]] = []
    final_feasible_pairs: list[tuple[int, int, float]] = []
    max_book_bids = max_book_asks = 0

    while True:
        economic_buy, economic_sell, buys, sells = _zi_options(
            players, state, cap, ceiling
        )
        economic_pairs = _zi_positive_pairs(economic_buy, economic_sell)
        feasible_pairs = _zi_executable_pairs(buys, sells)
        if not feasible_pairs:
            final_economic_pairs, final_feasible_pairs = economic_pairs, feasible_pairs
            if not economic_pairs:
                stop_reason = "induced_value_equilibrium"
            else:
                cash_block = any(
                    buyer not in buys
                    or min(economic_buy[buyer][1], float(state[buyer]["cash"]))
                    <= economic_sell[seller][1] + ZI_NUMERIC_TOLERANCE
                    for buyer, seller, _ in economic_pairs
                )
                ask_block = any(seller not in sells for _, seller, _ in economic_pairs)
                if cash_block and ask_block:
                    stop_reason = "cash_and_ask_ceiling_block_equilibrium"
                elif cash_block:
                    stop_reason = "cash_blocks_equilibrium"
                else:
                    stop_reason = "ask_ceiling_blocks_equilibrium"
            break
        active = [pid for pid in ids if pid in buys or pid in sells]
        if not active:
            raise RuntimeError("存在可成交配對，卻沒有可喊價交易者。")
        best_bid: tuple[int, float, int] | None = None
        best_ask: tuple[int, float, int] | None = None
        # NO maximum quote events. This loop exits only through actual execution.
        while True:
            pid = active[int(rng.integers(len(active)))]
            counters["n_quote_events_zi"] += 1
            event = counters["n_quote_events_zi"]
            old_bid, old_ask = best_bid, best_ask
            bid_quote = ask_quote = np.nan
            bid_action = ask_action = "not_feasible"
            bid_candidate: dict[str, Any] | None = None
            ask_candidate: dict[str, Any] | None = None

            if pid in buys:
                _, value, upper = buys[pid]
                if upper < value:
                    counters["n_cash_constrained_buy_quotes_zi"] += 1
                bid_quote = float(rng.uniform(0.0, upper))
                # numpy documents an extremely rare rounding path to ``high``;
                # enforce the teacher-specified strict inequality explicitly.
                if bid_quote >= upper:
                    bid_quote = float(np.nextafter(upper, -np.inf))
                counters["n_quote_submissions_zi"] += 1
                counters["n_bid_submissions_zi"] += 1
                if old_ask is not None and bid_quote >= old_ask[1]:
                    if old_ask[0] != pid:
                        bid_candidate = {
                            "buyer": pid, "seller": old_ask[0],
                            "price": old_ask[1], "incoming_quote": bid_quote,
                            "incoming_side": "buy", "standing_side": "sell",
                            "standing_event": old_ask[2],
                        }
                        bid_action = "cross_standing_ask"
                    else:
                        counters["n_self_crosses_not_executed_zi"] += 1
                        bid_action = "self_cross_not_executed"
                else:
                    bid_action = "eligible_for_book_update"

            if pid in sells:
                _, _, strict_lower = sells[pid]
                ask_quote = (
                    strict_lower
                    if strict_lower == ceiling
                    else float(rng.uniform(strict_lower, ceiling))
                )
                counters["n_quote_submissions_zi"] += 1
                counters["n_ask_submissions_zi"] += 1
                if old_bid is not None and ask_quote <= old_bid[1]:
                    if old_bid[0] != pid:
                        ask_candidate = {
                            "buyer": old_bid[0], "seller": pid,
                            "price": old_bid[1], "incoming_quote": ask_quote,
                            "incoming_side": "sell", "standing_side": "buy",
                            "standing_event": old_bid[2],
                        }
                        ask_action = "cross_standing_bid"
                    else:
                        counters["n_self_crosses_not_executed_zi"] += 1
                        ask_action = "self_cross_not_executed"
                else:
                    ask_action = "eligible_for_book_update"

            # Candidate order is stable for the audit only: incoming bid leg,
            # then incoming ask leg. With two candidates, both are economically
            # simultaneous and are settled from the same pre-event state.
            executed_trades = [
                candidate
                for candidate in (bid_candidate, ask_candidate)
                if candidate is not None
            ]
            if len(executed_trades) == 2:
                counters["n_dual_cross_activations_zi"] += 1

            if not executed_trades:
                if pid in buys:
                    if old_bid is None or bid_quote > old_bid[1]:
                        best_bid = (pid, float(bid_quote), event)
                        bid_action = "set_best_bid"
                    elif bid_quote == old_bid[1]:
                        counters["n_equal_price_quotes_retaining_earlier_zi"] += 1
                        counters["n_ignored_nonimproving_quotes_zi"] += 1
                        bid_action = "equal_bid_keep_earlier"
                    else:
                        counters["n_ignored_nonimproving_quotes_zi"] += 1
                        bid_action = "ignore_nonimproving_bid"
                if pid in sells:
                    if old_ask is None or ask_quote < old_ask[1]:
                        best_ask = (pid, float(ask_quote), event)
                        ask_action = "set_best_ask"
                    elif ask_quote == old_ask[1]:
                        counters["n_equal_price_quotes_retaining_earlier_zi"] += 1
                        counters["n_ignored_nonimproving_quotes_zi"] += 1
                        ask_action = "equal_ask_keep_earlier"
                    else:
                        counters["n_ignored_nonimproving_quotes_zi"] += 1
                        ask_action = "ignore_nonimproving_ask"

            max_book_bids = max(max_book_bids, int(best_bid is not None))
            max_book_asks = max(max_book_asks, int(best_ask is not None))
            if keep_details and ZI_KEEP_QUOTE_LOG:
                quote_log.append({
                    "quote_event": event, "trader_id": pid,
                    "bid_quote": bid_quote, "ask_quote": ask_quote,
                    "bid_action": bid_action, "ask_action": ask_action,
                    "dual_cross": len(executed_trades) == 2,
                    "executed_trade_count": len(executed_trades),
                    "best_bid_after": (
                        np.nan if executed_trades
                        else best_bid[1] if best_bid else np.nan
                    ),
                    "best_ask_after": (
                        np.nan if executed_trades
                        else best_ask[1] if best_ask else np.nan
                    ),
                })
            if ZI_PROGRESS_EVERY_QUOTES and event % ZI_PROGRESS_EVERY_QUOTES == 0:
                print(f"[ZI running] {spec['session_code']} round={spec['round_raw']} "
                      f"quotes={event:,}, trades={len(prices)} (no event cutoff)", flush=True)
            if not executed_trades:
                continue

            # Prepare every leg from one common pre-event snapshot. In a dual
            # cross, the selected participant is buyer in one leg and seller in
            # the other, but neither leg may use the other leg's incoming permit,
            # cash, or updated marginal value. Both contracts are settled together.
            holding_before = {trader: int(state[trader]["holding"]) for trader in ids}
            cash_before = {trader: float(state[trader]["cash"]) for trader in ids}
            holding_delta = {trader: 0 for trader in ids}
            cash_delta = {trader: 0.0 for trader in ids}
            prepared_trades: list[dict[str, Any]] = []
            for within_event_order, executed_trade in enumerate(executed_trades, 1):
                buyer = int(executed_trade["buyer"])
                seller = int(executed_trade["seller"])
                price = float(executed_trade["price"])
                standing_event = int(executed_trade["standing_event"])
                if buyer == seller or not standing_event < event:
                    raise RuntimeError("ZI 自成交或成交順序錯誤。")
                buy_ordinal, buyer_value, buyer_upper = buys[buyer]
                sell_ordinal, seller_cost, _ = sells[seller]
                if (
                    price > buyer_upper + ZI_NUMERIC_TOLERANCE
                    or price < seller_cost - ZI_NUMERIC_TOLERANCE
                ):
                    raise RuntimeError("ZI 成交不符合報價現金／單位價值限制。")
                if buyer_value <= seller_cost + ZI_NUMERIC_TOLERANCE:
                    raise RuntimeError("ZI 成交產生負的 pre-event induced-unit surplus。")
                bh, sh = holding_before[buyer], holding_before[seller]
                if sh <= 0 or bh >= cap:
                    raise RuntimeError("ZI 成交前持有量不合法。")
                buyer_true_gain = float(
                    players[buyer]["frontier"][bh + 1]
                    - players[buyer]["frontier"][bh]
                )
                seller_true_cost = float(
                    players[seller]["frontier"][sh]
                    - players[seller]["frontier"][sh - 1]
                )
                prepared_trades.append({
                    **executed_trade,
                    "within_event_order": within_event_order,
                    "buyer": buyer, "seller": seller, "price": price,
                    "buy_ordinal": buy_ordinal, "buyer_value": buyer_value,
                    "sell_ordinal": sell_ordinal, "seller_cost": seller_cost,
                    "buyer_holding_before": bh, "seller_holding_before": sh,
                    "buyer_true_gain": buyer_true_gain,
                    "seller_true_cost": seller_true_cost,
                })
                holding_delta[buyer] += 1
                holding_delta[seller] -= 1
                cash_delta[buyer] -= price
                cash_delta[seller] += price

            if len({trade["seller"] for trade in prepared_trades}) != len(prepared_trades):
                raise RuntimeError("同一原子報價事件要求同一賣方交付超過一單位。")
            final_event_holdings = {
                trader: holding_before[trader] + holding_delta[trader]
                for trader in ids
            }
            final_event_cash = {
                trader: cash_before[trader] + cash_delta[trader]
                for trader in ids
            }
            if any(not 0 <= final_event_holdings[trader] <= cap for trader in ids):
                raise RuntimeError("ZI 原子雙成交後持有量超出可行範圍。")
            if min(final_event_cash.values()) < -ZI_NUMERIC_TOLERANCE:
                raise RuntimeError("ZI 原子雙成交後買方現金為負。")

            # Remove every sold token before delivering any purchased token. This
            # prevents a simultaneous second leg from reselling a token received
            # only moments earlier in the same atomic event.
            for trade in prepared_trades:
                token = tokens[trade["seller"]].pop()
                trade["token"] = token
                trade["sold_unit_previously_acquired"] = (
                    token["origin_owner"] != trade["seller"]
                    or token["transfer_count"] > 0
                )
                token["transfer_count"] += 1
            for trade in prepared_trades:
                tokens[trade["buyer"]].append(trade["token"])
                state[trade["buyer"]]["gross_bought"] += 1
                state[trade["seller"]]["gross_sold"] += 1
                counters["n_resale_trades_zi"] += int(
                    trade["sold_unit_previously_acquired"]
                )
            for trader in ids:
                state[trader]["holding"] = final_event_holdings[trader]
                state[trader]["cash"] = final_event_cash[trader]

            if min(state[p]["cash"] for p in ids) < -ZI_NUMERIC_TOLERANCE:
                raise RuntimeError("ZI 買方現金為負。")
            if sum(int(state[p]["holding"]) for p in ids) != cap:
                raise RuntimeError("ZI 碳權不守恆。")
            if any(len(tokens[p]) != int(state[p]["holding"]) for p in ids):
                raise RuntimeError("ZI token 與持有量不一致。")
            if abs(sum(state[p]["cash"] for p in ids) - initial_total_cash) > 1e-7:
                raise RuntimeError("ZI 現金不守恆。")

            event_value_before = sum(
                players[trader]["induced_frontier"][holding_before[trader]]
                for trader in ids
            )
            event_value_after = sum(
                players[trader]["induced_frontier"][int(state[trader]["holding"])]
                for trader in ids
            )
            if event_value_after <= event_value_before + ZI_NUMERIC_TOLERANCE:
                raise RuntimeError("ZI 成交事件未提高 induced total value。")

            best_bid = best_ask = None
            for trade in prepared_trades:
                buyer, seller, price = (
                    trade["buyer"], trade["seller"], trade["price"]
                )
                buyer_value, seller_cost = (
                    trade["buyer_value"], trade["seller_cost"]
                )
                buyer_true_gain, seller_true_cost = (
                    trade["buyer_true_gain"], trade["seller_true_cost"]
                )
                token = trade["token"]
                prices.append(price)
                values.append(buyer_value)
                costs.append(seller_cost)
                counters["n_true_frontier_buyer_losses_zi"] += int(
                    buyer_true_gain - price < -ZI_NUMERIC_TOLERANCE
                )
                counters["n_true_frontier_seller_losses_zi"] += int(
                    price - seller_true_cost < -ZI_NUMERIC_TOLERANCE
                )
                if not keep_details:
                    continue
                trade_log.append({
                    "transaction_order": len(prices), "quote_event": event,
                    "within_quote_event_order": trade["within_event_order"],
                    "atomic_event_trade_count": len(prepared_trades),
                    "buyer_id": buyer, "seller_id": seller, "quantity": 1,
                    "price": price, "incoming_quote": trade["incoming_quote"],
                    "incoming_side": trade["incoming_side"],
                    "standing_side": trade["standing_side"],
                    "standing_quote_event": trade["standing_event"],
                    "buyer_value": buyer_value, "seller_cost": seller_cost,
                    "buyer_unit_ordinal_before_trade": trade["buy_ordinal"],
                    "seller_unit_ordinal_before_trade": trade["sell_ordinal"],
                    "buyer_holding_before": trade["buyer_holding_before"],
                    "seller_holding_before": trade["seller_holding_before"],
                    "buyer_holding_after": state[buyer]["holding"],
                    "seller_holding_after": state[seller]["holding"],
                    "buyer_cash_before": cash_before[buyer],
                    "seller_cash_before": cash_before[seller],
                    "buyer_cash_after": state[buyer]["cash"], "seller_cash_after": state[seller]["cash"],
                    "buyer_induced_unit_profit": buyer_value - price,
                    "seller_induced_unit_profit": price - seller_cost,
                    "buyer_true_frontier_gain": buyer_true_gain,
                    "seller_true_frontier_cost": seller_true_cost,
                    "buyer_true_frontier_profit": buyer_true_gain - price,
                    "seller_true_frontier_profit": price - seller_true_cost,
                    "token_id": token["token_id"],
                    "token_origin_owner": token["origin_owner"],
                    "token_transfer_count_after": token["transfer_count"],
                    "sold_unit_is_initial": not trade["sold_unit_previously_acquired"],
                    "sold_unit_previously_acquired": trade["sold_unit_previously_acquired"],
                    "n_live_bids_after_trade": 0, "n_live_asks_after_trade": 0,
                })
            break

    order = np.arange(1, len(prices) + 1, dtype=float)
    changes = pd.Series(prices, dtype=float).diff()
    rhos = {
        "rho_buyer_zi": safe_spearman(values, order),
        "rho_seller_zi": safe_spearman(costs, order),
        "rho_price_zi": safe_pearson(changes, changes.shift(1)),
    }
    if len(prices) < MIN_TRANSACTION_PACKAGES:
        rhos = {name: np.nan for name in rhos}
    final_rows = []
    for pid in ids:
        p, st = players[pid], state[pid]
        final_holding = int(st["holding"])
        net = final_holding - p["initial_holding"]
        target_net = p["equilibrium_net_purchase"]
        final_rows.append({
            "id_in_group": pid,
            "initial_equilibrium_direction": p["initial_equilibrium_direction"],
            "initial_holding": p["initial_holding"],
            "equilibrium_target_holding": p["equilibrium_target_holding"],
            "equilibrium_target_net_purchase": target_net,
            "physical_target_holding": p["physical_target_holding"],
            "physical_target_net_purchase": p["physical_net_purchase"],
            "gross_bought": st["gross_bought"], "gross_sold": st["gross_sold"],
            "net_purchase": net, "final_holding": final_holding,
            "n_initial_origin_tokens_final": sum(
                token["origin_owner"] == pid for token in tokens[pid]
            ),
            "n_acquired_origin_tokens_final": sum(
                token["origin_owner"] != pid for token in tokens[pid]
            ),
            "final_cash": st["cash"],
            "holding_minus_equilibrium_target": (
                final_holding - p["equilibrium_target_holding"]
            ),
            "holding_minus_physical_target": final_holding - p["physical_target_holding"],
        })
    final_frontier = sum(players[p]["frontier"][int(state[p]["holding"])] for p in ids)
    initial_frontier = sum(players[p]["frontier"][players[p]["initial_holding"]] for p in ids)
    final_induced = sum(players[p]["induced_frontier"][int(state[p]["holding"])] for p in ids)
    initial_induced = sum(players[p]["induced_frontier"][players[p]["initial_holding"]] for p in ids)
    equilibrium_reached = not final_economic_pairs
    realized_induced_gain = float(final_induced - initial_induced)
    pre_event_pair_surplus = float(sum(values) - sum(costs))
    if realized_induced_gain < -ZI_NUMERIC_TOLERANCE:
        raise RuntimeError("ZI 成交後 induced total value 反而下降。")
    if (
        counters["n_dual_cross_activations_zi"] == 0
        and abs(realized_induced_gain - pre_event_pair_surplus) > 1e-7
    ):
        raise RuntimeError("ZI 單筆成交價值加總與最終配置變化不一致。")
    induced_optimality_gap = spec["induced_optimal_total_value"] - final_induced
    if equilibrium_reached and induced_optimality_gap > 1e-7:
        raise RuntimeError("ZI 已宣告價值均衡，但仍未達 induced-value 全域最適配置。")
    result = {
        **rhos, **counters,
        "role_mode": ZI_ROLE_MODE, "n_trades_zi": len(prices),
        "dual_cross_rule": ZI_DUAL_CROSS_RULE,
        "equilibrium_reached": equilibrium_reached,
        "valid_for_95_region": (
            equilibrium_reached and all(np.isfinite(r) for r in rhos.values())
        ),
        "stop_reason": stop_reason, "quote_event_cutoff_used": False,
        "resale_allowed": True, "no_self_trade_violation": counters["n_self_trades_zi"] == 0,
        "max_live_bids": max_book_bids,
        "max_live_asks": max_book_asks,
        "remaining_economic_positive_pairs": len(final_economic_pairs),
        "remaining_executable_positive_pairs": len(final_feasible_pairs),
        "max_remaining_economic_surplus": max((x[2] for x in final_economic_pairs), default=0.0),
        "min_final_cash_zi": min(state[p]["cash"] for p in ids),
        "max_final_cash_zi": max(state[p]["cash"] for p in ids),
        "initial_total_cash_zi": initial_total_cash,
        "total_final_cash_zi": sum(state[p]["cash"] for p in ids),
        "cash_conservation_error_zi": sum(state[p]["cash"] for p in ids) - initial_total_cash,
        "permit_conservation_error_zi": sum(int(state[p]["holding"]) for p in ids) - cap,
        "initial_induced_total_value": initial_induced,
        "final_induced_total_value": final_induced,
        "induced_total_value_gain": final_induced - initial_induced,
        "induced_optimality_gap": induced_optimality_gap,
        "initial_physical_production_frontier": initial_frontier,
        "final_physical_production_frontier": final_frontier,
        "physical_frontier_gain": final_frontier - initial_frontier,
        "sum_induced_unit_surplus": realized_induced_gain,
        "sum_pre_event_trade_reference_surplus": pre_event_pair_surplus,
        "dual_cross_surplus_accounting_gap": (
            realized_induced_gain - pre_event_pair_surplus
        ),
        "total_absolute_holding_gap_from_equilibrium_target": sum(
            abs(r["holding_minus_equilibrium_target"]) for r in final_rows
        ),
        "n_buyer_eligible_trades_zi": len(prices),
        "n_seller_eligible_trades_zi": len(prices),
    }
    if keep_details:
        result["_trades"] = trade_log
        result["_players"] = final_rows
        result["_quotes"] = quote_log
    return result


_ZI_WORKER_SPECS: list[dict[str, Any]] | None = None


def _init_zi_worker(specs: list[dict[str, Any]]) -> None:
    global _ZI_WORKER_SPECS
    _ZI_WORKER_SPECS = specs


def _simulate_zi_task(
    specs: list[dict[str, Any]], task: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    simulation, spec_index = task
    spec = specs[spec_index]
    rng = np.random.default_rng(
        np.random.SeedSequence([ZI_RANDOM_SEED, 1, simulation])
    )
    result = simulate_one_zi_market(spec, rng, keep_details=True)
    prefix = {
        "simulation_id": simulation,
        "session_code": spec["session_code"],
        "round_raw": spec["round_raw"],
        "role_mode": ZI_ROLE_MODE,
    }
    trades = [{**prefix, **row} for row in result.pop("_trades")]
    final_players = [{**prefix, **row} for row in result.pop("_players")]
    quotes = [{**prefix, **row} for row in result.pop("_quotes")]
    outcome = {**prefix, "pool_index": spec_index, **result}
    return outcome, trades, final_players, quotes


def _zi_worker(
    task: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if _ZI_WORKER_SPECS is None:
        raise RuntimeError("ZI worker 尚未初始化市場設定。")
    return _simulate_zi_task(_ZI_WORKER_SPECS, task)


def simulate_zi_benchmark(
    subj: pd.DataFrame, group: pd.DataFrame,
    n_simulations: int = ZI_SIMULATIONS,
) -> pd.DataFrame:
    """Exactly N attempts sampled uniformly from ALL experimental round 1–6 keys.

    Never condition market selection on observed trade counts or valid simulated
    correlations. All N outcomes are saved, including zero-trade/undefined-rho
    outcomes. The existing ellipse code uses complete cases only at plotting time.
    Market sampling and each simulation have separate deterministic seeds. A fixed
    process pool parallelizes independent simulations only; within-market event
    order and economic rules are identical at every worker count.
    """
    if not isinstance(n_simulations, (int, np.integer)) or n_simulations < 1:
        raise ValueError("n_simulations 必須是正整數。")
    valid_subj = subj.loc[
        subj["app"].eq("Stage_CarbonTrading")
        & subj["round_raw"].between(FIGURE_ROUND_RAW_MIN, FIGURE_ROUND_RAW_MAX)
    ].copy()
    valid_group = group.loc[
        group["app"].eq("Stage_CarbonTrading")
        & group["round_raw"].between(FIGURE_ROUND_RAW_MIN, FIGURE_ROUND_RAW_MAX)
    ].copy()
    if valid_subj[MARKET_KEYS].isna().any().any() or valid_group[MARKET_KEYS].isna().any().any():
        raise ValueError("ZI 市場 key 有缺值；不可默默排除。")
    if valid_group.duplicated(MARKET_KEYS).any():
        raise ValueError("ZI group_lv_panel 同一 session × round_raw 重複。")
    subj_keys = set(map(tuple, valid_subj[MARKET_KEYS].drop_duplicates().to_numpy()))
    group_keys = set(map(tuple, valid_group[MARKET_KEYS].to_numpy()))
    if subj_keys != group_keys:
        raise ValueError(f"ZI 市場設定不完整：僅 subj={subj_keys-group_keys}；僅 group={group_keys-subj_keys}。")
    if not subj_keys:
        raise ValueError("沒有 round_raw 1–6 的 CTAT 市場設定。")
    specs = [build_zi_market_spec(rows) for _, rows in valid_subj.groupby(MARKET_KEYS, sort=True)]
    market_rng = np.random.default_rng(np.random.SeedSequence([ZI_RANDOM_SEED, 0]))
    sampled = market_rng.integers(0, len(specs), size=n_simulations)
    rows, trade_rows, player_rows, quote_rows = [], [], [], []
    pool_rows, value_rows, schedule_rows = [], [], []
    for index, spec in enumerate(specs):
        meta = {"session_code": spec["session_code"], "round_raw": spec["round_raw"]}
        pool_rows.append({
            **meta, "pool_index": index, "sampling_probability": 1.0 / len(specs),
            "sampled_count": int(np.count_nonzero(sampled == index)),
            "n_players": len(spec["players"]), "cap_total": spec["cap_total"],
            "induced_optimal_total_value": spec["induced_optimal_total_value"],
            "induced_minimum_net_quantity": spec["induced_minimum_net_quantity"],
            "physical_optimal_production_profit": spec["physical_optimal_production_profit"],
            "physical_minimum_net_quantity": spec["physical_minimum_net_quantity"],
            "equilibrium_allocation_source": spec["equilibrium_allocation_source"],
            "unit_model": spec["unit_model"],
        })
        for pid, p in spec["players"].items():
            value_rows.append({
                **meta, "id_in_group": pid,
                **{k: p[k] for k in (
                    "initial_holding", "equilibrium_target_holding",
                    "equilibrium_net_purchase", "initial_equilibrium_direction",
                    "physical_target_holding", "physical_net_purchase",
                    "marginal_values_reordered")},
                "can_submit_bids_when_holding_below_cap": True,
                "can_submit_asks_when_holding_above_zero": True,
                "resale_allowed": True,
            })
            for ordinal in range(1, spec["cap_total"] + 1):
                schedule_rows.append({
                    **meta, "id_in_group": pid, "holding_ordinal": ordinal,
                    "raw_average_permit_value": p["raw_schedule"][ordinal],
                    "priority_marginal_value": p["marginal_values"][ordinal],
                    "induced_total_value_at_holding": p["induced_frontier"][ordinal],
                    "physical_production_profit_at_holding": p["frontier"][ordinal],
                })

    tasks = [
        (simulation, int(sampled[simulation - 1]))
        for simulation in range(1, n_simulations + 1)
    ]
    n_workers = min(int(ZI_N_WORKERS), n_simulations)
    if n_workers == 1:
        iterator = (_simulate_zi_task(specs, task) for task in tasks)
    else:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_zi_worker,
            initargs=(specs,),
        )
        iterator = executor.map(_zi_worker, tasks, chunksize=ZI_PARALLEL_CHUNKSIZE)
    try:
        progress_every = max(1, min(100, n_simulations // 100 or 1))
        for completed, (outcome_row, trades, final_players, quotes) in enumerate(iterator, 1):
            rows.append(outcome_row)
            trade_rows.extend(trades)
            player_rows.extend(final_players)
            quote_rows.extend(quotes)
            if completed % progress_every == 0 or completed == n_simulations:
                print(
                    f"[ZI] {ZI_ROLE_MODE}: {completed:,}/{n_simulations:,} "
                    f"markets simulated with {n_workers} worker(s)",
                    flush=True,
                )
    finally:
        if n_workers != 1:
            executor.shutdown(wait=True)
    # All empirical input files remain unmodified. These are new simulation audits.
    audits = {
        "figure_2_zi_market_pool.csv": pd.DataFrame(pool_rows),
        "figure_2_zi_player_values.csv": pd.DataFrame(value_rows),
        "figure_2_zi_marginal_schedules.csv": pd.DataFrame(schedule_rows),
        "figure_2_zi_trade_audit.csv": pd.DataFrame(trade_rows),
        "figure_2_zi_final_player_audit.csv": pd.DataFrame(player_rows),
    }
    for filename, table in audits.items():
        if table.empty and filename.endswith("trade_audit.csv"):
            table = pd.DataFrame(columns=["simulation_id", "session_code", "round_raw", "buyer_id", "seller_id", "price", "quantity"])
        table.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    if ZI_KEEP_QUOTE_LOG:
        pd.DataFrame(quote_rows).to_csv(OUTPUT_DIR / "figure_2_zi_quote_audit.csv", index=False, encoding="utf-8-sig")
    outcome = pd.DataFrame(rows)
    outcome.to_csv(OUTPUT_DIR / "figure_2_zi_simulation_benchmark.csv", index=False, encoding="utf-8-sig")
    outcome["stop_reason"].value_counts(dropna=False).rename_axis("stop_reason").reset_index(name="n_simulations").to_csv(
        OUTPUT_DIR / "figure_2_zi_stop_reason_counts.csv", index=False, encoding="utf-8-sig")
    valid = int(outcome["valid_for_95_region"].sum())
    print(
        f"[ZI] Exactly {len(outcome):,} attempts; {valid:,} equilibrium complete-"
        "correlation draws. No replenishment/redraw."
    )
    if valid < 50:
        print("[ZI WARNING] Fewer than 50 complete draws: existing plotting rule omits the 95% ellipse.")
    return outcome


# ============================================================
# 5. Plotting helpers
# ============================================================

def plot_rank_histogram(
    ax: plt.Axes,
    market_panel: pd.DataFrame,
    rho_col: str,
    xlabel: str,
    panel_letter: str,
) -> None:
    # Each histogram observation is one market-period Spearman rho, as in the paper.
    early_rows = market_panel.loc[
        market_panel["period_group"].eq(EARLY_LABEL)
        & market_panel[rho_col].notna()
    ].copy()
    late_rows = market_panel.loc[
        market_panel["period_group"].eq(LATE_LABEL)
        & market_panel[rho_col].notna()
    ].copy()

    early = early_rows[rho_col]
    late = late_rows[rho_col]
    bins = np.linspace(-1, 1, 21)
    ax.hist(
        late,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=LATE_COLOR,
        label=LATE_LABEL,
    )
    ax.hist(
        early,
        bins=bins,
        density=True,
        alpha=0.55,
        color=EARLY_COLOR,
        edgecolor=EARLY_COLOR,
        label=EARLY_LABEL,
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylabel("Density")
    ax.set_xlabel(xlabel)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        -0.12,
        1.04,
        panel_letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
    )


def price_change_columns(mode: str) -> tuple[str, str]:
    if mode == "standardized":
        return "price_change_standardized", "lagged_price_change_standardized"
    if mode == "raw":
        return "price_change_raw", "lagged_price_change_raw"
    raise ValueError(
        f"未知的 price-change mode：{mode!r}；必須是 {PRICE_CHANGE_MODES}。"
    )


def fit_price_change_ols(
    price_pairs: pd.DataFrame,
    x_col: str,
    y_col: str,
):
    data = price_pairs.dropna(subset=[x_col, y_col]).copy()
    x = sm.add_constant(data[x_col], has_constant="add")
    groups = data["session_code"].astype(str) + "_" + data["round_raw"].astype(str)
    if groups.nunique() >= 2:
        result = sm.OLS(data[y_col], x).fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    else:
        result = sm.OLS(data[y_col], x).fit(cov_type="HC1")
    return data, result


def plot_price_change_panel(
    ax: plt.Axes,
    price_pairs: pd.DataFrame,
    mode: str,
    panel_letter: str = "c",
) -> dict[str, Any]:
    """Draw either standardized or raw ΔP_j using one fixed comparison sample."""
    x_col, y_col = price_change_columns(mode)
    all_data = price_pairs.dropna(
        subset=[
            "price_change_standardized",
            "lagged_price_change_standardized",
            "price_change_raw",
            "lagged_price_change_raw",
        ]
    ).copy()

    # The authors' public R code limits both standardized axes to [-4,4]. Use
    # that identical set of transaction pairs for the raw comparison too; this
    # isolates the effect of the denominator from the effect of sample trimming.
    selection_limit = PRICE_CHANGE_PLOT_LIMIT
    visible = all_data.loc[
        all_data["price_change_standardized"].between(
            -selection_limit,
            selection_limit,
        )
        & all_data["lagged_price_change_standardized"].between(
            -selection_limit,
            selection_limit,
        )
    ].copy()
    if visible.empty:
        raise ValueError("Figure 2c 在共同的標準化 [-4,4] 篩選內沒有觀察值。")

    # Hexagons, fitted line, Pearson r, OLS slope and N all use this same sample.
    data, result = fit_price_change_ols(visible, x_col=x_col, y_col=y_col)
    pooled_r = safe_pearson(data[x_col], data[y_col])
    ci_low, ci_high = fisher_ci(pooled_r, len(data))

    if mode == "standardized":
        plot_limit = selection_limit
        x_label = r"Standardized price change  $(P_j-P_{j-1})/P^C$"
        y_label = r"Lagged standardized change  $(P_{j-1}-P_{j-2})/P^C$"
        denominator_label = "with_eq_price_denominator"
    else:
        maximum_absolute_raw_change = float(
            np.nanmax(
                np.abs(
                    np.concatenate(
                        [data[x_col].to_numpy(float), data[y_col].to_numpy(float)]
                    )
                )
            )
        )
        plot_limit = max(1.0, 1.02 * maximum_absolute_raw_change)
        x_label = r"Raw price change  $P_j-P_{j-1}$"
        y_label = r"Lagged raw change  $P_{j-1}-P_{j-2}$"
        denominator_label = "without_denominator"

    image = ax.hexbin(
        data[x_col],
        data[y_col],
        gridsize=PRICE_CHANGE_HEXBINS,
        extent=(-plot_limit, plot_limit, -plot_limit, plot_limit),
        mincnt=1,
        cmap="Spectral_r",
        linewidths=0,
    )
    grid = np.linspace(-plot_limit, plot_limit, 200)
    fitted = result.params["const"] + result.params[x_col] * grid
    ax.plot(grid, fitted, color=REGRESSION_COLOR, linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.text(
        -0.12,
        1.04,
        panel_letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
    )

    return {
        "artist": image,
        "price_change_mode": mode,
        "denominator_treatment": denominator_label,
        "x_column": x_col,
        "y_column": y_col,
        "n_price_change_pairs": len(data),
        "pearson_r": pooled_r,
        "pearson_ci_low_fisher": ci_low,
        "pearson_ci_high_fisher": ci_high,
        "ols_intercept": float(result.params["const"]),
        "ols_intercept_se_cluster_market": float(result.bse["const"]),
        "ols_slope": float(result.params[x_col]),
        "ols_slope_se_cluster_market": float(result.bse[x_col]),
        "ols_slope_p_value_cluster_market": float(result.pvalues[x_col]),
        "display_ols_intercept": float(result.params["const"]),
        "display_ols_slope": float(result.params[x_col]),
        "display_axis_limit_abs": plot_limit,
        "comparison_sample_limit_abs_standardized": selection_limit,
        "n_visible_in_central_plot": len(data),
        "n_price_change_pairs_before_display_filter": len(all_data),
        "n_omitted_outside_display_limit": len(all_data) - len(data),
    }



def evaluate_kde_grid(
    x: np.ndarray,
    y: np.ndarray,
    grid_n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return None

    grid = np.linspace(-1, 1, grid_n)
    xx, yy = np.meshgrid(grid, grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def zi_column_for_actual(actual_col: str) -> str:
    if "buyer" in actual_col:
        return "rho_buyer_zi"
    if "seller" in actual_col:
        return "rho_seller_zi"
    if "price" in actual_col:
        return "rho_price_zi"
    raise ValueError(f"無法對應 ZI 欄位：{actual_col}")


def add_zi_95_region(
    ax: plt.Axes,
    zi_simulations: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    """Overlay the 95% covariance ellipse of complete ZI simulation draws.

    The uploaded script's covariance/chi-square ellipse is preserved unchanged.
    This is an approximation, not an exact replication of the default robust-t
    stat_ellipse in the paper's utils.R. Fewer than 50 complete draws are omitted.
    No invalid simulation is replaced to meet a desired count.
    """
    zi_x = zi_column_for_actual(x_col)
    zi_y = zi_column_for_actual(y_col)
    eligible = zi_simulations.loc[
        zi_simulations["valid_for_95_region"].astype(bool)
    ]
    data = eligible.dropna(
        subset=["rho_buyer_zi", "rho_seller_zi", "rho_price_zi"]
    )[[zi_x, zi_y]].copy()
    if len(data) < 50:
        return

    values = data.to_numpy(dtype=float)
    values = values[
        (values[:, 0] >= -1) & (values[:, 0] <= 1)
        & (values[:, 1] >= -1) & (values[:, 1] <= 1)
    ]
    if len(values) < 50:
        return

    center = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        return

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = float(
        np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    )
    radius = float(np.sqrt(stats.chi2.ppf(0.95, df=2)))
    ellipse = Ellipse(
        xy=(float(center[0]), float(center[1])),
        width=2 * radius * float(np.sqrt(eigenvalues[0])),
        height=2 * radius * float(np.sqrt(eigenvalues[1])),
        angle=angle,
        facecolor=ZI_COLOR,
        edgecolor=ZI_COLOR,
        linewidth=2.0,
        alpha=0.12,
        zorder=6,
    )
    ax.add_patch(ellipse)


def add_ma_an_predictions(ax, ma_an_benchmark, x_col, y_col):
    """Draw analytical reference loci exactly as official utils.R does.

    These are restrictions under the original model's assumptions, not a
    calibrated CTAT model. Repeated prices in expanded packages need not obey
    the original execution-level autocorrelation restrictions.
    """
    def coordinate(column, price):
        if "buyer" in column:
            return -np.ones_like(price)
        if "seller" in column:
            return np.ones_like(price)
        return price
    prices = np.linspace(0., 1., 101)
    ax.plot(coordinate(x_col, prices), coordinate(y_col, prices),
            color=AN_COLOR, linewidth=2, zorder=12)
    if "price" not in x_col and "price" not in y_col:
        ax.scatter([-1], [1], marker="s", facecolors="none",
                   edgecolors=AN_COLOR, s=75, zorder=12)
    zero = np.array([0.])
    ax.scatter(coordinate(x_col, zero), coordinate(y_col, zero),
               color=MA_COLOR, s=30, zorder=13)


def theory_benchmark_table(ma_an_benchmark=None):
    return pd.DataFrame([
        {"model": "MA", "rho_buyer": -1., "rho_seller": 1.,
         "rho_price_restriction": "=0"},
        {"model": "AN", "rho_buyer": -1., "rho_seller": 1.,
         "rho_price_restriction": ">0 (line drawn over [0,1])"},
    ]).assign(source="Lin (2020) Supplement B.3 pp26-27; official utils.R",
              status="Analytical reference, not estimated CTAT equilibrium",
              assumptions="Risk neutrality and symmetry; original market model")


def plot_pairwise_density(
    ax: plt.Axes,
    market_panel: pd.DataFrame,
    zi_simulations: pd.DataFrame,
    ma_an_benchmark: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
) -> None:
    data = market_panel.dropna(subset=[x_col, y_col]).copy()
    kde_result = evaluate_kde_grid(
        data[x_col].to_numpy(float),
        data[y_col].to_numpy(float),
    )

    if kde_result is not None:
        xx, yy, zz = kde_result
        ax.contourf(
            xx,
            yy,
            zz,
            levels=12,
            cmap="GnBu",
            alpha=0.90,
        )

    if not data.empty:
        ax.scatter(
            [data[x_col].mean()],
            [data[y_col].mean()],
            marker="D",
            s=30,
            facecolor=EMPIRICAL_MEAN_COLOR,
            edgecolor=EMPIRICAL_MEAN_COLOR,
            linewidth=0.7,
            zorder=11,
        )

    add_zi_95_region(ax, zi_simulations, x_col, y_col)
    add_ma_an_predictions(ax, ma_an_benchmark, x_col, y_col)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)



# ============================================================
# 6. Statistics and four complete sensitivity figures
# ============================================================

def early_late_statistics(
    market_panel: pd.DataFrame,
    rho_buyer_col: str,
    rho_seller_col: str,
    observation_level: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = market_panel.dropna(
        subset=[rho_buyer_col, rho_seller_col,
                "rho_price_unit_raw" if observation_level == "permit_unit"
                else "rho_price_package_raw"]
    ).copy()

    for measure_name, rho_col in [
        ("rho_seller", rho_seller_col),
        ("rho_buyer", rho_buyer_col),
    ]:
        early = complete.loc[
            complete["period_group"].eq(EARLY_LABEL),
            rho_col,
        ].dropna()
        late = complete.loc[
            complete["period_group"].eq(LATE_LABEL),
            rho_col,
        ].dropna()

        ks = stats.ks_2samp(early, late) if len(early) and len(late) else None
        for period_label, values in [(EARLY_LABEL, early), (LATE_LABEL, late)]:
            rows.append(
                {
                    "observation_level": observation_level,
                    "measure": measure_name,
                    "period_group": period_label,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "ks_statistic_early_vs_late": ks.statistic if ks else np.nan,
                    "ks_p_value_early_vs_late": ks.pvalue if ks else np.nan,
                }
            )

    return pd.DataFrame(rows)


def create_combined_figure(
    market_panel: pd.DataFrame,
    price_pairs: pd.DataFrame,
    zi_simulations: pd.DataFrame,
    ma_an_benchmark: pd.DataFrame,
    rho_buyer_col: str,
    rho_seller_col: str,
    price_change_mode: str,
    observation_level: str,
    title_suffix: str,
    footer_text: str,
    output_filename: str,
) -> dict[str, Any]:
    level_key = "unit" if observation_level == "permit_unit" else "package"
    rho_price_col = f"rho_price_{level_key}_{price_change_mode}"
    price_pairs = price_pairs.loc[
        price_pairs["observation_level"].eq(observation_level)
    ].copy()
    # The authors form a common complete-case sample for all three correlations
    # before drawing panels a, b and d.
    complete_market_panel = market_panel.dropna(
        subset=[rho_buyer_col, rho_seller_col, rho_price_col]
    ).copy()
    fig = plt.figure(figsize=(14.5, 10.2))
    outer = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1, 1.25],
        width_ratios=[1, 1.25],
        hspace=0.45,
        wspace=0.35,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[0:2, 1])

    plot_rank_histogram(
        ax_a,
        complete_market_panel,
        rho_seller_col,
        "Seller opportunity cost of permit vs transaction order",
        "a",
    )
    plot_rank_histogram(
        ax_b,
        complete_market_panel,
        rho_buyer_col,
        "Buyer marginal value of permit vs transaction order",
        "b",
    )
    c_result = plot_price_change_panel(
        ax_c,
        price_pairs,
        mode=price_change_mode,
        panel_letter="c",
    )
    fig.colorbar(
        c_result["artist"],
        ax=ax_c,
        label="Number of observations",
        fraction=0.046,
        pad=0.04,
    )

    inner_d = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[2, :],
        wspace=0.35,
    )
    d_axes = [fig.add_subplot(inner_d[0, i]) for i in range(3)]
    plot_pairwise_density(
        d_axes[0],
        complete_market_panel,
        zi_simulations,
        ma_an_benchmark,
        rho_buyer_col,
        rho_seller_col,
        r"$\rho_{buyer}$",
        r"$\rho_{seller}$",
    )
    plot_pairwise_density(
        d_axes[1],
        complete_market_panel,
        zi_simulations,
        ma_an_benchmark,
        rho_price_col,
        rho_seller_col,
        r"$\rho_{price}$",
        r"$\rho_{seller}$",
    )
    plot_pairwise_density(
        d_axes[2],
        complete_market_panel,
        zi_simulations,
        ma_an_benchmark,
        rho_buyer_col,
        rho_price_col,
        r"$\rho_{buyer}$",
        r"$\rho_{price}$",
    )
    d_axes[0].text(
        -0.18,
        1.04,
        "d",
        transform=d_axes[0].transAxes,
        fontweight="bold",
        fontsize=12,
    )
    theory_legend = [
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=5,
            markerfacecolor=EMPIRICAL_MEAN_COLOR,
            markeredgecolor=EMPIRICAL_MEAN_COLOR,
            label="CTAT",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=MA_COLOR, markeredgecolor=MA_COLOR,
            label="MA theory",
        ),
        Line2D(
            [0], [0], color=AN_COLOR, linewidth=2.0,
            label="AN theory",
        ),
        Line2D(
            [0], [0], color=ZI_COLOR, linewidth=2.0,
            label="ZI-C",
        ),
    ]
    d_axes[0].legend(
        handles=theory_legend,
        frameon=False,
        fontsize=7,
        loc="lower left",
    )

    fig.suptitle("Per permit unit" if observation_level == "permit_unit"
                 else "Per transaction", fontsize=15, y=0.99)
    fig.savefig(
        OUTPUT_DIR / output_filename,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    c_result = {key: value for key, value in c_result.items() if key != "artist"}
    c_result["observation_level_for_panels_a_b_d"] = observation_level
    c_result["rho_price_column_for_panel_d"] = rho_price_col
    c_result["output_filename"] = output_filename
    return c_result


def write_validation_report(market_panel, package_panel, unit_panel,
                            price_pairs, zi_simulations, ma_an_benchmark):
    lines = [
        "CTAT Figure 2 validation", "=" * 60,
        f"Markets: {len(market_panel)}; packages: {len(package_panel)}; units: {len(unit_panel)}",
        f"Reconstructed final holdings match: {market_panel.reconstruction_ok.all()}",
        "Rounds 1-6; markets with at least 4 packages. No zero changes removed.",
        "Unit price sequence repeats every executed price by its quantity.",
        "Value/cost unit ranks tie within simultaneous packages; price sequence expands every unit.",
        "Raw/standardized panels c share the standardized [-4,4] display filter within each level.",
        "Panel d uses level-specific market price correlations and common complete cases.",
        "Red diamond: unweighted mean of retained market-period correlations.",
        "MA/AN: original analytical reference, not CTAT estimation or simulation.",
        "Removed old greedy surplus matching: it is not a solution of MA or AN.",
        "MA: rho_buyer=-1, rho_seller=1, rho_price=0 under source assumptions.",
        "AN: rho_buyer=-1, rho_seller=1, rho_price>0; 0 is a graphical endpoint only.",
        "Source: Lin (2020), Supplement B.3 pp26-27 and OSF utils.R plot.theory.comp.",
        "https://osf.io/9mfws/ ; Wilson (1987); Friedman (1991).",
        "AN original implementation recipe: Friedman (1991), pp58-62, section4 PA1-4,",
        "equations (3)-(6), section5, and Appendix1:",
        "  1. Initialize risk-neutral agents, private values/costs and deadline T.",
        "  2. Specify remaining quote opportunities and prior bid/ask distributions.",
        "  3. Update posteriors after eligible quotes and acceptances (censored data).",
        "  4. Solve optimal-stopping reservation prices with terminal value=cost/value.",
        "  5. Accept a feasible standing offer; otherwise improve one's quote when optimal.",
        "  6. If no immediate action is optimal, advance time to the next action.",
        "Author-hosted PDF linked from https://leeps.ucsc.edu/papers/:",
        "https://drive.google.com/file/d/1MB26qwnZKY4JqR3SvYBsQQcfPwAtxlXc/view",
        "A CTAT AN implementation needs quote/acceptance logs, timing/rules, prior",
        "parameters, and an explicit multi-unit/role-switching extension. Trades alone",
        "do not identify this process. This script does not pretend to fit that model.",
        "MA mechanism (Lin B.3 summary of Wilson): specify common type distribution;",
        "solve sequential equilibrium with beliefs/strategies; waiting creates missed-",
        "trade risk; an offer improves until accepted. Greedy matching is not that solution.",
        "Original ordering restrictions require risk neutrality and symmetry.",
        "CTAT role switching, endogenous permit values and bundle complementarity differ.",
        "Expanded-package zero changes can also change theoretical price autocorrelation:",
        "the displayed source restrictions are references, not newly derived unit-level predictions.",
        f"ZI-C mode: {ZI_ROLE_MODE}; attempts: {len(zi_simulations)}.",
        f"ZI-C complete-rho draws: {int(zi_simulations.valid_for_95_region.sum())}.",
        "ZI samples ALL session x round_raw 1-6 settings, not the empirical retained sample.",
        "Exactly requested attempts; undefined correlations are retained and never resampled.",
        "No maximum quote events, no period timeout, no consecutive-failure cutoff.",
        "Each activation selects one participant; that participant submits both feasible sides.",
        "Every bid/ask is for one permit. A participant may buy and sell as holdings change.",
        "Bid: 0 <= bid < min(current next-unit value,cash).",
        "Ask: current last-unit value < ask <= 1000. Cash starts at 1000 and is updated.",
        "Only best bid and best ask live; ignored worse quotes do not alter draw intervals.",
        "Equal same-side quotes retain the earlier standing order.",
        "Incoming quotes compare only with pre-activation standing quotes; no self trade.",
        "Execution uses the pre-existing opposite quote and clears both book sides.",
        "If both simultaneous sides cross, both execute atomically from one pre-event state.",
        "Each dual-cross leg uses its own pre-existing opposite quote; then both book sides clear.",
        "Dual-cross CSV ordering is bid leg then ask leg for audit only; settlement is simultaneous.",
        "Acquired permit tokens may be resold; every transfer is recorded in the trade audit.",
        "Current buy value is marginal unit h+1; current sell value is marginal unit h.",
        "Descending average-permit values define the induced holding frontier.",
        "The DP allocation is an audit target only; it never restricts bid/ask direction.",
        "Average induced-unit no-loss is not guaranteed no-loss in integer production profits.",
        "True physical-frontier gains/costs are exported per trade for checking this difference.",
        "Equilibrium stop: no distinct-person transfer can strictly raise induced total value.",
        "Cash/ask-ceiling blockage is reported separately and is not labelled equilibrium.",
        f"Independent simulations use {min(ZI_N_WORKERS, len(zi_simulations))} worker(s); within-market logic is unchanged.",
        f"ZI stop counts: {zi_simulations.stop_reason.value_counts().to_dict()}",
        "This v2 mechanism follows the teacher-confirmed CTAT specification; it is not",
        "presented as a verbatim implementation recovered from Lin's public source code.",
        "ZI-C simulates one-unit trades only; it has no fitted multi-unit package-size process.",
    ]
    for level, panel in [("package", package_panel), ("permit_unit", unit_panel)]:
        pairs = price_pairs.loc[price_pairs.observation_level.eq(level)]
        lines.append(f"{level}: pairs={len(pairs)}, zero changes={panel.price_change_raw.eq(0).sum()}")
    (OUTPUT_DIR / "figure_2_four_way_validation.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trades, subj, group = load_raw_data()
    # Simulate first so all 1-6 settings are used even if no empirical market
    # meets the four-transaction plotting threshold. Audits are saved immediately.
    zi_simulations = simulate_zi_benchmark(
        subj, group, n_simulations=ZI_SIMULATIONS,
    )
    market_panel, package_panel, unit_panel, price_pairs = build_analysis_data(
        trades, subj, group,
    )
    ma_an_benchmark = theory_benchmark_table()

    market_panel.to_csv(
        OUTPUT_DIR / "figure_2_market_period_correlations_both_levels.csv",
        index=False,
        encoding="utf-8-sig",
    )
    package_panel.to_csv(
        OUTPUT_DIR / "figure_2_package_level_values_costs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unit_panel.to_csv(
        OUTPUT_DIR / "figure_2_unit_level_values_costs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    price_pairs.to_csv(
        OUTPUT_DIR / "figure_2_price_change_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    zi_simulations.to_csv(
        OUTPUT_DIR / "figure_2_zi_simulation_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )
    theory_benchmark_table(ma_an_benchmark).to_csv(
        OUTPUT_DIR / "figure_2_ma_an_theory_benchmarks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    package_stats = early_late_statistics(
        market_panel,
        rho_buyer_col="rho_buyer_package",
        rho_seller_col="rho_seller_package",
        observation_level="package",
    )
    unit_stats = early_late_statistics(
        market_panel,
        rho_buyer_col="rho_buyer_unit",
        rho_seller_col="rho_seller_unit",
        observation_level="permit_unit",
    )
    pd.concat([package_stats, unit_stats], ignore_index=True).to_csv(
        OUTPUT_DIR / "figure_2_early_late_statistics_both_levels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure_specs = [
        dict(rho_buyer_col=f"rho_buyer_{level}",
             rho_seller_col=f"rho_seller_{level}",
             observation_level="permit_unit" if level == "unit" else "package",
             price_change_mode=mode, title_suffix="", footer_text="",
             output_filename=f"figure_2abcd_{level}_{mode}_price_change.png")
        for level in ("package", "unit") for mode in ("standardized", "raw")
    ]
    price_statistics = [
        create_combined_figure(
            market_panel,
            price_pairs,
            zi_simulations,
            ma_an_benchmark,
            **figure_spec,
        )
        for figure_spec in figure_specs
    ]

    pd.DataFrame(price_statistics).to_csv(
        OUTPUT_DIR / "figure_2_price_change_regression_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_validation_report(
        market_panel,
        package_panel,
        unit_panel,
        price_pairs,
        zi_simulations,
        ma_an_benchmark,
    )

    print("[DONE] Four complete Figure 2 sensitivity versions were created.")
    for figure_spec in figure_specs:
        print(f"[FIGURE] {OUTPUT_DIR / figure_spec['output_filename']}")
    print("[CHECK] All Figure 2 panels use round_raw 1-6 only.")
    print("[CHECK] Figure 2c is exported with and without the eq_price denominator.")
    print("[CHECK] Both 2c modes use the same paper-rule sample and 100 hex bins.")
    print("[CHECK] Package and permit-unit versions are both exported.")
    print("[CHECK] a-b restrict seller costs to initial units and buyer values to non-initial units.")
    print("[CHECK] Dividing by eq_price does not alter within-market rho_price in panel d.")
    print(f"[CHECK] ZI version: {ZI_ROLE_MODE}; exactly {ZI_SIMULATIONS:,} market attempts.")
    print(f"[CHECK] Complete draws for ellipse: {int(zi_simulations.valid_for_95_region.sum()):,}.")
    print("[CHECK] No quote-event cutoff; resale is allowed; all books clear after execution.")
    print("[CHECK] One selected participant submits both feasible one-unit sides; no self trade.")
    print("[CHECK] See final-player audit for induced-equilibrium and physical-frontier targets.")
    print("[CHECK] MA/AN are source-theory references; no fitted CTAT equilibrium is claimed.")


if __name__ == "__main__":
    main()
