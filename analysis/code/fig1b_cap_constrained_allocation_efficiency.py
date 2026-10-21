from __future__ import annotations

"""
Figure 1b adaptation: realized-profit efficiency under a fixed permit cap.

Place this file in:
    ctat_analysis/analysis/code/

Inputs:
    ctat_analysis/analysis/input/trades_panel.csv
    ctat_analysis/analysis/input/subj_lv_panel.csv
    ctat_analysis/analysis/input/group_lv_panel.csv

Core efficiency definition (aligned with Lin et al. Fig. 1b's realized-profit idea):
    numerator   = ACTUAL aggregate net profit observed in the market-period,
                  sum(net_profit) across subjects, as requested for CTAT.
    denominator = maximum total production profit attainable in that same
                  market-period, subject to the observed total permit cap
                  (group_cap_total).

Therefore, 100% means observed production realized the maximum market production
profit feasible under that market-period's permit cap. The final-allocation
profit frontier is retained only as an audit variable, not as the numerator.

The contour lines in the figures are empirical 2-D kernel-density (KDE) contours:
contours farther inward correspond to higher local concentrations of market-period
observations. They are density contours, NOT confidence intervals.

The script also exports a Lin-style decomposition of allocation loss into:
    EM-inefficiency = loss associated with extra-marginal net reallocations;
    V-inefficiency  = loss associated with profitable net reallocations not realized.
Because CTAT permits can be resold, the decomposition uses the NET change from the
initial to final allocation. Counting gross trades would count the same permit more
than once and would not measure final allocative efficiency.
"""

import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import gaussian_kde


# ============================================================
# 0. Paths and analysis blocks
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = CODE_DIR.parent
INPUT_DIR = ANALYSIS_DIR / "input"
OUTPUT_DIR = ANALYSIS_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADES_PATH = INPUT_DIR / "trades_panel.csv"
SUBJ_PATH = INPUT_DIR / "subj_lv_panel.csv"
GROUP_PATH = INPUT_DIR / "group_lv_panel.csv"

MARKET_KEYS = ["session_code", "round_raw"]
# Retain the established output-column name for compatibility. Its numerator is
# now sum(net_profit), as documented above and in the figure labels.
EFFICIENCY_COL = "realized_production_efficiency"

ROUND_BLOCKS = [
    ("Rounds 1-3", 1, 3),
    ("Rounds 4-6", 4, 6),
    ("Rounds 7-9", 7, 9),
    ("Rounds 10-12", 10, 12),
    ("Rounds 13-15", 13, 15),
    ("Rounds 16-18", 16, 18),
]

BLOCK_STYLES = {
    "Rounds 1-3":   {"color": "#c05a93", "linestyle": "-"},
    "Rounds 4-6":   {"color": "#b05fa2", "linestyle": "--"},
    "Rounds 7-9":   {"color": "#956bb1", "linestyle": "-"},
    "Rounds 10-12": {"color": "#7b76bd", "linestyle": "--"},
    "Rounds 13-15": {"color": "#6880c2", "linestyle": "-"},
    "Rounds 16-18": {"color": "#5b84c4", "linestyle": "--"},
}


# ============================================================
# 1. General utilities
# ============================================================

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到輸入檔：{path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} 缺少欄位：{', '.join(sorted(missing))}")


def parse_disturbances(value: Any) -> np.ndarray:
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


def assign_round_block(round_raw: int) -> str | None:
    for label, lower, upper in ROUND_BLOCKS:
        if lower <= round_raw <= upper:
            return label
    return None


def as_nonnegative_int(value: float, name: str) -> int:
    if pd.isna(value):
        raise ValueError(f"{name} 不可為缺值。")
    rounded = int(round(float(value)))
    if not np.isclose(float(value), rounded):
        raise ValueError(f"{name} 必須是整數，目前為 {value}。")
    if rounded < 0:
        raise ValueError(f"{name} 不可小於 0，目前為 {rounded}。")
    return rounded


# ============================================================
# 2. Load panels
# ============================================================

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = read_csv(TRADES_PATH)
    subj = read_csv(SUBJ_PATH)
    group = read_csv(GROUP_PATH)

    require_columns(
        trades,
        {"session_code", "round_raw", "price", "eq_price"},
        "trades_panel.csv",
    )
    require_columns(
        subj,
        {
            "session_code",
            "round_raw",
            "app",
            "id_in_group",
            "permits",
            "current_permits",
            "disturbance_values",
            "marginal_cost_coefficient",
            "carbon_emission_per_unit",
            "max_production",
            "market_price",
            "group_cap_total",
            "net_profit",
        },
        "subj_lv_panel.csv",
    )
    require_columns(
        group,
        {
            "session_code",
            "round_raw",
            "app",
            "round",
            "first",
            "is_equal",
            "allocation_method",
        },
        "group_lv_panel.csv",
    )

    subj = subj.loc[subj["app"].eq("Stage_CarbonTrading")].copy()
    group = group.loc[group["app"].eq("Stage_CarbonTrading")].copy()

    for col in ["round_raw", "price", "eq_price"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    numeric_subj = [
        "round_raw",
        "id_in_group",
        "permits",
        "current_permits",
        "marginal_cost_coefficient",
        "carbon_emission_per_unit",
        "max_production",
        "market_price",
        "group_cap_total",
        "net_profit",
    ]
    for col in numeric_subj:
        subj[col] = pd.to_numeric(subj[col], errors="coerce")

    for col in ["round_raw", "round", "first", "is_equal"]:
        group[col] = pd.to_numeric(group[col], errors="coerce")

    return trades, subj, group


# ============================================================
# 3. Individual profit frontier V_i(h)
# ============================================================

def profit_frontier(row: pd.Series, total_cap: int) -> np.ndarray:
    """Return V_i(h) for h = 0, ..., total_cap.

    V_i(h) is subject i's maximum production profit when holding h permits.
    Production is restricted by:
        q <= max_production
        q * carbon_emission_per_unit <= h
    """

    emissions_per_output = as_nonnegative_int(
        row["carbon_emission_per_unit"],
        "carbon_emission_per_unit",
    )
    if emissions_per_output <= 0:
        raise ValueError("carbon_emission_per_unit 必須大於 0。")

    max_production = as_nonnegative_int(row["max_production"], "max_production")
    output_price = float(row["market_price"])
    mc_coefficient = float(row["marginal_cost_coefficient"])

    disturbances = parse_disturbances(row["disturbance_values"])
    if disturbances.size < max_production:
        raise ValueError(
            "disturbance_values 長度小於 max_production："
            f"id_in_group={row['id_in_group']}，"
            f"length={disturbances.size}，max_production={max_production}"
        )

    # Profit from producing q units, q = 0, ..., max_production.
    q_units = np.arange(1, max_production + 1, dtype=float)
    marginal_costs = mc_coefficient * q_units + disturbances[:max_production]
    production_profit = np.concatenate(
        [[0.0], np.cumsum(output_price - marginal_costs)]
    )

    frontier = np.zeros(total_cap + 1, dtype=float)
    for holding in range(total_cap + 1):
        feasible_q_max = min(
            max_production,
            holding // emissions_per_output,
        )
        frontier[holding] = float(
            np.max(production_profit[: feasible_q_max + 1])
        )

    return frontier


def permit_marginal_schedule(
    row: pd.Series,
    total_cap: int,
    frontier: np.ndarray | None = None,
) -> np.ndarray:
    """Return the production-profit value of permit ordinals 1..total_cap.

    If one output needs b permits, its complete profit increment is divided
    equally across the b permits.  For a large firm with b=2, a bundle gain of
    40 is therefore represented as 20 and 20, rather than 0 and 40.  The
    production frontier itself remains physical: output is possible only after
    all b permits have been assembled.
    """
    if frontier is None:
        frontier = profit_frontier(row, total_cap)

    b = as_nonnegative_int(
        row["carbon_emission_per_unit"],
        "carbon_emission_per_unit",
    )
    if b <= 0:
        raise ValueError("carbon_emission_per_unit 必須大於 0。")

    max_production = as_nonnegative_int(row["max_production"], "max_production")
    schedule = np.zeros(total_cap + 1, dtype=float)
    for output_unit in range(1, max_production + 1):
        block_start = (output_unit - 1) * b
        block_end = output_unit * b
        if block_end > total_cap:
            break
        block_gain = float(frontier[block_end] - frontier[block_start])
        schedule[block_start + 1 : block_end + 1] = block_gain / b
    return schedule


def lin_style_inefficiency_decomposition(
    rows: pd.DataFrame,
    initial_holdings: np.ndarray,
    final_holdings: np.ndarray,
    optimal_holdings: np.ndarray,
    marginal_schedules: list[np.ndarray],
    eq_price: float,
    initial_allocation_profit: float,
    final_allocation_profit: float,
    cap_constrained_optimal_profit: float,
    actual_net_profit: float,
) -> dict[str, Any]:
    """Adapt Lin et al.'s EM/V decomposition to a two-sided permit market.

    Lin et al. have fixed buyers and sellers and each induced unit can trade at
    most once.  CTAT participants can buy and sell, and a permit can be resold.
    The paper's gross-transaction counts would therefore over-count allocation
    changes in CTAT.  We use net movements from initial to final holdings:

      * net increases are buyer units that were not held initially;
      * net decreases are seller units that were held initially.

    The mixed-case formulas below reproduce the authors' public Stata code:
    missing inframarginal surplus is allocated between unexecuted volume
    (V-inefficiency) and displacement by extra-marginal units
    (EM-inefficiency). Amounts add to the loss in the averaged per-permit
    schedule. The physical bundle-allocation loss and its difference from the
    smoothed schedule are reported separately, as is the remaining gap to
    actual net profit.
    """
    if not np.isfinite(eq_price) or eq_price <= 0:
        raise ValueError(f"eq_price 必須大於 0，目前為 {eq_price}")

    optimal_buyer_values: list[float] = []
    optimal_seller_costs: list[float] = []
    actual_buyer_values: list[float] = []
    actual_seller_costs: list[float] = []

    for i, _ in rows.iterrows():
        initial_h = int(initial_holdings[i])
        final_h = int(final_holdings[i])
        optimal_h = int(optimal_holdings[i])
        schedule = marginal_schedules[i]

        if optimal_h > initial_h:
            optimal_buyer_values.extend(
                float(schedule[ordinal])
                for ordinal in range(initial_h + 1, optimal_h + 1)
            )
        elif optimal_h < initial_h:
            optimal_seller_costs.extend(
                float(schedule[ordinal])
                for ordinal in range(optimal_h + 1, initial_h + 1)
            )

        if final_h > initial_h:
            actual_buyer_values.extend(
                float(schedule[ordinal])
                for ordinal in range(initial_h + 1, final_h + 1)
            )
        elif final_h < initial_h:
            actual_seller_costs.extend(
                float(schedule[ordinal])
                for ordinal in range(final_h + 1, initial_h + 1)
            )

    eq_quantity_buy = len(optimal_buyer_values)
    eq_quantity_sell = len(optimal_seller_costs)
    actual_quantity_buy = len(actual_buyer_values)
    actual_quantity_sell = len(actual_seller_costs)
    if eq_quantity_buy != eq_quantity_sell:
        raise ValueError("最適配置的淨買入量與淨賣出量不一致。")
    if actual_quantity_buy != actual_quantity_sell:
        raise ValueError("最終配置的淨買入量與淨賣出量不一致。")

    eq_quantity = eq_quantity_buy
    actual_quantity = actual_quantity_buy
    optimal_consumer_surplus = float(
        sum(value - eq_price for value in optimal_buyer_values)
    )
    optimal_producer_surplus = float(
        sum(eq_price - cost for cost in optimal_seller_costs)
    )

    buyer_surpluses = np.asarray(actual_buyer_values, dtype=float) - eq_price
    seller_surpluses = eq_price - np.asarray(actual_seller_costs, dtype=float)

    buyer_positive = buyer_surpluses[buyer_surpluses >= 0]
    seller_positive = seller_surpluses[seller_surpluses >= 0]
    buyer_negative_loss = float(-buyer_surpluses[buyer_surpluses < 0].sum())
    seller_negative_loss = float(-seller_surpluses[seller_surpluses < 0].sum())

    infra_buyer = int(len(buyer_positive))
    infra_seller = int(len(seller_positive))
    missing_infra_buyer = int(eq_quantity - infra_buyer)
    missing_infra_seller = int(eq_quantity - infra_seller)
    extra_buyer = int(actual_quantity - infra_buyer)
    extra_seller = int(actual_quantity - infra_seller)

    buyer_surplus_diff = float(
        optimal_consumer_surplus - buyer_positive.sum()
    )
    seller_surplus_diff = float(
        optimal_producer_surplus - seller_positive.sum()
    )
    # EM/V is evaluated in the paper-style per-permit schedule. With b=2, a
    # half-complete bundle receives the requested averaged marginal value, so
    # this schedule loss can differ from the physical production-frontier loss.
    optimal_gains_from_reallocation = float(
        optimal_consumer_surplus + optimal_producer_surplus
    )
    realized_gains_from_final_allocation = float(
        np.sum(actual_buyer_values) - np.sum(actual_seller_costs)
    )
    allocation_loss = float(
        optimal_gains_from_reallocation
        - realized_gains_from_final_allocation
    )
    physical_allocation_loss = float(
        cap_constrained_optimal_profit - final_allocation_profit
    )
    tolerance = 1e-8

    if abs(allocation_loss) <= tolerance:
        em_amount = 0.0
        v_amount = 0.0
        decomposition_case = "efficient_final_allocation"
    elif actual_quantity >= eq_quantity:
        # Same trivial case as the authors' code: all loss is attributed to
        # displacement when volume reaches/exceeds the equilibrium quantity.
        em_amount = allocation_loss
        v_amount = 0.0
        decomposition_case = "all_em_quantity_at_or_above_equilibrium"
    elif extra_buyer == 0 and extra_seller == 0:
        em_amount = 0.0
        v_amount = allocation_loss
        decomposition_case = "all_v_undertrading_without_extramarginal_units"
    else:
        def split_side(
            surplus_diff: float,
            negative_loss: float,
            missing_infra: int,
            extra_units: int,
        ) -> tuple[float, float]:
            if missing_infra <= 0:
                return surplus_diff + negative_loss, 0.0
            unexecuted_infra = missing_infra - extra_units
            v_score = surplus_diff * unexecuted_infra / missing_infra
            em_score = surplus_diff - v_score + negative_loss
            return float(em_score), float(v_score)

        em_buyer, v_buyer = split_side(
            buyer_surplus_diff,
            buyer_negative_loss,
            missing_infra_buyer,
            extra_buyer,
        )
        em_seller, v_seller = split_side(
            seller_surplus_diff,
            seller_negative_loss,
            missing_infra_seller,
            extra_seller,
        )
        em_amount = em_buyer + em_seller
        v_amount = v_buyer + v_seller
        decomposition_case = "mixed_lin_formula"

    decomposed_loss = float(em_amount + v_amount)
    if decomposed_loss > tolerance:
        em_share = float(em_amount / decomposed_loss)
        v_share = float(v_amount / decomposed_loss)
    else:
        em_share = np.nan
        v_share = np.nan

    allocation_efficiency = (
        realized_gains_from_final_allocation / optimal_gains_from_reallocation
        if optimal_gains_from_reallocation > tolerance
        else np.nan
    )

    return {
        "eq_quantity_net_reallocation": float(eq_quantity),
        "actual_quantity_net_reallocation": float(actual_quantity),
        "optimal_consumer_surplus": optimal_consumer_surplus,
        "optimal_producer_surplus": optimal_producer_surplus,
        "buyer_negative_surplus_loss": buyer_negative_loss,
        "seller_negative_surplus_loss": seller_negative_loss,
        "infra_buyer_units": float(infra_buyer),
        "infra_seller_units": float(infra_seller),
        "extra_buyer_units": float(extra_buyer),
        "extra_seller_units": float(extra_seller),
        "missing_infra_buyer_units": float(missing_infra_buyer),
        "missing_infra_seller_units": float(missing_infra_seller),
        "optimal_gains_from_reallocation": optimal_gains_from_reallocation,
        "realized_gains_from_final_allocation": realized_gains_from_final_allocation,
        "allocation_efficiency_gains": allocation_efficiency,
        "allocation_inefficiency_amount": allocation_loss,
        "physical_allocation_inefficiency_amount": physical_allocation_loss,
        "bundle_smoothing_residual": float(
            physical_allocation_loss - allocation_loss
        ),
        "em_inefficiency_amount": float(em_amount),
        "v_inefficiency_amount": float(v_amount),
        "em_inefficiency_share": em_share,
        "v_inefficiency_share": v_share,
        "inefficiency_decomposition_case": decomposition_case,
        "decomposition_residual": float(allocation_loss - decomposed_loss),
        "cap_optimum_minus_actual_net_profit": float(
            cap_constrained_optimal_profit - actual_net_profit
        ),
        "final_frontier_minus_actual_net_profit": float(
            final_allocation_profit - actual_net_profit
        ),
    }


# ============================================================
# 4. Cap-constrained market optimum
# ============================================================

def optimize_cap_allocation(
    frontiers: list[np.ndarray],
    initial_holdings: np.ndarray,
    total_cap: int,
) -> tuple[float, np.ndarray]:
    """Maximize aggregate frontier value and return one optimal allocation.

    Profit is the primary objective. Among profit ties, the algorithm minimizes
    total movement from the initial allocation so valueless leftover permits do
    not create artificial equilibrium trading volume.
    """
    negative_infinity = -np.inf
    best_profit = np.full(total_cap + 1, negative_infinity, dtype=float)
    best_profit[0] = 0.0
    best_movement = np.full(total_cap + 1, np.inf, dtype=float)
    best_movement[0] = 0.0
    allocation_choices: list[np.ndarray] = []

    for player_index, one_frontier in enumerate(frontiers):
        updated = np.full(total_cap + 1, negative_infinity, dtype=float)
        updated_movement = np.full(total_cap + 1, np.inf, dtype=float)
        chosen_holding = np.full(total_cap + 1, -1, dtype=int)
        for allocated_so_far in range(total_cap + 1):
            if not np.isfinite(best_profit[allocated_so_far]):
                continue
            remaining = total_cap - allocated_so_far
            for holding in range(remaining + 1):
                total_holding = allocated_so_far + holding
                candidate = best_profit[allocated_so_far] + one_frontier[holding]
                candidate_movement = (
                    best_movement[allocated_so_far]
                    + abs(holding - initial_holdings[player_index])
                )
                profit_is_better = candidate > updated[total_holding] + 1e-10
                profit_is_tied = np.isclose(
                    candidate,
                    updated[total_holding],
                    atol=1e-10,
                    rtol=0,
                )
                movement_is_better = (
                    candidate_movement < updated_movement[total_holding]
                )
                if profit_is_better or (profit_is_tied and movement_is_better):
                    updated[total_holding] = candidate
                    updated_movement[total_holding] = candidate_movement
                    chosen_holding[total_holding] = holding
        best_profit = updated
        best_movement = updated_movement
        allocation_choices.append(chosen_holding)

    optimum = float(best_profit[total_cap])
    if not np.isfinite(optimum):
        raise ValueError("無法求得固定 group_cap_total 下的市場最大利潤。")

    optimal_holdings = np.zeros(len(frontiers), dtype=int)
    allocated_total = total_cap
    for i in range(len(frontiers) - 1, -1, -1):
        holding = int(allocation_choices[i][allocated_total])
        if holding < 0:
            raise ValueError("無法回推出固定總碳權下的最適配置。")
        optimal_holdings[i] = holding
        allocated_total -= holding
    if allocated_total != 0:
        raise ValueError("最適配置回推後的碳權總量不守恆。")
    return optimum, optimal_holdings


def calculate_market_efficiency(
    subject_rows: pd.DataFrame,
    eq_price: float,
) -> dict[str, Any]:
    rows = subject_rows.sort_values("id_in_group").reset_index(drop=True)

    cap_values = rows["group_cap_total"].dropna().unique()
    if len(cap_values) != 1:
        raise ValueError("同一市場回合的 group_cap_total 必須唯一。")
    total_cap = as_nonnegative_int(cap_values[0], "group_cap_total")

    initial_holdings = np.array(
        [as_nonnegative_int(v, "permits") for v in rows["permits"]],
        dtype=int,
    )
    final_holdings = np.array(
        [as_nonnegative_int(v, "current_permits") for v in rows["current_permits"]],
        dtype=int,
    )

    if int(initial_holdings.sum()) != total_cap:
        raise ValueError(
            "初始 permits 加總不等於 group_cap_total："
            f"{initial_holdings.sum()} != {total_cap}"
        )
    if int(final_holdings.sum()) != total_cap:
        raise ValueError(
            "交易後 current_permits 加總不等於 group_cap_total："
            f"{final_holdings.sum()} != {total_cap}"
        )
    if np.any(final_holdings > total_cap):
        raise ValueError("個人 current_permits 不可超過全市場 group_cap_total。")

    frontiers = [profit_frontier(row, total_cap) for _, row in rows.iterrows()]

    cap_constrained_optimal_profit, optimal_holdings = optimize_cap_allocation(
        frontiers,
        initial_holdings,
        total_cap,
    )

    initial_allocation_profit = float(
        sum(frontiers[i][holding] for i, holding in enumerate(initial_holdings))
    )
    final_allocation_profit = float(
        sum(frontiers[i][holding] for i, holding in enumerate(final_holdings))
    )

    # Audit only: sum of each subject's own maximum, ignoring the shared cap.
    # This is NOT the efficiency denominator.
    no_shared_cap_profit_upper_bound = float(
        sum(float(np.max(frontier)) for frontier in frontiers)
    )

    # Requested Fig. 1b numerator: the observed subject-level net_profit field.
    # Missing values are rejected instead of being silently skipped.
    if rows["net_profit"].isna().any():
        raise ValueError("同一市場回合的 net_profit 含缺值，無法計算實際總利潤。")
    actual_net_profit = float(rows["net_profit"].sum())

    if cap_constrained_optimal_profit > 0:
        efficiency = actual_net_profit / cap_constrained_optimal_profit
    else:
        efficiency = np.nan

    # Do not clip the lower tail: if a subject actually makes a production loss,
    # realized efficiency can in principle be below zero.  Values materially
    # above one indicate an accounting/data inconsistency because the denominator
    # is the cap-constrained maximum production profit.
    if np.isfinite(efficiency) and efficiency > 1 + 1e-6:
        raise ValueError(
            "實際 net_profit 加總超過固定總碳權下的理論最大利潤，請檢查 net_profit 定義或最佳化程序："
            f"efficiency={efficiency}"
        )
    if np.isfinite(efficiency) and efficiency > 1:
        efficiency = 1.0

    marginal_schedules = [
        permit_marginal_schedule(row, total_cap, frontiers[i])
        for i, (_, row) in enumerate(rows.iterrows())
    ]
    smoothed_frontiers = [
        np.concatenate([[0.0], np.cumsum(schedule[1:])])
        for schedule in marginal_schedules
    ]
    smoothed_optimal_profit, smoothed_optimal_holdings = optimize_cap_allocation(
        smoothed_frontiers,
        initial_holdings,
        total_cap,
    )
    inefficiency = lin_style_inefficiency_decomposition(
        rows=rows,
        initial_holdings=initial_holdings,
        final_holdings=final_holdings,
        optimal_holdings=smoothed_optimal_holdings,
        marginal_schedules=marginal_schedules,
        eq_price=float(eq_price),
        initial_allocation_profit=initial_allocation_profit,
        final_allocation_profit=final_allocation_profit,
        cap_constrained_optimal_profit=cap_constrained_optimal_profit,
        actual_net_profit=actual_net_profit,
    )

    return {
        "cap_total": float(total_cap),
        "initial_allocation_profit": initial_allocation_profit,
        "final_allocation_profit": final_allocation_profit,
        "actual_net_profit": actual_net_profit,
        "cap_constrained_optimal_profit": cap_constrained_optimal_profit,
        EFFICIENCY_COL: efficiency,
        "optimal_holdings": json.dumps(optimal_holdings.tolist()),
        "smoothed_cap_constrained_optimal_profit": smoothed_optimal_profit,
        "optimal_holdings_smoothed": json.dumps(
            smoothed_optimal_holdings.tolist()
        ),
        "no_shared_cap_profit_upper_bound": no_shared_cap_profit_upper_bound,
        "profit_lost_due_to_cap_constraint": (
            no_shared_cap_profit_upper_bound - cap_constrained_optimal_profit
        ),
        **inefficiency,
    }


# ============================================================
# 5. Smith's alpha
# ============================================================

def calculate_smith_alpha(trade_rows: pd.DataFrame) -> dict[str, float]:
    valid = trade_rows.dropna(subset=["price", "eq_price"]).copy()
    if valid.empty:
        return {"eq_price": np.nan, "n_trades": 0, "smith_alpha": np.nan}

    eq_values = valid["eq_price"].unique()
    if len(eq_values) != 1:
        raise ValueError(
            "同一 session_code × round_raw 出現多個 eq_price："
            f"{eq_values.tolist()}"
        )

    eq_price = float(eq_values[0])
    if eq_price <= 0:
        raise ValueError(f"eq_price 必須大於 0，目前為 {eq_price}")

    smith_alpha = float(
        np.sqrt(np.mean((valid["price"] - eq_price) ** 2)) / eq_price
    )

    return {
        "eq_price": eq_price,
        "n_trades": int(len(valid)),
        "smith_alpha": smith_alpha,
        "min_trade_price": float(valid["price"].min()),
        "mean_trade_price": float(valid["price"].mean()),
        "max_trade_price": float(valid["price"].max()),
    }


# ============================================================
# 6. Build one row per market-period
# ============================================================

def build_market_panel(
    trades: pd.DataFrame,
    subj: pd.DataFrame,
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_use = group.loc[group["round_raw"].between(1, 18)].copy()

    duplicate_count = group_use.groupby(MARKET_KEYS).size()
    duplicate_markets = duplicate_count.loc[duplicate_count.gt(1)]
    if not duplicate_markets.empty:
        raise ValueError(
            "group_lv_panel.csv 的 session_code × round_raw 不是唯一：\n"
            + duplicate_markets.to_string()
        )

    market_rows: list[dict[str, Any]] = []

    for _, meta in group_use.sort_values(MARKET_KEYS).iterrows():
        session_code = meta["session_code"]
        round_raw = int(meta["round_raw"])
        block = assign_round_block(round_raw)
        if block is None:
            continue

        subject_rows = subj.loc[
            subj["session_code"].eq(session_code)
            & subj["round_raw"].eq(round_raw)
        ].copy()
        trade_rows = trades.loc[
            trades["session_code"].eq(session_code)
            & trades["round_raw"].eq(round_raw)
        ].copy()

        if subject_rows.empty or trade_rows.empty:
            continue

        alpha_results = calculate_smith_alpha(trade_rows)
        efficiency_results = calculate_market_efficiency(
            subject_rows,
            eq_price=alpha_results["eq_price"],
        )

        market_rows.append(
            {
                "session_code": session_code,
                "round_raw": round_raw,
                "round": int(meta["round"]),
                "first": int(meta["first"]),
                "is_equal": int(meta["is_equal"]),
                "allocation_method": meta["allocation_method"],
                "round_block": block,
                **alpha_results,
                **efficiency_results,
            }
        )

    panel = (
        pd.DataFrame(market_rows)
        .sort_values(MARKET_KEYS)
        .reset_index(drop=True)
    )
    if panel.empty:
        raise ValueError("沒有可用的 round_raw 1-18 市場回合。")

    expected_rounds = set(range(1, 19))
    audit_rows: list[dict[str, Any]] = []
    for session_code, session_rows in panel.groupby("session_code", sort=True):
        present = set(session_rows["round_raw"].astype(int))
        audit_rows.append(
            {
                "session_code": session_code,
                "present_round_raw": ",".join(map(str, sorted(present))),
                "missing_round_raw": ",".join(
                    map(str, sorted(expected_rounds - present))
                ),
                "n_market_periods": len(session_rows),
            }
        )

    return panel, pd.DataFrame(audit_rows)


# ============================================================
# 7. Regression and density helpers
# ============================================================

def fit_line(group: pd.DataFrame):
    x = sm.add_constant(group[EFFICIENCY_COL], has_constant="add")

    n_clusters = group["session_code"].nunique()
    if n_clusters >= 2:
        result = sm.OLS(group["smith_alpha"], x).fit(
            cov_type="cluster",
            cov_kwds={"groups": group["session_code"]},
        )
        se_type = "cluster_session"
    else:
        result = sm.OLS(group["smith_alpha"], x).fit(cov_type="HC1")
        se_type = "HC1"

    x_grid = np.linspace(
        group[EFFICIENCY_COL].min(),
        group[EFFICIENCY_COL].max(),
        100,
    )
    y_grid = result.params["const"] + result.params[EFFICIENCY_COL] * x_grid
    return result, x_grid, y_grid, se_type


def draw_density_contours(
    ax: plt.Axes,
    group: pd.DataFrame,
    color: str,
    linestyle: str,
) -> None:
    if len(group) < 5:
        return
    if group[EFFICIENCY_COL].nunique() < 2 or group["smith_alpha"].nunique() < 2:
        return

    x = group[EFFICIENCY_COL].to_numpy(dtype=float)
    y = group["smith_alpha"].to_numpy(dtype=float)
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return

    x_pad = max((x.max() - x.min()) * 0.08, 0.02)
    y_pad = max((y.max() - y.min()) * 0.10, 0.05)
    x_grid = np.linspace(min(0.0, x.min() - x_pad), min(1.02, x.max() + x_pad), 150)
    y_grid = np.linspace(max(0, y.min() - y_pad), y.max() + y_pad, 150)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    z_min = float(np.nanmin(zz))
    z_max = float(np.nanmax(zz))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return

    levels = np.linspace(z_min + (z_max - z_min) * 0.30, z_max, 6)
    ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors=color,
        linestyles=linestyle,
        linewidths=1.0,
        alpha=0.90,
    )


def calculate_block_result(group: pd.DataFrame, block: str):
    row: dict[str, Any] = {
        "round_block": block,
        "n": len(group),
        "n_sessions": group["session_code"].nunique(),
        "mean_efficiency": group[EFFICIENCY_COL].mean(),
        "mean_smith_alpha": group["smith_alpha"].mean(),
        "median_smith_alpha": group["smith_alpha"].median(),
    }

    if len(group) >= 3 and group[EFFICIENCY_COL].nunique() >= 2:
        result, x_grid, y_grid, se_type = fit_line(group)

        if group["smith_alpha"].nunique() >= 2:
            pearson = stats.pearsonr(group[EFFICIENCY_COL], group["smith_alpha"])
            pearson_r = float(pearson.statistic)
            pearson_p = float(pearson.pvalue)
        else:
            pearson_r = np.nan
            pearson_p = np.nan

        row.update(
            {
                "intercept": result.params["const"],
                "intercept_se": result.bse["const"],
                "efficiency_slope": result.params[EFFICIENCY_COL],
                "efficiency_slope_se": result.bse[EFFICIENCY_COL],
                "efficiency_slope_p_value": result.pvalues[EFFICIENCY_COL],
                "standard_error_type": se_type,
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
                "r_squared": result.rsquared,
            }
        )
        return row, x_grid, y_grid

    row.update(
        {
            "intercept": np.nan,
            "intercept_se": np.nan,
            "efficiency_slope": np.nan,
            "efficiency_slope_se": np.nan,
            "efficiency_slope_p_value": np.nan,
            "standard_error_type": np.nan,
            "pearson_r": np.nan,
            "pearson_p_value": np.nan,
            "r_squared": np.nan,
        }
    )
    return row, None, None


# ============================================================
# 8. Figures
# ============================================================

def create_overlaid_figure(panel: pd.DataFrame) -> list[dict[str, Any]]:
    data = panel.dropna(subset=[EFFICIENCY_COL, "smith_alpha"]).copy()
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    result_rows: list[dict[str, Any]] = []

    for block, _, _ in ROUND_BLOCKS:
        group = data.loc[data["round_block"].eq(block)].copy()
        if group.empty:
            continue

        style = BLOCK_STYLES[block]
        draw_density_contours(ax, group, style["color"], style["linestyle"])
        result_row, x_grid, y_grid = calculate_block_result(group, block)
        result_rows.append(result_row)

        ax.plot(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            label=f"{block} (n={len(group)})",
        )
        if x_grid is not None:
            ax.plot(
                x_grid,
                y_grid,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.1,
            )

    ax.set_xlabel("Realized net-profit efficiency")
    ax.set_ylabel("Smith's alpha")
    ax.set_title(
        "Figure 1b adaptation: realized net-profit efficiency\n"
        "Actual net profit / cap-constrained market optimum"
    )
    ax.set_xlim(min(0.0, float(data[EFFICIENCY_COL].min()) - 0.02), 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, fontsize=9)

    fig.text(
        0.5,
        -0.025,
        "Numerator: actual profit = sum(net_profit). "
        "Denominator: maximum market production profit under the same group_cap_total. "
        "KDE contours show empirical observation density (inner contours = higher density; not confidence intervals).",
        ha="center",
        fontsize=8,
    )

    output_path = OUTPUT_DIR / "figure_1b_cap_constrained_efficiency_all_six_blocks_overlay.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {output_path}")
    return result_rows


def create_small_multiples(panel: pd.DataFrame) -> None:
    data = panel.dropna(subset=[EFFICIENCY_COL, "smith_alpha"]).copy()
    y_max = max(1.0, float(data["smith_alpha"].max()) * 1.08)
    x_min = min(0.0, float(data[EFFICIENCY_COL].min()) - 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)

    for ax, (block, _, _) in zip(axes.ravel(), ROUND_BLOCKS):
        group = data.loc[data["round_block"].eq(block)].copy()
        style = BLOCK_STYLES[block]

        if group.empty:
            ax.set_title(f"{block} (no data)")
            ax.set_xlim(x_min, 1.02)
            ax.set_ylim(0, y_max)
            continue

        draw_density_contours(ax, group, style["color"], style["linestyle"])
        _, x_grid, y_grid = calculate_block_result(group, block)
        if x_grid is not None:
            ax.plot(
                x_grid,
                y_grid,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
            )

        ax.scatter(
            group[EFFICIENCY_COL],
            group["smith_alpha"],
            s=20,
            alpha=0.30,
            color=style["color"],
        )
        ax.set_title(f"{block} (n={len(group)})")
        ax.set_xlim(x_min, 1.02)
        ax.set_ylim(0, y_max)

    for ax in axes[-1, :]:
        ax.set_xlabel("Realized net-profit efficiency")
    for ax in axes[:, 0]:
        ax.set_ylabel("Smith's alpha")

    fig.suptitle(
        "Figure 1b adaptation by round_raw block\n"
        "Realized net-profit efficiency",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.012,
        "Contours are empirical 2-D KDE density contours: inward contours indicate higher "
        "concentrations of market-period observations; they are not confidence intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    output_path = OUTPUT_DIR / "figure_1b_cap_constrained_efficiency_all_six_blocks_panels.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {output_path}")


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    trades, subj, group = load_raw_data()
    panel, audit = build_market_panel(trades, subj, group)

    panel.to_csv(
        OUTPUT_DIR / "figure_1b_cap_constrained_efficiency_plot_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        OUTPUT_DIR / "figure_1b_cap_constrained_efficiency_sample_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result_rows = create_overlaid_figure(panel)
    create_small_multiples(panel)

    pd.DataFrame(result_rows).to_csv(
        OUTPUT_DIR / "figure_1b_cap_constrained_efficiency_regression_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    inefficiency_columns = [
        "session_code",
        "round_raw",
        "round_block",
        "eq_price",
        "optimal_holdings",
        "optimal_holdings_smoothed",
        "cap_constrained_optimal_profit",
        "smoothed_cap_constrained_optimal_profit",
        "eq_quantity_net_reallocation",
        "actual_quantity_net_reallocation",
        "allocation_efficiency_gains",
        "allocation_inefficiency_amount",
        "physical_allocation_inefficiency_amount",
        "bundle_smoothing_residual",
        "em_inefficiency_amount",
        "v_inefficiency_amount",
        "em_inefficiency_share",
        "v_inefficiency_share",
        "inefficiency_decomposition_case",
        "decomposition_residual",
        "cap_optimum_minus_actual_net_profit",
        "final_frontier_minus_actual_net_profit",
    ]
    panel[inefficiency_columns].to_csv(
        OUTPUT_DIR / "figure_1b_inefficiency_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    block_counts = (
        panel.groupby("round_block")
        .size()
        .reindex([label for label, _, _ in ROUND_BLOCKS])
        .rename("n_market_periods")
    )

    print("[DONE] Figure 1b realized-net-profit efficiency analysis completed.")
    print("\nMarket-period counts:")
    print(block_counts.to_string())


if __name__ == "__main__":
    main()
