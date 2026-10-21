from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_round_report(
    session_code: str,
    round_num: int,
    player_id: int,
    base_dir: Path | str | None = None,
    dpi: int = 220,
) -> Path:
    """
    Generate a report image for a given session / round / player.
    - Input:
        CTAT/analysis/input/clean_2025-08-29.csv
        CTAT/analysis/input/clean_2025-08-29_trades_long.csv
    - Output:
        CTAT/analysis/output/round_report_{session}_r{round}_p{player}.png
    """
    # ---------- Paths ----------
    if base_dir is None:
        try:
            base_dir = Path(__file__).resolve().parents[2]  # .../CTAT
        except Exception:
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    in_dir = base_dir / "analysis" / "input"
    out_dir = base_dir / "analysis" / "output"
    in_long = in_dir / "clean_2025-08-29.csv"
    in_trades = in_dir / "clean_2025-08-29_trades_long.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ----------
    df_long = pd.read_csv(in_long, encoding="utf-8-sig")
    df_tr = pd.read_csv(in_trades, encoding="utf-8-sig")

    # ---------- Filter trades ----------
    t = df_tr[
        (df_tr["session_code"] == session_code)
        & (df_tr["round"] == round_num)
        & ((df_tr["buyer_id"] == player_id) | (df_tr["seller_id"] == player_id))
    ].copy()

    # Cash change from player's perspective
    t["Change"] = np.where(t["buyer_id"] == player_id,
                           -t["price"] * t["quantity"],
                           t["price"] * t["quantity"])

    # Sort & keep columns
    sort_cols = ["seconds"]
    if "price" in t.columns:
        sort_cols.append("price")
    t = t.sort_values(sort_cols, kind="mergesort")
    desired = ["round", "seconds", "buyer_id", "seller_id", "price", "quantity", "Change"]
    t = t[[c for c in desired if c in t.columns]].copy()

    # ---------- Long format info ----------
    l = df_long[
        (df_long["session_code"] == session_code)
        & (df_long["round"] == round_num)
        & (df_long["app"] == "Stage_CarbonTrading")
        & (df_long["participant_id_in_session"] == player_id)
    ].copy()

    if l.empty:
        raise ValueError("No matching record found in long-format data "
                         "(check app=Stage_CarbonTrading / session_code / round / player_id).")

    row = l.iloc[0]

    def pick(r, names, default=0):
        for n in names:
            if n in r and pd.notna(r[n]):
                return r[n]
        return default

    market_price = float(pick(row, ["market_price"]))
    production = float(pick(row, ["individual_production"]))
    total_cost = float(pick(row, ["total_cost", "production_total_cost", "total_production_cost", "cost_total"]))
    round_payoff = float(pick(row, ["round_payoff"]))

    cash_change_sum = float(t["Change"].sum()) if not t.empty else 0.0
    prod_profit = market_price * production - total_cost
    total_reward_calc = prod_profit + cash_change_sum
    diff_reward = total_reward_calc - round_payoff

    # ---------- Plot ----------
    fig = plt.figure(figsize=(10.5, 6.2), dpi=dpi)

    # Top: trade table
    ax1 = fig.add_axes([0.05, 0.47, 0.90, 0.48])
    ax1.axis("off")
    title = f"session={session_code} | round={round_num} | player_id={player_id}"
    ax1.text(0.0, 1.04, title, fontsize=13, fontweight="bold", transform=ax1.transAxes)

    if t.empty:
        ax1.text(0.02, 0.75, "No trades", fontsize=12)
        trade_table_data = []
        col_labels = ["round", "seconds", "buyer_id", "seller_id", "price", "quantity", "Change"]
    else:
        trade_table_data = t.values.tolist()
        col_labels = t.columns.tolist()

    tbl1 = ax1.table(cellText=trade_table_data, colLabels=col_labels,
                     loc="upper left", cellLoc="center")
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(11)
    tbl1.scale(1, 1.2)

    # Bottom: summary
    ax2 = fig.add_axes([0.05, 0.05, 0.90, 0.36])
    ax2.axis("off")

    # --- Left summary (production + trade cashflow) ---
    left_labels = ["player_id", "market_price", "production", "total_cost",
                   "net_profit_from_production", "net_trade_cashflow"]
    left_values = [player_id, f"{market_price:.0f}", f"{production:.0f}",
                   f"{total_cost:.0f}", f"{prod_profit:.0f}", f"{cash_change_sum:.0f}"]

    # --- Right summary (only payoff info) ---
    right_labels = ["total_payoff (calculated)",
                    "total_payoff",
                    "difference (calc - payoff)"]
    right_values = [f"{total_reward_calc:.0f}", f"{round_payoff:.0f}", f"{diff_reward:.0f}"]

    tbl2 = ax2.table(cellText=[[l, v] for l, v in zip(left_labels, left_values)],
                     colLabels=["", ""], cellLoc="center",
                     loc="upper left", bbox=[0.00, 0.00, 0.48, 1.00])
    tbl3 = ax2.table(cellText=[[l, v] for l, v in zip(right_labels, right_values)],
                     colLabels=["", ""], cellLoc="center",
                     loc="upper right", bbox=[0.52, 0.00, 0.48, 1.00])

    for tbl in (tbl2, tbl3):
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1, 1.2)

    out_path = out_dir / f"round_report_{session_code}_r{round_num}_p{player_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


# Example
generate_round_report("m8hhe0br", 4, 4)
generate_round_report("m8hhe0br", 4, 2)
generate_round_report("m8hhe0br", 4, 3)
generate_round_report("m8hhe0br", 4, 13)