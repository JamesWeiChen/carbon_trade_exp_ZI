"""
CTAT 市場層級：回歸 + test + 圖表
--------------------------------------------------------------------------------
用法：
1) 會自動到 analysis/input 下挑最新日期的market_level_clean_YYYY-MM-DD.csv。
2) 在檔案最後面 MODELS 放入想跑的 (formula, 'm1a/m1b/...') 後，直接執行本檔。
"""

from __future__ import annotations
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========== 路徑設定（自動抓最新 market_level_clean_YYYY-MM-DD.csv）=========
BASE = Path(__file__).resolve().parents[1]   # .../analysis
IN_DIR = BASE / "input"
OUT = BASE / "temp"
OUT.mkdir(parents=True, exist_ok=True)

def _pick_latest_market_csv(in_dir: Path) -> Path:
    """
    在 in_dir 中尋找：
      market_level_clean_YYYY-MM-DD.csv
    """
    candidates = []
    for p in in_dir.glob("market_level_clean*.csv"):
        m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", p.stem)
        if m:
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            candidates.append((date_str, p))

    print(f"：{in_dir.resolve()}")


    # 依日期 + mtime 排序後取最新
    candidates.sort(key=lambda t: (t[0], t[1].stat().st_mtime))
    latest_date = candidates[-1][0]
    same_date   = [p for d, p in candidates if d == latest_date]
    infile      = max(same_date, key=lambda p: p.stat().st_mtime)
    print(f"[LOAD] 將讀取：{infile.name}（日期={latest_date}）")
    return infile

INFILE = _pick_latest_market_csv(IN_DIR)

# ========== 建欄位==========
def build_columns_market(df: pd.DataFrame) -> pd.DataFrame:
    reg = df.copy()
    if "app" in reg.columns:
        reg = reg.query("app != 'Stage_MUDA'").copy()

    # 缺欄位補 0（避免公式出錯）
    for c in ["is_tax","is_trade","is_equal","tax_rate","carbon_multiplier","is_dominant"]:
        if c not in reg.columns:
            reg[c] = 0

    # 稅率與均衡價
    reg["tax_rate_low"]  = (reg["tax_rate"] == 3).astype(int)
    reg["tax_rate_high"] = (reg["tax_rate"] == 9).astype(int)
    reg["eq_price"]      = 6 * reg["carbon_multiplier"]
    reg["eq_price_low"]  = (reg["eq_price"] == 3).astype(int)
    reg["eq_price_high"] = (reg["eq_price"] == 9).astype(int)
    reg[["tax_rate_low", "tax_rate_high"]] = reg[["tax_rate_low", "tax_rate_high"]].fillna(0)
    reg[["eq_price_low", "eq_price_high"]] = reg[["eq_price_low", "eq_price_high"]].fillna(0)

    # 交互項
    reg["is_tax_x_is_dominant"]     = reg["is_tax"]   * reg["is_dominant"]
    reg["is_trade_x_is_dominant"]   = reg["is_trade"] * reg["is_dominant"]
    reg["is_tax_x_tax_rate_high"]   = reg["is_tax"]   * reg["tax_rate_high"]
    reg["is_tax_x_tax_rate_low"]    = reg["is_tax"]   * reg["tax_rate_low"]
    reg["is_trade_x_eq_price_low"]  = reg["is_trade"] * reg["eq_price_low"]
    reg["is_trade_x_eq_price_high"] = reg["is_trade"] * reg["eq_price_high"]
    reg["is_trade_x_is_equal"]      = reg["is_trade"] * reg["is_equal"]

    # NaN → 0
    to_fill = ["tax_rate_low","tax_rate_high","eq_price_low","eq_price_high",
               "is_tax_x_is_dominant","is_trade_x_is_dominant",
               "is_tax_x_tax_rate_high","is_tax_x_tax_rate_low",
               "is_trade_x_eq_price_low","is_trade_x_eq_price_high",
               "is_trade_x_is_equal"]
    for c in to_fill:
        if c in reg.columns:
            reg[c] = reg[c].fillna(0)
    return reg

# ========== 回歸收集器 ==========
COEF_ROWS: list[dict] = []   # 係數層級 rows
STAT_ROWS: list[dict] = []   # 模型層級 rows

# 推薦顯示順序（沿用你在市場稿中的清單）
PREFERRED_ORDER = [
    "is_tax", "is_trade", "is_dominant",
    "is_tax:is_dominant", "is_trade_x_is_equal",
    "is_tax_x_is_dominant","is_trade_x_is_dominant",
    "is_tax_x_tax_rate_low","is_tax_x_tax_rate_high",
    "is_trade_x_eq_price_low","is_trade_x_eq_price_high",
    "Intercept", "const"
]
ORDER_MAP = {name: i for i, name in enumerate(PREFERRED_ORDER, start=10)}

def norm_var(name: str) -> str:
    n = str(name).strip()
    return "Intercept" if n.lower() in {"const","intercept"} else n

def to_group(model_tag: str) -> str:
    m = re.match(r"(m\d+)", str(model_tag))
    return m.group(1) if m else "others"

all_summary_file = OUT / "reg_all_results_market.txt"
all_summary_file.write_text("", encoding="utf-8")  # 清空舊檔

def run_regression(data: pd.DataFrame, formula: str, model_tag: str = ""):
    print(f"[REG] {model_tag}: {formula}")
    model = smf.ols(formula, data=data).fit()

    # 文字輸出（summary / robust / VIF）
    with open(all_summary_file, "a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"[Model {model_tag}] regression: {formula}\n\n")
        f.write(model.summary().as_text()); f.write("\n\n")

        robust = model.get_robustcov_results(cov_type="HC3")
        f.write("[Robust SE: HC3] Summary\n")
        f.write(robust.summary().as_text()); f.write("\n\n")

        # VIF
        try:
            X = model.model.exog
            names_for_vif = model.model.exog_names
            vif_list = []
            for i, nm in enumerate(names_for_vif):
                try:
                    vif_val = variance_inflation_factor(X, i)
                except Exception:
                    vif_val = float("nan")
                vif_list.append((nm, vif_val))
            vif_df = pd.DataFrame(vif_list, columns=["term", "VIF"]).query("term != 'Intercept'")
            f.write("VIF：\n"); f.write(vif_df.to_string(index=False)); f.write("\n\n")
        except Exception as e:
            f.write(f"[VIF skipped] {e}\n\n")

    # （HC3 係數與統計）
    dep    = model.model.endog_names
    robust = model.get_robustcov_results(cov_type="HC3")
    names  = list(model.model.exog_names)
    params = robust.params
    bse    = robust.bse
    pvals  = robust.pvalues
    group  = to_group(model_tag)

    for i, nm in enumerate(names):
        v = norm_var(nm)
        COEF_ROWS.append({
            "group": group,
            "model": model_tag,
            "dep": dep,
            "var": v,
            "coef": float(params[i]),
            "p": float(pvals[i]),
            "se": float(bse[i]),
            "order": ORDER_MAP.get(v, 1000 + i)
        })

    STAT_ROWS.append({
        "group": group,
        "model": model_tag,
        "N": int(model.nobs),
        "ll": float(model.llf),
        "AIC": float(model.aic),
        "BIC": float(model.bic)
    })
    return model

# ========== 假設檢定（單卡 + 面板） ==========
HYP_OUT        = OUT / "hypo_tests_market";  HYP_OUT.mkdir(parents=True, exist_ok=True)
HYP_PANEL_OUT  = OUT / "hypo_panels_market"; HYP_PANEL_OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_H0     = "is_tax = is_trade"

def run_eq_test_png(model, model_tag: str, hypothesis: str = DEFAULT_H0, outdir: Path = HYP_OUT):
    rob = model.get_robustcov_results(cov_type="HC3")
    t_res = rob.t_test(hypothesis)
    f_res = rob.f_test(hypothesis)

    t_stat = float(t_res.tvalue); t_pval = float(t_res.pvalue)
    f_stat = float(f_res.fvalue); f_pval = float(f_res.pvalue)
    df_resid = int(rob.df_resid)
    dep = model.model.endog_names

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    ax.axis('off')
    y = 0.92; line_h = 0.13
    ax.text(0.02, y, "Linear Hypothesis Test", fontsize=14, fontweight="bold"); y -= line_h
    ax.text(0.02, y, f"Model: {model_tag}    Dep: {dep}", fontsize=11); y -= line_h*0.95
    ax.text(0.02, y, f"H0: {hypothesis}", fontsize=12); y -= line_h
    ax.text(0.02, y, f"t-test (HC3): t = {t_stat:.3f}, df = {df_resid}, p = {t_pval:.4f}", fontsize=12); y -= line_h*0.95
    ax.text(0.02, y, f"Wald F-test (HC3): F = {f_stat:.3f}, p = {f_pval:.4f}", fontsize=12); y -= line_h

    out_path = outdir / f"linTest_{model_tag}_{dep.replace('/','_')}.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[H-TEST] Saved: {out_path}")
    return {
        "tag": model_tag, "dep": dep, "H0": hypothesis,
        "t": t_stat, "tp": t_pval, "F": f_stat, "Fp": f_pval, "df": df_resid
    }

def save_hypothesis_panel(test_rows: list[dict], title_prefix: str, outdir: Path = HYP_PANEL_OUT):
    if not test_rows: return
    dep = test_rows[0]["dep"]
    n = len(test_rows)
    r, c = (2, 2) if n <= 4 else (int(math.ceil(math.sqrt(n))),) * 2

    fig, axes = plt.subplots(r, c, figsize=(12, 7.5), dpi=220)
    axes = np.array(axes).reshape(r, c)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n: continue
        row = test_rows[i]
        y = 0.95; line = 0.12
        ax.text(0.02, y, f"({i+1}) {row['tag']}", fontsize=12, fontweight="bold"); y -= line
        ax.text(0.02, y, f"H0: {row['H0']}", fontsize=11); y -= line
        ax.text(0.02, y, f"t (HC3) = {row['t']:.3f}, df = {row['df']}, p = {row['tp']:.4f}", fontsize=11); y -= line
        ax.text(0.02, y, f"Wald F (HC3) = {row['F']:.3f}, p = {row['Fp']:.4f}", fontsize=11); y -= line

    fig.suptitle(f"Linear Hypothesis Test • {title_prefix.upper()} • Dep: {dep}",
                 fontsize=14.5, fontweight="bold", y=0.99)
    out = outdir / f"linTest_panel_{title_prefix}_{dep.replace('/','_')}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[H-PANEL] Saved: {out}")

# ========== gen_table_market（合成圖表） ==========
MAX_COLS_PER_FIG = 6
PAGINATE = False
ONLY_USED_VARS = True
COEF_ONE_LINE  = False
LABEL_COL_W = 3.2
MODEL_COL_W = 2.3
ROW_H       = 0.72
COEF_FS     = 11.2
P_FS        = 10.2
LINE_GAP    = 0.15
TOP_SPACE   = 1.70
BOTTOM_SPACE= 1.15
HEAD_GAP    = 0.18
FOOT_GAP    = 0.32
DPI         = 220
TITLE       = "Regression Results"
ROBUST_NOTE = "(Robust SE: HC3)"

def stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ('†' if p < 0.10 else '')))

def render_group_tables(coef_df: pd.DataFrame, stat_df: pd.DataFrame,
                        outdir: Path = OUT,
                        max_cols: int = MAX_COLS_PER_FIG,
                        paginate: bool = PAGINATE):
    coef = coef_df.copy()
    stat = stat_df.copy()

    for col in ["model","dep","var"]:
        if col in coef.columns:
            coef[col] = coef[col].astype(str).str.strip()
    if "model" in stat.columns:
        stat["model"] = stat["model"].astype(str).str.strip()

    def _to_group(m):
        mm = re.match(r"(m\d+)", str(m))
        return mm.group(1) if mm else "others"

    coef["group"] = coef["model"].map(_to_group)
    stat["group"] = stat["model"].map(_to_group)

    def _render_one(gname: str, dep: str, models: list, page_idx: int = 1):
        gdf = coef[(coef["group"] == gname) & (coef["dep"] == dep)].copy()
        if gdf.empty: return

        sub = gdf[gdf["model"].isin(models)].copy()
        if sub.empty: return

        vals = {}
        used_vars_order = []
        for _, r in sub.iterrows():
            v = "Intercept" if str(r["var"]).lower() in {"const","intercept"} else str(r["var"])
            m = r["model"]
            vals.setdefault(v, {})[m] = (float(r["coef"]), float(r["p"]))
            if v not in used_vars_order:
                used_vars_order.append(v)

        has_order = "order" in sub.columns and sub["order"].notna().any()
        if has_order:
            order_df = (
                sub[["var","order"]]
                .drop_duplicates(subset=["var"])
                .assign(var=lambda d: d["var"].astype(str))
                .sort_values(["order","var"], na_position="last")
            )
            ordered = [("Intercept" if v.lower() in {"const","intercept"} else v) for v in order_df["var"].tolist()]
            var_list = [v for v in ordered if v in vals] if ONLY_USED_VARS else ordered
        else:
            var_list = used_vars_order if ONLY_USED_VARS else list(dict.fromkeys(
                ["Intercept" if str(v).lower() in {"const","intercept"} else str(v) for v in gdf["var"]]
            ))

        if "Intercept" in var_list:
            var_list = [v for v in var_list if v != "Intercept"] + ["Intercept"]

        ssub = stat[(stat["group"] == gname) & (stat["model"].isin(models))].set_index("model")

        n_rows = len(var_list) + 4
        width_units = LABEL_COL_W + len(models) * MODEL_COL_W
        fig_w = max(9.0, width_units * 0.82)
        fig_h = TOP_SPACE + n_rows * ROW_H + BOTTOM_SPACE

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
        ax.set_xlim(0, width_units)
        ax.set_ylim(0, n_rows + 2.0)
        ax.axis("off")

        x_label = 0.08
        x_cols  = [LABEL_COL_W + MODEL_COL_W*j + MODEL_COL_W/2 for j in range(len(models))]

        ax.text(0, n_rows + 1.62, TITLE, fontsize=13.5, fontweight="bold", ha="left", va="bottom")
        ax.text(0, n_rows + 1.28, f"Dependent variable: {dep}   {ROBUST_NOTE}".strip(),
                fontsize=10.8, ha="left", va="bottom")

        for j, _m in enumerate(models, start=1):
            ax.text(x_cols[j-1], n_rows + 0.82, f"({j})", ha="center", va="bottom",
                    fontsize=11.6, fontweight="bold")

        y0 = n_rows + 0.69
        ax.hlines(y0, 0, width_units, linewidth=1.4)
        ax.hlines(y0 - HEAD_GAP, 0, width_units, linewidth=1.0)

        y = n_rows + 0.35
        for v in var_list:
            y -= ROW_H
            ax.text(x_label, y, v, ha="left", va="center", fontsize=11.4)
            for j, m in enumerate(models):
                tup = vals.get(v, {}).get(m)
                if not tup: continue
                b, pval = tup
                if COEF_ONE_LINE:
                    ax.text(x_cols[j], y, f"{b:.3f}{stars(pval)} ({pval:.3f})",
                            ha="center", va="center", fontsize=COEF_FS)
                else:
                    ax.text(x_cols[j], y + LINE_GAP, f"{b:.3f}{stars(pval)}",
                            ha="center", va="center", fontsize=COEF_FS)
                    ax.text(x_cols[j], y - LINE_GAP, f"({pval:.3f})",
                            ha="center", va="center", fontsize=P_FS)

        ax.hlines(y - FOOT_GAP, 0, width_units, linewidth=1.1)
        y -= (FOOT_GAP + 0.10)

        stat_rows = [("N","N",False), ("ll","Log-likelihood",False), ("AIC","AIC",True), ("BIC","BIC",True)]
        for key, lab, italic in stat_rows:
            y -= ROW_H * (0.92 if key=="N" else 1.0)
            ax.text(x_label, y, lab, ha="left", va="center", fontsize=11.4,
                    fontstyle=("italic" if italic else "normal"))
            for j, m in enumerate(models):
                if m in ssub.index:
                    if key == "N":
                        val = ssub.loc[m, "N"]
                        s = f"{int(val.iloc[0] if hasattr(val, 'iloc') else val):,}"
                    else:
                        val = ssub.loc[m, key]
                        s = f"{float(val.iloc[0] if hasattr(val, 'iloc') else val):.1f}"
                    ax.text(x_cols[j], y, s, ha="center", va="center", fontsize=11.4)

        ax.text(0, -0.08, "† p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001",
                fontsize=9.6, ha="left", va="top")

        dep_safe = (dep or "dep").replace("/", "_")
        suffix = f"_p{page_idx}" if page_idx > 1 else ""
        out = outdir / f"table_{gname}_{dep_safe}_auto{suffix}_market.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved: {out}")

    for gname, gdf_group in coef.groupby("group", sort=False):
        for dep_val, gdf_dep in gdf_group.groupby("dep", sort=False):
            model_order_all = list(dict.fromkeys(gdf_dep["model"]))
            if not model_order_all: continue
            if not paginate:
                _render_one(gname, dep_val, model_order_all[:max_cols], page_idx=1)
            else:
                page = 1
                for i in range(0, len(model_order_all), max_cols):
                    _render_one(gname, dep_val, model_order_all[i:i+max_cols], page_idx=page)
                    page += 1

# ========== Pipeline ==========
def run_market_pipeline(models: list[tuple[str, str]], hypothesis: str = DEFAULT_H0):
    """
    models: [(formula, "m1a"), (formula, "m1b"), ...]
    """
    # 讀檔 + 建欄
    df = pd.read_csv(INFILE, encoding="utf-8-sig")
    reg = build_columns_market(df)
    print(f"[LOAD] market <= {INFILE.name} (N={len(reg):,})")

    # 逐模型回歸 + 檢定
    tests_by_group: dict[str, list[dict]] = {}
    for (formula, tag) in models:
        model = run_regression(reg, formula, tag)
        try:
            row = run_eq_test_png(model, tag, hypothesis=hypothesis)
            tests_by_group.setdefault(to_group(tag), []).append(row)
        except Exception as e:
            print(f"[WARN] 檢定失敗 {tag}: {e}")

    # 匯出 CSV (給合成表用)
    coef_df = pd.DataFrame(COEF_ROWS, columns=["group","model","dep","var","coef","p","se","order"])
    stat_df = pd.DataFrame(STAT_ROWS, columns=["group","model","N","ll","AIC","BIC"])

    for col in ["group","model","dep","var"]:
        if col in coef_df.columns:
            coef_df[col] = coef_df[col].astype(str).str.strip()

    coef_df.sort_values(["group","model","order","var"], inplace=True)
    stat_df.sort_values(["group","model"], inplace=True)

    coef_csv = OUT / "reg_table_coeffs_market.csv"
    stat_csv = OUT / "reg_table_stats_market.csv"

    coef_df.to_csv(OUT / "reg_table_coeffs_market.csv", index=False, encoding="utf-8-sig")
    stat_df.to_csv(OUT / "reg_table_stats_market.csv", index=False, encoding="utf-8-sig")
    print(f"[CSV] written:\n  {OUT / 'reg_table_coeffs_market.csv'}\n  {OUT / 'reg_table_stats_market.csv'}")

    # 合成 2x2 圖
    for g, rows in tests_by_group.items():
        rows_sorted = sorted(rows, key=lambda r: r["tag"])
        save_hypothesis_panel(rows_sorted, title_prefix=g)

    # 合成表
    render_group_tables(coef_df, stat_df, outdir=OUT)

    print(f"[DONE] 所有市場層級結果已輸出到：{OUT}；彙總txt在：{all_summary_file}")

# ========== 要跑的模型清單 ==========
if __name__ == "__main__":
    MODELS = [
        # m1* : group_emission
        ("group_emission ~ is_tax + is_trade + is_trade_x_is_equal", "m1a"),
        ("group_emission ~ is_tax + is_trade + is_trade_x_is_equal + is_tax_x_tax_rate_low + is_tax_x_tax_rate_high", "m1b"),
        ("group_emission ~ is_tax + is_trade + is_trade_x_is_equal + is_trade_x_eq_price_low + is_trade_x_eq_price_high", "m1c"),

        # m2* : group_profit
        ("group_profit ~ is_tax + is_trade + is_trade_x_is_equal", "m2a"),
        ("group_profit ~ is_tax + is_trade + is_trade_x_is_equal + is_tax_x_tax_rate_low + is_tax_x_tax_rate_high", "m2b"),
        ("group_profit ~ is_tax + is_trade + is_trade_x_is_equal + is_trade_x_eq_price_low + is_trade_x_eq_price_high", "m2c"),

        # m3* : group_production
        ("group_production ~ is_tax + is_trade + is_trade_x_is_equal", "m3a"),
        ("group_production ~ is_tax + is_trade + is_trade_x_is_equal + is_tax_x_tax_rate_low + is_tax_x_tax_rate_high", "m3b"),
        ("group_production ~ is_tax + is_trade + is_trade_x_is_equal + is_trade_x_eq_price_low + is_trade_x_eq_price_high", "m3c"),
    ]
    run_market_pipeline(MODELS, hypothesis=DEFAULT_H0)
