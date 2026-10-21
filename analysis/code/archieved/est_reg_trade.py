
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ===================== 路徑設定（抓最新的 trade_clean_CT_* 檔） =====================
BASE   = Path(__file__).resolve().parents[1]   # .../analysis
IN_DIR = BASE / "input"
OUT    = BASE / "temp"
OUT.mkdir(parents=True, exist_ok=True)

def _pick_latest_trade_csv(in_dir: Path) -> Path:
    pat = re.compile(r"^trade_clean_CT_(\d{4})-(\d{2})-(\d{2})_t(\d+)-(\d+)s\.csv$")
    candidates = []
    print(f"[SCAN] 掃描：{in_dir.resolve()}")
    for p in in_dir.glob("trade_clean_CT_*.csv"):
        m = pat.match(p.name)
        if m:
            y, mo, d, t0, t1 = m.groups()
            candidates.append(((f"{y}-{mo}-{d}", int(t0), int(t1)), p))
    if not candidates:
        found = [p.name for p in in_dir.glob("trade_clean_CT_*.csv")]
        raise FileNotFoundError(f"找不到 trade_clean_CT_YYYY-MM-DD_tX-Ys.csv；目前：{found}")

    for (d, s, e), p in sorted(candidates, key=lambda it: (it[0][0], it[0][1], it[0][2], it[1].stat().st_mtime)):
        print(f"  - 發現：{p.name}（日期={d}，t={s}-{e}s）")
    (_, _, _), infile = max(candidates, key=lambda it: (it[0][0], it[0][2], it[0][1], it[1].stat().st_mtime))
    print(f"[LOAD] 讀取：{infile.resolve()}")
    return infile

INFILE = _pick_latest_trade_csv(IN_DIR)

# ===================== 工具 =====================
def _coerce_01(s: pd.Series) -> pd.Series:
    """把 0/1/True/False/字串0/1 轉成 0/1（保留 NaN）。"""
    if s is None:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(s, errors="coerce")
    mask_bool = s.astype(str).str.lower().isin(["true", "false"])
    if mask_bool.any():
        out.loc[mask_bool] = s[mask_bool].astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    return out

def _value_check(df: pd.DataFrame, cols: list[str], title: str):
    print(f"\n[CHECK] {title}")
    for c in cols:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False).sort_index()
            print(f"  {c}: {dict(vc)}")
        else:
            print(f"  {c}: <不存在>")

def _vars_in_formula(formula: str) -> tuple[set[str], str]:
    """粗略解析公式，抓出自變數名稱集合與應變數名稱（僅支援 ~ 和 +）。"""
    left, right = formula.split("~", 1)
    dep = left.strip()
    toks = [t.strip() for t in right.replace(" ", "").split("+") if t.strip()]
    names = [t for t in toks if re.fullmatch(r"[A-Za-z_]\w*", t)]
    return set(names), dep

# ===================== 建欄位（交易層級，不做任何推估） =====================
def build_columns_trade(df: pd.DataFrame) -> pd.DataFrame:
    reg = df.copy()

    # 只在欄位存在時才處理；否則保留為 NaN（不推、不補）
    if "is_equal" in reg.columns:
        reg["is_equal"] = _coerce_01(reg["is_equal"]).astype("float64")
    else:
        reg["is_equal"] = np.nan

    if "eq_price_low" in reg.columns:
        reg["eq_price_low"] = _coerce_01(reg["eq_price_low"]).astype("float64")
    else:
        reg["eq_price_low"] = np.nan

    if "eq_price_high" in reg.columns:
        reg["eq_price_high"] = _coerce_01(reg["eq_price_high"]).astype("float64")
    else:
        reg["eq_price_high"] = np.nan

    # 交互項：只有在基底欄位都存在時才建立；否則維持 NaN
    if {"is_equal", "eq_price_low"}.issubset(reg.columns):
        reg["is_equal_x_eq_price_low"] = (_coerce_01(reg["is_equal"]) * _coerce_01(reg["eq_price_low"])).astype("float64")
    else:
        reg["is_equal_x_eq_price_low"] = np.nan

    if {"is_equal", "eq_price_high"}.issubset(reg.columns):
        reg["is_equal_x_eq_price_high"] = (_coerce_01(reg["is_equal"]) * _coerce_01(reg["eq_price_high"])).astype("float64")
    else:
        reg["is_equal_x_eq_price_high"] = np.nan

    _value_check(reg, ["is_equal", "eq_price_low", "eq_price_high",
                       "is_equal_x_eq_price_low", "is_equal_x_eq_price_high"],
                 "Dummy/交互項分布（NaN 代表來源資料沒有給到）")
    return reg

# ===================== 回歸集合與輸出 =====================
COEF_ROWS: list[dict] = []
STAT_ROWS: list[dict] = []

PREFERRED_ORDER = [
    "is_equal",
    "eq_price_low", "eq_price_high",
    "is_equal_x_eq_price_low", "is_equal_x_eq_price_high",
    "Intercept", "const"
]
ORDER_MAP = {name: i for i, name in enumerate(PREFERRED_ORDER, start=10)}

all_summary_file = OUT / "reg_all_results_trade.txt"
all_summary_file.write_text("", encoding="utf-8")  # 清空

def norm_var(name: str) -> str:
    n = str(name).strip()
    return "Intercept" if n.lower() in {"const", "intercept"} else n

def to_group(model_tag: str) -> str:
    m = re.match(r"(m\d+)", str(model_tag))
    return m.group(1) if m else "others"

def run_regression(data: pd.DataFrame, formula: str, model_tag: str = ""):
    print(f"[REG] {model_tag}: {formula}")
    model = smf.ols(formula, data=data).fit()

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
            "group": group, "model": model_tag, "dep": dep, "var": v,
            "coef": float(params[i]), "p": float(pvals[i]), "se": float(bse[i]),
            "order": ORDER_MAP.get(v, 1000 + i)
        })
    STAT_ROWS.append({
        "group": group, "model": model_tag,
        "N": int(model.nobs), "ll": float(model.llf),
        "AIC": float(model.aic), "BIC": float(model.bic)
    })
    return model

# ===================== 合成圖 =====================
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
    coef = coef_df.copy(); stat = stat_df.copy()
    for col in ["model", "dep", "var"]:
        if col in coef.columns: coef[col] = coef[col].astype(str).str.strip()
    if "model" in stat.columns: stat["model"] = stat["model"].astype(str).str.strip()

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
            v = "Intercept" if str(r["var"]).lower() in {"const", "intercept"} else str(r["var"])
            m = r["model"]
            vals.setdefault(v, {})[m] = (float(r["coef"]), float(r["p"]))
            if v not in used_vars_order: used_vars_order.append(v)

        has_order = "order" in sub.columns and sub["order"].notna().any()
        if has_order:
            order_df = (sub[["var", "order"]]
                        .drop_duplicates(subset=["var"])
                        .assign(var=lambda d: d["var"].astype(str))
                        .sort_values(["order", "var"], na_position="last"))
            ordered = [("Intercept" if v.lower() in {"const", "intercept"} else v)
                       for v in order_df["var"].tolist()]
            var_list = [v for v in ordered if v in vals] if ONLY_USED_VARS else ordered
        else:
            var_list = used_vars_order if ONLY_USED_VARS else list(dict.fromkeys(
                ["Intercept" if str(v).lower() in {"const", "intercept"} else str(v) for v in gdf["var"]]
            ))

        if "Intercept" in var_list:
            var_list = [v for v in var_list if v != "Intercept"] + ["Intercept"]

        ssub = stat[(stat["group"] == gname) & (stat["model"].isin(models))].set_index("model")

        n_rows = len(var_list) + 4
        width_units = LABEL_COL_W + len(models) * MODEL_COL_W
        fig_w = max(9.0, width_units * 0.82)
        fig_h = TOP_SPACE + n_rows * ROW_H + BOTTOM_SPACE

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
        ax.set_xlim(0, width_units); ax.set_ylim(0, n_rows + 2.0); ax.axis("off")
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
                ax.text(x_cols[j], y + 0.15, f"{b:.3f}{'***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else '†' if pval<0.10 else ''}",
                        ha="center", va="center", fontsize=COEF_FS)
                ax.text(x_cols[j], y - 0.15, f"({pval:.3f})",
                        ha="center", va="center", fontsize=P_FS)

        ax.hlines(y - FOOT_GAP, 0, width_units, linewidth=1.1)
        y -= (FOOT_GAP + 0.10)
        for key, lab, italic in [("N","N",False), ("ll","Log-likelihood",False),
                                 ("AIC","AIC",True), ("BIC","BIC",True)]:
            y -= ROW_H * (0.92 if key=="N" else 1.0)
            ax.text(x_label, y, lab, ha="left", va="center", fontsize=11.4,
                    fontstyle=("italic" if italic else "normal"))
            for j, m in enumerate(models):
                if m in ssub.index:
                    if key == "N":
                        val = ssub.loc[m, "N"]; s = f"{int(val.iloc[0] if hasattr(val, 'iloc') else val):,}"
                    else:
                        val = ssub.loc[m, key]; s = f"{float(val.iloc[0] if hasattr(val, 'iloc') else val):.1f}"
                    ax.text(x_cols[j], y, s, ha="center", va="center", fontsize=11.4)
        ax.text(0, -0.08, "† p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001",
                fontsize=9.6, ha="left", va="top")

        dep_safe = (dep or "dep").replace("/", "_")
        suffix = f"_p{page_idx}" if page_idx > 1 else ""
        out = outdir / f"table_{gname}_{dep_safe}_auto{suffix}_trade.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"[OK] Saved: {out}")

    for gname, gdf_group in coef.groupby("group", sort=False):
        for dep_val, gdf_dep in gdf_group.groupby("dep", sort=False):
            model_order_all = list(dict.fromkeys(gdf_dep["model"]))
            if not model_order_all: continue
            _render_one(gname, dep_val, model_order_all[:max_cols], page_idx=1)

# ===================== Pipeline =====================
def run_trade_pipeline(models: list[tuple[str, str]]):
    df = pd.read_csv(INFILE, encoding="utf-8-sig")
    reg = build_columns_trade(df)
    print(f"[LOAD] trade <= {INFILE.name} (N={len(reg):,})")

    for nm in ["eq_price_low", "eq_price_high"]:
        if nm in reg.columns and reg[nm].notna().any() == False:
            print(f"[WARN] {nm} 欄位存在但皆為 NaN；相關模型可能被跳過。")
        if nm not in reg.columns:
            print(f"[WARN] 缺少欄位：{nm}；相關模型將跳過。")

    # 若模型需要的欄位缺失 → 跳過
    present_cols = set(reg.columns)
    filtered_models: list[tuple[str, str]] = []
    for (formula, tag) in models:
        need_vars, dep = _vars_in_formula(formula)
        missing = [v for v in need_vars | {dep} if v not in present_cols]
        if missing:
            print(f"[SKIP] {tag}: 缺少欄位 {missing} → {formula}")
            continue
        filtered_models.append((formula, tag))

    if not filtered_models:
        print("[STOP] 沒有可執行的模型（皆因缺欄位被跳過）。")
        return

    # 逐模型回歸
    for (formula, tag) in filtered_models:
        _ = run_regression(reg, formula, tag)

    # 匯出 CSV
    coef_df = pd.DataFrame(COEF_ROWS, columns=["group","model","dep","var","coef","p","se","order"])
    stat_df = pd.DataFrame(STAT_ROWS, columns=["group","model","N","ll","AIC","BIC"])
    for col in ["group","model","dep","var"]:
        if col in coef_df.columns:
            coef_df[col] = coef_df[col].astype(str).str.strip()
    coef_df.sort_values(["group","model","order","var"], inplace=True)
    stat_df.sort_values(["group","model"], inplace=True)

    coef_csv = OUT / "reg_table_coeffs_trade.csv"
    stat_csv = OUT / "reg_table_stats_trade.csv"
    coef_df.to_csv(coef_csv, index=False, encoding="utf-8-sig")
    stat_df.to_csv(stat_csv, index=False, encoding="utf-8-sig")
    print(f"[CSV] written:\n  {coef_csv}\n  {stat_csv}")

    render_group_tables(coef_df, stat_df, outdir=OUT)
    print(f"[DONE] 交易層級結果已輸出到：{OUT}；彙總txt：{all_summary_file}")

# ========== 要跑的模型清單（改這裡就好） ==========
if __name__ == "__main__":
    MODELS = [
        ("price_vwap ~ is_equal", "m1a"),
        ("price_vwap ~ is_equal + eq_price_low + eq_price_high", "m1b"),
        ("price_vwap ~ is_equal + eq_price_low + eq_price_high + is_equal_x_eq_price_low + is_equal_x_eq_price_high", "m1c"),

        ("price_median ~ is_equal", "m2a"),
        ("price_median ~ is_equal + eq_price_low + eq_price_high", "m2b"),
        ("price_median ~ is_equal + eq_price_low + eq_price_high + is_equal_x_eq_price_low + is_equal_x_eq_price_high", "m2c"),

        ("price_diff_vwap ~ is_equal", "m3a"),
        ("price_diff_vwap ~ is_equal + eq_price_low + eq_price_high", "m3b"),
        ("price_diff_vwap ~ is_equal + eq_price_low + eq_price_high + is_equal_x_eq_price_low + is_equal_x_eq_price_high", "m3c"),

        ("price_diff_median ~ is_equal", "m4a"),
        ("price_diff_median ~ is_equal + eq_price_low + eq_price_high", "m4b"),
        ("price_diff_median ~ is_equal + eq_price_low + eq_price_high + is_equal_x_eq_price_low + is_equal_x_eq_price_high", "m4c"),
    ]
    run_trade_pipeline(MODELS)
