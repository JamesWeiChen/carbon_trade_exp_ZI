import pandas as pd
import statsmodels.formula.api as smf
from tabulate import tabulate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

def get_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def plot_regression_results(results, model_names, y_var, cluster_info, output_path):
    """
    繪製回歸結果的係數圖表
    
    Parameters:
    results: list of regression results
    model_names: list of model names
    y_var: dependent variable name
    cluster_info: string describing clustering method
    output_path: path to save the plot
    """
    # 使用更豐富的顏色方案（從附圖中提取）
    colors = ['#51ABDD', '#227D51', '#B5495B', '#FF6F61', '#9ACD32', '#87CEEB', 
              '#006400', '#DA70D6', '#4169E1', '#A0522D', '#008B8B', '#800080']
    
    # 收集所有變數名稱（包含截距項 Intercept）
    all_vars = []
    for res in results:
        for var in res.params.index:
            if var not in all_vars:
                all_vars.append(var)
    
    if not all_vars:
        return  # 沒有變數可以繪製
    
    # 準備資料
    coefficients = []
    std_errors = []
    p_values = []
    
    for res in results:
        coef_list = []
        se_list = []
        p_list = []
        for var in all_vars:
            if var in res.params.index:
                coef_list.append(res.params[var])
                se_list.append(res.bse[var])
                p_list.append(res.pvalues[var])
            else:
                coef_list.append(np.nan)
                se_list.append(np.nan)
                p_list.append(np.nan)
        coefficients.append(coef_list)
        std_errors.append(se_list)
        p_values.append(p_list)
    
    # 設定圖表參數
    x = np.arange(len(all_vars))
    # 每個變數群組總寬度（留點左右邊界）
    group_total_width = 0.8
    # 單一模型理論寬度（模型之間的中心點間距）
    bar_slot_width = group_total_width / max(len(results), 1)
    # 實際畫柱子的寬度，稍微縮小來產生模型之間的小間隔
    width = bar_slot_width * 0.85  # 15% 的間隔
    
    # 計算y軸範圍
    all_coefs = [c for coef_list in coefficients for c in coef_list if not np.isnan(c)]
    all_ses = [s for se_list in std_errors for s in se_list if not np.isnan(s)]
    
    if not all_coefs:
        return
    
    max_coef = max(all_coefs)
    min_coef = min(all_coefs)
    max_se = max(all_ses) if all_ses else 0
    
    # 為信賴區間留出空間
    y_range = max_coef - min_coef
    y_max = max_coef + max_se + 0.2 * y_range
    y_min = min_coef - max_se - 0.2 * y_range
    
    # 檢查是否需要截斷圖形（更嚴格的條件）
    coef_range = max_coef - min_coef
    total_range = y_max - y_min
    # 使用幅度比率並避免除以零
    denom = abs(min_coef) if abs(min_coef) > 1e-9 else 1e-9
    spread_ratio = abs(max_coef) / denom
    need_break = (coef_range > 10 and total_range > 20) or (coef_range > 5 and total_range > 15 and spread_ratio > 10)
    
    if need_break:
        # 使用截斷圖形（16:9 寬螢幕比例）
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12.8, 7.2))
        
        # 設定截斷範圍
        break_point = (max_coef + min_coef) / 2
        ax1.set_ylim(break_point, y_max)
        ax2.set_ylim(y_min, break_point - 0.5)
        
        # 隱藏連接部分的軸線
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(axis='x', which='both', bottom=False)
        ax2.spines['top'].set_visible(False)
        
        axes = [ax1, ax2]
        
        # 添加波浪線表示截斷
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
    else:
        # 使用普通圖形（16:9 寬螢幕比例）
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.set_ylim(y_min, y_max)
        axes = [ax]
    
    # 繪製柱狀圖
    error_kw = dict(ecolor='lightgray', lw=1, capsize=5, capthick=1)
    rects_list = []
    
    for i, (coefs, ses, model_name) in enumerate(zip(coefficients, std_errors, model_names)):
        color = colors[i % len(colors)]
        # 使用 slot 寬度來安排柱子的中心位置，確保柱子之間有間隔
        x_pos = x + (i - (len(results) - 1) / 2) * bar_slot_width
        
        for ax in axes:
            rects = ax.bar(x_pos, coefs, width, label=f"Model {i+1}", 
                          yerr=ses, capsize=5, color=color, error_kw=error_kw)
            if ax == axes[0]:  # 只在第一個軸上收集 rects
                rects_list.append((rects, p_values[i]))
    
    # 設定標籤和標題
    main_ax = axes[0] if need_break else ax
    main_ax.set_ylabel('Coefficients')
    title = f'Y = {y_var}'
    if cluster_info:
        title += f' | {cluster_info}'
    main_ax.set_title(title)
    
    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x)
    # 讓 Intercept 清楚可見
    xticklabels = [('Intercept' if v == 'Intercept' else v) for v in all_vars]
    bottom_ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    
    # 取消圖例顯示（依使用者需求）
    # main_ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=min(len(results), 6), frameon=False)
    
    # 移除上方和右方的軸線
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='black', linewidth=1)
    
    # 添加顯著性星號
    def add_significance_stars(rects, p_vals, ax):
        for rect, p_val in zip(rects, p_vals):
            if not np.isnan(p_val):
                height = rect.get_height()
                significance = get_stars(p_val)
                if significance:
                    y_offset = 3 if height >= 0 else -18
                    ax.annotate(significance,
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, y_offset),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top')
    
    for rects, p_vals in rects_list:
        for ax in axes:
            add_significance_stars(rects, p_vals, ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] 圖表已儲存至: {output_path}")

def custom_summary_table(results, model_names):
    first_vars = list(results[0].params.index)
    all_vars = list(first_vars)
    for res in results:
        for v in res.params.index:
            if v not in all_vars:
                all_vars.append(v)
    table = []
    for var in all_vars:
        row = [var]
        for res in results:
            if var in res.params.index:
                est = res.params[var]
                p = res.pvalues[var]
                stars = get_stars(p)
                row.append(f"{est:.3f}{stars}")
            else:
                row.append("")
        table.append(row)
        se_row = [""]
        for res in results:
            if var in res.bse.index:
                se = res.bse[var]
                se_row.append(f"({se:.3f})")
            else:
                se_row.append("")
        table.append(se_row)
    info_keys = ["N", "AIC", "BIC"]
    info_dicts = []
    for res in results:
        info = {
            "N": int(res.nobs),
            "AIC": f"{res.aic:.1f}",
            "BIC": f"{res.bic:.1f}"
        }
        info_dicts.append(info)
    for key in info_keys:
        row = [key]
        for info in info_dicts:
            row.append(str(info[key]))
        table.append(row)
    return table, all_vars

def reg_func(df, y_to_formulas, output_prefix, cluster_col=None, extra_effects_func=None):
    txt_outfile = output_prefix + "_reg_results.txt"
    tab_outfile = output_prefix + "_reg_results_tabulate.txt"
    
    # 設定輸出路徑
    output_dir = os.path.dirname(output_prefix)
    # 尋找專案根目錄（含有 analysis 與 doc 的那層）
    cur = os.path.abspath(output_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "analysis")) and os.path.isdir(os.path.join(cur, "doc")):
            project_root = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot find project root containing both 'analysis' and 'doc' folders.")
        cur = parent
    doc_dir = os.path.join(project_root, "doc", "input")
    os.makedirs(doc_dir, exist_ok=True)
    
    with open(txt_outfile, "w", encoding="utf-8") as f_txt, open(tab_outfile, "w", encoding="utf-8") as f_tab:
        for y, formulas in y_to_formulas.items():
            hc3_results, cluster_results, names = [], [], []
            wald_tests_hc3, wald_tests_cluster = [], []
            for i, fml in enumerate(formulas, 1):
                print(f"[REG] {y} model {i}: {fml}")
                names.append(f"({i})")
                # # 決定子樣本
                # if ("tax_rate_low" in fml or "tax_rate_high" in fml):
                #     if "is_tax" in df.columns:
                #         sub_df = df[df["is_tax"] == 1].copy()
                #     else:
                #         sub_df = df
                # elif ("eq_price_low" in fml or "eq_price_high" in fml):
                #     if "is_trade" in df.columns:
                #         sub_df = df[df["is_trade"] == 1].copy()
                #     else:
                #         sub_df = df  # trade 資料本身就都是交易制度
                # else:
                #     sub_df = df
                # 決定子樣本
                # 現在 tax_rate_low / tax_rate_high 不再代表「碳稅組」，
                # 而是交易制度內部的 3/6/9 水準，所以不要再切 is_tax 子樣本。
                rhs_terms = fml.split("~", 1)[1] if "~" in fml else fml
                formula_controls_trade_dummy = "is_trade" in rhs_terms
                if (
                    ("eq_price_low" in fml or "eq_price_high" in fml)
                    and not formula_controls_trade_dummy
                ):
                    if "is_trade" in df.columns:
                        sub_df = df[df["is_trade"] == 1].copy()
                    else:
                        sub_df = df
                else:
                    sub_df = df
                model = smf.ols(fml, data=sub_df, missing="drop")
                res_hc3 = model.fit(cov_type="HC3")
                hc3_results.append(res_hc3)
                # subject-level cluster: group by cluster_col (if provided)
                if cluster_col is not None:
                    if cluster_col not in sub_df.columns:
                        raise ValueError(f"'{cluster_col}' column required for clustering.")
                    used_groups = sub_df.loc[model.data.row_labels, cluster_col]
                    res_cluster = model.fit(cov_type="cluster", cov_kwds={"groups": used_groups})
                    cluster_results.append(res_cluster)
                # Wald tests:123123
                # if all(x in res_hc3.params.index for x in ["is_tax", "is_trade"]):
                #     wald1_hc3 = res_hc3.wald_test('is_tax = is_trade')
                #     wald1_cluster = res_cluster.wald_test('is_tax = is_trade') if cluster_col is not None else None
                # else:
                #     wald1_hc3 = None
                #     wald1_cluster = None
                # if all(x in res_hc3.params.index for x in ["is_tax", "is_trade", "is_tax:is_dominant", "is_trade:is_dominant"]):
                #     wald2_hc3 = res_hc3.wald_test('is_tax + is_tax:is_dominant = is_trade + is_trade:is_dominant')
                #     wald2_cluster = res_cluster.wald_test('is_tax + is_tax:is_dominant = is_trade + is_trade:is_dominant') if cluster_col is not None else None
                # else:
                #     wald2_hc3 = None
                #     wald2_cluster = None
                wald1_hc3 = None
                wald1_cluster = None
                wald2_hc3 = None
                wald2_cluster = None
                #########################################
                wald_tests_hc3.append((i, wald1_hc3, wald2_hc3))
                if cluster_col is not None:
                    wald_tests_cluster.append((i, wald1_cluster, wald2_cluster))
            
            # 繪製圖表
            base_filename = os.path.basename(output_prefix)
            
            for label, results, wald_tests in [
                ("HC3 Robust SE", hc3_results, wald_tests_hc3),
                (f"Clustered SE (by {cluster_col})", cluster_results, wald_tests_cluster) if cluster_col is not None else (None, None, None)
            ]:
                if results is None:
                    continue
                
                # 生成圖表
                plot_filename = f"{base_filename}_{y}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plot_regression_results(results, names, y, label, plot_path)
                
                # 複製圖表到 doc/input
                doc_plot_path = os.path.join(doc_dir, plot_filename)
                try:
                    import shutil
                    shutil.copy2(plot_path, doc_plot_path)
                except Exception as e:
                    print(f"Warning: Could not copy plot to doc/input: {e}")
                
                table, all_vars = custom_summary_table(results, names)
                # Write txt (aligned)
                f_txt.write(f"=== Outcome: {y} | {label} ===\n")
                col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
                header = [f"Y = {y}"] + names
                f_txt.write("  ".join(h.rjust(w) for h, w in zip(header, col_widths)) + "\n")
                for row in table:
                    f_txt.write("  ".join(str(cell).rjust(w) for cell, w in zip(row, col_widths)) + "\n")
                f_txt.write("\n")
                for idx, wald1, wald2 in wald_tests:
                    if wald1 is not None:
                        f_txt.write(f"[Model {idx}] Wald test: is_tax == is_trade:  chi2={float(wald1.statistic):.3f}, p={wald1.pvalue:.4f}\n")
                    if wald2 is not None:
                        f_txt.write(f"[Model {idx}] Wald test: is_tax+is_tax:is_dominant == is_trade+is_trade:is_dominant:  chi2={float(wald2.statistic):.3f}, p={wald2.pvalue:.4f}\n")
                # Simple effects/Wald (extra)
                if extra_effects_func is not None:
                    for idx, res in enumerate(results):
                        extra_lines = extra_effects_func(res, idx, label)
                        if extra_lines:
                            f_txt.write(f"--- Extra effects for model {names[idx]} ({label}) ---\n")
                            for line in extra_lines:
                                f_txt.write(line + "\n")
                            f_txt.write("\n")
                f_txt.write("\n")
                # Write tabulate txt
                f_tab.write(f"=== Outcome: {y} | {label} ===\n")
                f_tab.write(tabulate(table, headers=[f"Y = {y}"] + names, tablefmt="github", showindex=False) + "\n\n")
                for idx, wald1, wald2 in wald_tests:
                    if wald1 is not None:
                        f_tab.write(f"[Model {idx}] Wald test: is_tax == is_trade:  chi2={float(wald1.statistic):.3f}, p={wald1.pvalue:.4f}\n")
                    if wald2 is not None:
                        f_tab.write(f"[Model {idx}] Wald test: is_tax+is_tax:is_dominant == is_trade+is_trade:is_dominant:  chi2={float(wald2.statistic):.3f}, p={wald2.pvalue:.4f}\n")
                # Simple effects/Wald (extra)
                if extra_effects_func is not None:
                    for idx, res in enumerate(results):
                        extra_lines = extra_effects_func(res, idx, label)
                        if extra_lines:
                            f_tab.write(f"--- Extra effects for model {names[idx]} ({label}) ---\n")
                            for line in extra_lines:
                                f_tab.write(line + "\n")
                            f_tab.write("\n")
                f_tab.write("\n")
        f_txt.write("Standard errors in parentheses.\n")
        f_txt.write("* p<.05, ** p<.01, *** p<.001.\n\n")
        f_tab.write("Standard errors in parentheses.\n")
        f_tab.write("* p<.05, ** p<.01, *** p<.001.\n\n")
    
    # 複製文字結果到 doc/input
    try:
        import shutil
        doc_txt_path = os.path.join(doc_dir, os.path.basename(txt_outfile))
        doc_tab_path = os.path.join(doc_dir, os.path.basename(tab_outfile))
        shutil.copy2(txt_outfile, doc_txt_path)
        shutil.copy2(tab_outfile, doc_tab_path)
    except Exception as e:
        print(f"Warning: Could not copy results to doc/input: {e}")
    
    print(f"[DONE] 結果已輸出至: {txt_outfile}\nTabulate: {tab_outfile}")
    print(f"[PLOTS] 圖表已儲存至: {output_dir}")
    print(f"[COPY] 結果已複製至: {doc_dir}")
