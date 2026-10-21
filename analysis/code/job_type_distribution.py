# %% CTAT: distributions and exploratory comparisons by work type
"""Analyze participant characteristics and experimental outcomes by env_job_type.

Expected project layout (same convention as the other CTAT scripts):

    project/
      input/subj_lv_panel.csv
      code/ctat_job_type_distribution_analysis.py
      output/job_type_analysis/

The input is a participant-round panel. Counts and demographics are calculated
once per participant. Experimental outcomes are first summarized within each
participant, so a participant with many rounds is not counted as many people.
"""

from __future__ import annotations

import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu


# =============================================================================
# Part 0: Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "input", "subj_lv_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "job_type_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Part 1: Settings
# =============================================================================
PARTICIPANT_COL = "participant_code"
JOB_COL = "env_job_type"
EXPERIENCE_COL = "work_experience"

# Short English labels keep figures readable even on computers without a
# Chinese font. The original Chinese labels are retained in every data table.
JOB_SHORT_NAMES = {
    "企業內部永續、ESG、環安衛或環境管理人員": "Corporate ESG/EHS",
    "顧問、查證、會計師事務所或其他專業服務人員": "Professional services",
    "NGO、倡議、研究或政策推動相關人員": "NGO/research/policy",
    "其他與減碳、碳資訊或環境管理有關的工作者": "Other carbon/env.",
    "綠能、節能或環境技術解決方案公司人員": "Green/env. tech",
    "其他": "Other",
}

BASELINE_NUMERIC = [
    "age",
    "work_experience",
    "trade_experience",
    "emission_experience",
    "ind_risk",
    "performance_pressure",
    "prevention_climate",
    "carbon_trade_environment",
    "carbon_trade_mkt_power",
]

BASELINE_CATEGORICAL = [
    "male",
    "background",
    "econ_or_bz",
    "env",
    "work_experienced",
    "trade_experienced",
    "emission_experienced",
    "main_goal",
    "carbon_trade_efficiency",
    "carbon_trade_fairness",
]


# =============================================================================
# Part 2: Helpers
# =============================================================================
def bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return pd.Series(result, index=p_values.index)

    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(
        os.path.join(OUTPUT_DIR, filename),
        index=False,
        encoding="utf-8-sig",
    )


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("subj_lv_panel.csv 缺少必要欄位：" + ", ".join(missing))


def cramer_v_bias_corrected(table: pd.DataFrame) -> tuple[float, float, float]:
    """Return bias-corrected Cramer's V, minimum expected count, and share < 5."""
    chi2, _, _, expected = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    rows, cols = table.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    cols_corrected = cols - ((cols - 1) ** 2) / (n - 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    value = np.sqrt(phi2_corrected / denominator) if denominator > 0 else np.nan
    return float(value), float(expected.min()), float((expected < 5).mean())


def make_boxplot_grid(
    subject: pd.DataFrame,
    metrics: list[tuple[str, str]],
    job_order: list[str],
    filename: str,
    title: str,
    robust_display_limits: bool = False,
) -> None:
    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    positions = np.arange(1, len(job_order) + 1)

    for axis, (column, label) in zip(axes, metrics):
        groups = [
            pd.to_numeric(
                subject.loc[subject[JOB_COL] == job, column], errors="coerce"
            ).dropna().to_numpy()
            for job in job_order
        ]
        axis.boxplot(groups, positions=positions, widths=0.6, showfliers=False)
        rng = np.random.default_rng(20260920)
        for position, values in zip(positions, groups):
            if len(values):
                jitter = rng.normal(position, 0.045, len(values))
                axis.scatter(jitter, values, s=15, alpha=0.55, color="#2B6CB0")
        if robust_display_limits:
            pooled = np.concatenate([values for values in groups if len(values)])
            if len(pooled) >= 20:
                lower, upper = np.nanquantile(pooled, [0.01, 0.99])
                if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                    padding = (upper - lower) * 0.08
                    axis.set_ylim(lower - padding, upper + padding)
                    clipped = int(((pooled < lower) | (pooled > upper)).sum())
                    if clipped:
                        axis.text(
                            0.99,
                            0.02,
                            f"{clipped} extreme point(s) outside display range",
                            transform=axis.transAxes,
                            ha="right",
                            va="bottom",
                            fontsize=7,
                            color="#555555",
                        )
        axis.set_title(label)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"J{i + 1}" for i in range(len(job_order))])
        axis.grid(axis="y", alpha=0.25)

    for axis in axes[len(metrics) :]:
        axis.remove()

    fig.suptitle(title, fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Part 3: Load and validate the panel
# =============================================================================
if not os.path.exists(INPUT):
    raise FileNotFoundError(
        "找不到 input/subj_lv_panel.csv。請將資料檔放在專案的 input 資料夾，"
        "並命名為 subj_lv_panel.csv。"
    )

print(f"[LOAD] {INPUT}")
panel = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

require_columns(
    panel,
    [
        PARTICIPANT_COL,
        "session_code",
        "round_raw",
        "is_trade",
        "is_dominant",
        JOB_COL,
        EXPERIENCE_COL,
        "age",
        "net_profit",
        "production",
        "emission",
        "permit_gap",
        "net_permit_trade",
    ],
)

numeric_candidates = list(
    dict.fromkeys(
        BASELINE_NUMERIC
        + [
            "round_raw",
            "is_trade",
            "is_dominant",
            "net_profit",
            "production",
            "emission",
            "permit_gap",
            "net_permit_trade",
            "buy_completion_rate",
            "sell_completion_rate",
        ]
    )
)
for column in numeric_candidates:
    if column in panel.columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

if panel[PARTICIPANT_COL].isna().any():
    raise ValueError("participant_code 有缺值，無法可靠地計算受試者人數。")

if panel[JOB_COL].isna().any():
    panel[JOB_COL] = panel[JOB_COL].fillna("未填答")

# Verify that baseline variables do not change across rounds for the same person.
stable_candidates = [
    column
    for column in [JOB_COL] + BASELINE_NUMERIC + BASELINE_CATEGORICAL
    if column in panel.columns
]
stability = (
    panel.groupby(PARTICIPANT_COL)[stable_candidates]
    .nunique(dropna=False)
    .max()
)
unstable = stability[stability > 1]
if not unstable.empty:
    raise ValueError(
        "以下受試者背景欄位在同一人不同回合不一致："
        + ", ".join(unstable.index)
    )


# =============================================================================
# Part 4: Build one-row-per-participant data
# =============================================================================
sort_columns = [PARTICIPANT_COL, "round_raw"]
baseline = (
    panel.sort_values(sort_columns)
    .drop_duplicates(PARTICIPANT_COL)
    .copy()
)

# Relative profit compares a person with participants in the same
# session × experimental round × dominant/non-dominant role. This removes much
# of the mechanical profit difference caused by market condition and firm type.
cell_columns = ["session_code", "round_raw", "is_dominant"]
panel["net_profit_role_diff"] = panel["net_profit"] - panel.groupby(
    cell_columns, dropna=False
)["net_profit"].transform("mean")
panel["abs_permit_gap"] = panel["permit_gap"].abs()
panel["abs_net_permit_trade"] = panel["net_permit_trade"].abs()

outcome_specs = {
    "control_mean_net_profit": (0, "net_profit", "mean"),
    "trade_mean_net_profit": (1, "net_profit", "mean"),
    "control_mean_emission": (0, "emission", "mean"),
    "trade_mean_emission": (1, "emission", "mean"),
    "control_mean_production": (0, "production", "mean"),
    "trade_mean_production": (1, "production", "mean"),
    "trade_mean_relative_profit": (1, "net_profit_role_diff", "mean"),
    "trade_sd_net_profit": (1, "net_profit", "std"),
    "trade_mean_abs_permit_gap": (1, "abs_permit_gap", "mean"),
    "trade_mean_abs_net_permit_trade": (1, "abs_net_permit_trade", "mean"),
    "trade_mean_buy_completion": (1, "buy_completion_rate", "mean"),
    "trade_mean_sell_completion": (1, "sell_completion_rate", "mean"),
}

outcome_frames = []
for output_name, (trade_value, source_column, function) in outcome_specs.items():
    if source_column not in panel.columns:
        continue
    one = (
        panel.loc[panel["is_trade"] == trade_value]
        .groupby(PARTICIPANT_COL)[source_column]
        .agg(function)
        .rename(output_name)
    )
    outcome_frames.append(one)

participant_outcomes = pd.concat(outcome_frames, axis=1).reset_index()
subject = baseline.merge(
    participant_outcomes,
    how="left",
    on=PARTICIPANT_COL,
    validate="one_to_one",
)
subject["trade_minus_control_net_profit"] = (
    subject["trade_mean_net_profit"] - subject["control_mean_net_profit"]
)
subject["trade_minus_control_production"] = (
    subject["trade_mean_production"] - subject["control_mean_production"]
)
subject["job_type_short"] = subject[JOB_COL].map(JOB_SHORT_NAMES).fillna(
    subject[JOB_COL]
)

job_order = subject[JOB_COL].value_counts().index.tolist()
subject_count = subject[PARTICIPANT_COL].nunique()


# =============================================================================
# Part 5: Counts, mappings, and subject-level analysis file
# =============================================================================
counts = (
    subject.groupby([JOB_COL, "job_type_short"], dropna=False)
    .agg(n_participants=(PARTICIPANT_COL, "nunique"))
    .reset_index()
)
counts["share_of_participants"] = counts["n_participants"] / subject_count
counts = counts.sort_values("n_participants", ascending=False)
save_csv(counts, "job_type_counts.csv")

mapping = counts[[JOB_COL, "job_type_short", "n_participants"]].copy()
mapping.insert(0, "job_code", [f"J{i + 1}" for i in range(len(mapping))])
save_csv(mapping, "job_type_category_mapping.csv")

subject_output_columns = list(
    dict.fromkeys(
        [
            PARTICIPANT_COL,
            "session_code",
            "subject_id",
            JOB_COL,
            "job_type_short",
        ]
        + BASELINE_NUMERIC
        + BASELINE_CATEGORICAL
        + ["is_dominant"]
        + list(participant_outcomes.columns.drop(PARTICIPANT_COL))
        + ["trade_minus_control_net_profit", "trade_minus_control_production"]
    )
)
subject_output_columns = [
    column for column in subject_output_columns if column in subject.columns
]
save_csv(subject[subject_output_columns], "job_type_subject_level.csv")


# =============================================================================
# Part 6: Descriptive distributions by work type
# =============================================================================
derived_numeric = [
    "control_mean_net_profit",
    "trade_mean_net_profit",
    "trade_minus_control_net_profit",
    "trade_mean_relative_profit",
    "trade_sd_net_profit",
    "control_mean_production",
    "trade_mean_production",
    "trade_minus_control_production",
    "trade_mean_abs_permit_gap",
    "trade_mean_abs_net_permit_trade",
    "trade_mean_buy_completion",
    "trade_mean_sell_completion",
]
numeric_metrics = [
    column
    for column in BASELINE_NUMERIC + derived_numeric
    if column in subject.columns and subject[column].notna().any()
]

descriptive_rows = []
for job in job_order:
    job_data = subject.loc[subject[JOB_COL] == job]
    for metric in numeric_metrics:
        values = pd.to_numeric(job_data[metric], errors="coerce").dropna()
        descriptive_rows.append(
            {
                JOB_COL: job,
                "job_type_short": JOB_SHORT_NAMES.get(job, job),
                "n_participants_in_job_type": job_data[PARTICIPANT_COL].nunique(),
                "metric": metric,
                "n_nonmissing": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )

descriptive = pd.DataFrame(descriptive_rows)
save_csv(descriptive, "job_type_descriptive_stats.csv")


# =============================================================================
# Part 7: Exploratory omnibus tests and effect sizes
# =============================================================================
numeric_test_rows = []
for metric in numeric_metrics:
    groups = []
    included_jobs = []
    for job in job_order:
        values = pd.to_numeric(
            subject.loc[subject[JOB_COL] == job, metric], errors="coerce"
        ).dropna()
        if len(values) > 0:
            groups.append(values.to_numpy())
            included_jobs.append(job)

    if len(groups) < 2:
        continue
    statistic, p_value = kruskal(*groups)
    n = sum(len(values) for values in groups)
    k = len(groups)
    epsilon_squared = max(0.0, (statistic - k + 1) / (n - k)) if n > k else np.nan
    numeric_test_rows.append(
        {
            "metric": metric,
            "test": "Kruskal-Wallis",
            "n": n,
            "n_job_types": k,
            "statistic": statistic,
            "p_value": p_value,
            "epsilon_squared": epsilon_squared,
        }
    )

numeric_screening = pd.DataFrame(numeric_test_rows)
numeric_screening["fdr_q_value"] = bh_adjust(numeric_screening["p_value"])
numeric_screening["exploratory_flag"] = np.select(
    [numeric_screening["fdr_q_value"] < 0.05, numeric_screening["p_value"] < 0.05],
    ["FDR q < .05", "raw p < .05 only"],
    default="no clear difference",
)
numeric_screening = numeric_screening.sort_values(["fdr_q_value", "p_value"])
save_csv(numeric_screening, "job_type_numeric_screening.csv")

categorical_rows = []
for metric in [column for column in BASELINE_CATEGORICAL if column in subject.columns]:
    analysis = subject[[JOB_COL, metric]].dropna()
    table = pd.crosstab(analysis[JOB_COL], analysis[metric])
    if table.shape[0] < 2 or table.shape[1] < 2:
        continue
    statistic, p_value, dof, _ = chi2_contingency(table, correction=False)
    cramer_v, min_expected, share_expected_under_5 = cramer_v_bias_corrected(table)
    categorical_rows.append(
        {
            "metric": metric,
            "test": "Chi-square",
            "n": int(table.to_numpy().sum()),
            "n_job_types": table.shape[0],
            "n_categories": table.shape[1],
            "statistic": statistic,
            "degrees_of_freedom": dof,
            "p_value": p_value,
            "cramers_v_bias_corrected": cramer_v,
            "minimum_expected_count": min_expected,
            "share_expected_cells_under_5": share_expected_under_5,
            "small_cell_warning": min_expected < 5,
        }
    )

categorical_screening = pd.DataFrame(categorical_rows)
categorical_screening["fdr_q_value"] = bh_adjust(
    categorical_screening["p_value"]
)
categorical_screening = categorical_screening.sort_values(
    ["fdr_q_value", "p_value"]
)
save_csv(categorical_screening, "job_type_categorical_screening.csv")


# =============================================================================
# Part 8: Pairwise follow-up for promising numeric metrics
# =============================================================================
# Follow up outcomes with raw omnibus p < .10. The table still reports adjusted
# p-values, so a small unadjusted p-value is not presented as confirmatory proof.
promising_metrics = numeric_screening.loc[
    numeric_screening["p_value"] < 0.10, "metric"
].tolist()
pairwise_rows = []
for metric in promising_metrics:
    for job_a, job_b in combinations(job_order, 2):
        a = pd.to_numeric(
            subject.loc[subject[JOB_COL] == job_a, metric], errors="coerce"
        ).dropna()
        b = pd.to_numeric(
            subject.loc[subject[JOB_COL] == job_b, metric], errors="coerce"
        ).dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        statistic, p_value = mannwhitneyu(a, b, alternative="two-sided")
        pairwise_rows.append(
            {
                "metric": metric,
                "job_type_a": job_a,
                "job_type_b": job_b,
                "n_a": len(a),
                "n_b": len(b),
                "median_a": a.median(),
                "median_b": b.median(),
                "median_a_minus_b": a.median() - b.median(),
                "mann_whitney_u": statistic,
                "p_value": p_value,
            }
        )

pairwise = pd.DataFrame(pairwise_rows)
if not pairwise.empty:
    pairwise["fdr_q_within_metric"] = pairwise.groupby("metric")[
        "p_value"
    ].transform(bh_adjust)
    pairwise = pairwise.sort_values(
        ["metric", "fdr_q_within_metric", "p_value"]
    )
save_csv(pairwise, "job_type_pairwise_promising_numeric.csv")


# =============================================================================
# Part 9: Figures
# =============================================================================
plot_counts = counts.sort_values("n_participants", ascending=True)
fig, axis = plt.subplots(figsize=(10, 5.5))
labels = plot_counts["job_type_short"].tolist()
bars = axis.barh(labels, plot_counts["n_participants"], color="#2B6CB0")
for bar, value in zip(bars, plot_counts["n_participants"]):
    axis.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"n={value}", va="center")
axis.set_xlabel("Number of participants")
axis.set_title("Participants by work type")
axis.set_xlim(0, plot_counts["n_participants"].max() * 1.18)
axis.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "figure_job_type_counts.png"),
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)

make_boxplot_grid(
    subject,
    [
        ("age", "Age"),
        ("work_experience", "Years in current work type"),
        ("trade_experience", "Prior trading experience (years)"),
        ("emission_experience", "Prior emissions experience (years)"),
        ("ind_risk", "Risk preference"),
        ("performance_pressure", "Performance pressure"),
    ],
    job_order,
    "figure_job_type_background_distributions.png",
    "Participant background distributions by work type",
)

make_boxplot_grid(
    subject,
    [
        ("control_mean_net_profit", "Mean net profit: Control"),
        ("trade_mean_net_profit", "Mean net profit: Carbon Trading"),
        ("control_mean_emission", "Mean emissions: Control"),
        ("trade_mean_emission", "Mean emissions: Carbon Trading"),
        ("trade_mean_abs_permit_gap", "Mean absolute permit gap: Trading"),
        ("trade_mean_abs_net_permit_trade", "Mean absolute net permit trade"),
    ],
    job_order,
    "figure_job_type_outcome_distributions.png",
    "Experimental outcomes by work type",
    robust_display_limits=True,
)

# Profile heatmap: z-score each participant-level metric across all people,
# then average within work type. This compares shapes, not original units.
heatmap_metrics = [
    metric
    for metric in [
        "age",
        "work_experience",
        "ind_risk",
        "trade_mean_net_profit",
        "trade_minus_control_net_profit",
        "trade_mean_relative_profit",
        "trade_mean_abs_permit_gap",
        "trade_mean_abs_net_permit_trade",
    ]
    if metric in subject.columns
]
standardized = subject[[JOB_COL] + heatmap_metrics].copy()
for metric in heatmap_metrics:
    values = pd.to_numeric(standardized[metric], errors="coerce")
    sd = values.std(ddof=1)
    standardized[metric] = (values - values.mean()) / sd if sd > 0 else np.nan

profile = standardized.groupby(JOB_COL)[heatmap_metrics].mean().reindex(job_order)
fig, axis = plt.subplots(figsize=(13, 5.8))
image = axis.imshow(profile.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axis.set_yticks(np.arange(len(job_order)))
axis.set_yticklabels([JOB_SHORT_NAMES.get(job, job) for job in job_order])
axis.set_xticks(np.arange(len(heatmap_metrics)))
axis.set_xticklabels(heatmap_metrics, rotation=35, ha="right")
for row in range(profile.shape[0]):
    for column in range(profile.shape[1]):
        value = profile.iat[row, column]
        if np.isfinite(value):
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
axis.set_title("Standardized work-type profiles (mean participant z-score)")
fig.colorbar(image, ax=axis, label="Mean z-score")
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "figure_job_type_profile_heatmap.png"),
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# Part 10: Plain-text research screening summary
# =============================================================================
top_numeric = numeric_screening.head(8)
top_categorical = categorical_screening.head(8)

lines = [
    "CTAT work-type exploratory analysis",
    "=" * 45,
    f"Participants: {subject_count}",
    f"Panel rows: {len(panel)}",
    "",
    "Work-type counts",
]
for _, row in counts.iterrows():
    lines.append(f"- {row[JOB_COL]}: n={int(row['n_participants'])}")

lines.extend(
    [
        "",
        "Top numeric differences (Kruskal-Wallis)",
    ]
)
for _, row in top_numeric.iterrows():
    lines.append(
        f"- {row['metric']}: p={row['p_value']:.4f}, "
        f"FDR q={row['fdr_q_value']:.4f}, "
        f"epsilon^2={row['epsilon_squared']:.3f}"
    )

lines.extend(
    [
        "",
        "Top categorical associations (chi-square)",
    ]
)
for _, row in top_categorical.iterrows():
    warning = " [small expected cells]" if row["small_cell_warning"] else ""
    lines.append(
        f"- {row['metric']}: p={row['p_value']:.4f}, "
        f"FDR q={row['fdr_q_value']:.4f}, "
        f"Cramer's V={row['cramers_v_bias_corrected']:.3f}{warning}"
    )

lines.extend(
    [
        "",
        "Interpretation notes",
        "- Counts and demographics use one row per participant.",
        "- Outcome distributions use one participant mean, not all repeated rounds as independent observations.",
        "- Trading outcomes and permit metrics use is_trade == 1 only.",
        "- Relative trading profit subtracts the mean within session x round_raw x dominant role.",
        "- Small job groups and multiple comparisons make these results exploratory.",
        "- Use the screening tables to choose hypotheses; do not treat raw p < .05 alone as confirmation.",
    ]
)

with open(
    os.path.join(OUTPUT_DIR, "job_type_analysis_summary.txt"),
    "w",
    encoding="utf-8-sig",
) as file:
    file.write("\n".join(lines))

print(f"[DONE] Participants: {subject_count}")
print(f"[DONE] Output: {OUTPUT_DIR}")
