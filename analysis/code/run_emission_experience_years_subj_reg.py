# %% Emission-experience regressions with progressively added personal controls
import os

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Part 0. Settings and paths
# =============================================================================
EXPERIENCE_MODE = "years"
SOURCE_VARIABLE = "emission_experience"  # Original years, not a 0/1 column.
FOCAL_VARIABLE = "emission_experienced" if EXPERIENCE_MODE == "dummy" else SOURCE_VARIABLE
CONTROL_BLOCKS = [
    ["age"],
    ["male"],
    ["env"],
    ["ind_risk"],
    ["performance_pressure"],
    ["prevention_climate"],
]
personal_controls = list(dict.fromkeys(
    variable for block in CONTROL_BLOCKS for variable in block
))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.normpath(
    os.path.join(BASE_DIR, "..", "input", "subj_lv_panel.csv")
)
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "output"))
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, f"emission_experience_{EXPERIENCE_MODE}_results_tabulate.txt"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "Cannot find subj_lv_panel.csv. Expected location:\n"
        f"{INPUT_FILE}\n\n"
        "Put this script in analysis\\code and the CSV in analysis\\input."
    )


# =============================================================================
# Part 1. Read and prepare Carbon Trading participant-round data
# =============================================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = [
    "app", "participant_code", SOURCE_VARIABLE,
    "emission", "e_soc", "e_mkt", "round_raw",
    "is_equal", "first", "eq_price_low", "eq_price_high", "is_dominant",
] + personal_controls
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise KeyError(
        "Missing required columns: " + ", ".join(missing_columns)
    )

df = df.loc[df["app"].eq("Stage_CarbonTrading")].copy()

numeric_columns = [
    SOURCE_VARIABLE, "emission", "e_soc", "e_mkt", "round_raw",
    "is_equal", "first", "eq_price_low", "eq_price_high", "is_dominant",
] + personal_controls
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

years = df[SOURCE_VARIABLE].replace([np.inf, -np.inf], np.nan)
if years.lt(0).any():
    raise ValueError("emission_experience must contain nonnegative years.")
df[SOURCE_VARIABLE] = years
if EXPERIENCE_MODE == "dummy":
    # Preserve missing years as missing, rather than treating them as zero.
    df[FOCAL_VARIABLE] = years.gt(0).astype(float).where(years.notna())
elif EXPERIENCE_MODE != "years":
    raise ValueError("EXPERIENCE_MODE must be dummy or years.")

df["total_round"] = df["round_raw"]
df["is_equal_first"] = 1 - df["first"]
df["is_equal:is_equal_first"] = df["is_equal"] * df["is_equal_first"]
df["emission_diff_soc"] = df["emission"] - df["e_soc"]
df["emission_diff_mkt"] = df["emission"] - df["e_mkt"]

# =============================================================================
# Part 2. Model sequence
# (1) experience only
# (2) add all experimental-design controls
# (3) onward progressively add CONTROL_BLOCKS
# =============================================================================
design_controls = [
    "total_round", "is_equal", "is_equal_first",
    "is_equal:is_equal_first", "eq_price_low", "eq_price_high",
    "is_dominant",
]

model_regressors = []
current = [FOCAL_VARIABLE]
model_regressors.append(current.copy())

current += design_controls
model_regressors.append(current.copy())

for control_block in CONTROL_BLOCKS:
    current += control_block
    model_regressors.append(current.copy())

display_rows = ["Intercept"] + list(dict.fromkeys(
    variable for regressors in model_regressors for variable in regressors
))

outcomes = {
    "emission_diff_soc": "emission - e_soc",
    "emission_diff_mkt": "emission - e_mkt",
}


# =============================================================================
# Part 3. OLS, HC3 and participant-clustered standard errors
# =============================================================================
def significance_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def fit_one_model(outcome, regressors, covariance_type):
    needed = [outcome, "participant_code"] + regressors
    sample = (
        df[needed]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    y = sample[outcome].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(len(sample))]
        + [sample[column].to_numpy(dtype=float) for column in regressors]
    )
    parameter_names = ["Intercept"] + regressors

    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        raise ValueError(
            f"Insufficient observations or exact collinearity in {outcome}: "
            + ", ".join(regressors)
        )

    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta
    rss = float(residual.T @ residual)

    if covariance_type == "HC3":
        leverage = np.sum((X @ bread) * X, axis=1)
        adjusted_residual_sq = (residual / (1 - leverage)) ** 2
        meat = X.T @ (X * adjusted_residual_sq[:, None])
        covariance = bread @ meat @ bread
        reference_df = n - k
    elif covariance_type == "cluster":
        groups = sample["participant_code"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        cluster_count = len(unique_groups)
        meat = np.zeros((k, k))
        for group in unique_groups:
            use = groups == group
            score = X[use].T @ residual[use]
            meat += np.outer(score, score)
        correction = (
            (cluster_count / (cluster_count - 1))
            * ((n - 1) / (n - k))
        )
        covariance = correction * bread @ meat @ bread
        reference_df = cluster_count - 1
    else:
        raise ValueError(f"Unknown covariance type: {covariance_type}")

    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_values = np.divide(
        beta, standard_errors,
        out=np.full_like(beta, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2 * stats.t.sf(np.abs(t_values), df=reference_df)

    log_likelihood = -0.5 * n * (
        np.log(2 * np.pi) + 1 + np.log(rss / n)
    )
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + np.log(n) * k

    return {
        "params": pd.Series(beta, index=parameter_names),
        "bse": pd.Series(standard_errors, index=parameter_names),
        "pvalues": pd.Series(p_values, index=parameter_names),
        "nobs": n,
        "aic": aic,
        "bic": bic,
    }


def markdown_table(headers, rows):
    widths = []
    for column_index, header in enumerate(headers):
        values = [str(header)] + [str(row[column_index]) for row in rows]
        widths.append(max(len(value) for value in values))

    def format_row(row):
        cells = [str(value).ljust(widths[i]) for i, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join(
        [format_row(headers), separator]
        + [format_row(row) for row in rows]
    )


def build_regression_table(outcome, definition, covariance_type):
    models = [
        fit_one_model(outcome, regressors, covariance_type)
        for regressors in model_regressors
    ]
    headers = [f"Y = {outcome}"] + [
        f"({index})" for index in range(1, len(models) + 1)
    ]
    rows = []

    for parameter in display_rows:
        coefficient_row = [parameter]
        standard_error_row = [""]
        for model in models:
            if parameter in model["params"].index:
                coefficient = model["params"][parameter]
                standard_error = model["bse"][parameter]
                p_value = model["pvalues"][parameter]
                coefficient_row.append(
                    f"{coefficient:.3f}{significance_stars(p_value)}"
                )
                standard_error_row.append(f"({standard_error:.3f})")
            else:
                coefficient_row.append("")
                standard_error_row.append("")
        rows.extend([coefficient_row, standard_error_row])

    rows.append(["N"] + [str(model["nobs"]) for model in models])
    rows.append(["AIC"] + [f"{model['aic']:.1f}" for model in models])
    rows.append(["BIC"] + [f"{model['bic']:.1f}" for model in models])

    se_title = (
        "HC3 Robust SE"
        if covariance_type == "HC3"
        else "Clustered SE (by participant_code)"
    )
    title = f"=== Outcome: {outcome} ({definition}) | {se_title} ==="
    return title + "\n" + markdown_table(headers, rows)


# =============================================================================
# Part 4. Save one TXT containing both outcomes and both SE estimators
# =============================================================================
sections = []
for outcome, definition in outcomes.items():
    sections.append(build_regression_table(outcome, definition, "HC3"))
    sections.append(build_regression_table(outcome, definition, "cluster"))

model_notes = ["(1) focal experience only", "(2) + all design controls"]
model_notes += [
    f"({index}) + " + ", ".join(block)
    for index, block in enumerate(CONTROL_BLOCKS, start=3)
]
experience_note = (
    "emission_experienced = 1 if emission_experience > 0, 0 if years = 0; missing years remain missing."
    if EXPERIENCE_MODE == "dummy"
    else "emission_experience = original continuous years (no binning or log transformation)."
)
sections.append(
    "Standard errors in parentheses.\n"
    "* p<.05, ** p<.01, *** p<.001.\n"
    "Primary inference should use participant-clustered standard errors.\n"
    + experience_note + "\n"
    + "Model sequence: " + "; ".join(model_notes) + "."
)

complete_output = "\n\n\n".join(sections) + "\n"
with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
    output_file.write(complete_output)

print(f"[MODEL CHECK] focal variable = {FOCAL_VARIABLE}")
print(complete_output)
print(f"[DONE] Table TXT saved to: {OUTPUT_FILE}")
