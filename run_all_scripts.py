import os
import subprocess
import sys


# =====================================================
# 0. 專案根目錄
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_path: str) -> None:
    """
    執行單一 Python 程式。

    若檔案不存在，直接顯示清楚的錯誤；
    若程式執行失敗，subprocess 會中止整個流程。
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"找不到程式：\n{script_path}"
        )

    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    subprocess.run(
        [sys.executable, script_path],
        check=True,
    )


# =====================================================
# 1. 執行 build/code 下的資料轉換程式
# =====================================================

# build_code_dir = os.path.join(PROJECT_DIR, "build", "code")

# build_scripts = [
#     "gen_subj_and_group_panel.py",
#     "gen_trade_panel.py",
#     "gen_muda_trade_panel.py",
# ]

# for script in build_scripts:
#     script_path = os.path.join(build_code_dir, script)
#     run_script(script_path)


# =====================================================
# 2. 執行 analysis/code 下的分析程式
# =====================================================

analysis_code_dir = os.path.join(PROJECT_DIR, "analysis", "code")

analysis_scripts = [
    # "gen_trade_seconds_vs_price_diff.py",
    # "run_group_lv_reg.py",
    # "run_subj_lv_reg.py",
    # "run_trade_mkt_reg.py",
    # 'gen_trade_seconds_vs_price_diff.py',
    #'ctat_job_type_distribution_analysis.py',
    # "run_trade_experience_reg.py",
    # 'run_permit_gap_reg.py',
    # 'run_job_two_experience_reg.py',
    # 'run_emission_experience_years_subj_reg.py',
    # 'fig1b_cap_constrained_allocation_efficiency.py',
    'job_type_distribution.py'
]

for script in analysis_scripts:
    script_path = os.path.join(analysis_code_dir, script)
    run_script(script_path)


print("\n" + "=" * 80)
print("All scripts finished successfully.")
print("=" * 80)
