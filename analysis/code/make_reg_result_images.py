import os
import re
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "output"
SAVE_DIR = OUTPUT_DIR / "reg_result_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DOC_INPUT_DIR = BASE_DIR.parent.parent / "doc" / "input"
DOC_SAVE_DIR = DOC_INPUT_DIR / "reg_result_images"
DOC_SAVE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    "group_lv_reg_results_tabulate.txt",
    "subj_lv_reg_results_tabulate.txt",
    "trade_mkt_reg_results_tabulate.txt",
]


def sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-_.]", "", name)
    return name


def split_sections(text: str):
    """
    用 '=== Outcome:' 切各段
    """
    pattern = r"(=== Outcome:.*?)(?=\n=== Outcome:|\Z)"
    sections = re.findall(pattern, text, flags=re.S)
    return [s.strip() for s in sections if s.strip()]


def parse_header(section_text: str):
    """
    從段落第一行抓 outcome 與 label
    例如:
    === Outcome: emission_diff | HC3 Robust SE ===
    """
    first_line = section_text.splitlines()[0].strip()
    m = re.match(r"=== Outcome:\s*(.*?)\s*\|\s*(.*?)\s*===", first_line)
    if m:
        outcome = m.group(1).strip()
        label = m.group(2).strip()
        return outcome, label
    return "unknown_outcome", "unknown_label"


def render_text_to_image(text: str, title: str, save_path: Path):
    """
    把整段文字表格渲染成 PNG
    """
    lines = text.splitlines()

    # 粗估圖尺寸
    max_line_len = max((len(line) for line in lines), default=80)
    n_lines = len(lines)

    fig_width = min(max(10, max_line_len * 0.11), 24)
    fig_height = min(max(4, n_lines * 0.33), 40)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # 標題
    ax.text(
        0.01, 0.99, title,
        fontsize=14, fontweight="bold",
        va="top", ha="left",
        family="sans-serif",
        transform=ax.transAxes
    )

    # 內容
    ax.text(
        0.01, 0.95, text,
        fontsize=10,
        va="top", ha="left",
        family="monospace",
        transform=ax.transAxes
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    for filename in INPUT_FILES:
        infile = OUTPUT_DIR / filename
        if not infile.exists():
            print(f"[跳過] 找不到檔案: {infile}")
            continue

        print(f"[讀取] {infile}")
        text = infile.read_text(encoding="utf-8-sig")
        sections = split_sections(text)

        if not sections:
            print(f"[警告] 沒有找到可切分段落: {filename}")
            continue

        stem = infile.stem
        subdir = SAVE_DIR / stem
        subdir.mkdir(parents=True, exist_ok=True)
        doc_subdir = DOC_SAVE_DIR / stem
        doc_subdir.mkdir(parents=True, exist_ok=True)

        for i, section in enumerate(sections, start=1):
            outcome, label = parse_header(section)
            safe_outcome = sanitize_filename(outcome)
            safe_label = sanitize_filename(label)

            save_name = f"{i:02d}_{safe_outcome}_{safe_label}.png"
            save_path = subdir / save_name
            doc_save_path = doc_subdir / save_name

            title = f"{stem} | {outcome} | {label}"
            render_text_to_image(section, title, save_path)
            shutil.copy2(save_path, doc_save_path)
            print(f"[輸出] {save_path}")

    print(f"\n完成！圖片都在：{SAVE_DIR}")
    print(f"doc/input 副本在：{DOC_SAVE_DIR}")


if __name__ == "__main__":
    main()
