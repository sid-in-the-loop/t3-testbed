#!/usr/bin/env python3
"""
Build GAIA results table: Exact Match + LLM-as-a-judge for gpt-4o-mini and gpt-4.1-mini.
Reads from results/gaia_103/<model>/ and results/gaia_103_judged/<model>/ (or root for judged).
Output: one text file with the table filled from available data.
"""

import sys
from pathlib import Path

_GA_DIR = Path(__file__).resolve().parent.parent
GAIA_RUN = _GA_DIR / "results" / "gaia_103"
GAIA_JUDGED = _GA_DIR / "results" / "gaia_103_judged"

# Table row order and mapping: table_label -> condition id in CSVs (hyphen)
TABLE_CONDITIONS = [
    ("naive_t4", "naive-t4"),
    ("jaccard_o16", "jaccard-o16"),
    ("jaccard_o32", "jaccard-o32"),
    ("jaccard_o48", "jaccard-o48"),
    ("jaccard_o64", "jaccard-o64"),
    ("dense_o16", "dense-o16"),
    ("dense_o32", "dense-o32"),
    ("dense_o48", "dense-o48"),
    ("dense_o64", "dense-o64"),
]


def read_summary_csv(path: Path) -> dict:
    """Return dict condition -> {pass@1, pass@4, n}. Auto-detect pass@1/pass@1_llm, pass@4/pass@4_llm."""
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    cond_col = df.columns[0]
    col1 = next((c for c in df.columns if "pass" in c.lower() and "1" in c), df.columns[1])
    col4 = next((c for c in df.columns if "pass" in c.lower() and "4" in c), df.columns[2])
    n_col = "n_questions" if "n_questions" in df.columns else ("n" if "n" in df.columns else None)
    out = {}
    for _, row in df.iterrows():
        c = row[cond_col]
        try:
            n_val = int(row[n_col]) if n_col else 0
        except (ValueError, TypeError, KeyError):
            n_val = 0
        out[c] = {"pass@1": float(row[col1]), "pass@4": float(row[col4]), "n": n_val}
    return out


def main():
    out_lines = []

    for model_display, run_subdir, judged_subdir in [
        ("gpt-4o-mini", "gpt4o-mini", None),   # judged at root
        ("gpt-4.1-mini", "gpt-4.1-mini", "gpt-4.1-mini"),
    ]:
        run_dir = GAIA_RUN / run_subdir if run_subdir else GAIA_RUN
        run_csv = run_dir / "summary.csv"
        judged_dir = GAIA_JUDGED / judged_subdir if judged_subdir else GAIA_JUDGED
        judged_csv = judged_dir / "summary_llm.csv"

        em = read_summary_csv(run_csv)
        llm = read_summary_csv(judged_csv)

        out_lines.append(f"{model_display}\tExact Match\t\tLLM-as-a-judge")
        out_lines.append("\tpass@1\tpass@4\tpass@1\tpass@4")
        for label, cond_id in TABLE_CONDITIONS:
            e = em.get(cond_id, {})
            l = llm.get(cond_id, {})
            p1_em = f"{e['pass@1']:.3f}" if e else ""
            p4_em = f"{e['pass@4']:.3f}" if e else ""
            p1_llm = f"{l['pass@1']:.3f}" if l else ""
            p4_llm = f"{l['pass@4']:.3f}" if l else ""
            out_lines.append(f"{label}\t{p1_em}\t{p4_em}\t{p1_llm}\t{p4_llm}")
        out_lines.append("")
        out_lines.append("")

    return "\n".join(out_lines)


if __name__ == "__main__":
    table = main()
    out_path = _GA_DIR / "results" / "gaia_103" / "gaia_results_table.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"\nWritten to {out_path}", file=sys.stderr)
