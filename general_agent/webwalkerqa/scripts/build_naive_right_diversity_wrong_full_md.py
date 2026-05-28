"""
Build naive_right_diversity_wrong_full.md with FULL rollout trajectories (same style as
dense_o8_vs_naive_comparison.md) for samples where naive got it right and diversity got it wrong.

Uses small_benchmark_seeds_judged only (GAIA judged jsonl do not have rollout_details).
For each such question: Naive-t4 winning rollout (full trajectory) + each failed condition's
4 rollouts (full or compact by --min-turns-full).

Usage (from general_agent):
  python -m webwalkerqa.scripts.build_naive_right_diversity_wrong_full_md
"""

import argparse
from pathlib import Path

from .build_dense_naive_trajectory_md import (
    build_turns_from_rollout_detail,
    format_trajectory_section,
    load_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full trajectory MD for naive right / diversity wrong samples"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "results"
        / "small_benchmark_seeds_judged"
        / "gpt-4.1-mini_seed0",
        help="Directory containing per-benchmark subdirs with naive-t4 and diversity jsonls",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output MD path (default: results-dir / naive_right_diversity_wrong_full.md)",
    )
    parser.add_argument(
        "--min-turns-full",
        type=int,
        default=5,
        help="Full turn-by-turn only when rollout has more than this many turns (default 5).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Max number of questions to include (default 0 = all). Set to e.g. 5 for a quick preview.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    out_path = args.out or (results_dir / "naive_right_diversity_wrong_full.md")
    min_turns_full = args.min_turns_full
    max_questions = args.max_questions

    benchmarks = ["2wikimultihopqa", "musique", "simpleqa", "hotpotqa", "bamboogle"]
    conditions = ["naive-t4", "dense-o8", "dense-o16", "jaccard-o8", "jaccard-o16"]

    # Collect (benchmark, qid, nd, failed_conditions) for all "naive right, at least one diversity wrong"
    candidates = []
    for bench in benchmarks:
        paths = {c: results_dir / bench / f"{c}.jsonl" for c in conditions}
        if not all(p.exists() for p in paths.values()):
            continue
        rows = {c: load_jsonl(paths[c]) for c in conditions}
        for qid, nd in rows["naive-t4"].items():
            if (nd.get("pass_at_4_llm") or 0) < 1:
                continue
            failed = []
            for c in conditions:
                if c == "naive-t4":
                    continue
                dd = rows[c].get(qid)
                if not dd or (dd.get("pass_at_4_llm") or 0) == 0:
                    failed.append(c)
            if failed:
                candidates.append((bench, qid, nd, failed))

    if max_questions > 0:
        candidates = candidates[:max_questions]

    md_lines = []
    md_lines.append("# Diversity degrades performance: Naive right, diversity wrong (full trajectories)")
    md_lines.append("")
    md_lines.append("Samples where **naive-t4** had ≥1 correct but at least one diversity condition had 0 correct.")
    md_lines.append("Full rollout trajectories in the same format as `dense_o8_vs_naive_comparison.md`.")
    md_lines.append("")
    md_lines.append(f"**Trajectories:** Rollouts with **more than {min_turns_full} turns** get full turn-by-turn detail; shorter get compact.")
    md_lines.append("")
    md_lines.append(f"**Total: {len(candidates)} questions** (small_benchmark only; GAIA judged jsonl have no rollout_details).")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    for bench, qid, nd, failed_conditions in candidates:
        question = nd.get("question", "")
        gold = nd.get("answer_gt", "")
        rollout_details_n = nd.get("rollout_details") or []
        judged_n = nd.get("judged_rollouts") or []

        md_lines.append(f"## {bench} — {qid}")
        md_lines.append("")
        md_lines.append(f"**Question:** {question}")
        md_lines.append("")
        md_lines.append(f"**Gold answer:** `{gold}`")
        md_lines.append("")

        # Naive-t4 — winning rollout (first correct)
        winning_idx = None
        for i, j in enumerate(judged_n):
            if j:
                winning_idx = i
                break
        if winning_idx is not None and winning_idx < len(rollout_details_n):
            rd = rollout_details_n[winning_idx]
            turns = build_turns_from_rollout_detail(rd)
            final = rd.get("answer", "")
            md_lines.append("### Naive-t4 — ✅ Winning rollout (full trajectory)")
            md_lines.append("")
            md_lines.extend(
                format_trajectory_section(
                    turns,
                    f"Rollout {winning_idx}",
                    "CORRECT",
                    final,
                    min_turns_for_full=min_turns_full,
                )
            )
        md_lines.append("---")
        md_lines.append("")

        # Each failed condition: all 4 rollouts
        for cond in failed_conditions:
            cond_path = results_dir / bench / f"{cond}.jsonl"
            if not cond_path.exists():
                continue
            rows_c = load_jsonl(cond_path)
            dd = rows_c.get(qid)
            if not dd:
                continue
            rollout_details_d = dd.get("rollout_details") or []
            md_lines.append(f"### {cond} — ❌ All 4 rollouts (0 correct)")
            md_lines.append("")
            for i in range(4):
                if i >= len(rollout_details_d):
                    break
                rd = rollout_details_d[i]
                turns = build_turns_from_rollout_detail(rd)
                final = rd.get("answer", "")
                md_lines.extend(
                    format_trajectory_section(
                        turns,
                        f"Rollout {i}",
                        "WRONG",
                        final,
                        min_turns_for_full=min_turns_full,
                    )
                )
            md_lines.append("---")
            md_lines.append("")

    md_lines.append("")
    md_lines.append("**Note:** GAIA-103 gpt-4.1-mini_judged jsonl do not contain rollout_details; only small_benchmark has full trajectories.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(candidates)} questions)")


if __name__ == "__main__":
    main()
