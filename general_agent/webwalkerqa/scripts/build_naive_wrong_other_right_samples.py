"""
Build a single MD file listing ALL questions where naive-t4 got it wrong (pass@4_llm=0)
and at least one other condition got it right. Covers small_benchmark_seeds_judged + GAIA-103.

Output: naive_wrong_other_right_samples.md (in results dir)
"""

import json
from pathlib import Path


def load_jsonl(p: Path) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["question_id"]] = d
    return out


def first_winning_answer(row: dict) -> str:
    """From a condition row, return first correct rollout answer if any."""
    judged = row.get("judged_rollouts") or []
    answers = row.get("rollout_answers") or []
    for i, j in enumerate(judged):
        if j and i < len(answers):
            return answers[i] or ""
    return answers[0] if answers else ""


def main() -> None:
    ga_dir = Path(__file__).resolve().parent.parent.parent
    out_path = ga_dir / "results" / "small_benchmark_seeds_judged" / "gpt-4.1-mini_seed0" / "naive_wrong_other_right_samples.md"

    lines = []
    lines.append("# More samples: Naive wrong, some other condition right")
    lines.append("")
    lines.append("All questions where **naive-t4** had 0 correct (pass@4_llm) and at least one of")
    lines.append("**dense-o8**, **dense-o16**, **jaccard-o8**, **jaccard-o16** had ≥1 correct.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Small benchmark ----
    base_small = ga_dir / "results" / "small_benchmark_seeds_judged" / "gpt-4.1-mini_seed0"
    benchmarks = ["2wikimultihopqa", "musique", "simpleqa", "hotpotqa", "bamboogle"]
    conditions_small = ["naive-t4", "dense-o8", "dense-o16", "jaccard-o8", "jaccard-o16"]

    small_rows = []
    for bench in benchmarks:
        paths = {c: base_small / bench / f"{c}.jsonl" for c in conditions_small}
        if not all(p.exists() for p in paths.values()):
            continue
        rows = {c: load_jsonl(paths[c]) for c in conditions_small}
        for qid, nd in rows["naive-t4"].items():
            if (nd.get("pass_at_4_llm") or 0) != 0:
                continue
            winners = []
            winning_answer = ""
            for c in conditions_small:
                if c == "naive-t4":
                    continue
                dd = rows[c].get(qid)
                if dd and (dd.get("pass_at_4_llm") or 0) >= 1:
                    winners.append(c)
                    if not winning_answer:
                        winning_answer = first_winning_answer(dd)
            if winners:
                small_rows.append({
                    "benchmark": bench,
                    "question_id": qid,
                    "question": nd.get("question", ""),
                    "gold": nd.get("answer_gt", ""),
                    "winners": winners,
                    "winning_answer": winning_answer,
                })

    lines.append("## Small benchmark (gpt-4.1-mini_seed0)")
    lines.append("")
    lines.append(f"**Total: {len(small_rows)} questions**")
    lines.append("")
    lines.append("| Benchmark | Question ID | Question (excerpt) | Gold | Who got it right | Winning answer (excerpt) |")
    lines.append("|-----------|-------------|--------------------|------|------------------|---------------------------|")
    for r in small_rows:
        q_excerpt = (r["question"][:60] + "…") if len(r["question"]) > 60 else r["question"]
        gold_excerpt = (r["gold"][:30] + "…") if len(r["gold"]) > 30 else r["gold"]
        ans_excerpt = (r["winning_answer"][:40] + "…") if len(r["winning_answer"]) > 40 else (r["winning_answer"] or "—")
        winners_str = ", ".join(r["winners"])
        lines.append(f"| {r['benchmark']} | {r['question_id']} | {q_excerpt} | {gold_excerpt} | {winners_str} | {ans_excerpt} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- GAIA-103 ----
    base_gaia = ga_dir / "results" / "gaia_103" / "gpt-4.1-mini_judged"
    conditions_gaia = ["naive-t4", "jaccard-o16", "dense-o8", "dense-o16"]
    gaia_rows = []
    if base_gaia.exists():
        paths_gaia = {c: base_gaia / f"{c}.jsonl" for c in conditions_gaia}
        if all(p.exists() for p in paths_gaia.values()):
            rows_gaia = {c: load_jsonl(paths_gaia[c]) for c in conditions_gaia}
            for qid, nd in rows_gaia["naive-t4"].items():
                if (nd.get("pass_at_4_llm") or 0) != 0:
                    continue
                winners = []
                winning_answer = ""
                for c in conditions_gaia:
                    if c == "naive-t4":
                        continue
                    dd = rows_gaia[c].get(qid)
                    if dd and (dd.get("pass_at_4_llm") or 0) >= 1:
                        winners.append(c)
                        if not winning_answer:
                            winning_answer = first_winning_answer(dd)
                if winners:
                    gaia_rows.append({
                        "question_id": qid,
                        "question": nd.get("question", ""),
                        "gold": nd.get("answer_gt", ""),
                        "winners": winners,
                        "winning_answer": winning_answer,
                    })

    lines.append("## GAIA-103 (gpt-4.1-mini_judged)")
    lines.append("")
    lines.append(f"**Total: {len(gaia_rows)} questions**")
    lines.append("")
    lines.append("| Question ID | Question (excerpt) | Gold | Who got it right | Winning answer (excerpt) |")
    lines.append("|-------------|--------------------|------|------------------|---------------------------|")
    for r in gaia_rows:
        q_excerpt = (r["question"][:60] + "…") if len(r["question"]) > 60 else r["question"]
        gold_excerpt = (r["gold"][:30] + "…") if len(r["gold"]) > 30 else r["gold"]
        ans_excerpt = (r["winning_answer"][:40] + "…") if len(r["winning_answer"]) > 40 else (r["winning_answer"] or "—")
        winners_str = ", ".join(r["winners"])
        lines.append(f"| {r['question_id']} | {q_excerpt} | {gold_excerpt} | {winners_str} | {ans_excerpt} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**Grand total: {len(small_rows) + len(gaia_rows)} samples** where naive was wrong and some other condition got it right.")
    lines.append("")
    lines.append("For **full turn-by-turn trajectories** on a subset of these, see `dense_o8_vs_naive_comparison.md` (small benchmark) and `jaccard_vs_naive_comparison.md` (GAIA).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(small_rows)} small + {len(gaia_rows)} GAIA = {len(small_rows)+len(gaia_rows)} samples)")


if __name__ == "__main__":
    main()
