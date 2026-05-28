"""
Build an MD file listing samples where diversity degrades performance:
- Naive-t4 gets it right (≥1 correct), but diversity condition(s) get it wrong.

Two sections:
1. Strict: naive right, ALL diversity conditions wrong (diversity fully hurts).
2. Broader: naive right, at least one diversity condition wrong (which ones failed).

Sources: small_benchmark_seeds_judged (5 benchmarks) + GAIA-103 gpt-4.1-mini_judged.
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
    judged = row.get("judged_rollouts") or []
    answers = row.get("rollout_answers") or []
    for i, j in enumerate(judged):
        if j and i < len(answers):
            return answers[i] or ""
    return answers[0] if answers else ""


def main() -> None:
    ga_dir = Path(__file__).resolve().parent.parent.parent
    out_path = (
        ga_dir
        / "results"
        / "small_benchmark_seeds_judged"
        / "gpt-4.1-mini_seed0"
        / "naive_right_diversity_wrong_samples.md"
    )

    lines = []
    lines.append("# Diversity degrades performance: Naive right, diversity wrong")
    lines.append("")
    lines.append("Samples where **naive-t4** had ≥1 correct (pass@4_llm) but diversity condition(s)")
    lines.append("(dense-o8, dense-o16, jaccard-o8, jaccard-o16) had 0 correct.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Small benchmark ----
    base_small = ga_dir / "results" / "small_benchmark_seeds_judged" / "gpt-4.1-mini_seed0"
    benchmarks = ["2wikimultihopqa", "musique", "simpleqa", "hotpotqa", "bamboogle"]
    conditions_small = ["naive-t4", "dense-o8", "dense-o16", "jaccard-o8", "jaccard-o16"]

    strict_small = []
    broader_small = []
    for bench in benchmarks:
        paths = {c: base_small / bench / f"{c}.jsonl" for c in conditions_small}
        if not all(p.exists() for p in paths.values()):
            continue
        rows = {c: load_jsonl(paths[c]) for c in conditions_small}
        for qid, nd in rows["naive-t4"].items():
            if (nd.get("pass_at_4_llm") or 0) < 1:
                continue
            failed = []
            for c in conditions_small:
                if c == "naive-t4":
                    continue
                dd = rows[c].get(qid)
                if not dd or (dd.get("pass_at_4_llm") or 0) == 0:
                    failed.append(c)
            rec = {
                "benchmark": bench,
                "question_id": qid,
                "question": nd.get("question", ""),
                "gold": nd.get("answer_gt", ""),
                "naive_winning_answer": first_winning_answer(nd),
                "failed_conditions": failed,
            }
            if len(failed) == 4:
                strict_small.append(rec)
            elif failed:
                broader_small.append(rec)

    # ---- GAIA ----
    base_gaia = ga_dir / "results" / "gaia_103" / "gpt-4.1-mini_judged"
    conditions_gaia = ["naive-t4", "jaccard-o16", "dense-o8", "dense-o16"]
    strict_gaia = []
    broader_gaia = []
    if base_gaia.exists():
        paths_gaia = {c: base_gaia / f"{c}.jsonl" for c in conditions_gaia}
        if all(p.exists() for p in paths_gaia.values()):
            rows_gaia = {c: load_jsonl(paths_gaia[c]) for c in conditions_gaia}
            for qid, nd in rows_gaia["naive-t4"].items():
                if (nd.get("pass_at_4_llm") or 0) < 1:
                    continue
                failed = []
                for c in conditions_gaia:
                    if c == "naive-t4":
                        continue
                    dd = rows_gaia[c].get(qid)
                    if not dd or (dd.get("pass_at_4_llm") or 0) == 0:
                        failed.append(c)
                rec = {
                    "question_id": qid,
                    "question": nd.get("question", ""),
                    "gold": nd.get("answer_gt", ""),
                    "naive_winning_answer": first_winning_answer(nd),
                    "failed_conditions": failed,
                }
                if len(failed) == 3:
                    strict_gaia.append(rec)
                elif failed:
                    broader_gaia.append(rec)

    # ----- Section 1: Strict (diversity fully hurts) -----
    lines.append("## 1. Strict: Naive right, ALL diversity conditions wrong")
    lines.append("")
    lines.append("Naive got ≥1 correct; dense-o8, dense-o16, jaccard-o8, jaccard-o16 (and on GAIA the 3 run) all got 0.")
    lines.append("")
    total_strict = len(strict_small) + len(strict_gaia)
    lines.append(f"**Total: {total_strict}** (small: {len(strict_small)}, GAIA: {len(strict_gaia)})")
    lines.append("")
    if total_strict > 0:
        lines.append("| Source | ID | Question (excerpt) | Gold | Naive winning answer (excerpt) |")
        lines.append("|--------|-----|--------------------|------|----------------------------------|")
        for r in strict_small:
            q = (r["question"][:55] + "…") if len(r["question"]) > 55 else r["question"]
            g = (r["gold"][:30] + "…") if len(r["gold"]) > 30 else r["gold"]
            a = (r["naive_winning_answer"][:45] + "…") if len(r["naive_winning_answer"]) > 45 else (r["naive_winning_answer"] or "—")
            lines.append(f"| {r['benchmark']} | {r['question_id']} | {q} | {g} | {a} |")
        for r in strict_gaia:
            q = (r["question"][:55] + "…") if len(r["question"]) > 55 else r["question"]
            g = (r["gold"][:30] + "…") if len(r["gold"]) > 30 else r["gold"]
            a = (r["naive_winning_answer"][:45] + "…") if len(r["naive_winning_answer"]) > 45 else (r["naive_winning_answer"] or "—")
            lines.append(f"| GAIA | {r['question_id']} | {q} | {g} | {a} |")
    else:
        lines.append("*(None in this run.)*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ----- Section 2: Broader (naive right, at least one diversity wrong) -----
    lines.append("## 2. Broader: Naive right, at least one diversity condition wrong")
    lines.append("")
    lines.append("Naive got ≥1 correct; at least one of the diversity conditions got 0. Column **Failed** lists which.")
    lines.append("")
    total_broader = len(broader_small) + len(broader_gaia)
    lines.append(f"**Total: {total_broader}** (small: {len(broader_small)}, GAIA: {len(broader_gaia)})")
    lines.append("")
    lines.append("| Source | ID | Question (excerpt) | Gold | Failed (0 correct) | Naive answer (excerpt) |")
    lines.append("|--------|-----|--------------------|------|---------------------|------------------------|")
    for r in broader_small:
        q = (r["question"][:50] + "…") if len(r["question"]) > 50 else r["question"]
        g = (r["gold"][:25] + "…") if len(r["gold"]) > 25 else r["gold"]
        failed_str = ", ".join(r["failed_conditions"])
        a = (r["naive_winning_answer"][:35] + "…") if len(r["naive_winning_answer"]) > 35 else (r["naive_winning_answer"] or "—")
        lines.append(f"| {r['benchmark']} | {r['question_id']} | {q} | {g} | {failed_str} | {a} |")
    for r in broader_gaia:
        q = (r["question"][:50] + "…") if len(r["question"]) > 50 else r["question"]
        g = (r["gold"][:25] + "…") if len(r["gold"]) > 25 else r["gold"]
        failed_str = ", ".join(r["failed_conditions"])
        a = (r["naive_winning_answer"][:35] + "…") if len(r["naive_winning_answer"]) > 35 else (r["naive_winning_answer"] or "—")
        lines.append(f"| GAIA | {r['question_id']} | {q} | {g} | {failed_str} | {a} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**Grand total: {total_strict} strict + {total_broader} broader = {total_strict + total_broader} samples** where naive got it right and diversity (at least partly) degraded performance.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} (strict: {total_strict}, broader: {total_broader})")


if __name__ == "__main__":
    main()
