"""
Build dense_o8_vs_naive_comparison.md with FULL rollout trajectories (turn-by-turn)
from small_benchmark_seeds_judged jsonl files (which already store rollout_details
with turn_logs and full_responses).

Usage (from general_agent):
  python -m webwalkerqa.scripts.build_dense_naive_trajectory_md
  # Or with custom paths:
  python -m webwalkerqa.scripts.build_dense_naive_trajectory_md --results-dir results/small_benchmark_seeds_judged/gpt-4.1-mini_seed0 --out path/to/comparison.md
"""

import argparse
import json
from pathlib import Path


def build_turns_from_rollout_detail(rollout_detail: dict) -> list[dict]:
    """Convert rollout_details[i] (turn_logs + full_responses) into same schema as qualitative_analysis turns."""
    turn_logs = rollout_detail.get("turn_logs", [])
    full_responses = rollout_detail.get("full_responses", [])
    response_by_turn = {r["turn"]: r for r in full_responses}

    turns_schema = []
    for log in turn_logs:
        turn_idx = log["turn"]
        resp = response_by_turn.get(turn_idx, {})
        tokens = resp.get("tokens", {}) or {}
        if "query" in log:
            action = "search"
            query = log["query"]
            search_results = log.get("search_result", "") or ""
            response = log.get("response", "") or resp.get("response", "")
        elif "answer" in log:
            action = "answer"
            query = None
            search_results = None
            response = log.get("response", "") or resp.get("response", "")
        else:
            action = "answer"
            query = None
            search_results = None
            response = log.get("response", "") or resp.get("response", "")
        turns_schema.append({
            "turn_idx": turn_idx,
            "thought": "",
            "action": action,
            "query": query,
            "search_results": search_results,
            "response": response or "",
            "prompt_tokens": tokens.get("prompt", 0),
            "completion_tokens": tokens.get("completion", 0),
        })
    return turns_schema


def format_trajectory_section(
    turns: list[dict],
    rollout_label: str,
    result_label: str,
    final_answer: str,
    min_turns_for_full: int = 5,
) -> list[str]:
    """
    Format one trajectory. If len(turns) > min_turns_for_full, output full turn-by-turn;
    otherwise output a compact summary (turns used, queries, final answer).
    """
    lines = []
    lines.append(f"#### {rollout_label} — {result_label}")
    lines.append(f"**Final answer:** {final_answer}")
    lines.append("")
    n_turns = len(turns)
    if n_turns <= min_turns_for_full:
        # Compact: turns count, one-line per turn, no full results/response
        lines.append(f"**Turns:** {n_turns} (compact; full trajectory only when > {min_turns_for_full} turns)")
        lines.append("")
        for turn in turns:
            idx = turn["turn_idx"]
            if turn.get("action") == "search":
                q = turn.get("query") or ""
                lines.append(f"- **Turn {idx}:** search → \"{q[:100]}{'…' if len(q) > 100 else ''}\"")
            else:
                lines.append(f"- **Turn {idx}:** answer")
        lines.append("")
        return lines
    # Full trajectory
    for turn in turns:
        idx = turn["turn_idx"]
        lines.append(f"**Turn {idx}**")
        thought = turn.get("thought") or "(none)"
        lines.append(f"- **Thought:** {thought}")
        if turn.get("action") == "search":
            lines.append(f"- **Action:** search → \"{turn.get('query', '')}\"")
            sr = turn.get("search_results") or ""
            if len(sr) > 8000:
                sr = sr[:8000] + "\n... [truncated]"
            lines.append(f"- **Results:** {sr}")
        else:
            lines.append(f"- **Action:** answer")
        resp = turn.get("response", "")
        if len(resp) > 6000:
            resp = resp[:6000] + "\n... [truncated]"
        lines.append(f"- **Response:** {resp}")
        lines.append("")
    return lines


def load_jsonl(p: Path) -> dict[str, dict]:
    """Load jsonl into question_id -> row."""
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["question_id"]] = d
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense-o8 vs naive comparison MD with full trajectories")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "results" / "small_benchmark_seeds_judged" / "gpt-4.1-mini_seed0",
        help="Directory containing per-benchmark subdirs with naive-t4.jsonl and dense-o8.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output MD path (default: results-dir / dense_o8_vs_naive_comparison.md)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1,
        help="Max number of example questions per benchmark with full trajectories (default 1). Use 2 or 3 for more samples.",
    )
    parser.add_argument(
        "--min-turns-full",
        type=int,
        default=5,
        help="Only write full turn-by-turn trajectory in MD when rollout has more than this many turns (default 5). Shorter rollouts get a compact summary.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir
    out_path = args.out or (results_dir / "dense_o8_vs_naive_comparison.md")
    max_examples = max(1, args.max_examples)
    min_turns_full = args.min_turns_full

    # Discover all (benchmark, qid) where naive wrong and dense-o8 right; keep up to max_examples per benchmark
    benchmarks = ["2wikimultihopqa", "musique", "simpleqa", "hotpotqa", "bamboogle"]
    benchmarks_and_qids: list[tuple[str, str]] = []
    for bench in benchmarks:
        naive_path = results_dir / bench / "naive-t4.jsonl"
        dense_path = results_dir / bench / "dense-o8.jsonl"
        if not naive_path.exists() or not dense_path.exists():
            continue
        naive_rows = load_jsonl(naive_path)
        dense_rows = load_jsonl(dense_path)
        candidates = []
        for qid, nd in naive_rows.items():
            if (nd.get("pass_at_4_llm") or 0) != 0:
                continue
            dd = dense_rows.get(qid)
            if dd and (dd.get("pass_at_4_llm") or 0) >= 1:
                candidates.append(qid)
        for qid in candidates[:max_examples]:
            benchmarks_and_qids.append((bench, qid))

    md_lines = []
    md_lines.append("# Dense-o8 vs Naive-t4: Where Dense Got It Right and Naive Failed")
    md_lines.append("")
    md_lines.append("**Source:** small_benchmark_seeds_judged (gpt-4.1-mini_seed0). Correctness via LLM judge.")
    md_lines.append("")
    md_lines.append(f"**Trajectories:** Rollouts with **more than {min_turns_full} turns** get full turn-by-turn detail (thought, action, results, response). Shorter rollouts get a compact summary (turn count + one line per turn).")
    md_lines.append("")
    md_lines.append(f"Full **turn-by-turn rollout trajectories** for {max_examples} example(s) per benchmark: one winning dense-o8 rollout and all four naive-t4 rollouts per question.")
    md_lines.append("")
    md_lines.append("| Benchmark        | Example question_id(s)     |")
    md_lines.append("|------------------|-------------------------|")
    seen_bench = set()
    for bench, qid in benchmarks_and_qids:
        if bench not in seen_bench:
            qids_for_bench = [q for b, q in benchmarks_and_qids if b == bench]
            md_lines.append(f"| {bench} | {', '.join(qids_for_bench)} |")
            seen_bench.add(bench)
    if "hotpotqa" not in seen_bench:
        md_lines.append("| hotpotqa         | *(none in this run)*    |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    for bench, qid in benchmarks_and_qids:
        naive_path = results_dir / bench / "naive-t4.jsonl"
        dense_path = results_dir / bench / "dense-o8.jsonl"
        if not naive_path.exists() or not dense_path.exists():
            continue
        naive_rows = load_jsonl(naive_path)
        dense_rows = load_jsonl(dense_path)
        if qid not in naive_rows or qid not in dense_rows:
            continue
        nd = naive_rows[qid]
        dd = dense_rows[qid]
        question = nd.get("question", "")
        gold = nd.get("answer_gt", "")
        md_lines.append(f"## {bench} — {qid}")
        md_lines.append("")
        md_lines.append(f"**Question:** {question}")
        md_lines.append("")
        md_lines.append(f"**Gold answer:** `{gold}`")
        md_lines.append("")

        # Winning dense-o8 rollout (first judged correct)
        judged_d = dd.get("judged_rollouts") or []
        rollout_details_d = dd.get("rollout_details") or []
        winning_idx = None
        for i, j in enumerate(judged_d):
            if j:
                winning_idx = i
                break
        if winning_idx is not None and winning_idx < len(rollout_details_d):
            rd = rollout_details_d[winning_idx]
            turns = build_turns_from_rollout_detail(rd)
            final = rd.get("answer", "")
            md_lines.append("### Dense-o8 — ✅ Winning rollout (full trajectory)")
            md_lines.append("")
            md_lines.extend(format_trajectory_section(turns, f"Rollout {winning_idx}", "CORRECT", final, min_turns_for_full=min_turns_full))
        md_lines.append("---")
        md_lines.append("")

        # All 4 naive-t4 rollouts (full trajectories)
        md_lines.append("### Naive-t4 — ❌ All 4 rollouts (full trajectories)")
        md_lines.append("")
        rollout_details_n = nd.get("rollout_details") or []
        for i in range(4):
            if i >= len(rollout_details_n):
                break
            rd = rollout_details_n[i]
            turns = build_turns_from_rollout_detail(rd)
            final = rd.get("answer", "")
            md_lines.extend(format_trajectory_section(turns, f"Rollout {i}", "WRONG", final, min_turns_for_full=min_turns_full))
        md_lines.append("---")
        md_lines.append("")

    md_lines.append("")
    md_lines.append("**Note:** HotpotQA had no question in this run where dense-o8 was correct and naive-t4 was wrong.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
