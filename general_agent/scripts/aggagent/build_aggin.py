"""Convert saved T3 rollouts into AggAgent input format.

Reads:
  results/main_table_clueweb_t8/<model>/<dataset>/<cond_disk>/run_<seed>/
    trajectories/naive_parallel_T8/<q_id>.json   (one file per question, k threads inside)
    naive_parallel_T8.jsonl                       (sibling, has judged_rollouts per thread)

Writes:
  aggin/<model>/<dataset>/<cond>/run_<seed>/thread_<i>/<q_id>.json
  where each file is one trajectory in AggAgent's expected schema:
    {question, prediction, auto_judge:{correctness, extracted_final_answer, confidence},
     cost:{rollout,tool}, messages:[OpenAI-format]}

cond names on disk vary by model: naive_parallel|naive_k4 -> naive_k4;
                                  diversity_parallel|div_k4 -> div_k4.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


DISK_TO_COND = {
    "naive_parallel": "naive_k4",
    "naive_k4": "naive_k4",
    "diversity_parallel": "div_k4",
    "div_k4": "div_k4",
    "naive_k8": "naive_k8",
    "div_k8": "div_k8",
}

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching the web. "
    "Issue search queries via the `search` tool, then provide a final answer."
)


def build_messages(thread: dict, question: str) -> list[dict]:
    """Convert one thread (turn_logs) into OpenAI-format messages."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    call_idx = 0
    for log in thread.get("turn_logs", []):
        if "query" in log:
            call_idx += 1
            call_id = f"call_{call_idx}"
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": log["query"]}),
                    },
                }],
            })
            msgs.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": "search",
                "content": log.get("search_result", ""),
            })
        elif "answer" in log or "tagged_answer" in log:
            ans = log.get("tagged_answer") or log.get("answer") or ""
            msgs.append({
                "role": "assistant",
                "content": f"<answer>{ans}</answer>",
            })
    return msgs


def load_judgments(jsonl_path: Path) -> dict:
    """jsonl rows: {question_id, judged_rollouts: [bool, ...], ...}.
    Return dict question_id -> list[bool]."""
    out = {}
    if not jsonl_path.exists():
        return out
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id") or row.get("question")
            if qid is None:
                continue
            out[qid] = row.get("judged_rollouts", [])
    return out


def find_jsonl(seed_dir: Path) -> Path | None:
    """Find the *.jsonl sibling (e.g. naive_parallel_T8.jsonl)."""
    cands = [p for p in seed_dir.iterdir() if p.suffix == ".jsonl"]
    return cands[0] if cands else None


def find_trajectory_dir(seed_dir: Path) -> Path | None:
    """Find the deep dir holding <q_id>.json files."""
    t = seed_dir / "trajectories"
    if not t.exists():
        return None
    subs = [p for p in t.iterdir() if p.is_dir()]
    if not subs:
        return None
    return subs[0]  # there's only ever one


def process_seed(
    model: str, dataset: str, cond_disk: str, seed_dir: Path,
    aggin_root: Path, overwrite: bool = False,
) -> tuple[int, int]:
    """Process one (model, dataset, cond, seed). Returns (n_questions, n_threads_written)."""
    cond = DISK_TO_COND[cond_disk]
    seed_name = seed_dir.name  # run_1 / run_2 / ...

    traj_dir = find_trajectory_dir(seed_dir)
    if traj_dir is None:
        return (0, 0)
    jsonl = find_jsonl(seed_dir)
    judgments = load_judgments(jsonl) if jsonl else {}

    q_files = sorted(p for p in traj_dir.iterdir() if p.suffix == ".json")
    n_q = 0
    n_thr = 0
    missing_judge = 0

    for q_file in q_files:
        with open(q_file) as f:
            d = json.load(f)
        qid = d.get("question_id") or q_file.stem
        question = d.get("question", "")
        gold = d.get("gold_answer", "")
        threads = d.get("threads", [])
        per_thread_judged = judgments.get(qid)
        if per_thread_judged is None or len(per_thread_judged) != len(threads):
            missing_judge += 1
            per_thread_judged = [None] * len(threads)

        for i, th in enumerate(threads):
            ans = th.get("answer", "")
            judged = per_thread_judged[i]
            correctness = (
                "correct" if judged is True
                else "incorrect" if judged is False
                else "unknown"
            )
            out = {
                "question": question,
                "question_id": qid,
                "gold_answer": gold,
                "prediction": ans,
                "auto_judge": {
                    "correctness": correctness,
                    "extracted_final_answer": ans,
                    "confidence": None,  # filled by score_confidence.py
                },
                "cost": {"rollout": 0.0, "tool": 0.0},
                "messages": build_messages(th, question),
                "_meta": {
                    "model": model,
                    "dataset": dataset,
                    "condition": cond,
                    "cond_disk": cond_disk,
                    "seed": seed_name,
                    "thread_id": th.get("thread_id", i),
                },
            }
            out_dir = aggin_root / model / dataset / cond / seed_name / f"thread_{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{qid}.json"
            if out_path.exists() and not overwrite:
                continue
            with open(out_path, "w") as f:
                json.dump(out, f, ensure_ascii=False)
            n_thr += 1
        n_q += 1

    if missing_judge:
        print(f"    [warn] {missing_judge}/{n_q} questions missing/mismatched judged_rollouts in {jsonl}")
    return (n_q, n_thr)


def discover_targets(
    results_root: Path, models: list[str], conds: list[str],
    datasets: list[str] | None = None, seeds: list[str] | None = None,
) -> list[tuple]:
    """Return list of (model, dataset, cond_disk, seed_dir)."""
    out = []
    disk_for = defaultdict(list)
    for k, v in DISK_TO_COND.items():
        disk_for[v].append(k)

    for model in models:
        mroot = results_root / model
        if not mroot.exists():
            print(f"[skip] no dir for model {model}")
            continue
        for ds_dir in sorted(p for p in mroot.iterdir() if p.is_dir()):
            if datasets and ds_dir.name not in datasets:
                continue
            for cond in conds:
                for cdisk in disk_for[cond]:
                    cdir = ds_dir / cdisk
                    if not cdir.exists():
                        continue
                    for seed_dir in sorted(p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("run_")):
                        if seeds and seed_dir.name not in seeds:
                            continue
                        out.append((model, ds_dir.name, cdisk, seed_dir))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="/home/ssmurali/t3-testbed/results/main_table_clueweb_t8")
    p.add_argument("--aggin-root", default="/home/ssmurali/t3-testbed/aggin")
    p.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-4b", "qwen3-1.7b", "gemma3-4b", "gemma3-12b"])
    p.add_argument("--conds", nargs="+", default=["naive_k4", "div_k4"],
                   choices=["naive_k4", "div_k4", "naive_k8", "div_k8"])
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Filter to these datasets (e.g. bamboogle hle).")
    p.add_argument("--seeds", nargs="+", default=None,
                   help="Filter to these seed dirs (e.g. run_1).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Only process this many (model,ds,cond,seed) tuples (smoke test).")
    args = p.parse_args()

    results_root = Path(args.results_root)
    aggin_root = Path(args.aggin_root)
    aggin_root.mkdir(parents=True, exist_ok=True)

    targets = discover_targets(results_root, args.models, args.conds, args.datasets, args.seeds)
    if args.limit:
        targets = targets[:args.limit]
    print(f"Found {len(targets)} (model, dataset, cond, seed) tuples")

    total_q = 0
    total_t = 0
    for model, dataset, cond_disk, seed_dir in targets:
        cond = DISK_TO_COND[cond_disk]
        print(f"  {model}/{dataset}/{cond} ({cond_disk})/{seed_dir.name}", flush=True)
        n_q, n_t = process_seed(model, dataset, cond_disk, seed_dir, aggin_root, overwrite=args.overwrite)
        print(f"    -> {n_q} questions, {n_t} threads written")
        total_q += n_q
        total_t += n_t

    print(f"\nDone. total questions: {total_q}, total threads: {total_t}")


if __name__ == "__main__":
    main()
