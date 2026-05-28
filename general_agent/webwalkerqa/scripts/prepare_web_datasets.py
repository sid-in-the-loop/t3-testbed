"""
Prepare subsampled dataset files for the Table-2 (Serper, web-reasoning) experiments.
  - webwalker_sub.json  (250 of 680)
  - hle_sub.json         (250 of 500)
  - browsecomp_sub.json  (250 of 1266, from CSV)
  - gaia_full.json       (103, identical to existing GAIA.json — just consistent naming)

Deterministic (seed=42). IDs prefixed: webwalker-, hle-, browsecomp-, gaia-.
"""
import argparse
import csv
import json
import random
from pathlib import Path

OUT_DIR = Path("/home/ssmurali/t3-testbed/general_agent/data/main_table")


def _load_json(p: Path) -> list:
    with open(p) as f:
        return json.load(f)


def _write_json(p: Path, records: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def sub_webwalker(n: int, rng: random.Random) -> None:
    src = OUT_DIR / "webwalker.json"
    if not src.exists():
        raise FileNotFoundError(src)
    data = _load_json(src)
    rng.shuffle(data)
    picked = data[:n]
    for i, r in enumerate(picked):
        if not str(r.get("id", "")).startswith("webwalker-"):
            r["id"] = f"webwalker-{r['id']}"
    _write_json(OUT_DIR / "webwalker_sub.json", picked)
    print(f"webwalker_sub.json: {len(picked)} records")


def sub_hle(n: int, rng: random.Random) -> None:
    src = OUT_DIR / "hle.json"
    if not src.exists():
        raise FileNotFoundError(src)
    data = _load_json(src)
    rng.shuffle(data)
    picked = data[:n]
    for r in picked:
        if not str(r.get("id", "")).startswith("hle-"):
            r["id"] = f"hle-{r['id']}"
    _write_json(OUT_DIR / "hle_sub.json", picked)
    print(f"hle_sub.json: {len(picked)} records")


def sub_browsecomp(n: int, rng: random.Random) -> None:
    src = OUT_DIR / "browse_comp_test_set.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    with open(src) as f:
        rows = list(csv.DictReader(f))
    # BrowseComp CSV: columns include at least question/answer. Figure out on the fly.
    # Try common columns: problem/answer, question/answer, prompt/answer
    def _col(row, candidates):
        for c in candidates:
            if c in row and row[c]:
                return row[c]
        return ""
    recs = []
    for i, row in enumerate(rows):
        q = _col(row, ["problem", "question", "prompt", "Prompt", "Question"])
        a = _col(row, ["answer", "Answer", "gold", "ground_truth"])
        if not q or not a:
            continue
        recs.append({"id": f"browsecomp-{i}", "question": q, "answer": a})
    rng.shuffle(recs)
    picked = recs[:n]
    _write_json(OUT_DIR / "browsecomp_sub.json", picked)
    print(f"browsecomp_sub.json: {len(picked)} records (from {len(rows)} CSV rows)")


def sub_gaia() -> None:
    src = OUT_DIR / "GAIA.json"
    if not src.exists():
        raise FileNotFoundError(src)
    data = _load_json(src)
    for r in data:
        if not str(r.get("id", "")).startswith("gaia-"):
            r["id"] = f"gaia-{r['id']}"
    _write_json(OUT_DIR / "gaia_full.json", data)
    print(f"gaia_full.json: {len(data)} records (full, IDs prefixed gaia-)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-webwalker", type=int, default=250)
    p.add_argument("--n-hle", type=int, default=250)
    p.add_argument("--n-browsecomp", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    sub_webwalker(args.n_webwalker, random.Random(args.seed))
    sub_hle(args.n_hle, random.Random(args.seed + 1))
    sub_browsecomp(args.n_browsecomp, random.Random(args.seed + 2))
    sub_gaia()
    print("\nAll datasets prepared under", OUT_DIR)


if __name__ == "__main__":
    main()
