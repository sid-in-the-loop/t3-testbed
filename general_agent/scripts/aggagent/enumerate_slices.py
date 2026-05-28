"""Write a manifest of all (model, dataset, cond, seed) slices under aggin/.

Each line: <idx>\t<model>\t<dataset>\t<cond>\t<seed>\t<aggin_dir>\t<aggout_dir>
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggin-root", default="/data/user_data/ssmurali/aggin")
    p.add_argument("--aggout-root", default="/data/user_data/ssmurali/aggout")
    p.add_argument("--out", default="/data/user_data/ssmurali/aggregation_manifest.tsv")
    p.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-4b", "qwen3-1.7b", "gemma3-4b", "gemma3-12b"])
    p.add_argument("--conds", nargs="+", default=["naive_k4", "div_k4"])
    args = p.parse_args()

    aggin = Path(args.aggin_root)
    aggout = Path(args.aggout_root)
    rows = []
    for model in args.models:
        mroot = aggin / model
        if not mroot.exists():
            continue
        for ds in sorted(p for p in mroot.iterdir() if p.is_dir()):
            for cond in args.conds:
                cdir = ds / cond
                if not cdir.exists():
                    continue
                for seed in sorted(p for p in cdir.iterdir() if p.is_dir() and p.name.startswith("run_")):
                    aggin_dir = seed
                    aggout_dir = aggout / model / ds.name / cond / seed.name
                    rows.append((model, ds.name, cond, seed.name, aggin_dir, aggout_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, (m, d, c, s, ai, ao) in enumerate(rows):
            f.write(f"{i}\t{m}\t{d}\t{c}\t{s}\t{ai}\t{ao}\n")
    print(f"Wrote {len(rows)} slices to {out}")
    print(f"Suggested sbatch array: --array=0-{len(rows)-1}%50")


if __name__ == "__main__":
    main()
