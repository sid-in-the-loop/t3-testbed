# T3 Paper Details

## Experimental Results
<!-- numbers, accuracy scores, conditions, benchmarks -->
- [2026-03-20] GAIA (n=103), Budget B (sequential 20 turns, parallel 5 turns each, k=4): Sequential pass@1=0.175, Naive k=4 oracle=0.252/synthesis=0.126, Diversity k=4 oracle=0.359/synthesis=0.155
- [2026-03-20] GAIA (n=103), Budget 2B and 4B results: TBD — placeholder pending paste

## Story & Framing
<!-- what the paper claims, what changed, key findings -->
- [2026-03-22] Motivation flow: Figure 1 shows sequential vs pass@k only → show pass@k has low diversity (no KDEs yet) → hypothesize diversity helps → main table proves it empirically → then show KDE plots to explain why the method works
- [2026-03-22] Don't call it "oracle pass@k" — just use "pass@k" (without synthesis). The oracle overloading from original paper is confusing
- [2026-03-22] Report pass@k as primary metric; synthesis results go in appendix
- [2026-03-22] No need for formulas for pass@k or synthesis — don't plug in math for the sake of it

## TODO
<!-- things to run, things to write, things to fix -->
- [2026-03-20] Fill in Budget 2B and 4B Figure 1 numbers for GAIA (n=103)
- [2026-03-20] Generate Figure 1 plots (left: accuracy bars, right: QPD distributions from existing CSVs)
- [2026-03-20] Run k-sweep ablation
- [2026-03-20] Run pool-size ablation
- [2026-04-05] Re-run all models at T=12 turns (currently have T=5 results — advisor flagged this as a likely cause of low numbers)
- [2026-04-05] Run qwen3-1.7b (server submitted, in progress)
- [2026-04-05] Run gpt-oss-20b (OSS, needs vLLM — clarify exact HF model name)
- [2026-04-05] Run gemini-2.5-flash (API, need litellm support + key)
- [2026-04-05] Run gpt-4.1-mini (API)
- [2026-04-05] Investigate agent design from AReaL simple search agent (https://github.com/cxcscmu/AReaL/tree/main/examples/search_agent/tongyi_deepresearch_simple) — advisor says WebWalker can reach >30% with Serper; current agent gets 3%/7%
- [2026-04-05] Compare against baselines: Search-R1-Base (7B), R1-Searcher (7B), DeepResearcher (7B), BehaviorPrime (1.7B), ORBIT (4B) — all pass@1 only

## Section Notes
<!-- per-section writing decisions, what's locked, what's placeholder -->
- [2026-03-22] Remove all em-dash constructions (— this type —) from writing
- [2026-03-22] Don't introduce KDE plots (query diversity, inter-turn similarity) before the method is explained
- [2026-03-22] Consider sampling datasets (e.g. 512 questions per dataset) — check Jiahe's behavior priming paper for precedent

## Decisions Log
<!-- design decisions made and why, so we don't revisit them -->
- [2026-04-05] Final model set: qwen3-1.7b, qwen3-4b, qwen3-8b, gpt-oss-20b (vLLM), gpt-4o-mini, gemini-2.5-flash — 3 OSS via vLLM, 2 closed API, 1 OSS 20B via vLLM
- [2026-04-05] Final turn budget: T=12 per thread, k=4 parallel → sequential gets 48 turns. Previous T=5 results are preliminary only.
- [2026-04-05] Conditions: sequential, naive_parallel, diversity_parallel (greedy-Jaccard, pool=16)
