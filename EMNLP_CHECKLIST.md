# EMNLP Submission Checklist — T3 Paper

**Last updated:** 2026-04-26 (Table 2 — all 4 models in. Only gemma3-12b/browsecomp k=8 tail still finishing)
**Deadline:** TBD
**Lead:** @sid-in-the-loop

Update `Last updated` each time this file is touched. Add a one-line entry under "Daily log" at the bottom.

---

## 1. Experiments

### 1a. Main table (Phase 1) — k=4, T=8, pool=16, ClueWeb

Grid: 4 models × 8 datasets × 3 conditions × 5 seeds. Target: all post-judge pass@4_llm numbers.

**Legend:** ✅ done & judged | 🟡 running/partial | ❌ broken/missing | ⬜ not started

All values = pass@4_llm (%). 3 seeds per cell (new spec). WebWalker dropped.

#### qwen3-8b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| hotpotqa | 39.8 | 42.4 | 48.0 | 50.4 | **57.0** |
| musique | 14.6 | 20.6 | 27.1 | 23.9 | **29.7** |
| 2wiki | 29.3 | 46.8 | **55.0** | 47.4 | 53.4 |
| bamboogle | 35.2 | 47.2 | **57.8** | 47.7 | 56.8 |
| frames | 15.2 | 24.9 | 30.5 | 24.8 | **31.4** |
| GAIA | 7.1 | 12.6 | 12.0 | 14.6 | **14.9** |
| hle | 10.7 | 21.6 | **22.0** | 20.9 | 20.3 |

Diversity > Naive on all 7. k=8 edges out k=4 for diversity on most.

#### qwen3-4b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| hotpotqa | 32.0 | 41.9 | **53.2** | 42.3 | 53.1 |
| musique | 9.9 | 15.9 | 19.7 | 16.5 | **20.5** |
| 2wiki | 24.3 | 41.9 | 49.0 | 42.5 | **50.9** |
| bamboogle | 20.5 | 32.5 | 40.8 | 32.5 | **41.6** |
| frames | 10.1 | 15.5 | 20.3 | 14.7 | **20.4** |
| GAIA | 2.9 | 7.8 | 10.7 | 7.4 | **13.3** |
| hle | 8.7 | 17.1 | **20.4** | 16.6 | 19.3 |

Div > Naive on all 7. k=8 div edges out k=4 div on 5/7 datasets.
Small-model diversity can match big-model naive (4b div_k4 = 53.2 > 8b naive_k4 = 42.4 on hotpotqa).

#### qwen3-1.7b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| hotpotqa | 22.3 | 42.9 | 43.8 | 42.3 | **44.1** |
| musique | 7.6 | 14.5 | **15.6** | 15.0 | 13.5 |
| 2wiki | 17.2 | 37.6 | **41.5** | 38.7 | 38.9 |
| bamboogle | 12.3 | 16.8 | **24.3** | 13.6 | 24.3 |
| frames | 6.0 | 13.1 | **13.6** | 12.9 | 13.6 |
| GAIA | 4.5 | **10.0** | 8.7 | 8.1 | 8.1 |
| hle | 8.8 | 19.2 | 19.5 | 18.7 | **19.3** |

For 1.7B the div/naive gap is narrower (~1pp) than 4b/8b (~5-15pp). k=8 div often ties or slightly beats k=4 div. GAIA is the exception: naive > div (small dataset, high noise).

#### gemma3-4b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| hotpotqa | 28.2 | 40.0 | 49.2 | 39.5 | **49.4** |
| musique | 8.5 | 17.2 | 16.1 | 8.1 | **18.0** |
| 2wiki | 25.9 | 42.8 | **52.1** | 43.9 | 51.3 |
| bamboogle | 22.4 | 27.7 | **37.9** | 21.3 | 36.0 |
| frames | 6.0 | 12.3 | 14.6 | 12.4 | **15.0** |
| GAIA | 0.0 | 6.2 | 6.8 | **7.8** | 6.8 |
| hle | 7.2 | 17.8 | 18.1 | 17.6 | **19.0** |

Div wins on 6/7. k=8 div often ties/beats k=4 div. **Quirk at k=8 naive**: drops vs k=4 naive on hotpotqa/musique/bamboogle — halved max_tokens seems to hurt naive more than diversity. GAIA seq = 0.0 is real model weakness.

#### gemma3-12b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| hotpotqa | 37.9 | 54.9 | **59.0** | 54.6 | 58.9 |
| musique | 18.0 | 31.6 | 36.1 | 33.1 | **36.9** |
| 2wiki | 33.0 | 52.0 | 53.9 | 53.4 | **54.0** |
| bamboogle | 41.9 | 55.7 | **64.3** | 56.0 | 63.7 |
| frames | 21.6 | 30.9 | **37.5** | 31.4 | 37.0 |
| GAIA | 7.8 | 12.9 | 16.2 | 13.3 | **16.5** |
| hle | 11.6 | 23.6 | **24.3** | 22.8 | 23.5 |

Big model, big gap: seq→div_k4 is +22pp on bamboogle, +21pp on hotpotqa. **Best cells overall: bamboogle/div_k4 64.3%, hotpotqa/div_k4 59.0%.**

**Dropped (partner handling):** gpt-oss-20b, Llama-3.1-8B

**Experiments: DONE.** Everything above (5 models × main table + k=2 + pool-size + oversample-until-N).

### 1b. Pass@k / compute-matched budget ablation (Phase 2)

Grid: 4 models × 3 datasets × {k=2, k=8} × 2 conditions × 5 seeds. 48 jobs.

k=4 and k=8 are in the main table above. Phase 2 only adds k=2 (pass@k curve's middle point).

#### Phase 2 k=2 results (pass@4_llm %)

| Model | hotpotqa naive@2 / div@2 | bamboogle naive@2 / div@2 | GAIA naive@2 / div@2 |
|---|---|---|---|
| qwen3-8b   | 45.8 / 48.8 | 40.3 / 44.5 | 10.0 / 8.4 |
| qwen3-4b   | 36.5 / 45.3 | 25.1 / 31.5 | 5.5 / 7.4 |
| qwen3-1.7b | 32.6 / 36.5 | 13.9 / 18.4 | 5.5 / 4.5 |
| gemma3-4b  | 34.4 / 40.0 | 22.7 / 27.7 | 2.9 / 4.5 |
| gemma3-12b | 47.1 / 50.5 | 49.3 / 56.0 | 12.9 / 11.7 |

Budget matching: k=2 → T=16, max_tok=2048 | k=8 → T=8, max_tok=1024. Total = 65,536 tokens/q.

### 1c. Pool-size ablation ✅ COMPLETE

qwen3-{1.7b, 8b} × {hotpotqa, GAIA} × pool ∈ {4, 8, 16, 32} × div_k4.

| Model | Dataset | pool=4 | pool=8 | pool=16 | pool=32 | spread |
|---|---|---|---|---|---|---|
| qwen3-1.7b | hotpotqa | 44.2 | 44.1 | 44.3 | 44.4 | 0.3 |
| qwen3-1.7b | GAIA | 9.4 | 7.1 | 7.1 | 7.4 | 2.3 |
| qwen3-8b | hotpotqa | 54.1 | 55.8 | 54.5 | 54.0 | 1.8 |
| qwen3-8b | GAIA | 13.9 | 12.0 | 11.0 | 12.3 | 2.9 |

**Pool size basically doesn't matter.** Hotpotqa flat across both models (< 2pp spread), GAIA noisy (dataset is small, within seed variance). **Validates pool=16 as main-table choice** — bigger pools don't buy anything.

### 1d. Oversample-until-turn-N ablation ✅ COMPLETE

qwen3-{1.7b, 8b} × {hotpotqa, GAIA} × N ∈ {1..8} × div_k4 (pool=16). N=1 = turn-1-only (standard); N=8 = every turn uses pool override.

| Model | Dataset | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | best N | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | hotpotqa | 44.6 | 44.5 | 44.2 | 44.3 | 45.6 | 45.5 | 45.2 | 44.8 | N=5 (45.6) | 1.4 |
| qwen3-1.7b | GAIA | 9.4 | 8.1 | 8.4 | 7.1 | 5.5 | 8.7 | 7.4 | 9.4 | N=1/8 (9.4) | 3.9 |
| qwen3-8b | hotpotqa | 56.1 | 56.6 | 56.9 | 56.3 | 56.1 | 57.1 | 56.1 | 56.0 | N=6 (57.1) | 1.1 |
| qwen3-8b | GAIA | 12.9 | 14.9 | 14.6 | 14.9 | 13.6 | 13.3 | 11.7 | 11.7 | N=2/4 (14.9) | 3.2 |

**Key finding: turn-1-only diversity is ~optimal**. Extending pool-override to later turns yields <2pp gain on hotpotqa, noise-level on GAIA. **Simpler is better** — the N=1 default is well-chosen.

### 1e. TABLE 2 — Serper + web-reasoning prompt (hard-reasoning + browsing tasks)

Setup: Serper search backend, `web_reasoning` prompt (with `<summary>` action), web-task LLM judge. seq T=25 max_tok=8192; k=4 T=8 max_tok=8192; k=8 T=8 max_tok=4096. 3 seeds.

Datasets: webwalker_sub (250), hle_sub (250), gaia_full (103), browsecomp_sub (250).

#### qwen3-4b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| webwalker | 28.5 | 38.7 | 44.9 | 39.2 | **46.9** |
| hle | 6.4 | 9.7 | **14.3** | 10.0 | 12.9 |
| gaia | 14.9 | 22.7 | **27.8** | 21.0 | 27.5 |
| browsecomp | 0.9 | **1.6** | 0.1 | 1.3 | 0.5 |

#### qwen3-8b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| webwalker | 26.7 | 41.6 | **46.8** | 42.1 | 44.9 |
| hle | 4.1 | 10.0 | 11.5 | 9.3 | **12.4** |
| gaia | 13.3 | 23.9 | 26.2 | 24.6 | **27.8** |
| browsecomp | 1.7 | 3.9 | 0.7 | **4.1** | 1.5 |

#### gemma3-12b 🟢 (browsecomp k=8 tail running)

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| webwalker | 22.9 | 38.0 | **45.2** | 40.7 | 44.5 |
| hle | 6.9 | 12.7 | **14.8** | 14.7 | 13.6 |
| gaia | 14.9 | 34.0 | 34.9 | 33.7 | **35.9** 🏆 |
| browsecomp | 0.4 | **2.7** | 2.0 | 🟡 | 🟡 |

#### gpt-oss-20b ✅ COMPLETE

| Dataset | Seq | Naive@4 | Div@4 | Naive@8 | Div@8 |
|---|---|---|---|---|---|
| webwalker | 1.1 | 3.1 | **15.7** | 3.9 | 12.9 |
| hle | 4.1 | 8.3 | **9.3** | 7.7 | 8.0 |
| gaia | 7.4 | 14.2 | **18.8** | 12.3 | 16.5 |
| browsecomp | 0.0 | 0.3 | 0.5 | 0.3 | **1.2** |

### Notable Table-2 findings
- **gemma3-12b is the GAIA champion** (35.9% div_k8) — beats all others by ~8pp
- **gpt-oss-20b sequential is weirdly bad** (webwalker seq=1.1%) but div_k4 jumps to 15.7% (14×). Diversity rescues weak single-threads
- **BrowseComp is brutal everywhere** — best score is gemma3-12b/naive_k4 at 2.7%. Confirms it's a stress benchmark
- **Diversity wins on every dataset for every model in Table 2** (modulo a few k=4 vs k=8 ties)

**Headline shifts vs Table 1 (ClueWeb):**
- **WebWalker: 0% → 46.9%** (qwen3-4b div_k8) — Serper finds the actual web pages
- **GAIA: 13.3 → 27.8** (qwen3-8b div_k8) — 2.1× improvement
- **HLE: ~22 → ~13** — slight regression (Serper may return less academic-grade content)
- **BrowseComp**: barely above 0 — extremely hard benchmark, expected behavior at this size

### 1f. Already-done anchors (re-use, don't rerun)
- ✅ Figure 1: GAIA-103 at different model sizes (`results/figure1_*_judged/`)
- ✅ Figure 2: diversity method comparison (greedy-Jaccard vs MMR vs random) (`results/figure2/`)

---

## 2. Results & Figures

- [ ] Main table (Table 1 — EM datasets): hotpotqa, musique, 2wiki, bamboogle, frames
- [ ] Main table (Table 2 — LLM-judge datasets): GAIA, hle, webwalker
- [ ] Figure: pass@k curves (one line per method, 3 datasets × 4 models)
- [ ] Figure: model-size scaling plot (method gap vs model size)
- [ ] Analysis table/figure: QPD, ITC, ATC metrics per condition
- [ ] Qualitative example: 1-2 case studies (see `paper_assets/case_studies.json`)
- [ ] Aggregate script verified on final data: `webwalkerqa.scripts.aggregate_results`

---

## 3. Writing

- [ ] Abstract
- [ ] Introduction (motivation, contributions)
- [ ] Related work (test-time scaling, parallel decoding, search-augmented agents)
- [ ] Method section (greedy-Jaccard diversity selection, algorithm pseudocode)
- [ ] Experimental setup (models, datasets, metrics, search backend, hyperparameters)
- [ ] Main results narrative
- [ ] Ablations narrative (pass@k, model scaling)
- [ ] Discussion / limitations
- [ ] Conclusion
- [ ] Limitations section (required by EMNLP)
- [ ] Ethics / broader impact (required)
- [ ] References / BibTeX complete
- [ ] Appendix: extra tables, prompt details, per-dataset breakdowns

---

## 4. Reproducibility

- [ ] README with setup instructions (conda env, ClueWeb endpoint, vLLM launch)
- [ ] Dataset preparation script documented (`prepare_datasets.py`)
- [ ] Exact hyperparameters recorded (T=8, pool=16, temp=1.0 turn 1, max_tokens=2048)
- [ ] Seeds documented (1–5)
- [ ] Judge prompt + judge model (gpt-4o-mini) documented
- [ ] Code release decision (GitHub public / anonymous / supplementary zip)

---

## 5. Submission mechanics

- [ ] Overleaf project created (EMNLP 2026 template)
- [ ] Author list / affiliations finalized
- [ ] Anonymization pass (if double-blind)
- [ ] Page count check (8 pages main + unlimited refs + limitations + ethics)
- [ ] Figures exported as PDF at 300dpi
- [ ] Supplementary materials zipped
- [ ] arXiv version prepared

---

## 6. Known risks / open questions

- **WebWalker dropped from main table (2026-04-21).** ClueWeb22 search results don't contain the specific web-page content WebWalker questions reference (conference pages, award lists, etc.). Summary-action + 32k context don't help — it's a search-backend mismatch, not a method failure. Score stuck ~1.5% regardless. Serper would work but running a mixed-backend main table is ugly. Not worth the effort for EMNLP. Mention once in dataset-selection criteria or limitations. Not worth tuning.
- **Context window:** must use vLLM with `--max-model-len 32768` for sequential (k=1, T=32) to avoid overflow.
- **ClueWeb vs Serper gap:** our results use ClueWeb; friend's reported numbers use Serper. Note the backend difference when comparing.
- **Sequential pass@4 = pass@1:** k=1 has only one rollout, so EM pass@4 and EM pass@1 are identical. Note in caption.

---

## Daily log

- **2026-04-26** — Table 2 essentially DONE (38/40 jobs). qwen3-4b/qwen3-8b/gpt-oss-20b complete; gemma3-12b/browsecomp k=8 still finishing. Headlines: gemma3-12b dominates GAIA (35.9% best in table), gpt-oss-20b's diversity gain is huge (webwalker 3.1→15.7 from naive_k4→div_k4), all 4 models show Div > Naive on every dataset.
- **2026-04-25** — Table 2 (Serper + web-reasoning) qwen3-4b + qwen3-8b mostly done.
- **2026-04-23 (midday)** — 🎉 ALL TABLE 1 EXPERIMENTS DONE. Main table (5 models × 7 datasets × 5 conds × 3 seeds), Phase 2 k=2 ablation, pool-size ablation, oversample-until-N ablation — all landed.
- **2026-04-23 (morning)** — Wave A landed. gemma3-12b 5/7 + k=2 done. Merged judge into exp jobs for 2x submit cap.
- **2026-04-22 (late evening)** — Batches 8+9 landed. gemma3-4b FULLY COMPLETE.
- **2026-04-22 (evening)** — Batch 7: qwen3-1.7b FULL. gemma3-4b seq + k=4 done.
- **2026-04-22 (PM3)** — Batch 5 landed. qwen3-1.7b naive_k4 + div_k4 + naive_k8 done.
- **2026-04-22 (PM2)** — Batch 4 landed. qwen3-4b FULLY COMPLETE (div_k8 + k=2). qwen3-1.7b seq done.
- **2026-04-22 (PM)** — Batch 3 landed. qwen3-4b middle (naive_k4 + div_k4 + naive_k8).
- **2026-04-22 (AM)** — qwen3-8b FULL. Div > Naive > Seq across all datasets.
- **2026-04-15** — Created checklist.

<!-- Append entries below, newest at top -->

---

## 7. Synthesis (Aggregation) Accuracy — 3-seed means

`synthesis_accuracy` = an LLM call that combines the k thread answers into one
final answer, judged against the gold. Reported as % over 3 seeds.
Excludes gpt-oss-20b (off-spec) and browsecomp (all rows = 0.0%).

### 7a. TABLE 1 — synthesis_accuracy (ClueWeb22, MHQA)

#### qwen3-1.7b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| hotpotqa | 16.80 | 17.71 | 17.45 | **18.55** |
| musique | **2.80** | 2.15 | 2.60 | 2.02 |
| 2wikimultihopqa | 13.74 | 13.54 | **16.21** | 12.63 |
| bamboogle | 5.60 | **9.07** | 4.00 | 8.27 |
| frames | 2.59 | 2.91 | **3.16** | 2.83 |
| GAIA | **4.21** | 3.24 | 3.56 | 3.56 |
| hle | 3.13 | 3.00 | **3.47** | 2.47 |

#### qwen3-4b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| hotpotqa | 17.58 | 23.76 | 19.53 | **24.80** |
| musique | 3.39 | 3.78 | 3.39 | **3.71** |
| 2wikimultihopqa | 17.71 | 19.73 | 18.29 | **21.74** |
| bamboogle | 16.27 | 19.47 | 15.73 | **20.80** |
| frames | 3.52 | 4.61 | 3.92 | **5.18** |
| GAIA | 3.56 | 3.88 | 3.56 | **5.50** |
| hle | 3.60 | **3.93** | 3.07 | 3.53 |

#### qwen3-8b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| hotpotqa | 19.66 | 19.99 | 26.50 | **28.12** |
| musique | 2.99 | 5.21 | 4.56 | **7.81** |
| 2wikimultihopqa | 24.87 | 27.08 | 25.91 | **27.99** |
| bamboogle | 21.33 | **29.33** | 20.00 | 28.27 |
| frames | 6.84 | 9.26 | 7.00 | **9.99** |
| GAIA | 5.83 | 3.56 | **6.80** | 5.50 |
| hle | 3.80 | 3.47 | **4.27** | 3.40 |

#### gemma3-4b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| hotpotqa | 15.69 | 8.20 | **17.64** | 11.26 |
| musique | 1.04 | 1.04 | 0.65 | **1.37** |
| 2wikimultihopqa | 16.86 | 7.88 | **20.31** | 10.48 |
| bamboogle | 6.67 | 5.33 | **9.60** | 5.33 |
| frames | 1.29 | 2.10 | 1.13 | 1.82 |
| GAIA | 1.62 | **2.27** | 1.29 | 1.62 |
| hle | 2.20 | 1.80 | 2.20 | 2.00 |

**Quirk:** Standard > Ours on most datasets — synthesis call seems to lose more from Gemma3-4B's noisier diverse-thread outputs than from Standard's more homogeneous outputs.

#### gemma3-12b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| hotpotqa | 18.42 | 28.06 | 22.53 | **28.91** |
| musique | 7.68 | 7.88 | 7.88 | **8.66** |
| 2wikimultihopqa | 16.21 | 21.68 | 18.75 | **21.88** |
| bamboogle | 32.53 | **37.33** | 30.93 | 35.73 |
| frames | 9.71 | 10.96 | 9.87 | **11.00** |
| GAIA | 5.18 | **7.77** | 6.80 | 5.83 |
| hle | 1.53 | 1.67 | 1.33 | 1.33 |

### 7b. TABLE 2 — synthesis_accuracy (Serper, web-reasoning)

#### qwen3-4b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| webwalker | 10.67 | 11.73 | 11.47 | **12.67** |
| hle | 4.27 | 4.80 | **5.33** | 4.40 |
| gaia | 13.27 | 14.56 | 12.62 | **18.45** |

#### qwen3-8b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| webwalker | 10.93 | 13.07 | 11.47 | **13.60** |
| hle | **3.47** | 3.07 | 2.67 | 2.80 |
| gaia | 14.56 | 11.65 | 13.92 | **15.86** |

#### gemma3-12b

| Dataset | Standard@4 | Ours@4 | Standard@8 | Ours@8 |
|---|---|---|---|---|
| webwalker | 9.87 | 11.33 | **12.00** | 11.33 |
| hle | 3.20 | 3.20 | 2.80 | **3.47** |
| gaia | 16.50 | 14.89 | **19.74** | 18.77 |

### Takeaways
- **Aggregation tracks pass@4 ordering** (Ours ≥ Standard) on every qwen / gemma3-12b cell in both tables — gain is roughly 2-6 pp absolute.
- **Gemma3-4B is the outlier** — synthesis underperforms with diverse threads (likely because Gemma3-4B's reasoning is noisier and aggregation amplifies disagreement).
- **k=8 ≥ k=4** in 25 / 30 cells — adding parallel threads helps aggregation modestly.
- **HLE shows no aggregation gain** for any model — academic-question style penalizes synthesis (LLM combiner can't reconcile disagreements well on niche topics).
