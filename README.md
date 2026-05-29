<h1 align="center">Beyond Parallel Sampling:<br>Diverse Query Initialization for Parallel Agentic Search</h1>

<div align="center">
<a href="https://github.com/sid-in-the-loop">Sid Murali</a>*,
Ethan Chi*,
<a href="#">João Coelho</a>

Language Technologies Institute, Carnegie Mellon University
</div>

<div align="center">

[![Paper](https://img.shields.io/badge/EMNLP-2026-blue.svg?style=flat)](#citation)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](./LICENSE)

</div>

---

## The Problem

Parallel agentic search runs *k* threads independently and aggregates their answers. The intuition is that more threads cover more of the search space. In practice, threads tend to issue near-identical turn-1 queries — we call this **anchor collapse**.

<div align="center">
  <img src="paper_assets/figures/fig2/qpd_density.png" width="340" />
  <p><i>Turn-1 query diversity (QPD) under standard parallel sampling vs. DivInit (Qwen3-8B, k=4). Standard sampling clusters near QPD≈0.2; DivInit forces spread toward QPD≈0.85.</i></p>
</div>

The clustering matters because the first query anchors the entire trajectory. Once threads retrieve similar evidence early, they stay coupled — subsequent reasoning and search remain on the same narrow path regardless of temperature.

<div align="center">
  <img src="paper_assets/figures/fig1/turn1_imprint.png" width="300" />
  <p><i>Turn-1 QPD vs. full-trajectory diversity, per question. Standard threads cluster at low QPD and low trajectory diversity. DivInit shifts the distribution to the high-QPD, high-diversity region. Positive correlation in both conditions (ρ=0.42 / 0.35) confirms that turn-1 retrieval acts as an anchor throughout the search.</i></p>
</div>

## Method: DivInit

**DivInit** is a single training-free intervention at turn 1. Rather than launching *k* independent threads, DivInit:

1. Issues **one** shared LLM call that generates a pool of *n* candidate queries  
2. Selects *k < n* seeds via **MMR** (greedy max-min Jaccard distance, λ=0)  
3. Runs one thread per selected seed — everything from turn 2 is unchanged

```
Standard:   k × T  LLM calls    (k independent turn-1 calls)
DivInit:    1 + k(T−1) calls    (one shared pool call, k−1 fewer than standard)
```

No fine-tuning, no reward model. Selection runs in milliseconds on token-level Jaccard. Compatible with any ReAct-style agent loop.

## Results

<div align="center">
  <img src="paper_assets/figures/fig3/passk_sweep.png" width="380" />
  <p><i>pass@k vs. number of parallel threads k, averaged across benchmarks. The gap between DivInit and standard parallel sampling widens as k grows.</i></p>
</div>

DivInit consistently improves pass@k across five open-weight models and eight benchmarks, with **+5–7 points on multi-hop QA** at matched compute (Table 1 below). Gains scale with model capacity — near-zero at 1.7B, largest at 8B — suggesting a capacity floor below which models cannot act productively on varied seeds.

<div align="center">
  <img src="paper_assets/figures/fig4/gain_vs_size.png" width="420" />
  <p><i>Absolute pass@4 gain (DivInit − Standard) per dataset and model size (Qwen3 1.7B/4B/8B). Gains are consistent across datasets and grow with model scale.</i></p>
</div>

**Main table (pass@4, %)** — Standard (S) vs. DivInit (DI):

| Model | HpQA | MuSi | 2Wiki | Bambo | FRAMES | Avg (MHQA) | GAIA | HLE | WebWalker | Avg (Web) |
|-------|------|------|-------|-------|--------|------------|------|-----|-----------|-----------|
| Qwen3-1.7B | 42.9 / **43.8** | 14.5 / **15.6** | 37.6 / **41.5** | 16.8 / **24.3** | 13.1 / **13.6** | 25.0 / **27.8** | — | — | — | — |
| Qwen3-4B | 41.9 / **53.2** | 15.9 / **19.7** | 41.9 / **49.0** | 32.5 / **40.8** | 15.5 / **20.4** | 29.5 / **36.6** | 22.7 / **27.8** | 9.7 / **14.3** | 38.7 / **44.9** | 23.7 / **29.0** |
| Qwen3-8B | 50.4 / **57.0** | 23.9 / **29.7** | 46.3 / **55.1** | 47.7 / **57.6** | 24.8 / **30.8** | 38.6 / **46.0** | 26.0 / **30.2** | 10.0 / **14.1** | 41.6 / **46.8** | 25.2 / **28.2** |
| Gemma3-4B | 40.0 / **49.2** | 17.2 / 16.1 | 42.8 / **52.2** | 27.7 / **37.9** | 12.3 / **14.7** | 28.0 / **34.0** | — | — | — | — |
| Gemma3-12B | 54.9 / **59.1** | 31.6 / **36.1** | 52.0 / **53.9** | 55.7 / **64.3** | 31.0 / **37.5** | 45.0 / **50.2** | 34.0 / **35.0** | 12.7 / **14.8** | 38.0 / **45.2** | 28.2 / **31.6** |

## Repository Structure

| Path | Description |
|------|-------------|
| `general_agent/webwalkerqa/` | Core agent, DivInit method, evaluation loop |
| `general_agent/data/main_table/` | Benchmark datasets (8 benchmarks as `.json`) |
| `general_agent/scripts/` | SLURM launchers for all experiments |
| `results/main_table/` | Main table results (clueweb and serper backends) |
| `results/ablations/` | Oversample, pool-size, and temperature ablations |
| `paper_assets/figures/` | All paper figures with generation scripts |
| `AggAgent/` | AggAgent submodule (Lee et al., 2026) |

## Getting Started

**Install:**
```bash
cd general_agent
pip install -e .
pip install litellm sentence-transformers httpx python-dotenv
```

**API keys** — create `general_agent/.env`:
```
OPENAI_API_KEY=...
SERPER_API_KEY=...
```

**Quick run (DivInit on HotpotQA):**
```bash
cd general_agent
python -m webwalkerqa.run.run_main_table \
  --model openai/gpt-4o-mini \
  --dataset data/main_table/hotpotqa.json \
  --condition diversity_parallel \
  --output-dir ../results/my_run
```

## Reproducing Results

**Open models** (Qwen3, Gemma3 via vLLM) — see [`general_agent/HOW-TO-VLLM.md`](general_agent/HOW-TO-VLLM.md):
```bash
bash scripts/launch_vllm_server.sh Qwen/Qwen3-8B 8003
bash scripts/submit_main_table_open.sh qwen3-8b
```

**Closed models** (GPT-4o-mini, Gemini):
```bash
bash scripts/submit_main_table.sh gpt-4o-mini
```

**Ablations:**
```bash
bash scripts/submit_oversample_ablation.sh   # oversample-until-turn-N ablation
bash scripts/submit_poolsize.sh              # pool size sweep
bash scripts/submit_temperature_sweep.sh     # temperature sweep
```

## Citation

```bibtex
@inproceedings{murali2026divinit,
  title     = {Beyond Parallel Sampling: Diverse Query Initialization for Parallel Agentic Search},
  author    = {Murali, Sid and Chi, Ethan and Coelho, Jo{\~a}o},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
