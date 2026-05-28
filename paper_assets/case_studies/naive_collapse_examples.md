# Naive-collapse case studies

Three illustrative examples from the naive-parallel rollouts where all 4 threads
issued the same turn-1 query (QPD = 0.000) and all 4 final answers were judged wrong.
Drawn from the Bamboogle benchmark; trajectories sit under
`results/main_table_clueweb_t8/`.

---

## Case 1 — Confusing "father of X" with X itself

**Question:** *Who was the father of the father of empiricism?*
**Gold answer:** Sir Nicholas Bacon
**Model:** Qwen3-8B, seed 1 · **File:** `bamboogle/naive_parallel/run_1/.../bamboogle-2596.json`
**Turn-1 QPD:** 0.000 — all 4 threads queried `"father of the father of empiricism"`

### What happened

All four threads issued the same query, retrieved the same Wikipedia
snippets about Empiricism (John Locke, Descartes, Hume), and arrived at the same
conceptual mistake: identifying **Francis Bacon** (who *is* the father of
empiricism) as the answer, instead of finding **his** father (Sir Nicholas Bacon).
None of the four threads ever rephrased the query to dig past the surface match.

### All 4 turn-1 queries (identical)

```
father of the father of empiricism
```

### All 4 final answers

| Thread | Final answer (summarised) |
|---|---|
| 0 | "…**Francis Bacon**. Bacon is regarded as the founder of the empiricist tradition…" |
| 1 | "…**Francis Bacon**. Bacon is regarded as one of the founders of British empiricism…" |
| 2 | "…**Francis Bacon**. Bacon is regarded as the founder of the empirical method…" |
| 3 | "…**Francis Bacon**. He is regarded as the founder of the empirical method…" |

---

## Case 2 — Retrieval mismatch leads to identical wrong person

**Question:** *Who was the second wife of the founder of CNN?*
**Gold answer:** Jane Shirley Smith
**Model:** Qwen3-8B, seed 1 · **File:** `bamboogle/naive_parallel/run_1/.../bamboogle-2458.json`
**Turn-1 QPD:** 0.000 — all 4 threads queried `"second wife of the founder of CNN"`

### What happened

All four threads issued the same query and ClueWeb returned a doc
about **Louis Rukeyser** (a CNN financial commentator, not the founder) and his
wife Alexandra Gill. Without diversity, every thread re-issued the same query
again (some up to 5 times!) and converged on Alexandra Gill / Jane Turner — none
identified Jane Shirley Smith.

### All 4 turn-1 queries (identical)

```
second wife of the founder of CNN
```

### Re-query behaviour (the smoking gun)

Without diversity pressure, threads simply repeat themselves:

- Thread 0: queries `"second wife of the founder of CNN"` 4× across turns 1–4, then answers at turn 5
- Thread 1: queries the same 3× across turns 1–3, then answers at turn 4
- Thread 2: queries the same 2× across turns 1–2, then answers at turn 3
- Thread 3: queries the same 3× across turns 1–3, then answers at turn 4

### All 4 final answers

| Thread | Final answer (summarised) |
|---|---|
| 0 | "…**Alexandra Gill**. She was married to Louis Rukeyser…" |
| 1 | "…**Alexandra Gill**. She was married to Louis Rukeyser, who was a commentator for CNN…" |
| 2 | "…**Alexandra Gill**. She was married to Louis Rukeyser…" |
| 3 | "…the second wife of Ted Turner, the founder of CNN, was **Jane Turner**…" |

---

## Case 3 — Threads diverge in numbers but all wrong

**Question:** *How much protein in four boiled egg yolks?*
**Gold answer:** 10.8 (grams)
**Model:** Qwen3-4B, seed 2 · **File:** `bamboogle/naive_k4/run_2/.../bamboogle-2393.json`
**Turn-1 QPD:** 0.000 — all 4 threads queried `"protein in four boiled egg yolks"`

### What happened

All four threads issued the same query and ClueWeb returned a snippet
about egg-yolk composition that did not state the per-yolk protein content. With
no other evidence to draw on, each thread improvised a different base number for
"protein per yolk" (0.6 g, 0.875 g, 0.9 g, or "indeterminate") and multiplied by 4.
None of the four landed within 7 g of the correct 10.8 g answer.

### All 4 turn-1 queries (identical)

```
protein in four boiled egg yolks
```

### All 4 final answers

| Thread | Base assumed (g/yolk) | Final answer |
|---|---|---|
| 0 | 0.875 | "approximately **3.5 g** of protein" |
| 1 | 0.6   | "approximately **3.6 g** of protein" |
| 2 | —     | "it is not possible to determine the exact amount…" |
| 3 | 0.9   | "approximately **3.6 g** of protein" |

---

## Why these three together

Each case isolates a different failure mode under naive collapse:

| Case | Failure mode |
|---|---|
| 1. Francis Bacon | All threads make the same *conceptual* error and don't rephrase |
| 2. CNN second wife | All threads get the same *retrieval mismatch* and re-query identically |
| 3. Egg yolk protein | All threads improvise from the same *missing-evidence* state |

In all three, DIFFUSE's diverse-query selection would have routed at least one
thread to a different starting point: e.g. "Francis Bacon parents," "Ted Turner
biography wives," or "boiled egg nutrition facts" — any of which retrieves
documents containing the correct answer in their top-3.
