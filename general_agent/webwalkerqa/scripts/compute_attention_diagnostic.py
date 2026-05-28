"""
Phase B.1 — Attention diagnostic figure for the T3 paper.

Goal: 2x5 grid of attention heatmaps (Naive vs Diversity, 5 attention-binning variants)
to pick the cleanest variant for the paper Fig 3.

Question: gaia-62 (qwen3-8b, GAIA, seed 1)
Conditions: naive_k4, div_k4 (k=4 only)

Variants computed at the moment the model is about to emit a `<search>` query
(prompt with appended "<search>"). Each variant is a 6x6 (source-turn × attended-turn)
matrix, row-normalized.

  A: span_j = <search>q_j</search> tokens. ALL layers averaged.
  B: span_j = <search>q_j</search> + <information>r_j</information> tokens. ALL layers.
  C: span_j = <search>q_j</search>. Source = average across all tokens of the appended
              "<search>" string (smoother). ALL layers.
  D: same as A, LAST layer only.
  E: same as A, MIDDLE layer only (num_layers // 2).

Re-runnable. Saves attention tensors + matrices + plot.

Output:
  paper_assets/fig3_diagnostic_variants.pdf
  paper_assets/fig3_diagnostic_variants.png
  paper_assets/attn_cache/<cond>_t<thread>_turn<t>_attn.npz   (cached forward passes)
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ────────────── config ──────────────
QID                 = "gaia-97"   # multi-turn fact lookup, 4/4 naive + 2/4 div clean
MODEL_NAME          = "Qwen/Qwen3-8B"
TURNS               = 6
RESULTS_ROOT        = Path("/home/ssmurali/t3-testbed/results/main_table_web_serper/qwen3-8b/gaia")
OUT_DIR             = Path("/home/ssmurali/t3-testbed/paper_assets")
PDF_PATH            = OUT_DIR / "fig3_diagnostic_variants.pdf"
PNG_PATH            = OUT_DIR / "fig3_diagnostic_variants.png"
ATTN_CACHE          = OUT_DIR / "attn_cache"
SEED                = 1
SEARCH_OPEN         = "<search>"
SEARCH_CLOSE        = "</search>"
INFO_OPEN           = "<information>"
INFO_CLOSE          = "</information>"
HISTORY_MARKER      = "History Turns:"

CONDITIONS = [
    ("naive_k4", "naive_parallel_T8",      "Naive"),
    ("div_k4",   "diversity_parallel_T8",  "Diversity"),
]

# Cream-to-navy single-hue cmap
CMAP = LinearSegmentedColormap.from_list(
    "cream_navy", ["#FAF6E9", "#A8C7D6", "#3D5A7C", "#16223C"], N=256
)


# ────────────── data + spans ──────────────
def load_thread_full_responses(condition: str, traj_subdir: str, qid: str) -> List[List[Dict]]:
    """Return list of length 4: each entry is a thread's full_responses list."""
    fp = RESULTS_ROOT / condition / f"run_{SEED}" / "trajectories" / traj_subdir / f"{qid}.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    d = json.load(open(fp))
    return [t.get("full_responses", []) for t in d.get("threads", [])]


def is_clean_thread(full_responses: List[Dict], turns: int = TURNS) -> bool:
    """Thread is 'clean' if it issued a <search> at every one of the first `turns` turns
    (no <summary> compaction, no early <answer>)."""
    if len(full_responses) < turns:
        return False
    for f in full_responses[:turns]:
        r = f.get("response", "").lower()
        if "<search>" not in r: return False
        if "<summary>" in r:    return False
        if "<answer>" in r:     return False
    return True


def find_history_spans(prompt: str) -> List[Dict[str, Tuple[int, int]]]:
    """For each past-turn (search, information) pair after "History Turns:" in the prompt,
    return char-offset ranges. List length = number of past turns in this prompt."""
    h_idx = prompt.find(HISTORY_MARKER)
    if h_idx < 0:
        # If no marker, fall back to the last N pairs of <search>/<information> in prompt
        history_part = prompt
        offset = 0
    else:
        offset = h_idx
        history_part = prompt[h_idx:]

    spans = []
    pos = 0
    while True:
        s_open = history_part.find(SEARCH_OPEN, pos)
        if s_open < 0: break
        s_close = history_part.find(SEARCH_CLOSE, s_open)
        if s_close < 0: break
        s_close += len(SEARCH_CLOSE)
        i_open = history_part.find(INFO_OPEN, s_close)
        i_close = -1
        if i_open >= 0:
            i_close = history_part.find(INFO_CLOSE, i_open)
            if i_close >= 0:
                i_close += len(INFO_CLOSE)
        spans.append({
            "search":      (offset + s_open,  offset + s_close),
            "information": (offset + i_open,  offset + i_close) if (i_open >= 0 and i_close > 0) else None,
        })
        pos = (i_close if i_close > 0 else s_close)
    return spans


def char_range_to_token_range(offset_mapping: List[Tuple[int, int]],
                              char_lo: int, char_hi: int) -> Tuple[int, int]:
    """Inclusive token-index range covering [char_lo, char_hi)."""
    tok_lo, tok_hi = -1, -1
    for ti, (cs, ce) in enumerate(offset_mapping):
        if cs == ce == 0: continue  # skip special-token markers
        # token overlaps [char_lo, char_hi)
        if ce > char_lo and cs < char_hi:
            if tok_lo < 0: tok_lo = ti
            tok_hi = ti
    return tok_lo, tok_hi


# ────────────── attention ──────────────
_ATTN_CAPTURE: Dict = {"src_idxs": None, "per_layer": []}


def _install_attention_capture():
    """Monkey-patch transformers Qwen3 eager_attention_forward to capture only the
    attention rows at _ATTN_CAPTURE['src_idxs'] (a python list of token positions),
    then return None for attn_weights so the model doesn't keep the full L×L tensor.
    This avoids OOM when output_attentions=True."""
    import transformers.models.qwen3.modeling_qwen3 as q3
    if getattr(q3, "_t3_patched", False):
        return
    orig = q3.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        attn_output, attn_weights = orig(module, query, key, value, attention_mask,
                                         scaling, dropout, **kwargs)
        idxs = _ATTN_CAPTURE.get("src_idxs")
        if idxs is not None and attn_weights is not None:
            # attn_weights: [B=1, H, S, S]
            sliced = attn_weights[0, :, idxs, :].detach().to(torch.float32).cpu()
            _ATTN_CAPTURE["per_layer"].append(sliced)
        # Drop the full tensor immediately
        return attn_output, None

    q3.eager_attention_forward = patched
    q3._t3_patched = True


def load_model():
    """Load Qwen3-8B in bf16 with eager attention + monkey-patched attention capture."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {MODEL_NAME} (bfloat16, eager attention, hooked capture)…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    kwargs = dict(attn_implementation="eager", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, **kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    _install_attention_capture()
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}, "
          f"num_heads = {model.config.num_attention_heads}, "
          f"device = {next(model.parameters()).device}", flush=True)
    return tok, model


@torch.no_grad()
def run_forward_pass(tok, model, prompt_with_marker: str, capture_src_idxs: List[int]):
    """Tokenize + forward, capturing attention only at the given source positions per layer.
    Returns:
      input_ids   : torch.Tensor [1, L]
      offset_map  : List[Tuple[int,int]]
      attentions  : torch.Tensor [num_layers, num_heads, len(capture_src_idxs), L]   (fp32, CPU)
    """
    enc = tok(prompt_with_marker, return_tensors="pt", return_offsets_mapping=True,
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    offset_map = enc["offset_mapping"][0].tolist()
    L = input_ids.shape[-1]
    # Normalize negative indices
    norm_idxs = [(i if i >= 0 else L + i) for i in capture_src_idxs]
    _ATTN_CAPTURE["src_idxs"] = norm_idxs
    _ATTN_CAPTURE["per_layer"] = []
    try:
        _ = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    finally:
        _ATTN_CAPTURE["src_idxs"] = None
    A = torch.stack(_ATTN_CAPTURE["per_layer"], dim=0)   # [num_layers, H, len(src), L]
    _ATTN_CAPTURE["per_layer"] = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return input_ids, offset_map, A


def _summed_attention(A: torch.Tensor, src_local_idxs: List[int],
                      tgt_token_lo: int, tgt_token_hi: int,
                      layer_slice: slice) -> float:
    """A has shape [num_layers, H, len(captured_src), L]; src_local_idxs index INTO the
    captured-src dim (0..len(captured_src)-1). Sum over [tgt_lo..tgt_hi] target tokens,
    average over (layers, heads, selected src positions)."""
    if tgt_token_lo < 0 or tgt_token_hi < tgt_token_lo: return 0.0
    if not src_local_idxs: return 0.0
    sub = A[layer_slice, :, :, tgt_token_lo:tgt_token_hi+1]      # [Ll, H, Csrc, T]
    sub_summed_tgt = sub.sum(dim=-1)                              # [Ll, H, Csrc]
    sub_at_src = sub_summed_tgt[..., src_local_idxs]              # [Ll, H, len(local)]
    return float(sub_at_src.mean())


def compute_variants_for_turn(
    tok, model, prompt_str: str, source_turn: int, num_layers: int
) -> Optional[Dict[str, np.ndarray]]:
    """Append "<search>" to prompt; capture attention from the appended-<search> tokens
    (so we can do both 'last token' and 'span-avg' variants from one forward pass).
    Returns {variant: row-vector of length TURNS}, NaN for out-of-range cells."""
    prompt_marker = prompt_str + SEARCH_OPEN
    n_marker_tokens = len(tok(SEARCH_OPEN, add_special_tokens=False)["input_ids"])
    # capture src positions = the last n_marker_tokens (these are the appended <search> tokens)
    capture_idxs = [-(n_marker_tokens - i) for i in range(n_marker_tokens)]   # e.g. [-2, -1]
    input_ids, offset_map, A = run_forward_pass(tok, model, prompt_marker, capture_idxs)
    # A.shape = [num_layers, num_heads, n_marker_tokens, L]
    LAST_LOCAL = [n_marker_tokens - 1]      # index INTO captured-src dim
    SPAN_LOCAL = list(range(n_marker_tokens))

    # Find history spans in the original prompt
    spans = find_history_spans(prompt_str)
    if len(spans) < source_turn - 1:
        return None

    rows = {v: np.full(TURNS, np.nan, dtype=np.float64) for v in ("A","B","C","D","E")}
    for j in range(1, source_turn):
        sp = spans[j - 1]
        s_lo, s_hi = char_range_to_token_range(offset_map, *sp["search"])
        if sp["information"]:
            i_lo, i_hi = char_range_to_token_range(offset_map, *sp["information"])
            sb_lo, sb_hi = (s_lo, max(s_hi, i_hi))
        else:
            sb_lo, sb_hi = s_lo, s_hi
        rows["A"][j-1] = _summed_attention(A, LAST_LOCAL, s_lo, s_hi, slice(None))
        rows["B"][j-1] = _summed_attention(A, LAST_LOCAL, sb_lo, sb_hi, slice(None))
        rows["C"][j-1] = _summed_attention(A, SPAN_LOCAL, s_lo, s_hi, slice(None))
        rows["D"][j-1] = _summed_attention(A, LAST_LOCAL, s_lo, s_hi, slice(num_layers-1, num_layers))
        mid = num_layers // 2
        rows["E"][j-1] = _summed_attention(A, LAST_LOCAL, s_lo, s_hi, slice(mid, mid+1))
    return rows


def row_normalize(M: np.ndarray) -> np.ndarray:
    """Row-normalize, treating NaN entries as 0 for the sum."""
    out = np.full_like(M, np.nan)
    for i in range(M.shape[0]):
        row = M[i]
        valid = ~np.isnan(row)
        s = np.nansum(row)
        if s > 0:
            out[i, valid] = row[valid] / s
    return out


def compute_thread_matrices(tok, model, full_responses: List[Dict],
                            desc: str = "") -> Optional[Dict[str, np.ndarray]]:
    """Build a 6x6 matrix per variant for one thread."""
    if len(full_responses) < TURNS:
        return None
    num_layers = model.config.num_hidden_layers
    matrices = {v: np.full((TURNS, TURNS), np.nan, dtype=np.float64) for v in ("A","B","C","D","E")}
    for t in tqdm(range(1, TURNS + 1), desc=desc, leave=False, ncols=80):
        prompt_str = full_responses[t - 1].get("prompt", "")
        rows = compute_variants_for_turn(tok, model, prompt_str, t, num_layers)
        if rows is None:
            return None
        for v, vec in rows.items():
            matrices[v][t - 1, :] = vec
    return {v: row_normalize(M) for v, M in matrices.items()}


# ────────────── stats ──────────────
def _flatten_valid(M: np.ndarray) -> np.ndarray:
    return M[~np.isnan(M)].ravel()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0: return 0.0
    da = float(np.linalg.norm(a)); db = float(np.linalg.norm(b))
    if da == 0 or db == 0: return 0.0
    return float(np.dot(a, b) / (da * db))


def mean_pairwise_cosine(thread_matrices: List[np.ndarray]) -> float:
    flats = [_flatten_valid(m) for m in thread_matrices]
    n = len(flats)
    if n < 2: return float("nan")
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(cosine_sim(flats[i], flats[j]))
    return float(np.mean(vals))


# ────────────── plot ──────────────
def plot_grid(per_cond_per_variant: Dict[str, Dict[str, np.ndarray]],
              variant_titles: List[Tuple[str, str]],
              cond_labels: List[Tuple[str, str]],
              question_id: str) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(11.0, 4.6), dpi=300, constrained_layout=True)

    vmin, vmax = 0.0, 1.0
    for ri, (cond_key, cond_label) in enumerate(cond_labels):
        for ci, (vkey, vtitle) in enumerate(variant_titles):
            ax = axes[ri, ci]
            M = per_cond_per_variant[cond_key][vkey]
            im = ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax,
                           aspect="equal", interpolation="nearest")
            ax.set_xticks(range(TURNS)); ax.set_xticklabels(range(1, TURNS+1))
            ax.set_yticks(range(TURNS)); ax.set_yticklabels(range(1, TURNS+1))
            if ri == 0:
                ax.set_title(vtitle, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{cond_label}\nSource turn", fontsize=8)
            else:
                ax.set_ylabel("Source turn", fontsize=7)
            if ri == nrows - 1:
                ax.set_xlabel("Attended turn", fontsize=7)
            ax.tick_params(length=2, pad=1)
            for s in ("top","right","left","bottom"):
                ax.spines[s].set_color("#333333")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Row-normalized attention", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Attention bin variant comparison — qwen3-8b GAIA Q={question_id}", fontsize=10)

    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ────────────── main ──────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qid", default=QID)
    p.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ATTN_CACHE.mkdir(parents=True, exist_ok=True)

    tok, model = load_model()

    per_cond_thread_matrices: Dict[str, List[Dict[str, np.ndarray]]] = {}

    cond_iter = tqdm(CONDITIONS, desc="conditions", ncols=80)
    for cond_key, traj_subdir, cond_label in cond_iter:
        cond_iter.set_postfix_str(cond_label)
        threads_full_responses = load_thread_full_responses(cond_key, traj_subdir, args.qid)
        # Filter to threads that did <search> at every one of the first TURNS turns
        kept_idx = [i for i, fr in enumerate(threads_full_responses) if is_clean_thread(fr)]
        tqdm.write(f"  {cond_label}: {len(kept_idx)}/{len(threads_full_responses)} clean threads "
                   f"(idx={kept_idx})")
        thread_matrices: List[Dict[str, np.ndarray]] = []
        thread_iter = tqdm([(i, threads_full_responses[i]) for i in kept_idx],
                           desc=f"  {cond_label} threads", leave=False, ncols=80)
        for ti, full_responses in thread_iter:
            cache_path = ATTN_CACHE / f"{args.qid}_{cond_key}_t{ti}_T{TURNS}.npz"
            if cache_path.exists() and not args.force:
                npz = np.load(cache_path)
                tm = {v: npz[v] for v in ("A","B","C","D","E")}
                thread_iter.set_postfix_str(f"thread {ti}: cached")
            else:
                tm = compute_thread_matrices(
                    tok, model, full_responses,
                    desc=f"    {cond_label}/t{ti} turns"
                )
                if tm is None:
                    tqdm.write(f"    ⚠️  {cond_label} thread {ti}: insufficient turns; skipping")
                    continue
                np.savez(cache_path, **tm)
                thread_iter.set_postfix_str(f"thread {ti}: computed")
            thread_matrices.append(tm)
        per_cond_thread_matrices[cond_key] = thread_matrices

    # Average across the 4 threads per (cond, variant)
    per_cond_per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    for cond_key, _, _ in CONDITIONS:
        tms = per_cond_thread_matrices[cond_key]
        per_cond_per_variant[cond_key] = {}
        for v in ("A","B","C","D","E"):
            arr = np.stack([tm[v] for tm in tms], axis=0)   # [n_threads, 6, 6]
            per_cond_per_variant[cond_key][v] = np.nanmean(arr, axis=0)

    # Stats: mean pairwise cosine sim of the 4 thread matrices, per (cond, variant)
    print("\n=== Mean pairwise cosine similarity across the 4 threads (lower under div = more diverse attention) ===")
    print(f"{'variant':<24} {'naive_k4':>10} {'div_k4':>10} {'naive - div':>12}")
    for v, label in [("A","A: <search>, all layers"),
                     ("B","B: <s>+<info>, all layers"),
                     ("C","C: <search>, span-avg src"),
                     ("D","D: <search>, last layer"),
                     ("E","E: <search>, middle layer")]:
        n_sim = mean_pairwise_cosine([tm[v] for tm in per_cond_thread_matrices["naive_k4"]])
        d_sim = mean_pairwise_cosine([tm[v] for tm in per_cond_thread_matrices["div_k4"]])
        print(f"  {label:<24} {n_sim:>10.4f} {d_sim:>10.4f} {n_sim - d_sim:>+12.4f}")

    # Plot
    variant_titles = [
        ("A", "A: <search>, all layers"),
        ("B", "B: <search>+<info>, all layers"),
        ("C", "C: <search>, span-avg"),
        ("D", "D: <search>, last layer"),
        ("E", "E: <search>, middle layer"),
    ]
    cond_labels = [(k, lbl) for (k, _, lbl) in CONDITIONS]
    plot_grid(per_cond_per_variant, variant_titles, cond_labels, args.qid)
    print(f"\nWrote PDF: {PDF_PATH}")
    print(f"Wrote PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()
