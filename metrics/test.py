import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def md(src):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True)
    })

def code(src):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True)
    })

# =====================================================================
# 0. Title + config
# =====================================================================
md("""# Depth Reindexing: Cross-Model Layer Correspondence

**Goal.** Given two models A and B, find the correspondence function φ mapping each layer
of A to its counterpart in B. If φ is monotone, A and B pass through the same
representational stages in the same order. If φ is consistent across *every* pair of
models in the benchmark, this suggests a **universal depth coordinate** shared across
architectures and training objectives.

This notebook runs the full pipeline end to end:

1. Load per-layer embeddings (trained + untrained/random-init controls) for every model.
2. Verify row ordering (`object_id`) is identical across all layers and models.
3. Compute calibrated-MKNN alignment matrices between every layer pair.
4. Recover the correspondence φ via DTW, with an anchored and an open-ended variant.
5. Bootstrap an uncertainty band over random half-samples (k = 5, 10, 20, 50).
6. Test universality: overlay all pairwise φ curves + composition-error table.
7. Run the untrained control (random-init) and the pixel-PCA Layer-0 sanity check.
8. Print a concise, auto-generated summary of findings.

Models (from `extract_layerwise_colm.py`, `HCVYM5w6Gn/colm-results`):
`ar_affine`, `ar_aim`, `mae_affine`, `mae_aim` × `{001M, 021M, 100M}`, each with a
trained and an untrained (step-0 / scratch-init) version.
""")

code("""# --- 0. Config & imports ---------------------------------------------------
from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Directory containing <model>_<trained|untrained>_blocks_layerwise.parquet
# (output of extract_layerwise_colm.py)
EMB_DIR = Path("./reindex_embeddings")

# Model family × scale grid actually extracted upstream
FAMILIES = ["ar_affine", "ar_aim", "mae_affine", "mae_aim"]
SCALES   = ["001M", "021M", "100M"]
MODELS   = [f"{fam}_{sc}" for fam in FAMILIES for sc in SCALES]

K_MKNN        = 10      # neighbourhood size for calibrated MKNN
N_NULL        = 10      # permutation-null draws per (layer_i, layer_j) cell
BOOT_KS       = [5, 10, 20, 50]   # subsample sizes for Step 3 (uncertainty)
N_BOOT_REPEAT = 10                # k repeats per bootstrap setting

plt.rcParams["figure.dpi"] = 110
print(f"Configured {len(MODELS)} models: {MODELS}")
""")

# =====================================================================
# 1. Load data + fix row ordering
# =====================================================================
md("""## 1. Load embeddings & fix row ordering

Every parquet has one row per galaxy (`object_id` = `dr8_id`) and one `list[float]`
column per layer: `encoder`, `h.00 .. h.NN`, `ln_f`, and `pixel_pca` (the raw-pixel
PCA "Layer 0" control, identical across models by construction).

**Row ordering must be identical across every layer and every model.** We assert
this explicitly before doing anything else — a silent mismatch here would
invalidate every downstream alignment score.
""")

code("""# --- 1. Load all parquets, assert consistent object_id ordering ------------

def load_model_layers(model_name: str, tag: str, emb_dir: Path = EMB_DIR):
    \"\"\"Load one (model, trained|untrained) parquet -> (object_ids, {layer_name: (N,D) array}).\"\"\"
    path = emb_dir / f"{model_name}_{tag}_blocks_layerwise.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run extract_layerwise_colm.py first, "
            f"or point EMB_DIR at the right directory."
        )
    df = pl.read_parquet(path)
    object_ids = df["object_id"].to_numpy()
    layer_cols = [c for c in df.columns if c != "object_id"]
    layers = {c: np.array(df[c].to_list(), dtype=np.float32) for c in layer_cols}
    return object_ids, layers


REFERENCE_IDS = None
ALL_DATA: dict[tuple[str, str], dict[str, np.ndarray]] = {}   # (model, tag) -> {layer: (N,D)}
LAYER_ORDER: dict[tuple[str, str], list[str]] = {}             # (model, tag) -> ordered layer names

def ordered_layer_names(layers: dict[str, np.ndarray]) -> list[str]:
    \"\"\"encoder, h.00..h.NN (numeric), ln_f, pixel_pca last (kept separate as a control).\"\"\"
    names = list(layers)
    def keyfn(n):
        if n == "encoder": return (-1, 0)
        if n == "ln_f":    return (10_000, 0)
        if n == "pixel_pca": return (10_001, 0)
        if n.startswith("h."):
            return (int(n.split(".")[1]), 0)
        return (9_999, 0)
    return sorted(names, key=keyfn)

missing = []
for model in MODELS:
    for tag in ("trained", "untrained"):
        try:
            oids, layers = load_model_layers(model, tag)
        except FileNotFoundError as e:
            missing.append(str(e))
            continue
        if REFERENCE_IDS is None:
            REFERENCE_IDS = oids
        else:
            assert np.array_equal(oids, REFERENCE_IDS), (
                f"{model} [{tag}]: object_id ordering mismatch vs reference — "
                f"downstream alignment scores would be meaningless."
            )
        ALL_DATA[(model, tag)] = layers
        LAYER_ORDER[(model, tag)] = ordered_layer_names(layers)

if missing:
    print(f"[warn] {len(missing)} model/tag combos not found on disk yet "
          f"(run extract_layerwise_colm.py to produce them). Continuing with what's loaded.")
    for m in missing[:5]:
        print("   -", m)

N = len(REFERENCE_IDS) if REFERENCE_IDS is not None else 0
print(f"\\nLoaded {len(ALL_DATA)} (model, tag) combinations, N={N} galaxies each.")
for (model, tag), names in LAYER_ORDER.items():
    print(f"  {model:20s} [{tag:9s}] -> {len(names)} layers: {names}")
""")

# =====================================================================
# 2. Calibrated MKNN alignment matrix
# =====================================================================
md("""## 2. Calibrated MKNN Alignment Matrix

For two sets of per-layer embeddings, we compare every layer of A against every
layer of B using **mutual k-nearest-neighbour overlap**: for each galaxy, do the
two embedding spaces agree on which other galaxies are its nearest neighbours?
The raw overlap score is corrected against a **permutation null** (shuffle the
galaxy identities in B and recompute) so brightness in the heatmap is measured
in **σ above chance**, not raw overlap fraction — this matters because chance
overlap depends on N and k and isn't automatically zero.
""")

code("""# --- 2. Calibrated MKNN + alignment matrix ----------------------------------

def _mknn_raw(a: np.ndarray, b: np.ndarray, k: int) -> float:
    d1, d2 = cosine_distances(a), cosine_distances(b)
    np.fill_diagonal(d1, np.inf)
    np.fill_diagonal(d2, np.inf)
    nn1 = np.argsort(d1, axis=1)[:, :k]
    nn2 = np.argsort(d2, axis=1)[:, :k]
    overlap = np.mean([len(set(nn1[i]) & set(nn2[i])) for i in range(len(a))])
    return overlap / k


def mknn_calibrated(z1: np.ndarray, z2: np.ndarray, k: int = K_MKNN,
                     n_null: int = N_NULL, seed: int = 0) -> tuple[float, float, float]:
    \"\"\"Returns (score_above_chance, raw_score, chance_mean).\"\"\"
    raw = _mknn_raw(z1, z2, k)
    local_rng = np.random.default_rng(seed)
    nulls = np.array([
        _mknn_raw(z1, z2[local_rng.permutation(len(z2))], k)
        for _ in range(n_null)
    ])
    chance = nulls.mean()
    chance_std = nulls.std() + 1e-12
    return (raw - chance) / chance_std, raw, chance


def alignment_matrix(layers_A: dict[str, np.ndarray], names_A: list[str],
                      layers_B: dict[str, np.ndarray], names_B: list[str],
                      k: int = K_MKNN, subset_idx: np.ndarray | None = None,
                      seed: int = 0, verbose: bool = True):
    \"\"\"Sigma-above-chance MKNN matrix M[i, j] for layer i of A vs layer j of B.

    subset_idx restricts to a subsample of galaxies (for bootstrap uncertainty).
    Returns (M, chance_raw_mean) — chance_raw_mean is reported once for the
    heatmap's 'chance baseline' annotation.
    \"\"\"
    L_A, L_B = len(names_A), len(names_B)
    M = np.zeros((L_A, L_B))
    chance_raws = []
    for i, na in enumerate(names_A):
        a = layers_A[na]
        if subset_idx is not None:
            a = a[subset_idx]
        for j, nb_ in enumerate(names_B):
            b = layers_B[nb_]
            if subset_idx is not None:
                b = b[subset_idx]
            sigma, raw, chance = mknn_calibrated(a, b, k=k, seed=seed + i * L_B + j)
            M[i, j] = sigma
            chance_raws.append(chance)
        if verbose:
            print(f"  row {i+1}/{L_A} ({na})")
    return M, float(np.mean(chance_raws))
""")

# =====================================================================
# 3. Step 1 — raw correspondence + Step 2 — DTW
# =====================================================================
md("""## 3. Step 1 — Raw Correspondence Curve, Step 2 — DTW Smoothing

**Step 1.** For each layer *i* of A, φ_raw(i) = argmax_j M[i, j] — its best-matching
layer in B. We test monotonicity with the Spearman rank correlation between layer
index *i* and φ_raw(i).

**Step 2.** The raw curve is noisy, so we replace it with the optimal *monotone*
path through the similarity grid using Dynamic Time Warping: cost `C = 1 - M`,
cumulative-cost recurrence `D[i,j] = C[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])`,
then trace back from the final cell. We compute **both**:

- **Anchored DTW** — forces layer 0 ↔ layer 0 and last ↔ last (the default: input is
  input, final output is final output).
- **Open-ended DTW** — no forced endpoints, alignment floats freely.

If the two agree closely, the matching is solid; if they diverge, the anchoring is
doing real work and pacing conclusions should be read more cautiously.
""")

code("""# --- 3. Raw correspondence (Step 1) -----------------------------------------

def raw_correspondence(M: np.ndarray) -> tuple[np.ndarray, float, float]:
    \"\"\"phi_raw(i) = argmax_j M[i,j]; returns (phi_raw, spearman_rho, p_value).\"\"\"
    phi_raw = np.argmax(M, axis=1)
    rho, p = spearmanr(np.arange(len(phi_raw)), phi_raw)
    return phi_raw, float(rho), float(p)


# --- DTW (Step 2): anchored + open-ended ------------------------------------

def dtw_path_anchored(cost: np.ndarray) -> list[tuple[int, int]]:
    \"\"\"Standard anchored DTW: forces (0,0) -> (R-1,C-1).\"\"\"
    R, C = cost.shape
    D = np.full((R, C), np.inf)
    D[0, 0] = cost[0, 0]
    for i in range(1, R):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]
    for j in range(1, C):
        D[0, j] = D[0, j - 1] + cost[0, j]
    for i in range(1, R):
        for j in range(1, C):
            D[i, j] = cost[i, j] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    path, i, j = [(R - 1, C - 1)], R - 1, C - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            m = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if m == 0: i -= 1; j -= 1
            elif m == 1: i -= 1
            else: j -= 1
        path.append((i, j))
    return path[::-1]


def dtw_path_open_ended(cost: np.ndarray) -> list[tuple[int, int]]:
    \"\"\"Open-ended DTW: free start/end on the B (column) axis.

    Still monotone in i, still starts at i=0 and ends at i=R-1, but does NOT
    force j(0)=0 or j(R-1)=C-1 — the best starting/ending column is chosen
    by the optimisation itself.
    \"\"\"
    R, C = cost.shape
    D = np.full((R, C), np.inf)
    D[0, :] = cost[0, :]                       # any starting column allowed
    for i in range(1, R):
        for j in range(C):
            candidates = [D[i - 1, j]]
            if j > 0:
                candidates.append(D[i - 1, j - 1])
                candidates.append(D[i, j - 1])
            D[i, j] = cost[i, j] + min(candidates)
    j_end = int(np.argmin(D[R - 1, :]))
    path, i, j = [(R - 1, j_end)], R - 1, j_end
    while i > 0:
        opts = []
        opts.append((D[i - 1, j], (i - 1, j)))
        if j > 0:
            opts.append((D[i - 1, j - 1], (i - 1, j - 1)))
            opts.append((D[i, j - 1], (i, j - 1)))
        _, (ni, nj) = min(opts, key=lambda t: t[0])
        i, j = ni, nj
        path.append((i, j))
    return path[::-1]


def path_to_relative(path: list[tuple[int, int]], L_A: int, L_B: int) -> np.ndarray:
    \"\"\"Collapse a DTW path into phi: relative depth in B for each layer of A, in [0,1].\"\"\"
    d = defaultdict(list)
    for i, j in path:
        d[i].append(j)
    phi = np.array([np.median(d[i]) for i in range(L_A)])
    return phi / max(L_B - 1, 1)


def path_open_agreement(path_anchored, path_open, L_A) -> float:
    \"\"\"Average |anchored - open| gap in relative-depth units — flags when
    anchoring is doing substantial work vs. genuine structure.\"\"\"
    d_a, d_o = defaultdict(list), defaultdict(list)
    for i, j in path_anchored: d_a[i].append(j)
    for i, j in path_open:     d_o[i].append(j)
    L_B = max(max(v) for v in list(d_a.values()) + list(d_o.values())) + 1
    gaps = []
    for i in range(L_A):
        if i in d_a and i in d_o:
            gaps.append(abs(np.median(d_a[i]) - np.median(d_o[i])) / max(L_B - 1, 1))
    return float(np.mean(gaps)) if gaps else float("nan")
""")

# =====================================================================
# 4. Step 3 — uncertainty via bootstrap
# =====================================================================
md("""## 4. Step 3 — Bootstrap Uncertainty

We repeat the whole alignment + DTW procedure on `k` random half-samples of the
dataset (default k=10 repeats), for subsample sizes **5, 10, 20, 50** galaxies, and
plot the resulting correspondence paths as an uncertainty band around the
full-data solution. A robust correspondence should barely move across
subsamples and across neighbourhood sizes.

Note: "k" here plays double duty in the source material — `k` = MKNN neighbourhood
size (fixed at `K_MKNN=10` throughout) **and** `k` = number of bootstrap repeats /
subsample size swept below. We keep MKNN's `k` fixed and sweep subsample size per
the Step-3 spec (5, 10, 20, 50).
""")

code("""# --- 4. Bootstrap over subsample sizes ---------------------------------------

def bootstrap_correspondence(layers_A, names_A, layers_B, names_B,
                              subsample_sizes=BOOT_KS, n_repeat=N_BOOT_REPEAT,
                              k=K_MKNN, seed=0):
    \"\"\"For each subsample size, draw n_repeat random half-samples (or samples of
    that literal size, whichever the paper's convention — here: literal
    subsample_size galaxies, matching 'k random half-samples' generalised to a
    sweep over sample size) and recompute the anchored-DTW correspondence.
    Returns {subsample_size: (L_A, n_repeat) array of phi curves}.
    \"\"\"
    N_total = len(next(iter(layers_A.values())))
    L_A, L_B = len(names_A), len(names_B)
    out = {}
    local_rng = np.random.default_rng(seed)
    for size in subsample_sizes:
        size = min(size, N_total)
        curves = np.zeros((n_repeat, L_A))
        for r in range(n_repeat):
            idx = local_rng.choice(N_total, size=size, replace=False)
            M_b, _ = alignment_matrix(layers_A, names_A, layers_B, names_B,
                                       k=min(k, size - 1), subset_idx=idx,
                                       seed=local_rng.integers(1e6), verbose=False)
            path_b = dtw_path_anchored(1.0 - M_b)
            curves[r] = path_to_relative(path_b, L_A, L_B)
        out[size] = curves
        print(f"  subsample size={size}: {n_repeat} repeats done")
    return out
""")

# =====================================================================
# 5. Step 4 — full run for one pair (heatmap + correspondence plot)
# =====================================================================
md("""## 5. Step 4 — Full Run for One Model Pair

Produces the two-panel figure:

- **Left:** MKNN alignment heatmap (σ above chance) with the anchored-DTW path
  overlaid in red, plus the pixel-PCA "Layer 0" strip at the bottom as a sanity
  strip.
- **Right:** the same correspondence redrawn on a 0–1 relative-depth scale for
  both models, gray diagonal = "both models pace identically," pink band =
  bootstrap uncertainty, green dashed = pixel-PCA-as-real-node control.

We also report the **area between φ and the diagonal** (Step 4's deviation
quantification) and the anchored-vs-open-ended agreement gap (Step 2 caveat).
""")

code("""# --- 5. Full pairwise run: heatmap + correspondence + deviation -------------

def run_pair(name_A: str, name_B: str, tag: str = "trained",
             k: int = K_MKNN, boot_sizes=BOOT_KS, n_boot=N_BOOT_REPEAT,
             save_dir: Path = Path("."), show: bool = True):
    \"\"\"Full Step-4 pipeline for one (model_A, model_B) pair at a given tag
    (trained or untrained). Returns a result dict consumed by later steps.
    \"\"\"
    layers_A, names_A = ALL_DATA[(name_A, tag)], LAYER_ORDER[(name_A, tag)]
    layers_B, names_B = ALL_DATA[(name_B, tag)], LAYER_ORDER[(name_B, tag)]

    # keep pixel_pca out of the primary matrix (added back as an overlay control)
    core_A = [n for n in names_A if n != "pixel_pca"]
    core_B = [n for n in names_B if n != "pixel_pca"]

    M, chance_raw = alignment_matrix(layers_A, core_A, layers_B, core_B, k=k)
    L_A, L_B = len(core_A), len(core_B)

    phi_raw, rho, p_rho = raw_correspondence(M)

    path_anchored = dtw_path_anchored(1.0 - M)
    path_open     = dtw_path_open_ended(1.0 - M)
    phi_anchored  = path_to_relative(path_anchored, L_A, L_B)
    phi_open      = path_to_relative(path_open, L_A, L_B)
    open_gap      = path_open_agreement(path_anchored, path_open, L_A)

    # pixel-PCA as a real node inserted at position 0 in both A and B stacks
    names_A_pca = ["pixel_pca"] + core_A
    names_B_pca = ["pixel_pca"] + core_B
    M_pca, _ = alignment_matrix(layers_A, names_A_pca, layers_B, names_B_pca,
                                 k=k, verbose=False)
    path_pca = dtw_path_anchored(1.0 - M_pca)
    phi_pca  = path_to_relative(path_pca, L_A + 1, L_B + 1)

    # bootstrap uncertainty (Step 3)
    boot = bootstrap_correspondence(layers_A, core_A, layers_B, core_B,
                                     subsample_sizes=boot_sizes, n_repeat=n_boot, k=k)
    # use the largest available subsample band for the main plot's shaded region
    biggest = max(boot)
    band = boot[biggest]

    rel_i = np.linspace(0, 1, L_A)
    deviation_area = float(np.trapz(np.abs(phi_anchored - rel_i), rel_i))

    # --- Plot -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    im = ax1.imshow(M, origin="lower", aspect="auto", cmap="viridis")
    pi, pj = zip(*path_anchored)
    ax1.plot(pj, pi, "r-", lw=2, label="DTW (anchored)")
    poi, poj = zip(*path_open)
    ax1.plot(poj, poi, color="white", ls=":", lw=1.5, label="DTW (open-ended)")
    plt.colorbar(im, ax=ax1, label="MKNN score (σ above chance)")
    ax1.set_xlabel(f"{name_B} layer"); ax1.set_ylabel(f"{name_A} layer")
    ax1.set_title(f"{name_A} vs {name_B} [{tag}]\\nρ={rho:.2f}, p={p_rho:.3f}  "
                  f"(chance raw≈{chance_raw:.3f})")
    ax1.legend(loc="lower right", fontsize=8)

    ax2.fill_between(rel_i, np.percentile(band, 5, axis=0), np.percentile(band, 95, axis=0),
                      alpha=0.3, color="steelblue", label=f"90% CI (n={biggest})")
    ax2.plot(rel_i, phi_anchored, color="steelblue", lw=2, label="anchored DTW")
    ax2.plot(rel_i, phi_open, color="steelblue", lw=1.2, ls=":", label="open-ended DTW")
    rel_i_pca = np.linspace(0, 1, L_A + 1)
    ax2.plot(rel_i_pca, phi_pca, color="green", lw=1.5, ls="--", label="+pixel-PCA node")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="diagonal")
    ax2.set_xlabel(f"Relative depth ({name_A})"); ax2.set_ylabel(f"Relative depth ({name_B})")
    ax2.set_title(f"Correspondence φ — deviation area={deviation_area:.3f}")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = save_dir / f"dr_{name_A}_vs_{name_B}_{tag}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(
        M=M, phi_raw=phi_raw, phi=phi_anchored, phi_open=phi_open, phi_pca=phi_pca,
        rho=rho, p_rho=p_rho, deviation_area=deviation_area,
        open_anchor_gap=open_gap, chance_raw=chance_raw,
        boot=boot, L_A=L_A, L_B=L_B, tag=tag,
    )
""")

# =====================================================================
# 6. Step 5 — universality
# =====================================================================
md("""## 6. Step 5 — Universality Across All Model Pairs

Two complementary tests:

- **Overlay** — plot every pairwise φ curve on the same relative-depth axes. If
  they share a common shape, the stage sequence looks universal.
- **Composition** — for every model triple (A, B, C), check whether
  φ_AC ≈ φ_BC ∘ φ_AB. A small average composition error indicates a shared
  universal depth coordinate across all models simultaneously (not just
  pairwise-consistent).
""")

code("""# --- 6. Universality: run every pair, overlay, composition error ------------

def run_all_pairs(models: list[str], tag: str = "trained", k: int = K_MKNN,
                   boot_sizes=BOOT_KS, n_boot=N_BOOT_REPEAT,
                   save_dir: Path = Path("."), show: bool = False):
    \"\"\"Runs run_pair for every ordered pair of distinct models. Returns
    {(A,B): result_dict}. show=False by default to avoid flooding the notebook
    with N*(N-1) figures — set True if you want every heatmap inline.
    \"\"\"
    available = [m for m in models if (m, tag) in ALL_DATA]
    if len(available) < len(models):
        print(f"[warn] only {len(available)}/{len(models)} models available for tag={tag}")
    results = {}
    pairs = list(itertools.permutations(available, 2))
    for a, b in pairs:
        print(f"[{tag}] {a} -> {b}")
        results[(a, b)] = run_pair(a, b, tag=tag, k=k, boot_sizes=boot_sizes,
                                    n_boot=n_boot, save_dir=save_dir, show=show)
    return results


def overlay_plot(all_results: dict, tag: str, save_path: Path = Path("overlay.pdf")):
    fig, ax = plt.subplots(figsize=(8, 6))
    for (a, b), r in all_results.items():
        rel_i = np.linspace(0, 1, len(r["phi"]))
        ax.plot(rel_i, r["phi"], alpha=0.35, label=f"{a}→{b}")
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="diagonal")
    ax.set_xlabel("Relative depth (A)"); ax.set_ylabel("Relative depth (B)")
    ax.set_title(f"Correspondence functions — all pairs [{tag}]")
    if len(all_results) <= 12:
        ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def compose_error(phi_AB: np.ndarray, phi_BC: np.ndarray, phi_AC: np.ndarray,
                   n: int = 100) -> float:
    \"\"\"|| phi_AC - phi_BC(phi_AB) || averaged over a common relative-depth grid.\"\"\"
    t = np.linspace(0, 1, n)
    f = lambda phi: np.interp(t, np.linspace(0, 1, len(phi)), phi)
    fab, fbc, fac = f(phi_AB), f(phi_BC), f(phi_AC)
    return float(np.mean(np.abs(np.interp(fab, t, fbc) - fac)))


def composition_table(all_results: dict, models: list[str]) -> pd.DataFrame:
    rows = []
    for A, B, C in itertools.permutations(models, 3):
        if all(k in all_results for k in [(A, B), (B, C), (A, C)]):
            err = compose_error(all_results[(A, B)]["phi"],
                                 all_results[(B, C)]["phi"],
                                 all_results[(A, C)]["phi"])
            rows.append({"A": A, "B": B, "C": C, "composition_error": err})
    df = pd.DataFrame(rows).sort_values("composition_error")
    return df
""")

code("""# --- Run universality for the TRAINED models --------------------------------
# NOTE: this is O(n_models^2) alignment matrices, each O(L^2) MKNN calls with
# permutation nulls -- can be slow for the full 12-model grid. Start with a
# small subset (e.g. the 021M scale across all 4 families) and widen once the
# pipeline is validated.

DEMO_MODELS = [f"{fam}_021M" for fam in FAMILIES]   # 4 models, 12 ordered pairs
SAVE_DIR = Path("./dr_figures"); SAVE_DIR.mkdir(exist_ok=True)

results_trained = run_all_pairs(DEMO_MODELS, tag="trained", save_dir=SAVE_DIR, show=False)
overlay_plot(results_trained, tag="trained", save_path=SAVE_DIR / "overlay_trained.pdf")

comp_df_trained = composition_table(results_trained, DEMO_MODELS)
print(f"\\nComposition error (trained): mean={comp_df_trained.composition_error.mean():.4f} "
      f"± {comp_df_trained.composition_error.std():.4f}")
comp_df_trained.head(10)
""")

# =====================================================================
# 7. Step 6 — controls (untrained + pixel-PCA)
# =====================================================================
md("""## 7. Step 6 — Controls

- **Untrained (random-init) control.** Run the identical pipeline on the
  `untrained` tag (step-0 checkpoints, or seeded scratch-init fallback per
  `extract_layerwise_colm.py`'s `fetch_untrained`). Any correspondence that
  survives here reflects architecture/input-statistics effects, not learning.
- **Pixel-PCA Layer 0.** Already threaded through `run_pair` above as the green
  dashed "+pixel-PCA node" curve — a raw-pixel PCA control inserted as a real
  graph node, serving as the sanity check that the recovered ridge isn't an
  artifact of endpoint-anchoring.
""")

code("""# --- 7. Untrained control: same pairs, tag='untrained' -----------------------

results_untrained = run_all_pairs(DEMO_MODELS, tag="untrained", save_dir=SAVE_DIR, show=False)
overlay_plot(results_untrained, tag="untrained", save_path=SAVE_DIR / "overlay_untrained.pdf")

comp_df_untrained = composition_table(results_untrained, DEMO_MODELS)
print(f"\\nComposition error (untrained): mean={comp_df_untrained.composition_error.mean():.4f} "
      f"± {comp_df_untrained.composition_error.std():.4f}")

# Side-by-side rho comparison: trained vs untrained, per pair
rho_compare = pd.DataFrame([
    {"pair": f"{a}→{b}",
     "rho_trained": results_trained[(a, b)]["rho"],
     "rho_untrained": results_untrained.get((a, b), {}).get("rho", np.nan),
     "deviation_trained": results_trained[(a, b)]["deviation_area"],
     "deviation_untrained": results_untrained.get((a, b), {}).get("deviation_area", np.nan)}
    for (a, b) in results_trained
])
rho_compare.sort_values("rho_trained", ascending=False)
""")

# =====================================================================
# 8. Step 7 — deliverable / monotonicity table / summary
# =====================================================================
md("""## 8. Step 7 — Deliverable: Monotonicity Table & Summary

Final consolidated outputs:

- Monotonicity statistics (ρ, p) for every pair, trained vs untrained.
- The composition-error tables computed above.
- An auto-generated concise summary answering exactly the five questions the
  spec asks for: ridge existence, monotonicity, universality, where the
  principal depth warping occurs, and survival under the untrained control.
""")

code("""# --- 8. Monotonicity table ----------------------------------------------------

def monotonicity_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for (a, b), r in all_results.items():
        rows.append({
            "A": a, "B": b, "rho": r["rho"], "p": r["p_rho"],
            "deviation_area": r["deviation_area"],
            "open_anchor_gap": r["open_anchor_gap"],
        })
    return pd.DataFrame(rows).sort_values("rho", ascending=False)

mono_trained   = monotonicity_table(results_trained)
mono_untrained = monotonicity_table(results_untrained)

print("=== Monotonicity (trained) ===")
display(mono_trained)
print("\\n=== Monotonicity (untrained) ===")
display(mono_untrained)
""")

code("""# --- 8b. Auto-generated summary ------------------------------------------------

def summarize(mono_trained: pd.DataFrame, mono_untrained: pd.DataFrame,
              comp_trained: pd.DataFrame, comp_untrained: pd.DataFrame,
              rho_ridge_threshold: float = 0.7,
              p_threshold: float = 0.05,
              composition_threshold: float = 0.1) -> str:
    mean_rho_t = mono_trained["rho"].mean()
    mean_rho_u = mono_untrained["rho"].mean()
    frac_sig_t = (mono_trained["p"] < p_threshold).mean()
    ridge_exists = mean_rho_t > rho_ridge_threshold and frac_sig_t > 0.5
    monotone = ridge_exists  # monotonicity is essentially what rho captures here
    mean_comp_t = comp_trained["composition_error"].mean()
    universal = mean_comp_t < composition_threshold
    worst_pair = mono_trained.iloc[0] if ridge_exists else None
    biggest_warp = mono_trained.reindex(
        mono_trained["deviation_area"].abs().sort_values(ascending=False).index
    ).iloc[0]
    survives_control = mean_rho_t > (mean_rho_u + 0.3)   # trained clearly beats untrained

    lines = [
        "SUMMARY",
        "=======",
        f"1. Correspondence ridge exists: {'YES' if ridge_exists else 'NO'} "
        f"(mean ρ_trained={mean_rho_t:.2f}, {frac_sig_t*100:.0f}% of pairs significant at p<{p_threshold}).",
        f"2. Monotone: {'YES' if monotone else 'NO'} — same conclusion as (1), since ρ directly "
        f"measures rank-monotonicity of the raw correspondence curve.",
        f"3. Universal across model pairs: {'YES' if universal else 'NO'} "
        f"(mean composition error={mean_comp_t:.3f}, threshold={composition_threshold}).",
        f"4. Principal depth warping occurs for the pair {biggest_warp['A']} → {biggest_warp['B']} "
        f"(deviation area={biggest_warp['deviation_area']:.3f}); positive area means B reaches "
        f"each representational stage at a relatively deeper layer than A.",
        f"5. Survives untrained control: {'YES' if survives_control else 'NO'} "
        f"(mean ρ_trained={mean_rho_t:.2f} vs mean ρ_untrained={mean_rho_u:.2f}) — "
        f"{'the ridge is learned structure, not an architecture/input-statistics artifact.' if survives_control else 'the ridge may partly reflect architecture rather than learning — investigate further.'}",
    ]
    return "\\n".join(lines)

print(summarize(mono_trained, mono_untrained, comp_df_trained, comp_df_untrained))
""")

md("""---
### Next steps / widening the analysis

- Swap `DEMO_MODELS` (currently the 4 families at the 021M scale) for the full
  12-model grid (`MODELS`) once the pipeline is validated — expect runtime to
  scale roughly with the square of the model count times layers².
- Sweep `BOOT_KS` finer if the uncertainty bands look unstable at small
  subsample sizes.
- If a pair shows a large `open_anchor_gap`, treat its `deviation_area` /
  pacing conclusion as directional rather than precise (per Step 2 caveat).
""")

with open("C:\\Users\\dunli\\layerwise-analysis\\metrics\\depth_reindexing.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written: {len(nb['cells'])} cells")