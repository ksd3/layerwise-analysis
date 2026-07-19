#!/usr/bin/env python
"""Extract layer-wise embeddings from the colm-results tokenizer-grid checkpoints.

Depth-reindexing pilot: 3 trained 100M models (ar_affine, ar_aim, mae_affine)
plus their untrained (step-0 / random-init) controls, on N seeded-random
DESI-LS DR8 256x256 grz galaxy postage stamps.

Output: one parquet per (model, trained|untrained) at
    <out-dir>/<model>_<tag>_blocks_layerwise.parquet
schema: object_id (int64) + one list<float> column per layer
    encoder      tokenizer output          (present if hookable)
    h.00..h.NN   transformer block outputs (token-mean pooled)
    ln_f         final layernorm           (present if hookable)
    pixel_pca    raw-pixel PCA 'Layer 0'   (identical across models by design)

Pooling convention: mean over token dim — matches astropt's own
generate_embeddings pooling and pu extract_layerwise _generic_pool for astropt.

Usage:
    python extract_layerwise_colm.py --out-dir ./reindex_embeddings \
        [--models ar_affine_100M ar_aim_100M mae_affine_100M my_new_model] \
        [--n 100] [--seed 42] [--no-untrained] \
        [--dataset Smith42/galaxies] [--revision v2.0] \
        [--local-dir ~/hf_data]

    Untrained (random-init) controls are extracted by default alongside
    every trained model; pass --no-untrained to skip them.

    Default checkpoint resolution is always repo discovery (latest step_*.pt
    under checkpoints/<name>/) for every model in --models. Pass --ckpt
    NAME=path to pin an explicit checkpoint instead (repo-relative path, or
    an existing local file to skip download entirely). The repo's file
    listing is fetched once and cached, regardless of how many models or
    resolutions (latest/untrained) are requested.
"""
from __future__ import annotations

import argparse
import json
import dataclasses
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torchvision import transforms
from sklearn.decomposition import PCA

# Hugging Face ecosystem imports
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

# ---------------------------------------------------------------------------
# Fork import with a loud, actionable failure
# ---------------------------------------------------------------------------
try:
    from astropt.local_datasets import GalaxyImageDataset
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
except ImportError as e:
    sys.exit(f"[fatal] cannot import astropt fork: {e}\n"
             "Install the scaling-study astropt fork (repo with config/pythia-like/) "
             "via `pip install -e .` — PyPI astropt will NOT load these checkpoints.")

@contextmanager
def timed(label: str, timings: list[dict] | None = None):
    """Print elapsed wall time for a block; optionally append to a running
    log (list of {label, seconds} dicts) for a summary table at the end."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[time] {label}: {dt:.1f}s")
    if timings is not None:
        timings.append({"label": label, "seconds": dt})


REPO = "HCVYM5w6Gn/colm-results"
DEFAULT_LOCAL_DIR = "~/hf_data"
# DEFAULT_MODELS = ["ar_affine_100M", "ar_aim_100M", "mae_affine_100M"]
DEFAULT_MODELS = [
    "ar_affine_001M",
    "ar_affine_021M",
    "ar_affine_100M",
    "ar_aim_001M",
    "ar_aim_021M",
    "ar_aim_100M",
    "mae_affine_001M",
    "mae_affine_021M",
    "mae_affine_100M",
    "mae_aim_001M",
    "mae_aim_021M",
    "mae_aim_100M",
]
# DEFAULT_MODELS = [
#     # "ar_affine_001M",
#     "ar_affine_021M",
#     "ar_aim_021M",
#     "mae_affine_021M",
#     "mae_aim_021M",
#     # "ar_affine_100M",
#     # "ar_aim_001M",
#     # "ar_aim_100M",
#     # "mae_affine_001M",
#     # "mae_affine_100M",
#     # "mae_aim_001M",
#     # "mae_aim_100M",
# ]

# Cache the full repo file listing so a 12-model (or larger) --models run
# fetches it ONCE, not once per (model, which) resolution — list_repo_files
# returns every file in the whole repo regardless of which model you ask
# about, so re-fetching per model was pure waste.
_REPO_FILES_CACHE: dict[str, list[str]] = {}
_REPO_SIZES_CACHE: dict[str, dict[str, int]] = {}  # repo -> {path: size_bytes}


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _repo_files(repo: str) -> list[str]:
    if repo not in _REPO_FILES_CACHE:
        print(f"[list] fetching file listing for {repo} (once, cached)")
        api = HfApi()
        # list_repo_tree gives us sizes; fall back to list_repo_files if unavailable
        try:
            tree = list(api.list_repo_tree(repo, repo_type="dataset", recursive=True))
            sizes: dict[str, int] = {}
            paths: list[str] = []
            for item in tree:
                p = getattr(item, "path", None)
                s = getattr(item, "size", None)
                t = getattr(item, "type", None)
                if p and t != "directory":
                    paths.append(p)
                    if s is not None:
                        sizes[p] = s
            _REPO_FILES_CACHE[repo] = paths
            _REPO_SIZES_CACHE[repo] = sizes
        except Exception:
            _REPO_FILES_CACHE[repo] = list(api.list_repo_files(repo, repo_type="dataset"))
            _REPO_SIZES_CACHE[repo] = {}
    return _REPO_FILES_CACHE[repo]


def _ckpt_size(repo: str, path: str) -> str:
    """Return human-readable size for a repo-relative checkpoint path, or '?' if unknown."""
    sizes = _REPO_SIZES_CACHE.get(repo, {})
    b = sizes.get(path)
    return _fmt_bytes(b) if b is not None else "?"


def _dataset_size_str(dataset: str, revision: str) -> str:
    """Best-effort total size of a HF dataset (sum of all data files)."""
    try:
        info = HfApi().dataset_info(dataset, revision=revision)
        total = 0
        found = False
        # card_data siblings both have size_in_bytes
        siblings = getattr(info, "siblings", None) or []
        for s in siblings:
            sz = getattr(s, "size", None)
            if sz:
                total += sz
                found = True
        if found and total > 0:
            return _fmt_bytes(total)
        # fallback: cardData.dataset_info.dataset_size
        card = getattr(info, "card_data", None) or {}
        ds_size = (card.get("dataset_info") or {}).get("dataset_size")
        if ds_size:
            return _fmt_bytes(int(ds_size))
    except Exception:
        pass
    return "?"


def print_download_plan(args, overrides: dict[str, str]) -> None:
    """Print a full manifest of what will be downloaded before touching the network."""
    # ensure repo file listing + sizes are populated (fetched once here)
    _repo_files(REPO)

    print()
    print("=" * 65)
    print("DOWNLOAD PLAN  (sizes from HF metadata, before any download)")
    print("=" * 65)

    # Dataset
    ds_size = _dataset_size_str(args.dataset, args.revision)
    print(f"\n  Dataset  : {args.dataset}@{args.revision}")
    print(f"  Total    : {ds_size}  (streamed — only first {args.pool_size} rows read)")

    # Checkpoints
    print(f"\n  Checkpoints from {REPO}:")
    total_ckpt_bytes = 0
    already_local = 0
    local_dir = str(Path(args.local_dir).expanduser())
    sizes = _REPO_SIZES_CACHE.get(REPO, {})

    def _local_path(rel: str) -> Path:
        return Path(local_dir) / rel

    rows_to_print: list[tuple[str, str, str, str]] = []  # model, tag, rel_path, size_str
    for name in args.models:
        for which in (["latest", "untrained"] if args.untrained else ["latest"]):
            try:
                if which == "latest" and overrides and name in overrides:
                    val = overrides[name]
                    loc = Path(val).expanduser()
                    if loc.exists():
                        rows_to_print.append((name, which, str(loc), "local"))
                        already_local += 1
                        continue
                    rel = val
                else:
                    steps = _list_step_files(REPO, name)
                    if not steps:
                        rows_to_print.append((name, which, "NOT FOUND", "?"))
                        continue
                    _, rel = steps[0] if which == "untrained" else steps[-1]

                sz_bytes = sizes.get(rel)
                sz_str = _fmt_bytes(sz_bytes) if sz_bytes else "?"
                loc = _local_path(rel)
                if loc.exists():
                    sz_str += "  [cached]"
                    already_local += 1
                else:
                    if sz_bytes:
                        total_ckpt_bytes += sz_bytes
                rows_to_print.append((name, which, rel, sz_str))
            except Exception as e:
                rows_to_print.append((name, which, f"ERROR: {e}", "?"))

    col_w = max((len(r[0]) for r in rows_to_print), default=10) + 2
    for model, which, rel, sz in rows_to_print:
        tag = f"[{which}]"
        print(f"    {model:<{col_w}} {tag:<12} {sz:<18}  {rel}")

    if total_ckpt_bytes:
        print(f"\n  Net checkpoint download : {_fmt_bytes(total_ckpt_bytes)}"
              f"  ({already_local} already cached)")
    else:
        print(f"\n  All checkpoints already cached locally.")

    print("=" * 65)
    print()


def _list_step_files(repo: str, model_name: str) -> list[tuple[int, str]]:
    """List (step_int, repo_path) for every step_*.pt under checkpoints/<model_name>/
    in the dataset repo, sorted ascending by step. Filters the cached
    whole-repo listing rather than re-fetching it."""
    prefix = f"checkpoints/{model_name}/"
    files = _repo_files(repo)
    out = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".pt"):
            stem = f[len(prefix):-len(".pt")]
            if stem.startswith("step_"):
                try:
                    out.append((int(stem[len("step_"):]), f))
                except ValueError:
                    continue
    out.sort(key=lambda t: t[0])
    return out


def discover_checkpoint(repo: str, model_name: str, which: str = "latest") -> str:
    """Resolve a checkpoint repo-path for model_name by listing the HF repo.
    This is the DEFAULT resolution for every model (which='latest' = max
    step; 'untrained' = min step, expected to be 0) — add a checkpoint under
    checkpoints/<name>/ in the repo and it's picked up automatically, no code
    change, and discovery always finds the true latest step rather than a
    possibly-stale pinned filename."""
    steps = _list_step_files(repo, model_name)
    if not steps:
        sys.exit(f"[fatal] no checkpoints found under checkpoints/{model_name}/ "
                 f"in {repo} — check the model name / that it's been uploaded")
    step, path = steps[0] if which == "untrained" else steps[-1]
    if which == "untrained" and step != 0:
        print(f"[warn] {model_name}: smallest available step is {step}, not 0 — "
              f"'untrained' control is only approximately random-init")
    print(f"[discover] {model_name} ({which}): step_{step:08d}.pt "
          f"({len(steps)} checkpoints available)")
    return path


def resolve_checkpoint(repo: str, model_name: str, which: str = "latest",
                       overrides: dict[str, str] | None = None) -> str | Path:
    """Default: auto-discover the latest (or, for 'untrained', the smallest)
    step from the repo listing. If the user passed an explicit checkpoint via
    --ckpt NAME=path for this model (and which=='latest'), that overrides
    discovery: either a repo-relative path (still downloaded through
    hf_hub_download) or an existing local file (used directly, no download).
    Returns a Path for a local override, else a repo-relative string."""
    if which == "latest" and overrides and model_name in overrides:
        val = overrides[model_name]
        local = Path(val).expanduser()
        if local.exists():
            print(f"[override] {model_name}: using local file {local}")
            return local
        print(f"[override] {model_name}: using repo path {val}")
        return val
    return discover_checkpoint(repo, model_name, which)


def parse_ckpt_overrides(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated --ckpt NAME=path arguments into a dict."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"[fatal] --ckpt expects NAME=path, got: {p!r}")
        name, path = p.split("=", 1)
        out[name] = path
    return out

def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("./reindex_embeddings"))
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   help="Model names under checkpoints/<name>/ in the HF repo. "
                        "Default resolution is always repo discovery (latest "
                        "step_*.pt) — pass --ckpt to pin a specific file instead.")
    p.add_argument("--ckpt", action="append", metavar="NAME=path",
                   help="Override the checkpoint used for a specific model "
                        "instead of auto-discovering the latest step. path is "
                        "either a repo-relative path (e.g. "
                        "checkpoints/ar_affine_100M/step_00026483.pt, still "
                        "downloaded via hf_hub_download) or an existing local "
                        "file (used directly, no download). Repeatable, e.g. "
                        "--ckpt ar_affine_100M=checkpoints/ar_affine_100M/"
                        "step_00013000.pt --ckpt ar_aim_100M=/my/local/ckpt.pt")
    p.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR,
                   help="Local cache dir for downloaded checkpoints "
                        f"(default: {DEFAULT_LOCAL_DIR})")
    p.add_argument("--n", type=int, default=500,
                   help="number of galaxies (seeded-random)")
    p.add_argument("--pool-size", type=int, default=5000,
                   help="deterministic stream prefix to sample --n from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="Smith42/galaxies",
                   help="HF dataset of DESI-LS DR8 256x256 grz stamps (RGB PNG via PIL).")
    p.add_argument("--revision", default="v2.0",
                   help="Dataset revision. v2.0 ships dr8_id metadata directly "
                        "per row (also available as a root-dir parquet, "
                        "joinable on dr8_id, if a separate metadata pull is "
                        "ever needed).")
    p.add_argument("--untrained", dest="untrained", action="store_true",
                   default=True,
                   help="extract the untrained control per model "
                        "(step-0 checkpoint if on HF, else seeded scratch init). "
                        "ON by default; pass --no-untrained to skip it.")
    p.add_argument("--no-untrained", dest="untrained", action="store_false",
                   help="skip the untrained control (extract trained models only)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pca-dim", type=int, default=64,
                   help="pixel-PCA dimensionality for the Layer-0 control")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors scripts/train.py resume + model_utils.load_astropt)
# ---------------------------------------------------------------------------
def build_modality_registry(cfg: dict):
    """Reconstruct the 'images' ModalityRegistry exactly as scripts/train.py
    does at train time — it is NOT saved in these checkpoints (only
    'model', 'model_args', 'iter_num', 'config' are present).

    train.py:
        modalities = [ModalityConfig(name="images",
            input_size=patch_size*patch_size*n_chan, patch_size=patch_size,
            loss_weight=1.0, embed_pos=True, pos_input_size=1)]
        modality_registry = ModalityRegistry(modalities)
    """
    from astropt.model import ModalityConfig, ModalityRegistry
    patch_size = cfg["patch_size"]
    n_chan = cfg["n_chan"]
    mc = ModalityConfig(
        name="images",
        input_size=patch_size * patch_size * n_chan,
        patch_size=patch_size,
        loss_weight=1.0,
        embed_pos=True,
        pos_input_size=1,
    )
    return ModalityRegistry([mc])


def load_ckpt_model(path: str, device: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    cfg = ckpt.get("config")
    if cfg is None:
        raise KeyError(
            f"{path}: checkpoint has no 'config' key and no 'modality_registry' "
            f"key — cannot recover patch_size/n_chan to rebuild the modality "
            f"registry. Keys present: {list(ckpt)}"
        )
    registry = build_modality_registry(cfg)
    # drop any model_args keys this fork's GPTConfig doesn't know (fwd compat)
    known = {f.name for f in dataclasses.fields(GPTConfig)}
    unknown = set(model_args) - known
    if unknown:
        print(f"[warn] dropping unknown model_args keys: {sorted(unknown)} "
              f"(GPTConfig fields: install matching fork version if wrong)")
    gcfg = GPTConfig(**{k: v for k, v in model_args.items() if k in known},
                     modalities=list(registry.modalities.values()))
    model = GPT(gcfg, registry)
    sd = ckpt["model"]
    pref = "_orig_mod."
    for k in list(sd):
        if k.startswith(pref):
            sd[k[len(pref):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"[load] {path}: n_layer={gcfg.n_layer} tokeniser={gcfg.tokeniser} "
          f"patch_size={cfg['patch_size']} n_chan={cfg['n_chan']} "
          f"iter={ckpt.get('iter_num', '?')}")
    return model, model_args


def scratch_init_model(model_args: dict, cfg: dict, seed: int, device: str) -> torch.nn.Module:
    """Untrained control fallback: seeded random init, same architecture."""
    torch.manual_seed(seed)
    registry = build_modality_registry(cfg)
    known = {f.name for f in dataclasses.fields(GPTConfig)}
    gcfg = GPTConfig(**{k: v for k, v in model_args.items() if k in known},
                     modalities=list(registry.modalities.values()))
    model = GPT(gcfg, registry).to(device).eval()
    print(f"[init] scratch untrained model (seed={seed})")
    return model


def fetch_untrained(repo: str, model_name: str, trained_ckpt_path: str,
                    seed: int, device: str, local_dir: str) -> torch.nn.Module:
    """Prefer the smallest-step ('untrained', ideally step 0) checkpoint on HF.
    Fall back to seeded scratch init with the trained model's model_args/config
    — architecture identical, weights random. Works for any model name via
    discover_checkpoint, not just the pinned ones."""
    try:
        rel = discover_checkpoint(repo, model_name, which="untrained")
        p = hf_hub_download(repo, rel, repo_type="dataset", local_dir=local_dir)
        model, _ = load_ckpt_model(p, device)
        print(f"[untrained] {model_name}: using {rel} from HF")
        return model
    except Exception as e:
        print(f"[untrained] {model_name}: step-0 not available ({type(e).__name__}); "
              f"falling back to scratch init")
        ckpt = torch.load(trained_ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        if cfg is None:
            raise KeyError(f"{trained_ckpt_path}: no 'config' key, cannot build "
                           f"scratch-init fallback (need patch_size/n_chan)")
        return scratch_init_model(ckpt["model_args"], cfg, seed, device)


# ---------------------------------------------------------------------------
# Data: seeded-random N from a deterministic stream prefix
# ---------------------------------------------------------------------------
def load_galaxies(dataset: str, n: int, pool: int, seed: int, revision: str = "v2.0"):
    """Deterministic selection: take the first `pool` rows of the (unshuffled)
    stream, then seeded-choice `n` of them.

    object_id = dr8_id from the dataset's own metadata (Smith42/galaxies v2.0
    ships this directly per-row; also available as a root-dir parquet joinable
    on dr8_id if a separate metadata pull is ever needed). This replaces the
    earlier stream-prefix-position fallback — a REAL stable key, not a
    positional stand-in, which is what the row-alignment check actually wants.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, revision=revision, split="train", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if i >= pool:
            break
        rows.append(r)
    if len(rows) < n:
        sys.exit(f"[fatal] stream prefix has {len(rows)} < n={n} rows")
    if "dr8_id" not in rows[0]:
        sys.exit(f"[fatal] 'dr8_id' not found in {dataset}@{revision} row keys "
                 f"({list(rows[0].keys())}) — check the revision pin")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(rows), size=n, replace=False))
    picked = [rows[int(i)] for i in sel]
    object_ids = [r["dr8_id"] for r in picked]
    if "image" not in rows[0]:
        img_like = [k for k in rows[0] if "image" in k.lower() or "flux" in k.lower()]
        sys.exit(f"[fatal] 'image' key not found in row (keys={list(rows[0].keys())}); "
                 f"candidates: {img_like} — update the 'image' field name used "
                 f"in patchify_batch/pixel_pca to match v2.0's actual schema")
    print(f"[data] {dataset}@{revision}: selected {n}/{pool} rows, seed={seed}, "
          f"object_id='dr8_id'")
    return picked, object_ids


def _normalise(x: torch.Tensor) -> torch.Tensor:
    """Exact match to scripts/train.py's normalise(x, use_hf=True) transform,
    applied per-token (dim=1) after patchify. Input arrives as a numpy array
    from einops.rearrange inside process_galaxy; converted to tensor here."""
    if not torch.is_tensor(x):
        x = torch.from_numpy(x).to(torch.float32)
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def patchify_batch(rows, registry, device):
    """Exact replication of scripts/train.py's process_galaxy_wrapper:

        patch_galaxy = func(np.array(galdict["image"]).swapaxes(0, 2))

    where func = GalaxyImageDataset.process_galaxy, which internally
    einops-rearranges into patches, applies the normalise transform, and
    spiralises. images are PIL RGB (H, W, C) uint8 (confirmed: 256x256x3);
    swapaxes(0,2) -> (C, W, H), which is dimensionally (C, H, W) since the
    images are square — matches the repo's own convention exactly, not a
    guess."""
    galproc = GalaxyImageDataset(
        None, spiral=True,
        transform={"images": transforms.Compose([transforms.Lambda(_normalise)])},
        modality_registry=registry,
    )
    ims, poss = [], []
    for r in rows:
        arr = np.array(r["image"]).swapaxes(0, 2)   # (H,W,C) uint8 -> (C,W,H)
        patched = galproc.process_galaxy(arr)         # patchify + normalise + spiralise
        if not torch.is_tensor(patched):
            patched = torch.as_tensor(patched)
        ims.append(patched.to(torch.float))
        poss.append(torch.arange(0, len(patched), dtype=torch.long))
    return torch.stack(ims).to(device), torch.stack(poss).to(device)


# ---------------------------------------------------------------------------
# Layer-wise hooks: all transformer blocks + encoder + ln_f, token-mean pooled
# ---------------------------------------------------------------------------
def resolve_hook_targets(model) -> dict[str, torch.nn.Module]:
    targets = {}
    enc = getattr(model, "encoders", None)                 # fork attribute
    if enc is not None and "images" in enc:
        targets["encoder"] = enc["images"]
    elif hasattr(model.transformer, "wte"):                # older lineage
        targets["encoder"] = model.transformer.wte
    for i, blk in enumerate(model.transformer.h):
        targets[f"h.{i:02d}"] = blk
    if hasattr(model.transformer, "ln_f"):
        targets["ln_f"] = model.transformer.ln_f
    return targets


def extract_layers(model, images, positions, n_layer: int) -> dict[str, np.ndarray]:
    """Hook + forward. Guards against the fork's middle-layer early-break
    forward path: if generate_embeddings stops early (captured blocks <
    n_layer), rerun with an explicit full-stack forward."""
    targets = resolve_hook_targets(model)
    captured: dict[str, torch.Tensor] = {}

    def mk_hook(name):
        def h(mod, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t) and t.dim() >= 2:
                captured[name] = (t.mean(dim=1) if t.dim() == 3 else t).float().detach().cpu()
        return h

    hooks = [m.register_forward_hook(mk_hook(n)) for n, m in targets.items()]
    inputs = {"images": images, "images_positions": positions}
    try:
        with torch.no_grad():
            model.generate_embeddings(inputs)
        got_blocks = sum(k.startswith("h.") for k in captured)
        if got_blocks < n_layer:
            print(f"[warn] generate_embeddings captured only {got_blocks}/{n_layer} "
                  f"blocks (early-break forward path?) — running explicit full forward")
            captured.clear()
            with torch.no_grad():
                _explicit_full_forward(model, inputs)
            got_blocks = sum(k.startswith("h.") for k in captured)
        assert got_blocks == n_layer, \
            f"captured {got_blocks} blocks, expected {n_layer} — hooking broken, aborting"
    finally:
        for h in hooks:
            h.remove()
    return {k: v.numpy() for k, v in captured.items()}


def _explicit_full_forward(model, inputs):
    """Manual full-depth forward mirroring the fork's generate_embeddings,
    minus any early break. Hooks fire as modules execute."""
    embs, pos_embs = [], []
    for mod_name in model.modality_registry.names():
        embs.append(model.encoders[mod_name](inputs[mod_name]))
        pos_embs.append(model.embedders[mod_name](inputs[mod_name + "_positions"]))
    x = model.transformer.drop(torch.cat(embs, 1) + torch.cat(pos_embs, 1))
    for block in model.transformer.h:
        x = block(x)
    if hasattr(model.transformer, "ln_f"):
        x = model.transformer.ln_f(x)
    return x


# ---------------------------------------------------------------------------
# Pixel-PCA Layer 0 (shared across models; display strip + robustness node)
# ---------------------------------------------------------------------------
def pixel_pca(rows, dim: int, seed: int) -> np.ndarray:
    """Raw-pixel PCA 'Layer 0'. rows[i]['image'] is a PIL RGB image
    (confirmed 256x256x3); downsample to 64x64 and flatten."""
    from sklearn.decomposition import PCA
    flat = []
    for r in rows:
        im = r["image"].convert("RGB").resize((64, 64))
        flat.append(np.asarray(im, dtype=np.float32).reshape(-1) / 255.0)
    X = np.stack(flat)
    d = min(dim, X.shape[0] - 1, X.shape[1])
    Z = PCA(n_components=d, random_state=seed).fit_transform(X)
    print(f"[pca] pixel Layer-0: {X.shape} -> {Z.shape}")
    return Z


# ---------------------------------------------------------------------------
def write_parquet(out: Path, object_ids, layers: dict[str, np.ndarray],
                  pca: np.ndarray):
    cols = {"object_id": object_ids}
    for name in sorted(layers):
        cols[name] = [row.tolist() for row in layers[name]]
    cols["pixel_pca"] = [row.tolist() for row in pca]
    pl.DataFrame(cols).write_parquet(out)
    print(f"[write] {out}  layers={sorted(layers)}")


def main():
    run_t0 = time.perf_counter()
    timings: list[dict] = []

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = str(Path(args.local_dir).expanduser())
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    overrides = parse_ckpt_overrides(args.ckpt)
    from huggingface_hub import hf_hub_download

    print_download_plan(args, overrides)

    with timed("load_galaxies", timings):
        rows, object_ids = load_galaxies(args.dataset, args.n, args.pool_size,
                                         args.seed, args.revision)
    with timed("pixel_pca", timings):
        pca = pixel_pca(rows, args.pca_dim, args.seed)

    for name in args.models:
        with timed(f"{name}: resolve+download checkpoint", timings):
            resolved = resolve_checkpoint(REPO, name, which="latest", overrides=overrides)
            ckpt_path = (str(resolved) if isinstance(resolved, Path)
                        else hf_hub_download(REPO, resolved, repo_type="dataset",
                                            local_dir=local_dir))
        variants = [("trained", lambda: load_ckpt_model(ckpt_path, args.device)[0])]
        if args.untrained:
            variants.append(("untrained",
                             lambda: fetch_untrained(REPO, name, ckpt_path, args.seed,
                                                     args.device, local_dir)))
        for tag, make in variants:
            t0 = time.perf_counter()
            with timed(f"{name} [{tag}]: load model", timings):
                model = make()
            n_layer = len(model.transformer.h)
            registry = model.modality_registry
            all_layers: dict[str, list[np.ndarray]] = {}
            with timed(f"{name} [{tag}]: forward+hook ({len(rows)} rows, "
                      f"bs={args.batch_size})", timings):
                for s in range(0, len(rows), args.batch_size):
                    batch = rows[s:s + args.batch_size]
                    ims, poss = patchify_batch(batch, registry, args.device)
                    out = extract_layers(model, ims, poss, n_layer)
                    for k, v in out.items():
                        all_layers.setdefault(k, []).append(v)
            layers = {k: np.concatenate(v) for k, v in all_layers.items()}
            for k, v in layers.items():
                assert v.shape[0] == len(rows), f"{k}: {v.shape[0]} != {len(rows)} rows"
            with timed(f"{name} [{tag}]: write_parquet", timings):
                write_parquet(args.out_dir / f"{name}_{tag}_blocks_layerwise.parquet",
                              object_ids, layers, pca)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
            timings.append({"label": f"{name} [{tag}]: TOTAL",
                           "seconds": time.perf_counter() - t0})

    total = time.perf_counter() - run_t0
    print("\n" + "=" * 60)
    print(f"TIMING SUMMARY  (total: {total/60:.1f} min)")
    print("=" * 60)
    for row in timings:
        print(f"  {row['seconds']:8.1f}s  {row['label']}")
    print("=" * 60)
    timing_path = args.out_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump({"total_seconds": total, "steps": timings}, f, indent=2)
    print(f"[write] {timing_path}")

    print("[done]")

if __name__ == "__main__":
    main()