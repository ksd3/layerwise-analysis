import gc
import inspect
import io
import os
import random
import time

import torch
from tqdm import tqdm

import colm  # repo-root sibling; reused for _get/_put/_ls and everything below
from colm import (
    CONFIGS, PROBE_TARGETS, DEVICE, RESULTS_REPO, RIDGE_LAMBDA, TRAIN_FRAC,
    PROBE_LEASE_S, CKPT_UPLOAD_S, STARTUP_JITTER_S,
    build_model, layerwise_embeddings, load_eval_set, _spiral_fn,
    is_done, mark_done, claim, release, Lease, log,
)

# ------------------------------------------------------------------ config knobs
MIN_N = 200                 # need at least this many labelled galaxies for a probe
EVAL_SPLIT = os.environ.get("COLM_REL_SPLIT", "test")   # TEST ONLY (per request)
PARTIAL_VAR = "photo_z"     # quantity partialled out for the redshift control
SAFE_UPLOAD_S = 30 * 60     # sparse periodic parquet upload (keep HF commits low under 12-way fan-out)
FAIL_BACKOFF_S = 5 * 60     # sleep after ANY failed iteration -> a failure can never spin into a 429 storm
MAX_FAILS = 8               # consecutive failures -> give up (also rides out a rate-limit window)

# the named scaling relations to track (members are columns/derived labels).
# edit freely; the all-pairs RSA below does not depend on this list.
RELATIONS = [
    ("mass_size",       "mass_med_photoz",  "est_petro_th50_kpc"),
    ("mass_quenching",  "mass_med_photoz",  "ssfr_med_photoz"),    # sSFR-mass (main sequence proxy)
    ("color_mass",      "color_gr",         "mass_med_photoz"),
    ("color_magnitude", "color_gr",         "mag_abs_r_photoz"),
    ("size_luminosity", "mag_abs_r_photoz", "est_petro_th50_kpc"),
    ("mass_luminosity", "mass_med_photoz",  "mag_abs_r_photoz"),
]


# ------------------------------------------------------------------ small math
def _corr(a, b, mask):
    a = a[mask].float(); b = b[mask].float()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float("nan") if d <= 0 else float(a @ b / d)


def _corr_full(a, b):
    a = a.float() - a.float().mean(); b = b.float() - b.float().mean()
    d = a.norm() * b.norm()
    return float("nan") if d <= 0 else float(a @ b / d)


def _cos(a, b):
    d = a.norm() * b.norm()
    return float("nan") if d <= 0 else float(a @ b / d)


def _resid(y, z, mask):
    """Residual of y after a linear fit on z, over masked rows (returns len==mask.sum())."""
    yy = y[mask].float(); zz = z[mask].float()
    zc = zz - zz.mean(); var = (zc * zc).sum()
    b = (zc * (yy - yy.mean())).sum() / var if var > 0 else torch.zeros((), device=yy.device)
    return yy - (yy.mean() + b * zc)


def _r2_scalar(x, y, mask, gen):
    """Held-out R^2 of predicting y from the single scalar x (1-D linear)."""
    idx = mask.nonzero(as_tuple=True)[0]; n = idx.numel()
    if n < MIN_N:
        return float("nan")
    perm = idx[torch.randperm(n, generator=gen, device=idx.device)]
    ntr = int(TRAIN_FRAC * n); tr, te = perm[:ntr], perm[ntr:]
    xt = x[tr].float(); yt = y[tr].float()
    xc = xt - xt.mean(); var = (xc * xc).sum()
    b = (xc * (yt - yt.mean())).sum() / var if var > 0 else torch.zeros((), device=xt.device)
    a = yt.mean() - b * xt.mean()
    pe = a + b * x[te].float(); yte = y[te].float()
    sst = ((yte - yte.mean()) ** 2).sum() + 1e-12
    return float(1 - ((yte - pe) ** 2).sum() / sst)


def _layer_probes(Xs, labels, masks, gen):
    """Ridge-probe every param at this (already-standardized) layer.
    Returns {param: {w, yhat (all-N predictions), r2 (held-out)}}. Shared feature
    standardization => the w's live in one space, so cosines across params are meaningful."""
    out = {}
    D = Xs.shape[1]
    eye = RIDGE_LAMBDA * torch.eye(D, device=Xs.device)
    for p, y in labels.items():
        m = masks[p]; n = int(m.sum())
        if n < MIN_N:
            continue
        idx = m.nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(n, generator=gen, device=idx.device)]
        ntr = int(TRAIN_FRAC * n); tr, te = perm[:ntr], perm[ntr:]
        ytr = y[tr].float(); ybar = ytr.mean()
        Xtr = Xs[tr]
        w = torch.linalg.solve(Xtr.T @ Xtr + eye, Xtr.T @ (ytr - ybar))
        yhat = Xs @ w + ybar                      # predictions for ALL galaxies
        yte = y[te].float()
        sst = ((yte - yte.mean()) ** 2).sum() + 1e-12
        r2 = float(1 - ((yhat[te] - yte) ** 2).sum() / sst)
        out[p] = {"w": w, "yhat": yhat, "r2": r2, "ybar": ybar}
    return out


# ------------------------------------------------------------------ tier 0 (no GPU)
def _emergence_rows(name):
    """Read the probe's results.parquet (TEST rows) and, per param, find the
    checkpoint where decodability first reaches half its final value."""
    import polars as pl
    p = f"results/{name}/results.parquet"
    if p not in colm._ls(p):
        return []
    df = pl.read_parquet(io.BytesIO(colm._get(p)))
    if "probe_set" in df.columns:
        df = df.filter(pl.col("probe_set") == "test")
    rows = []
    for param in df["param"].unique().to_list():
        d = df.filter(pl.col("param") == param)
        last = d["step"].max()
        bl = d.filter(pl.col("step") == last).sort("r2", descending=True)
        if not bl.height:
            continue
        layer = int(bl["layer"][0]); r2_final = float(bl["r2"][0])
        dl = d.filter(pl.col("layer") == layer).sort("step")
        emerge = None
        if r2_final > 0.05:
            thr = 0.5 * r2_final
            for s, r in zip(dl["step"].to_list(), dl["r2"].to_list()):
                if r >= thr:
                    emerge = int(s); break
        rows.append(dict(block="emergence", name=param, layer=layer,
                         r2_final=r2_final, emerge_step=emerge))
    return rows


# ------------------------------------------------------------------ tiers 1 & 2 (GPU)
def _analyze_checkpoint(name, step, emb, labels, masks, z, zmask, param_order):
    rows = []
    N, L, _ = emb.shape
    allmask = torch.ones(N, dtype=torch.bool, device=DEVICE)
    W_l, MU_l, SD_l, YB_l, R2_l = [], [], [], [], []      # per-layer direction tensors
    for layer in range(L):
        gen = torch.Generator(device=DEVICE); gen.manual_seed(1234 + step * 97 + layer)
        Xl = emb[:, layer, :].to(DEVICE).float()
        mu = Xl.mean(0); sd = Xl.std(0) + 1e-6
        Xs = (Xl - mu) / sd
        probes = _layer_probes(Xs, labels, masks, gen)

        # TIER 1 -- RSA over the native targets (predicted vs true correlation web)
        names = [p for p in PROBE_TARGETS if p in probes]
        vp, vt = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                mboth = masks[a] & masks[b]
                if int(mboth.sum()) < MIN_N:
                    continue
                vt.append(_corr(labels[a], labels[b], mboth))
                vp.append(_corr(probes[a]["yhat"], probes[b]["yhat"], allmask))
        rsa = float("nan")
        if len(vp) >= 10:
            rsa = _corr_full(torch.tensor(vp), torch.tensor(vt))
        rows.append(dict(block="rsa", step=step, layer=layer, n_pairs=len(vp), rsa=rsa))

        # TIERS 1 & 2 -- per named relation
        for relname, A, B in RELATIONS:
            if A not in probes or B not in probes:
                continue
            mboth = masks[A] & masks[B]
            true_corr = _corr(labels[A], labels[B], mboth) if int(mboth.sum()) >= MIN_N else float("nan")
            pred_corr = _corr(probes[A]["yhat"], probes[B]["yhat"], allmask)
            cosine = _cos(probes[A]["w"], probes[B]["w"])
            incremental = probes[B]["r2"] - _r2_scalar(probes[A]["yhat"], labels[B], masks[B], gen)
            mz = mboth & zmask
            tpc = ppc = float("nan")
            if int(mz.sum()) >= MIN_N:
                tpc = _corr_full(_resid(labels[A], z, mz), _resid(labels[B], z, mz))
                ppc = _corr_full(_resid(probes[A]["yhat"], z, mz), _resid(probes[B]["yhat"], z, mz))
            rows.append(dict(block="relation", step=step, layer=layer, name=relname, a=A, b=B,
                             r2_a=probes[A]["r2"], r2_b=probes[B]["r2"],
                             pred_corr=pred_corr, true_corr=true_corr, cosine=cosine,
                             incremental_b_given_a=incremental,
                             pred_partial_corr=ppc, true_partial_corr=tpc))

        # keep the ACTUAL directions for every label (fixed param order, all layers/steps)
        D = Xs.shape[1]
        Wl = torch.zeros(len(param_order), D, device=DEVICE)
        ybl = torch.zeros(len(param_order), device=DEVICE)
        r2l = torch.full((len(param_order),), float("nan"))
        for pi, p in enumerate(param_order):
            if p in probes:
                Wl[pi] = probes[p]["w"]
                ybl[pi] = probes[p]["ybar"]
                r2l[pi] = probes[p]["r2"]
        W_l.append(Wl.half().cpu()); MU_l.append(mu.half().cpu()); SD_l.append(sd.half().cpu())
        YB_l.append(ybl.cpu()); R2_l.append(r2l)

        del Xl, Xs, probes
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    step_dirs = {"w": torch.stack(W_l), "mu": torch.stack(MU_l), "sd": torch.stack(SD_l),
                 "ybar": torch.stack(YB_l), "r2": torch.stack(R2_l)}
    return rows, step_dirs


def _upload(out_path, rows, name):
    import polars as pl
    cols = sorted({k for r in rows for k in r})
    norm = [{c: r.get(c) for c in cols} for r in rows]
    buf = io.BytesIO()
    pl.DataFrame(norm).write_parquet(buf)
    colm._put(out_path, buf.getvalue(), msg=f"relations {name} ({len(rows)} rows)")


def _upload_dirs(name, cfg, param_order, step_ids, dirs_steps):
    """Persist every probe direction (label x layer x step) as one small .pt sidecar."""
    if not dirs_steps:
        return
    n_layers = dirs_steps[0]["w"].shape[0]
    payload = {
        "config": name, "n_embd": cfg["n_embd"], "params": param_order,
        "steps": step_ids, "layers": list(range(n_layers)),  # layer 0 = raw patch+pos embeddings
        "note": "per-(step,layer) standardized feature space; raw-space dir = w/sd",
        "w": torch.stack([d["w"] for d in dirs_steps]),        # (S, L, P, D) fp16
        "mu": torch.stack([d["mu"] for d in dirs_steps]),      # (S, L, D)    fp16
        "sd": torch.stack([d["sd"] for d in dirs_steps]),      # (S, L, D)    fp16
        "ybar": torch.stack([d["ybar"] for d in dirs_steps]),  # (S, L, P)
        "r2": torch.stack([d["r2"] for d in dirs_steps]),      # (S, L, P)
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    colm._put(f"relations/{name}/directions.pt", buf.getvalue(),
              msg=f"directions {name} ({len(step_ids)}x{n_layers}x{len(param_order)})")


def analyze_config(cfg, lease, eval_patches, eval_labels):
    name = cfg["name"]
    out_path = f"relations/{name}/analysis.parquet"
    steps = sorted({int(p.rsplit("step_", 1)[-1][:-3]) for p in colm._ls(f"checkpoints/{name}/step_")})
    if not steps:
        log(f"relations {name}: no checkpoints; skipping")
        mark_done("relations", name)
        return

    # labels -> GPU once (+ derived colors), with finite-masks; z for the redshift control
    labels = {k: v.to(DEVICE) for k, v in eval_labels.items()}
    labels["color_gr"] = labels["mag_abs_g_photoz"] - labels["mag_abs_r_photoz"]
    labels["color_rz"] = labels["mag_abs_r_photoz"] - labels["mag_abs_z_photoz"]
    masks = {k: torch.isfinite(v) for k, v in labels.items()}
    z, zmask = labels[PARTIAL_VAR], masks[PARTIAL_VAR]
    param_order = [p for p in labels if int(masks[p].sum()) >= MIN_N]   # every decodable label

    rows = _emergence_rows(name)        # TIER 0, free
    dirs_steps, dirs_ids = [], []       # ALL probe directions, per successful checkpoint
    log(f"relations {name}: {len(steps)} checkpoints, {len(param_order)} decodable labels, "
        f"{len(rows)} emergence rows")
    last_upload = time.time()
    for step in tqdm(steps, desc=f"relations {name}", unit="ckpt", ascii=True):
        lease.beat()
        try:
            ck = torch.load(io.BytesIO(colm._get(f"checkpoints/{name}/step_{step:08d}.pt")),
                            map_location=DEVICE, weights_only=False)
            model, _, _ = build_model(cfg)
            model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
            emb = layerwise_embeddings(model, eval_patches)   # (N, L, n_embd) cpu fp16
            del model, ck
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            r, step_dirs = _analyze_checkpoint(name, step, emb, labels, masks, z, zmask, param_order)
            rows += r
            dirs_steps.append(step_dirs); dirs_ids.append(step)
            del emb
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            log(f"relations {name}: step {step} skipped ({e})")
            continue
        if time.time() - last_upload >= SAFE_UPLOAD_S:
            _upload(out_path, rows, name)
            last_upload = time.time()

    _upload(out_path, rows, name)
    _upload_dirs(name, cfg, param_order, dirs_ids, dirs_steps)
    mark_done("relations", name)
    log(f"relations {name}: DONE ({len(rows)} rows, {len(dirs_ids)} direction sets)")


# ------------------------------------------------------------------ worker loop
def _load_eval(split, spiral_fn):
    """Call colm.load_eval_set adaptively: some colm.py versions take only `split`
    (they build the spiral internally), others take (split, spiral_fn). Pass spiral_fn
    only when it's a required positional arg, so this worker tracks whatever's on the node."""
    try:
        required = [p for p in inspect.signature(load_eval_set).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    except (TypeError, ValueError):
        required = [None]
    return load_eval_set(split, spiral_fn) if len(required) >= 2 else load_eval_set(split)


def find_rel_work():
    names = list(CONFIGS)
    random.shuffle(names)
    for name in names:
        if is_done("probe", name) and not is_done("relations", name) \
                and claim("relations", name, PROBE_LEASE_S):
            return name
    return None


def main():
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set (set it in the .sh wrapper)"
    log(f"relations worker up | device={DEVICE} | repo={RESULTS_REPO} | split={EVAL_SPLIT}")
    if STARTUP_JITTER_S:
        delay = random.uniform(0, STARTUP_JITTER_S)
        log(f"startup jitter: sleeping {delay:.0f}s")
        time.sleep(delay)
    spiral_fn, _ = _spiral_fn()
    eval_cache = None        # TEST eval set (patches, labels), loaded lazily once
    fails = 0                # consecutive failures -> back off then abort (never spin into a 429 storm)

    while True:
        try:
            name = find_rel_work()
        except Exception as e:
            fails += 1
            log(f"find_work error ({e}); backing off {FAIL_BACKOFF_S}s [{fails}/{MAX_FAILS}]")
            if fails >= MAX_FAILS:
                log("too many find_work failures -- aborting")
                break
            time.sleep(FAIL_BACKOFF_S)
            continue
        if name is None:
            log("nothing claimable -- all done or in flight; exiting")
            break

        lease = Lease("relations", name)
        ok = False
        try:
            if eval_cache is None:
                log(f"loading eval set ({EVAL_SPLIT})...")
                eval_cache = _load_eval(EVAL_SPLIT, spiral_fn)
            analyze_config(CONFIGS[name], lease, *eval_cache)
            ok = True
        except Exception as e:
            import traceback
            log(f"relations {name} FAILED: {e}\n{traceback.format_exc()}")
        finally:
            release("relations", name)        # frees the lease for another worker
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        if ok:
            fails = 0
            continue
        # any failure: back off so claim/release can't storm HF; give up after MAX_FAILS
        fails += 1
        if fails >= MAX_FAILS:
            log(f"{fails} consecutive failures -- aborting")
            break
        log(f"backing off {FAIL_BACKOFF_S}s before next claim [{fails}/{MAX_FAILS}]")
        time.sleep(FAIL_BACKOFF_S)


if __name__ == "__main__":
    main()
