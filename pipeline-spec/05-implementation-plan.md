# Implementation Plan: Turning the v5 Spec into a Runnable, Cluster-Scale Pipeline

> **Status:** plan for PI review (2026-06-26). Companion to
> [`01-methodology.md`](01-methodology.md), [`02-script-spec.md`](02-script-spec.md),
> [`03-v4-walkthrough.md`](03-v4-walkthrough.md),
> [`04-downstream-analysis-interface.md`](04-downstream-analysis-interface.md).
>
> Adds: (1) an **adversarial review** of the five docs vs the stated goals, (2) a **gap
> analysis** vs PI feedback, (3) the **locked decisions**, (4) a corrected **data-access
> matrix** + HF/MMU inventory, (5) the **phased P0->P2 plan** with scripts + acceptance
> criteria, (6) **open decisions** in plain language. External facts verified by research
> agents (HF dataset pages, lsdb/hats API, NCSA Delta/DeltaAI docs, Legacy DR10 docs);
> v4 code read directly.

## 0. Locked decisions (review + PI sign-off)
- **Scope:** **galaxies first** (P1); stars + AGN as **P2**. 1.58M is the *whole-project*
  volume goal (reached once P2 adds stars/AGN), not a galaxy-stage requirement.
- **Objective:** **maximize object count, quality-gated** — volume is the objective but
  every retained pair must pass match-quality gates. Target **near-0% false-match rate**
  (v4 reported 0.02%/0.06%); shaky batches are **stamped `low_confidence`, never silently
  dropped**.
- **Compute:** **data-gen on x86 Delta `cpu` partition** (real CPU queue, no wasted GPU,
  x86 => no aarch64 wheel risk); **PRH analysis on DeltaAI** (`ghx4`, H100s). Both mount
  shared **/taiga Lustre**, so one release tree is visible to both clusters.
- **Galaxy images:** **HSC** (full HATS) first; **Legacy DR10** via the **Flatiron MMU
  Globus HDF5** (pre-cut 160px griz, 124M galaxies — the high-throughput channel already
  exists), with **brick-download + local `Cutout2D`** as fallback for z>21 / northern.
- **Storage:** Parquet with Arrow `fixed_size_list<float32,N>` (consumable directly by
  doc 04's `load_dataset(streaming=True)`); WebDataset deferred to P2 only if needed.
- **Output repo:** `UniverseTBD/omnisky-v5` (project org; personal namespace for test runs).

---

## 1. Adversarial review of the five docs

**Verdict:** the v5 spec is strong and honest — every self-critique was verified against
the actual v4 code (`false_match_test.py:23,80,85` = RA-only +30" nearest-only on
**un-propagated** positions, and its `run_real_match` skips `propagate_coords` so it does
not mirror production; `base.py:26-29` `is_cached()` = "any non-empty shard dir is done";
`finalize.py:169` HF "upload" = `create_repo` only; `run_pipeline_v3.sh:5-11` requests
`--gpus=1 --partition=ghx4` for a 2-day `ThreadPoolExecutor(max_workers=6)` CPU job).
The review targets what is **still** weak.

### Critical (feasibility)
- **C1 — "Run data-gen on CPU, not GPU" is infeasible *on DeltaAI*.** DeltaAI has **no
  CPU-only partition** (every job allocates a GH200 GPU). Fix: run data-gen on the **x86
  Delta `cpu` partition**; reserve DeltaAI GPUs for the analysis side-car.
- **C2 — Streaming assumes compute-node outbound internet, which is undocumented** on both
  Delta and DeltaAI. Must be **probed on a compute node** before any architecture lock;
  fallback is login/DTN + Globus pre-staging to /taiga.
- **C3 — The lsdb-over-`hf://` bet is committed before proof.** Doc 03 admits the only
  reference (`TobiasPitters/mmu-crossmatch`) is "a single-pixel, two-catalog,
  positional-only demo... not a pipeline." Must validate one-pixel end-to-end first.
- **C4 — Scope was never reconciled** (galaxy-only paper vs stars+galaxies+AGN). *Resolved*
  by section 0: galaxies first, stars/AGN P2.

### High (correctness / consistency)
- **H1 — `apply_space_motion` silently *zeroes* high-PM motion when distance is missing.**
  Without parallax/distance, ERFA substitutes a 1e-7" proxy parallax => for fast nearby
  stars it implies v~c, zeros the velocity, and the position **does not move** (only an
  `ErfaWarning`). The highest-PM stars get *no* correction. Fix (P2): **mandate
  distance/parallax**, use `format='jyear'`, define a missing/negative-parallax policy.
- **H2 — Two partition axes can't both hold.** section 3.2 mixes row-partitioning (row%K)
  and source-partitioning; and `finalize_shard`'s >=2-instrument join needs *all* sources
  for a shard => a **cross-source barrier** that breaks "embarrassingly parallel, zero
  coordination." Fix: explicit 2-D work model + defined finalize barrier (section 5).
- **H3 — "Single authoritative manifest, built once" has no cross-user creation protocol**
  => two users can build divergent manifests. Fix: **build-or-attach** under an `O_EXCL`
  `release.lock` (section 5).
- **H4 — DONE markers + claim ledger + Lustre token-bucket = a small-file metadata storm**
  on the filesystem NCSA says to avoid (route small files to per-node NVMe /tmp). Fix:
  source-ownership or a single rate-limit **broker**; keep marker churn off Lustre.
- **H5 — Producer/consumer storage mismatch** (WebDataset vs doc 04's
  `load_dataset(streaming=True)`). *Resolved* by section 0: Parquet + `fixed_size_list`.

### Medium / Low
- **M1** volume-vs-quality conflict -> *resolved* (quality-gated volume).
- **M2** `verify_markers` re-hashing every shard = multi-TB Lustre re-read; use write-time
  digest + sampled rehash + periodic deep audit.
- **M3** false-match gate had no numeric threshold or fail-action -> defined in section 8.
- **M4** reproducibility: pin **HF dataset revision SHAs** per catalog (not just names).
- **M5** schema carries no physical-label columns (z, M*, sSFR) the regress tier needs.
- **M6** no node-hour/cost budget or stopping rule for a max-volume run.
- **L1** cross-population `global_object_id` collision -> namespace the ID by population.
- **L2** type-safety deprioritizes source adapters, where corruption originates -> add
  runtime boundary validation at adapter outputs.
- **L3** spec source-file drift: doc 02 lists `legacy.py`/`galex.py`; v4 has
  `galex_images.py` and **no** `legacy.py` (Legacy came via `flatiron`).
- **L4** v4 dataset card claims OmniSky is "publicly available" but upload was
  `create_repo`-only -> assume **never shipped**; the card is aspirational.

## 2. Gap analysis vs PI feedback

| PI concern | v5 response | Residual gap (this plan's fix) |
|---|---|---|
| False-match severity (#1) | sec 3.4.1 motion-aware MC protocol | Thresholds + fail-action (sec 8); H1 distance trap (sec 7) |
| Multi-cluster consistency | sec 3.2 partitions, atomic writes, state machine | Axis conflict + finalize barrier (H2); manifest race (H3); Lustre churn (H4) -> sec 5 |
| CPU vs GPU throughput | sec 3.8 CPU-only; GPU side-car | Infeasible on DeltaAI -> run on x86 Delta (C1) |
| DONE-marker robustness | sec 3.2 checksummed markers + audit | Full-rehash cost (M2) -> write-time digest + sampled rehash |
| Dataset availability | App. A matrix; lsdb/HATS | Compute-node internet (C2) + full-vs-preview (C3) -> Phase 0 probe |
| Scope (galaxies vs stars) | not addressed | Resolved: galaxies first, stars/AGN P2 (C4) |
| Objective (volume vs science) | sec 1 picks volume | Resolved: quality-gated volume (M1) |

---

## 3. Architecture

### 3.1 Compute topology
```
x86 Delta (cpu partition, internet) --writes--> /taiga/<alloc>/omnisky/release/v5 <--reads-- DeltaAI GH200 (ghx4: extract/CKA/mNN)
        DATA GENERATION (this plan)             shared center-wide Lustre (both mount)        DOWNSTREAM PRH ANALYSIS (doc 04)
```
Data-gen never needs a GPU -> x86 Delta `cpu`. DeltaAI reserved for the GPU side-car.
Fallback if no Delta allocation: DeltaAI + forced GPU + pre-stage (decided at Phase 0).

### 3.2 Access model
Full data = **`UniverseTBD/mmu_*` (HATS via lsdb)**; **`MultimodalUniverse/*` = ~1k-row
previews — never build on them.** `_healpix_29` (NESTED order-29) confirmed; MMU publishes
**10" margin catalogs** that lsdb auto-loads from the **collection path**. Pin `lsdb/hats
>=0.9` and per-catalog **HF revision SHAs**. Non-MMU: ZTF over `s3://ipac-irsa-ztf` (anon),
Legacy over Flatiron Globus HDF5, CDS XMatch as a single un-fanned job.

**Canonical source list = the UniverseTBD "Multimodal Universe HATS" collection**
(`huggingface.co/collections/UniverseTBD/multimodal-universe-hats`). Members span **three
orgs** — `UniverseTBD/`, `hugging-science/` (APOGEE DR17, Legacy-south, MaNGA), `LSDB/`
(VIPERS) — so `open_catalog` paths must use the correct org, not a blanket `UniverseTBD/`.

---

## 4. Data-access matrix (verified) + dataset inventory

### 4.1 Source matrix

| Source (modality) | Pop | Full? / Host | Access protocol | Creds / rate limit | Volume | Strategy . Phase |
|---|---|---|---|---|---|---|
| `mmu_desi_edr_sv3` (spectra) | gal/agn | **Full HATS**, UniverseTBD/HF | lsdb collection (+margin) | none / none | 1-10M | seed + spectra . **P1** |
| `mmu_desi_provabgs` (tabular SED) | gal | **Full HATS** | lsdb | none | 0.1-1M | seed + labels . **P1** |
| `mmu_hsc_pdr3_dud_22.5` (images) | gal | **Full HATS** (deep/UD, mag<22.5) | lsdb | none | 0.1-1M | primary galaxy image . **P1** |
| Legacy DR10 (images) | all | HF `hugging-science/mmu_legacysurvey_dr10_south_21` (**Preview ~89k — verify full**); full = Flatiron MMU HDF5 (84 TB, 124M, 160px griz, z<21 south) | lsdb if full, else **Globus** + `h5py` (or brick + `Cutout2D`) | none | large | lsdb if full / else Globus pre-stage . **P1/P2** |
| `Smith42/desi_hsc_crossmatched` / `legacysurvey_hsc` | gal | Full (20k / ~100k) | `load_dataset` | none | 20k-100k | **validation oracle** . **P1** |
| `mmu_sdss_sdss` (spectra) | gal/star | **Full HATS** | lsdb | none | 1-10M | spectra . P1/P2 |
| `mmu_jwst_*` (images) | gal | Full, small fields | lsdb | none | 1k-100k | optional image add . P2 |
| `mmu_gaia_gaia` (BP/RP + astrometry) | star | HATS; props say 122M/48% sky, 10" margin, **but collection viewer shows "Preview 62.3k" -> verify full-vs-preview in Phase 0** | lsdb | none | 122M? | PM source . **P2** |
| APOGEE DR17 (seed+spectra) | star | **Full HATS** `hugging-science/mmu_apogee_dr17` (720k) | lsdb | none | ~0.72M | **lsdb stream** (no local FITS needed) . **P2** |
| ZTF DR23/24 (light curves) | star/agn | **Full HATS**, `s3://ipac-irsa-ztf` (us-east-1) | pyarrow/lsdb `anon=True` | none | large | S3 stream/pre-stage . **P2** |
| `mmu_tess_spoc` (light curves) | star | Full HATS | lsdb | none | 0.1-1M | P2 |
| GALAH DR3 (spectra) | star | preview-only; datacentral.org.au (AU) | VO SSA | mild; ~250 ms RTT | small | pre-stage . P2 |
| 2MASS / unWISE / GALEX (images) | all | IRSA | astroquery SIA/cutout | public; mild | - | cutout . P2 |
| CDS XMatch (Gaia verify) | star | CDS Strasbourg | VO XMatch | **403 + IP-ban if fanned**; <=100 MB anon / 500 MB reg; 2M-row cap; 180" | - | **single job, never fanned** . P2 |
| SDSS DR16Q (AGN seed) | agn | local / `mmu_sdss` | local FITS / lsdb | none | ~0.75M | seed . P2 |

### 4.2 HF/MMU specifics (preview vs full)
- **`MultimodalUniverse/*`** = ~1k-row **previews** (`load_dataset(streaming=True)` works
  there). **`UniverseTBD/mmu_*`** = **full HATS** (no dataset script -> use lsdb / direct
  Parquet over `hf://`).
- `mmu_gaia_gaia`: `hats_col_healpix_order=29`, `hats_nrows=122302572`,
  `moc_sky_fraction=0.48`, margin `mmu_gaia_gaia_10arcs`. Partitions coarsest at **order-4**
  but **adaptive to order-10**.
- **HATS availability (rechecked vs the collection):** now on HF as HATS —
  `hugging-science/mmu_apogee_dr17` (720k), `hugging-science/mmu_manga`, `LSDB/mmu_vipers_w1/w4`;
  Legacy-south is on HF as a **preview** (`hugging-science/mmu_legacysurvey_dr10_south_21`),
  full still via Flatiron Globus HDF5. **Still not HATS:** GALAH (datacentral) and the SDSS
  DR16Q AGN seed (local FITS) -> convert with `hats-import` (sec 4.5).

### 4.3 Smith42 pre-cross-matched (validation oracle, not the product)
| Dataset | Modalities | Rows | Use |
|---|---|---|---|
| `Smith42/desi_hsc_crossmatched` | DESI spectra + HSC images | ~20k | reproduce-their-matches check (P1) |
| `Smith42/legacysurvey_hsc_crossmatched` | LS + HSC images | ~100k | concordance check |
| `Smith42/sdss_gaia_crossmatched` | SDSS spectra + Gaia | ~25k | star check (P2) |

### 4.4 lsdb access pattern (canonical)
```python
import lsdb
px = lsdb.PixelSearch((4, 257))                       # order-4 tile (auto-fetches finer leaves)
desi = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_desi_edr_sv3",
                         search_filter=px, columns=["ra","dec","_healpix_29","flux","ivar"])
hsc  = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5",
                         search_filter=px, columns=["ra","dec","image"])   # collection path -> margin auto-loaded
matched = desi.crossmatch(hsc, radius_arcsec=1.0, n_neighbors=5).compute()  # n>1 surfaces ambiguity
```
Caveat: first `open_catalog` reads `point_map.fits` (~15-20 s/catalog) — budget it; mitigate with `PixelSearch`.

### 4.5 HATS-first: reuse existing matches, convert the rest, flag what we can't
To run **one uniform lsdb crossmatch**, convert non-HATS sources to HATS with `hats-import`
before matching (SDSS DR16Q AGN seed, GALAH spectra, any local FITS not on HF). Already
HATS — all MMU `mmu_*`, APOGEE DR17, ZTF (`s3://ipac-irsa-ztf`) — need no conversion.
**Flag explicitly** any source we cannot convert (license/format/volume), fall back to a
custom reader, and record the reason in the matrix (sec 4.1). Minimize bespoke readers so
lsdb does the join everywhere it can.

---

## 5. Orchestration, parallelism & graceful failure (fixes H2/H3/H4)
- **2-D work units `(source x seed-shard)`**, each tagged a **concurrency class**:
  `parallel` (lsdb/S3 — row-partition freely) vs `owned` (CDS/cutouts — one owner, never
  fanned). Resolves the row-vs-source contradiction.
- **Manifest build-or-attach (H3):** first submitter takes `release.lock` via
  `O_CREAT|O_EXCL`, runs `plan_work_units.py`, writes `manifest/` + `manifest_hash`; later
  submitters **attach** to the existing hash (abort on inputs-hash mismatch).
- **Explicit finalize barrier (H2):** `finalize_shard(S)` runs only when **all** source x S
  units show `complete` markers — the >=2-modality join needs every source present.
- **Rate-limit coordination (H4):** prefer **source-ownership**; if a service must be
  shared, use a **single broker job** (one fetch proxy), not a Lustre token-bucket.
- **Atomic writes:** heavy scratch on /tmp (NVMe) -> final via
  `tmp->fsync->os.replace->fsync(dir)` on Lustre -> DONE marker (one per unit). **Resume
  state machine** (pending/interrupted/suspicious/stale/corrupt/complete).
- **Recovery:** dead-owner tmp GC; checksum-mismatch -> re-run; stale (hash != current) ->
  re-run; **never** silently skip partial sources (kills v4's `is_cached` bug).

## 6. probe_sources.py (cold vs warm, not just pings)
Per source: **cold-path** (first open incl. lsdb MOC read ~15-20 s), **warm-path** (repeat
pixel fetch), **sustained throughput** (N>=200 units -> MB/s & units/s), **429/403
behavior**, **credential check**, **HATS coverage** for the seed footprint. **Must run on a
compute node** to settle the internet question. Emits `probe_report.json` -> global
per-service caps.

## 7. Crossmatch methodology
- **Don't reinvent matching.** Prefer, in order: (1) **already-cross-matched datasets** —
  Smith42 `desi_hsc`, `legacysurvey_hsc`, `sdss_gaia` — where they cover a pairing (use
  as-is, no matching); (2) **lsdb's built-in `crossmatch`** (KdTreeCrossmatch) for HATS
  catalogs sharing `_healpix_29` (the spatial join is solved); (3) **custom matching only**
  where neither exists. We own epoch handling + adjudication + provenance, **not** the join.
- `global_object_id` = order-29 NESTED index of the **seed** position (continuous with
  `_healpix_29`), **namespaced by population** (fixes L1). **P1 galaxies: pm=0 -> no
  `apply_space_motion`** (major simplification).
- lsdb `crossmatch(n_neighbors>1, radius_arcsec)` on the **collection path** (auto margin) +
  per-source radii + explicit adjudication (unambiguous -> best `sep/sigma` -> flags ->
  smallest-sep tiebreak); emit `match_ambiguous`, `n_candidates_within_radius`.
- **P2 stars (H1 fix):** `apply_space_motion` **with distance/parallax**, `format='jyear'`,
  defined missing/negative-parallax policy.
- Raw payloads preserved; normalization documented for train time (doc 04 / card).
- **Validate against Smith42** pre-cross-matched sets (reproduce their matches within tol).

## 8. Validation & false-match suite (fixes M3)
- **Mirror the production matcher exactly** (incl. epoch correction for stars). MC
  **random-direction** offsets (N~100) on an annulus scaled to confusion
  (`radius (+) mu.dt (+) parallax (+) sigma_astro`; for P1 galaxies -> `sigma_astro (+)
  density`). **PM-scramble null** = stars only (P2). **Stratify** by **|b| x local density**
  (x PM-bin for stars). **Analytic Rayleigh+background** cross-check.
- **Gate + fail-action:** include if false-match rate **< 0.1%** per source x |b| bin
  (target near-0%); bins above -> **route to `low_confidence` config** (kept + labeled, not
  dropped). Hard-block only on schema/uniqueness/marker failures.
- **Release gate (all required):** `verify_markers` green . `validate_release` green .
  false-match under threshold . `global_object_id` unique . >=2-modality coverage .
  **Smith42 concordance** (P1) . dataset card present.

## 9. Type / memory / reproducibility (fixes H5/M2/M4/L2)
- **Storage:** Arrow `fixed_size_list<float32,N>` in Parquet (HSC 160x160, DESI fixed-len)
  -> directly consumed by doc 04's `load_dataset(streaming=True)`. WebDataset deferred.
- **Integrity (M2):** write-time `xxh3` of a canonical per-column digest + size + rowcount
  in each marker; `verify_markers` checks those + **sampled** full-file rehash + periodic
  deep audit — no multi-TB re-read every upload.
- **Reproducibility (M4):** pin per-catalog **HF revision SHAs** + code git SHA +
  manifest/schema hashes + output checksums (all in markers + release manifest).
- **Typing:** mypy-strict on `config/ids/matching/coordination/schemas/io` **plus runtime
  boundary validation at every source-adapter output** (L2).

## 10. Phased plan — modules . tests . acceptance

### Phase 0 — Feasibility spike (local + tiny cluster job). **Gates everything.**
- **Build:** conda env (x86 Delta; aarch64 DeltaAI for side-car only), `probe_sources.py` v1.
- **Test:** compute-node `curl` to HF/S3/CDS on **Delta and DeltaAI**; one-pixel
  `mmu_desi x mmu_hsc` lsdb crossmatch (+margin) -> `.compute()`; reproduce
  `Smith42/desi_hsc` matches on an overlapping pixel.
- **Accept:** internet verdict recorded (-> stream vs pre-stage) . one-pixel crossmatch
  reproduces Smith42 within tolerance . env reproducible . per-service caps drafted.

### Phase 1 — Galaxy backbone (thin slice -> full galaxy run)
- **Build (modules):** `config.py`, `schemas.py`, `ids.py`, `matching.py` (pm=0 path),
  `io.py` (atomic + markers), `coordination.py` (build-or-attach + static partitions +
  barrier), `healpix.py`, `sources/base.py`, `sources/lsdb_mmu.py`,
  `sources/legacy_hdf5.py` (Flatiron Globus). **Scripts:** `build_seed_catalogs.py`,
  `plan_work_units.py`, `run_source_shard.py`, `finalize_shard.py`, `finalize_release.py`,
  `validate_release.py`, `verify_markers.py`, `false_match_report.py`, `upload_hf.py`.
  Sources: DESI spectra, HSC images, Legacy (Flatiron) images.
- **Test:** `TEST_MODE` end-to-end (~5 obj/pop) in CI; small-cluster run on a few HEALPix
  pixels.
- **Accept:** re-run with all markers = **no-op**; corrupt a final file -> forced re-run;
  two disjoint `--partition-id` submissions -> no clobber + caps respected; every object
  `n_modalities>=2` + unique id; `verify_markers`/`validate_release` green; false-match
  under threshold; **HF upload load-back verified**; cost dry-run within budget.

### Phase 2 — Scale + stars/AGN + extras
- Full galaxy run (all pixels); add **stars** (APOGEE local seed, Gaia/CDS verify,
  `apply_space_motion`+distance), **AGN** (SDSS DR16Q); ZTF/TESS light curves; PM-scramble
  false-match; WebDataset/broker/token-bucket **only if** P1 proved them necessary;
  HSC/JWST extra image adds. Push program total past 1.58M.
- **Accept:** full-release gates green across all populations; stars pass PM-aware
  false-match strata.

## 11. Open decisions for PI (plain language)
**Only three need the PI's judgment; the rest are defaults to rubber-stamp.**
1. **Galaxy images — HSC-only or also Legacy?** HSC = cleaner but less sky; Legacy (via
   Flatiron Globus HDF5) = many more galaxies, more setup. *Rec:* HSC first, add Legacy in
   P2. *(Leaning Legacy-via-Globus; confirm footprint scope.)*
2. **Volume ambition — is "beat 1.58M" a whole-project goal (after stars/AGN)?**
   *Confirmed: yes.*
3. **Match strictness + shaky matches.** *Confirmed: aim near-0%; stamp shaky as
   `low_confidence`, keep don't drop.*
4. *(fact)* **Delta (x86) allocation** present? Assumed yes (user has H200/Delta + H100/DeltaAI).
5. *(default)* **Format** = Parquet + `fixed_size_list`. Confirmed.
6. *(default)* **Repo** = `UniverseTBD/omnisky-v5`; single uploader.
7. *(default)* **Releases to pin:** DESI EDR (`mmu_desi_edr_sv3`), HSC PDR3 DUD,
   `mmu_desi_provabgs`; P2: Gaia DR3, APOGEE allStar DR17, GALAH DR3, SDSS DR16Q, ZTF DR23,
   TESS SPOC — exact revision SHAs locked in Phase 0.
8. *(default)* **Singles** (<2 modalities): drop from main; optional separate config.

## 12. References
- Huh et al. 2024 — Platonic Representation Hypothesis, arXiv:2405.07987.
- Duraphe, Smith, Sourav, Wu 2025 — The Platonic Universe, arXiv:2509.19453.
- Parker, Lanusse et al. 2024 — AstroCLIP, arXiv:2310.03024.
- Smith et al. 2024 — AstroPT, arXiv:2405.14930; multimodal ext. arXiv:2503.15312.
- Multimodal Universe 2024 — arXiv:2412.02527; github.com/MultimodalUniverse/MultimodalUniverse.
- lsdb docs.lsdb.io; HATS hats.readthedocs.io; Legacy DR10 legacysurvey.org/dr10/.
