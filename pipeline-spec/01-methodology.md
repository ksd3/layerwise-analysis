# Methodology: A Cluster-Scale Cross-Matched Multimodal Dataset for the Platonic Universe

> **Status:** draft for PI review (v5). Companion docs:
> [`02-script-spec.md`](02-script-spec.md) (the scripts),
> [`03-v4-walkthrough.md`](03-v4-walkthrough.md) (how the existing code works),
> [`04-downstream-analysis-interface.md`](04-downstream-analysis-interface.md) (what
> the dataset must support for the PRH analysis).
>
> **v5 changes (this revision)** address PI feedback: a real *false-match protocol*
> that accounts for real+apparent motion (§3.4.1); a *distributed execution &
> consistency model* for concurrent multi-user jobs on a shared filesystem (§3.2);
> a *resource & throughput model* that reasons about CPU/GPU/network (§3.8);
> *hardened DONE markers* with integrity verification (§3.2); and a *Data Access
> Matrix* answering "stream from afar vs. S3/HF" per source (Appendix A).

## 1. Goal & scientific framing
Build the largest possible **clean, cross-matched, multimodal** catalog of celestial
objects (stars, galaxies, AGN), where **each retained object is observed by ≥2
instruments** and carries, where available, **images + spectra + light curves +
tabular** features. Upload to HuggingFace. The dataset extends the *Platonic Universe*
program (arXiv:2509.19453, Duraphe, Smith, Sourav, Wu — "Do Foundation Models See the
Same Sky?"), which tests whether models trained on different astronomical
modalities/instruments converge to a shared representation (Platonic Representation
Hypothesis, Huh et al. 2024, arXiv:2405.07987).

**Why this dataset shape is the right one for the science** (from the downstream
analysis, see doc 04): the convergence metrics (**mutual k-NN alignment** and **CKA**)
require the **same N physical objects encoded by both models**. Precedent: AstroCLIP
(arXiv:2310.03024) aligned image↔spectrum on **197,632** galaxies matched by shared
TARGETID; the PRH paper got stable trends with **N≈1,000–1,024 paired objects** at
k=10. **Scale target (PI decision): maximize raw object count** — match or beat v4's
1.58M-object build — while every design choice below still optimizes for
**trustworthy pairs** (match *quality* gates inclusion; volume is the objective).

## 2. Starting point: extend & harden `v4/OmniSky`, don't rebuild
The `layerwise-analysis/v4` pipeline ("OmniSky") already produced a 1.58M-object,
12-survey dataset and is the right backbone (see doc 03). The science logic
(seed-and-match, HEALPix pre-filter, epoch propagation, streaming shards, the
shifted-catalog false-match test) is **sound and worth keeping**. What is "scratch"
and must be re-architected: **cluster concurrency across concurrent submissions,
resumability, global object identity, the (missing) HF upload, array storage/typing,
the false-match test (which today tests a *different matcher* than production), and
remote data access (today raw HTTP; v5 uses lsdb/HATS over HF + S3).** This document
specifies the hardened design; doc 02 specifies the scripts.

## 3. Core architectural decisions

### 3.1 Identity: one `global_object_id`, assigned once
- **Canonical frame/epoch:** ICRS, **J2016.0** (Gaia DR3 reference).
- **ID:** **nested HEALPix index at order 29 (`nside = 2^29`)** of the seed position,
  packed as signed `int64` → `global_object_id`. Deterministic, idempotent, compact,
  fast integer joins; sub-milliarcsec cell size makes collisions negligible (assert
  uniqueness at build time and **fail the build** on any collision). **This is not an
  arbitrary choice:** MMU v1 catalogs already carry a native `_healpix_29` column
  (order-29 NESTED), so our global ID is continuous with the ecosystem we draw from.
- **Assignment rule (critical):** compute the ID **once, from the seed catalog**, at
  J2016.0 (stars propagated there via `apply_space_motion`; galaxies/AGN at fixed
  ICRS). **Never recompute it from a downstream survey's coordinates** — two surveys
  report slightly different positions for the same object (and `_healpix_29` will
  therefore differ between catalogs), so recomputing would split identity. All source
  matches attach to the seed's ID.
- **Provenance:** keep every survey-native ID as its own column (`seed_native_id`,
  `gaia_source_id`, `apogee_id`, `sdss_specobjid`, …). Global ID = join key;
  native IDs = provenance. Persist `seed_ra_deg_j2016`, `seed_dec_deg_j2016`,
  `seed_epoch_jyear`.

### 3.2 Distributed execution & consistency model (multi-submission, shared FS)
**The PI's question:** "I submit a job, you submit a job — do they run in parallel and
write to the same database safely?" **Operating assumption (confirmed):** one cluster
(NCSA DeltaAI), **multiple independent SLURM submissions by multiple users sharing one
Lustre filesystem**, writing to one release tree. We design for that explicitly.

**Substrate guarantees we rely on (and ones we don't).** On a single Lustre filesystem,
`os.replace()` (rename) is atomic and `open(O_CREAT|O_EXCL)` is atomic for exclusive
creation — we build on these. We **do not** rely on `flock`/`fcntl` locks (Lustre
needs the `-o flock` mount option, it is advisory, and it is fragile across nodes).

**Ownership model — primary: disjoint partitioning by construction (no runtime locks).**
A single `plan_work_units.py` builds the authoritative manifest *once*. Each submission
is launched with `--partition-id k --num-partitions K`; a job processes only manifest
rows where `row_index % K == k` (or a contiguous block range). Two submissions with
different `k` **never target the same output path** ⇒ zero coordination at runtime,
embarrassingly parallel, and trivially correct even if more people join later. This is
the recommended default for v1.

**Ownership model — optional: a claim ledger (for dynamic load-balancing).** If static
partitions become load-imbalanced, switch to atomic claiming: to process row *i*, a
worker does `os.open(claims/<manifest_hash>/<i>.claim, O_CREAT|O_EXCL)`; success =
ownership (write `{jobid, array_task, host, pid, start_ts, heartbeat_ts}`), failure
(`FileExistsError`) = already owned, skip. A periodic heartbeat refreshes `heartbeat_ts`.
A reaper reclaims a claim only if its SLURM job is absent from `squeue` **and** the
heartbeat is older than a TTL (e.g. 2× interval), by deleting the stale claim. Use this
**only if** disjoint partitioning proves too rigid — it adds a ledger to reason about.

**Atomic, verified writes (every work unit).** Write
`shard.parquet.tmp.<jobid>.<arraytask>.<pid>` → `fsync(file)` → `os.replace()` to the
deterministic final path → `fsync(dir)` → **then** write the DONE marker atomically
(tmp → fsync → `os.replace` → `fsync(dir)`), **strictly ordered after** the data is
durable. A crash between the two leaves "final, no marker" = a *suspicious* state that
re-runs safely (idempotent overwrite).

**DONE markers prove the bytes are correct and current** (the PI's "barebones" fix). A
marker (`shard.done.json`) records: manifest hash, seed version, source release, schema
version, **code commit (git SHA)**, **row count**, **output byte size**, and an
**output content checksum** (xxh3/sha256 of the file, or of a canonical per-column
digest), plus host/jobid/timestamp. Resume is a **state machine**, not "directory
exists":
| State | Detection | Action |
|---|---|---|
| pending | no final, no marker | run |
| interrupted | tmp only (no final) | GC tmp if owner dead; run |
| suspicious | final, no marker | re-run (overwrite) |
| stale | final+marker, but marker hash ≠ current manifest/schema/code | re-run |
| corrupt | final+marker, hashes match, but file checksum ≠ marker | re-run |
| complete | final+marker, hashes & checksum match | **skip (no-op)** |

A release-level audit (`verify_markers.py`, gates upload) independently re-derives that
every manifest row is `complete`, with no orphan tmp files and no suspicious/stale/
corrupt states. **This is the "is *done* actually done?" test** — and it doubles as the
shared completion ledger across submissions.

**Global per-service rate limiting (the consequence of multi-submission + max-volume).**
Remote services (PS1, Legacy cutouts, CDS XMatch, GALAH, IRSA) limit the **sum** of
requests across *all* concurrent jobs and *all* users — per-job caps are insufficient,
and two users hammering CDS/PS1 at once risks throttling or an IP ban. Two options:
- **v1 (recommended): partition *sources* across submitters** (you own cutouts, PI owns
  spectra) so no two submitters hit the same service simultaneously — no shared state.
- **Upgrade: a shared token-bucket per service**, persisted on Lustre (atomic update via
  `O_EXCL` temp + rename; avoid sqlite-on-Lustre locking). Each worker acquires a token
  before a remote call. Rates are seeded from `probe_sources.py` and published limits.

**HF write model (the other "database").** Concurrent `upload_folder`/`create_commit`
to one HF repo cause git/LFS commit races (HTTP 412, non-fast-forward). Rule: a **single
designated uploader job** runs `upload_hf.py` *after* all builds finish and the audit
passes; it is never run by two submitters at once. (If incremental upload is ever
needed, each writer commits **disjoint path-prefixes** via `create_commit` with
backoff on 412 — but v1 uses the single-uploader model.)

**RAM/memory budget (the literal "memory model").** Each array task owns one seed
row-shard (≤ `--shard-size`, default 50k objects) × one source; peak working set =
the shard's positions + one source partition + matched payloads, bounded to a few GB.
`finalize_shard.py` joins **per shard** (no global shuffle) in 50k slabs. Narrow dtypes
(`float32` payloads, `int64`/`float64` for ids/coords), Arrow nested types (never pandas
object columns). A GH200 node's ~480 GB unified memory dwarfs a single task, so we pack
**many array tasks per node**.

### 3.3 HEALPix + lsdb/HATS as the access & match layer (PI decision: adopt lsdb)
**For MMU sources, read and match with lsdb/HATS, not raw HTTP.** MMU v1 publishes full
catalogs as HATS under `hf://datasets/UniverseTBD/mmu_{catalog}_{subcatalog}`,
partitioned at HEALPix **order-4 / nside-16 (NESTED)** with finer data-dependent leaves.
Access pattern (validated by the `TobiasPitters/mmu-crossmatch` Space):
```python
import lsdb
cat = lsdb.open_catalog(
    "hf://datasets/UniverseTBD/mmu_gaia_gaia",
    search_filter=lsdb.PixelSearch([(order, pix)]),   # partition pushdown
    columns=["ra", "dec", ...],                        # column pushdown
)
```
This reads **only the needed partitions and columns** from HF's S3/CDN — the modern
replacement for v4's HTTP reads off `users.flatironinstitute.org`. Two critical notes:
- **lsdb crossmatch is positional only** (default `n_neighbors=1`, `radius_arcsec`). So
  **we own epoch propagation**: propagate the seed to each source's epoch *before*
  handing coordinates to lsdb (§3.4), and set **`n_neighbors>1`** to surface ambiguity
  (the "all-within-radius" behavior of §3.4), then adjudicate ourselves.
- **Margin cache fixes boundary loss:** load each cell **plus a 1–2′ margin** (lsdb's
  margin cache, or neighbor cells via `astropy-healpix`) so an object near a partition
  edge isn't dropped when its counterpart sits in the adjacent cell. This is exactly the
  bug in MMU's naive per-cell `cross_match_datasets` and the demo's "do not overlap"
  sub-pixel case.

**Fail fast, never silently degrade.** v4's `compute_catalog_healpix` returns `None`
(→ full all-sky scan) if `healpy` is missing. v5 makes HEALPix/lsdb a **hard dependency**
and aborts if unavailable. Standardize: `lsdb`/`hats` for partitioned catalog access,
`astropy-healpix` for ad-hoc indexing/neighbors, `healpy` only for map utilities.
Non-MMU sources (ZTF on S3, cutout/VO services) keep purpose-built readers but reuse the
same pre-filter + column-projection discipline (doc 03 §3).

### 3.4 Matching: `apply_space_motion` + radius search + explicit adjudication
- **Proper motion:** replace the hand-rolled propagation with astropy
  `SkyCoord.apply_space_motion()` (rigorous spherical motion). Document edge cases:
  missing PM/RV, extragalactic `pm=0` *by assumption* (flag it), sign of Δt, the
  `cos(δ)` convention, near-pole instability, and long-time-span surveys vs a single
  mean epoch (use a catalog reference epoch, or per-detection times, where available).
- **Candidate retrieval:** use **`search_around_sky`** (all within radius) for custom
  readers, or **lsdb `crossmatch(..., n_neighbors>1)`** for MMU sources — **not**
  nearest-only — so ambiguity is visible. Then adjudicate: within source-specific radius
  → prefer unambiguous → prefer best `sep / positional_uncertainty` → use quality flags →
  smallest separation only as final tie-break. Emit `match_ambiguous: bool` and
  `n_candidates_within_radius: int`.
- **Per-source radii**, not one global 3″: tight for precise catalogs
  (Gaia/SDSS/DESI/Legacy/PS1), looser only where justified (GALEX, 2MASS, unWISE).

#### 3.4.1 Proving matches: the hardened false-match protocol (the PI's #1 point)
**Why v4's test is not enough.** `false_match_test.py` shifts the seed **+30″ in RA
only** and matches with **nearest-only on *un-propagated* positions** — i.e. it
benchmarks a *different, simpler matcher than production*, and a fixed 30″ is comparable
to the multi-decade proper-motion displacement of high-PM stars. So it (a) fails to
exercise the real failure mode, and (b) for a high-PM star, a 30″ shift can land on that
star's *true* counterpart at a different epoch, contaminating the "false" estimate. 30″
is **not** "clearly beyond any real displacement" once you admit proper motion and
parallax. The reported 0.02%/0.06% numbers therefore don't characterize the real pipeline.

**The v5 protocol (account for real *and* apparent motion):**
1. **Mirror production exactly** — same `apply_space_motion` to the source epoch, same
   radius search / lsdb match, same adjudication. The FMR must be the FMR of the *actual*
   matcher.
2. **Monte-Carlo random-direction offsets** — apply N≈100 offsets at random position
   angles (not one fixed RA shift); report the *distribution* (mean / 95th / worst), not
   a point estimate.
3. **Offset scaled to the physical confusion scale** — draw offset magnitudes from an
   annulus with inner radius `r_min ≳ match_radius ⊕ (μ_max · Δt_baseline) ⊕ ϖ_max ⊕
   σ_astrometry`: combine the match radius, **real motion** (proper motion × the
   survey's actual epoch baseline), **apparent motion** (parallax amplitude ϖ), and
   astrometric error. For nearby/high-PM stars this is tens of arcsec, so `r_min` is set
   per population/PM-bin — not a blanket 30″.
4. **PM-direction-scramble null (the most severe, most physical test)** — keep each
   star's PM *magnitude*, randomize its *direction*, re-propagate, re-match. This
   directly measures "if my epoch correction points the wrong way, how often do I grab a
   false neighbor?" — exactly the PI's worry.
5. **Stratify** by **PM-magnitude bin × local source density × |b| (galactic latitude)** —
   the high-PM tail and crowded low-|b| fields are where matches break; an aggregate
   number hides them.
6. **Analytic cross-check** — fit the match-separation histogram as real (Rayleigh core
   from astrometric error) + background (rising ∝ r from field density; Sutherland &
   Saunders 1992 / Budavári & Szalay 2008); integrate the background within the radius.
   Two independent estimators agreeing ⇒ trustworthy.
7. **Gate the upload** on a per source × population × PM-bin × |b|-bin threshold.

### 3.5 Storage: raw values, typed schema, heavy payloads as shards
- **Store raw** pixel/flux/time values. Normalization is a documented **train-time**
  transform per modality (§4), never baked in.
- **Explicit versioned `pyarrow.Schema`** (don't let pandas infer):
  `global_object_id:int64`, seed coords/epoch, per-source presence flags, native IDs,
  per-source `match_sep_arcsec` + ambiguity, modality/instrument counts, split,
  provenance/version.
- **Fixed-length arrays → Arrow `fixed_size_list<float32, N>`** (64×64→4096-flat;
  APOGEE 7514). **Variable/heavy payloads** (images, variable spectra, light curves) →
  **WebDataset tar shards** referenced by key, OR Arrow `list<struct<...>>` if inline.
  Avoid pandas object-list columns. `float32` payloads/separations; `float64` only for
  coordinates/times.

### 3.6 "≥2 instruments" is an explicit, enforced rule
After the per-shard merge, count **distinct instruments/surveys with valid raw
payloads**; the seed row alone does **not** count, and metadata-only attachments do
**not** count as an instrument. Persist `n_instruments_present` and an
`instrument_presence_mask`. Drop (or route to a "singles" split) objects with <2.

### 3.7 Upload is a first-class, verified phase
A dedicated `upload_hf.py` uses `HfApi.create_repo` + **`upload_folder`** for the release
tree, writes the dataset card, then **verifies** with a clean
`load_dataset(..., streaming=True)` / `hf_hub_download` round-trip. Run by a **single
designated uploader** after the release audit passes (§3.2). If single-repo size hurts,
split into a manifest repo + per-modality shard repos. (v4 only ever called
`create_repo` — the data was **never actually uploaded**; this phase fixes that.)

### 3.8 Resource & throughput model (the PI's #3 point: CPU vs GPU)
**Cross-matching is network- and CPU-bound; GPUs do not help the bottleneck.** The
angular match itself is trivial — astropy's (and lsdb's) k-d tree matches 1.5M × millions
of sources in seconds–minutes. Wall-clock is dominated by **fetching data** and the
**latency/rate limits of per-object services**. Therefore:
- **Run data-gen on CPU resources, not GPU.** v4's SLURM script requests
  `--gpus=1 --partition=ghx4` (a GH200) for a job that never touches the GPU — it sits
  idle for ~2 days. On DeltaAI (GH200 / **ARM aarch64**), prefer a CPU partition; if
  confined to GH200 nodes, leave the Hopper GPU unallocated and pack many array tasks
  onto the 72-core Grace CPU to overlap network waits. **Reserve GPUs for the downstream
  PRH side-car** (doc 04): embedding extraction, CKA's O(N²) Gram matrices, mutual-kNN
  (FAISS-GPU genuinely shines there).
- **Did we look for GPU cross-match? Yes — it exists but won't help us.** GPU spatial NN
  is real (FAISS-GPU, cuML `NearestNeighbors`, RAPIDS cuSpatial; map RA/Dec→unit vectors,
  Euclidean NN). By **Amdahl's law** it accelerates <1% of our wall-clock; the ~99% is
  network. lsdb's parallelism is **Dask (CPU)**, which is the right model here. The honest
  line for the paper: *"GPU xmatch exists; it doesn't move a network-bound workload."*
- **aarch64 caveat:** verify `healpy`/`pyarrow`/`lsdb`/`hats`/`astropy-healpix` wheels on
  ARM (most via conda-forge; flag any needing source builds).
- **Throughput budget (sizing).** Long poles: per-object cutouts (PS1, Legacy), CDS
  XMatch (France; star-seed Gaia verification), GALAH (Australia). Estimate per service
  `wall ≈ N / effective_req_per_sec`, `effective = min(service_limit, concurrency /
  latency)`, capped **globally** across submitters (§3.2). S3/HATS sources (ZTF, MMU-on-HF)
  are bandwidth-bound (tens of MB/s) and parallelize freely. `probe_sources.py` measures
  these **before** we commit cluster hours (Appendix A).

## 4. Normalization recipes (documented, applied at train time)
- **Images (HDR):** raw counts/flux stored. Recommended: per-band `asinh` stretch or
  `astropy.visualization` `ZScaleInterval`; never naive 0–1 min–max (it crushes faint
  structure against bright cores — the HDR failure flagged). Example code in the card.
- **Spectra:** raw flux + inverse-variance/mask stored; continuum-normalization optional
  downstream (APOGEE is already continuum-normalized; document the inconsistency).
- **Light curves:** raw times/flux(mag)/errors; standardize per object/band downstream;
  document time systems (ZTF HMJD vs TESS BTJD).

## 5. Correctness & bias checklist (the "don't ruin the experiment" list)
1. **Epoch propagation present and correct** for every source (per-survey epoch table;
   `apply_space_motion`). High-PM stars are the main hazard; galaxies pm=0.
2. **False-match rate via the §3.4.1 protocol** — production-mirroring, MC random
   directions, PM-scramble null, stratified by PM × density × |b|, gated at upload.
3. **Ambiguity surfaced** (`match_ambiguous`, `n_candidates_within_radius`).
4. **Missingness is typed:** *outside footprint* vs *queried, no detection* vs *failed QC*.
5. **Selection biases inherited** from parent surveys (APOGEE → bright giants; PROVABGS →
   DESI footprint, z<0.6; DR16Q → spectroscopic). Document them.
6. **Spatial (HEALPix) train/val/test split**, never per-object random.
7. **Uniqueness of `global_object_id` asserted**; **≥2-instrument rule enforced**.
8. **Determinism:** fixed seeds, pinned survey releases, manifest/schema/code hashes +
   output checksums in every DONE marker.

## 6. What "done" looks like
A versioned release on HuggingFace: a Parquet **manifest** + (WebDataset) payload shards,
an explicit schema, a dataset card with normalization recipes and per-source coverage/
false-match tables, and a reproducible pipeline (seed → plan → run → finalize → validate
→ upload) runnable as SLURM **arrays** with full resume, **verified by a release-level
marker audit before upload** (§3.2).

## 7. Open questions for the PI
- **Resolved (this revision):** execution model = one cluster, concurrent multi-user
  submissions on shared Lustre (disjoint partitions by construction); scale target =
  maximize object count; MMU access/match layer = lsdb/HATS over HF.
- **Global rate-limit policy:** partition sources across submitters (v1) vs a shared
  token-bucket (upgrade) — which for the first real run?
- **New surveys/modalities** beyond v4's 12 — add HSC/JWST imaging (both on MMU/HF) to
  strengthen the galaxy image↔spectrum pairing the PRH analysis most needs?
- **False-match gate thresholds** per population/PM-bin/|b|-bin.
- **Payload storage** — WebDataset shards vs everything-in-Parquet.
- **Singles handling** — drop <2-instrument objects, or publish as a separate config?

---

## Appendix A — Data Access Matrix (which sources stream from afar vs. S3/HF)
*"How we process it" follows directly from where the data lives. Confirm volumes/limits
empirically with `probe_sources.py` before the first full run.*

| Source (modality) | Pop. | Host & location | Protocol / partitioning | Rate limit | v5 strategy |
|---|---|---|---|---|---|
| MMU Gaia BP/RP (spectra) | star | `hf://UniverseTBD/mmu_gaia*` (HF S3/CDN) | HATS order-4; lsdb PixelSearch+cols | none | **lsdb stream** |
| MMU TESS (light curves) | star/agn | HF (UniverseTBD) | HATS; lsdb | none | **lsdb stream** |
| MMU DESI (spectra) | galaxy/agn | HF (UniverseTBD) | HATS; lsdb | none | **lsdb stream** |
| MMU Legacy Survey (images) | all | HF (UniverseTBD) | HATS; lsdb | none | **lsdb stream** (replaces v4 cutout svc) |
| MMU HSC / JWST (images) | galaxy | HF (UniverseTBD) | HATS; lsdb | none | optional add for image↔spectrum |
| ZTF (light curves) | star/agn | `s3://ipac-irsa-ztf` (AWS us-east-1) | HATS; pyarrow+S3 anon | none | **S3 stream** (~75 MB/s from Delta) |
| PS1 (images) | all | ps1images.stsci.edu (STScI) | per-object cutout HTTP | yes | thread pool + **global cap**; prefer MMU image where covered |
| unWISE (images) | all | unwise.me | coadd tiles HTTP | mild | HEALPix pre-filter |
| GALEX / 2MASS (images) | all/star | IRSA (Caltech) | HTTP | mild | HEALPix pre-filter |
| GALAH (spectra) | star | datacentral.org.au (**Australia**) | HTTP + VO SSA, per-object | yes (far) | small N; throttle; **global cap** |
| APOGEE allStar (seed+spectra) | star | MAST + **pre-staged local** (Globus) | local FITS | n/a | local read (fast) |
| SDSS spectra | galaxy/agn | **pre-staged local** dir | local FITS | n/a | local read (fast) |
| CDS XMatch (Gaia verify of star seed) | star | CDS Strasbourg (**France**) | VO XMatch service | yes (strict) | batch; **never fan across array**; global cap |

**Key consequence:** several v4 *remote cutout* sources — Legacy Survey images
especially — are **also** available as MMU HATS on HF, so v5 replaces rate-limited cutout
calls with `lsdb`-over-`hf://` reads wherever MMU coverage suffices, collapsing the
network long-pole. `probe_sources.py` confirms per-source coverage and throughput.
