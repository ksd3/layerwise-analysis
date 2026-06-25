# Script Spec: Hardened Cross-Match Pipeline (v5)

> Implements [`01-methodology.md`](01-methodology.md). Keeps v4's survey logic
> (doc 03), rewrites the fragile architecture layers. Target: SLURM **array** jobs on
> CPU resources, **multi-submission-safe** on a shared filesystem, full resume with
> integrity-verified DONE markers, lsdb/HATS access for MMU sources, verified HF upload.

## 1. Repository layout
```
mmu/                          # importable core library (type-checked, mypy-strict)
  config.py                   # frozen dataclass config; env parsing; validate once
  schemas.py                  # versioned pyarrow.Schema; column names; feature defs
  ids.py                      # canonical J2016.0 coords + global_object_id (healpix-29 int64)
  matching.py                 # apply_space_motion, radius search / lsdb n_neighbors>1, adjudication, per-source radii
  coordination.py             # NEW: manifest partition assignment, optional claim ledger, global per-service token bucket
  io.py                       # atomic write (tmp->fsync->os.replace->fsync dir), DONE markers w/ checksum, shard paths
  healpix.py                  # astropy-healpix + lsdb/hats indexing + neighbor/margin queries (fail-fast)
  rate_limit.py               # per-source concurrency caps + backoff + jitter (wraps coordination's global bucket)
  normalize.py                # (doc only) reference asinh/zscale/continuum recipes for the card
  sources/
    base.py                   # DataSource ABC: load_targets -> fetch_candidates -> match -> emit
    lsdb_mmu.py               # NEW: generic lsdb/HATS reader for hf://UniverseTBD/mmu_* (margin cache, col pushdown)
    ztf.py                    # S3 HATS (pyarrow + anonymous S3)
    ps1.py legacy.py galex.py twomass.py unwise.py galah.py   # custom cutout/VO/HTTP readers (kept)
    apogee.py sdss.py         # local pre-staged FITS readers (kept)
scripts/                      # thin CLI entrypoints (one responsibility each)
  build_seed_catalogs.py
  plan_work_units.py
  probe_sources.py            # NEW: measure latency/throughput/rate-limits/coverage per source
  run_source_shard.py
  finalize_shard.py
  finalize_release.py
  validate_release.py
  verify_markers.py           # NEW: release-level DONE-marker audit (gates upload)
  false_match_report.py
  upload_hf.py
slurm/                        # *.sbatch array wrappers (CPU partitions)
  build_seeds.sbatch run_source_array.sbatch finalize_array.sbatch
  validate_release.sbatch upload_hf.sbatch probe_sources.sbatch
tests/                        # unit tests incl. TEST_MODE end-to-end on ~5 objects/pop
```
Output tree (deterministic; content-hash in metadata, not paths):
```
release/v5/
  seeds/population={star,galaxy,agn}/seed.parquet         # + global_object_id
  manifest/work_units.parquet                             # the single authoritative manifest
  claims/<manifest_hash>/<row>.claim                      # only if dynamic claiming is enabled
  source=<name>/population=<pop>/shard=000123.parquet(.done.json)
  final/population=<pop>/shard=000123.parquet(.done.json)
  release/manifest.parquet  +  payloads/*.tar (WebDataset)  +  README.md
```

## 2. Data flow (seven phases, each resumable)
```
probe_sources (once, ahead of time)
build_seed_catalogs -> plan_work_units -> run_source_shard (array)
   -> finalize_shard (array) -> finalize_release + validate_release + verify_markers
   -> upload_hf (single designated uploader)
```

## 3. Script contracts

### `scripts/probe_sources.py`  (NEW — "test pings, done right")
- **Does:** for each source in the Data Access Matrix (doc 01 App. A), measure **median
  latency, sustained throughput, observed rate-limit/429 behavior, total volume estimate,
  and HEALPix/HATS coverage** against a small seed sample. For MMU/lsdb sources, confirm
  `hf://UniverseTBD/mmu_*` partitions open and report per-pixel download time.
- **Out:** `probe_report.json` + a Markdown table; feeds per-source concurrency caps and
  the global token-bucket rates. **Run before committing cluster hours.**
- **CLI:** `--sources all --sample 2000 --out probe_report.json`.

### `scripts/build_seed_catalogs.py`
- **In:** APOGEE allStar (SNR>50, Gaia-XMatch-verified, PM), DESI PROVABGS, SDSS DR16Q.
- **Does:** build per-population seeds; propagate stars to **J2016.0** via
  `apply_space_motion`; galaxies/AGN PM=0; compute `global_object_id`
  (`ids.assign_global_id`); **assert uniqueness** (fail on collision); keep native IDs.
- **Out:** `seeds/population=*/seed.parquet` (versioned). Cached/idempotent.

### `scripts/plan_work_units.py`
- **Does:** split each seed into deterministic row-shards (`--shard-size 50000`); cross
  with the active source list → emit the **single authoritative** `manifest/work_units.parquet`
  (`population, source, shard_id, row_start, row_end, n_objects, seed_path, output_path`)
  + a recorded `manifest_hash`.
- **CLI:** `--seeds ... --sources <from probe> --shard-size 50000 --out manifest/`.

### `scripts/run_source_shard.py`  (the SLURM array workhorse)
- **In:** one manifest row, selected by **partition-aware** indexing:
  `--task-id $SLURM_ARRAY_TASK_ID --partition-id k --num-partitions K` → processes rows
  where `row_index % K == k` (disjoint by construction; doc 01 §3.2). Optional
  `--claim` enables the dynamic ledger via `coordination.claim(row)`.
- **Does:** load that seed shard; for MMU sources use `sources.lsdb_mmu` (PixelSearch +
  column pushdown + margin cache); for others the custom reader. Propagate seed →
  source epoch (`apply_space_motion`), match via `matching.match(...)`
  (`search_around_sky` or lsdb `n_neighbors>1` + adjudication + per-source radius), emit
  typed rows `{global_object_id, <prefixed cols>, *_match_sep_arcsec, match_ambiguous,
  n_candidates_within_radius}`.
- **Out:** one shard via `io.atomic_write` + `io.write_done(...)` (checksum + provenance).
  **Skips if a `complete` DONE marker matches** current manifest/schema/code + checksum
  (doc 01 §3.2 state machine).
- **Concurrency:** acquire a token from `rate_limit`/`coordination` before each remote
  call (caps respected **globally** across submissions). Local/S3/HATS: high; remote
  services: 2–8 + backoff. **Never** fan CDS XMatch/Vizier across the array.

### `scripts/finalize_shard.py`  (array, per seed shard)
- **Does:** for one `(population, shard_id)`, join all `source=*/.../shard=<id>` on
  `global_object_id`; dedup (closest *after* adjudication); compute modality &
  **instrument** counts; **enforce ≥2 instruments**; assign HEALPix(nside=8) split; write
  `final/...` against the versioned schema (+ DONE marker w/ checksum).
- **Note:** per-shard join ⇒ no global shuffle, bounded memory.

### `scripts/finalize_release.py`
- Aggregate final shards into `manifest.parquet`; pack heavy payloads into WebDataset
  `payloads/*.tar`; emit counts + dataset-card inputs.

### `scripts/validate_release.py` + `scripts/verify_markers.py` + `scripts/false_match_report.py`
- `validate_release`: streaming QA (schema conformance, `global_object_id` uniqueness,
  coord sanity, sampled array shapes, modality/instrument coverage, split integrity,
  typed-missingness, PM sanity).
- `verify_markers` **(NEW, gates upload):** independently re-derive that **every**
  manifest row is `complete` — final+marker present, manifest/schema/code hashes match,
  **file checksum matches the marker**, no orphan `.tmp` files, no suspicious/stale/
  corrupt states. This is the "is *done* actually done?" audit (doc 01 §3.2).
- `false_match_report` **(rewritten to doc 01 §3.4.1):** **mirrors the production matcher**
  (incl. `apply_space_motion`); **Monte-Carlo random-direction offsets** scaled to the
  confusion scale (`radius ⊕ μ·Δt ⊕ ϖ ⊕ σ_astro`); a **PM-direction-scramble null**;
  stratified by **PM-bin × density × |b|**; plus the analytic background cross-check.
  Writes CSV + card table; **gates upload** on per source × pop × PM-bin × |b|-bin thresholds.

### `scripts/upload_hf.py`  (single designated uploader)
- `HfApi.create_repo(exist_ok=True)` → **`upload_folder(release/v5/release)`** → write
  README/card → **verify** via `load_dataset(..., streaming=True)` + `hf_hub_download`
  spot-check. Idempotent. **Run once, by one user, after `verify_markers` passes** — never
  two uploaders against one repo concurrently (doc 01 §3.2).

## 4. SLURM pattern (CPU, arrays, multi-submission-safe, throttled)
```bash
# probe first (cheap), then build seeds
sbatch slurm/probe_sources.sbatch
sbatch slurm/build_seeds.sbatch
# fan out: one array task per manifest row. %50 caps *this job's* concurrency;
# the global per-service token bucket caps the SUM across all concurrent submissions.
# Two users co-run by passing disjoint partitions:
sbatch --array=0-$((N-1))%50 slurm/run_source_array.sbatch   # user A: --partition-id 0 --num-partitions 2
sbatch --array=0-$((N-1))%50 slurm/run_source_array.sbatch   # user B: --partition-id 1 --num-partitions 2
# finalize per shard, then audit + validate, then a single upload
sbatch --array=0-$((S-1))%64 slurm/finalize_array.sbatch
sbatch slurm/validate_release.sbatch && sbatch slurm/upload_hf.sbatch
```
- **Partition (CPU, not GPU):** request a CPU partition; do **not** request `--gpus`
  (doc 01 §3.8 — the data path never uses one). If only GH200 nodes are available, omit
  `--gpus`, request many CPUs, and pack array tasks per node. Note the **aarch64** env.
- `run_source_array.sbatch` body:
  `python -m scripts.run_source_shard --manifest ... --task-id $SLURM_ARRAY_TASK_ID --partition-id $PID --num-partitions $K`.

## 5. Type/memory safety (pragmatic)
- `@dataclass(frozen=True, slots=True)` config; parse/validate once; **no pydantic**.
- `mypy` strict on `config/ids/matching/coordination/schemas/io`; lighter on messy I/O
  adapters; `numpy.typing.NDArray` where shapes matter.
- Runtime validation **only at boundaries** (after raw read, before shard write, before
  marker write, before upload).
- Memory: row-shard ownership + streaming + narrow dtypes + Arrow nested types (no pandas
  object payloads); pandas only for small table ops, `pyarrow` for write paths.

## 6. Keep / rewrite from v4
- **Keep:** survey fetch/query logic in `sources/*` (ZTF S3, cutout/VO readers, local
  APOGEE/SDSS); seed selection; HEALPix pre-filter & column projection; the
  shifted-catalog *concept*; `validate_dataset.py` checks.
- **Rewrite:** identity (`ids.py`); matching (`matching.py`, + lsdb for MMU);
  **orchestration** (single manifest + partition-aware SLURM arrays + `coordination.py`);
  **resume** (integrity-checked DONE markers + `verify_markers.py`); MMU access
  (`lsdb_mmu.py` over `hf://`, replacing raw Flatiron HTTP); finalize (per-shard); upload
  (`upload_hf.py`, single uploader); storage (typed schema + WebDataset); **false-match
  test** (doc 01 §3.4.1 — production-mirroring, motion-aware).

## 7. Build order (matches methodology P0→P2)
**P0:** `probe_sources`; seed row-shard ownership + partition-aware indexing;
`global_object_id`; `apply_space_motion`; radius search/lsdb + adjudication; atomic
writes + checksummed DONE + `verify_markers`; real upload.
**P1:** per-shard finalize; explicit pyarrow schema; per-source radii; lsdb `lsdb_mmu`
reader + margin cache; motion-aware false-match report; global token bucket.
**P2:** HATS/LSDB layouts for heavy local catalogs; WebDataset payloads; HSC/JWST image
adds; richer ambiguity/footprint QC; dynamic claim ledger (if needed).

## 8. Acceptance criteria
- `probe_sources` report exists and per-source caps are set from it.
- TEST_MODE end-to-end (≈5 objects/pop) passes in CI.
- Re-running any array with all DONE markers present is a **no-op**; corrupting a final
  file (checksum mismatch) forces a re-run.
- Two concurrent submissions with disjoint `--partition-id` produce no duplicate/clobbered
  outputs and respect global service rate limits.
- Every released object has `n_instruments_present ≥ 2` and a unique `global_object_id`.
- `verify_markers` green; `validate_release` green; motion-aware false-match report under
  threshold; HF load-back verified by the single uploader.
