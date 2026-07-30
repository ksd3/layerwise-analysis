# Worked Walkthrough: How the `v4/OmniSky` Pipeline Works

> **Audience:** a junior researcher who wants to learn how to build and cross-match
> large multimodal astronomical datasets *performantly*. This explains the existing
> `layerwise-analysis/v4` code — what each piece does, **why** it is shaped that way,
> and which patterns are worth stealing. Critiques and fixes live in
> [`01-methodology.md`](01-methodology.md); this doc is "how it works today."

## 0. The one-paragraph mental model

No observatory stores *the same object's* IR spectrum, UV image, and optical light
curve together — each survey only knows its own observations, indexed by its own IDs
at its own epoch. So you **build** the joined dataset in four moves:

1. **Seed.** Pick a reference catalog per population (stars, galaxies, AGN) giving a
   clean list of real objects with good coordinates. Defines *which objects exist*
   and is the source of truth for `object_id`, `ra`, `dec`, and (for stars) proper
   motion.
2. **Fan out & match.** For every other survey ("source"), find which of its
   observations land on your seed objects on the sky, and attach them. Each source
   writes its matches to its own shards.
3. **Merge.** Join all per-source shards back onto the seed catalog by `object_id`,
   one population at a time, in memory-bounded chunks.
4. **Validate & upload.** Check correctness, then push to HuggingFace.

```
   SEED CATALOGS          PER-SOURCE FAN-OUT            MERGE          SHIP
 stars(APOGEE+Gaia)   flatiron(Gaia/TESS/DESI)      finalize      validate +
 galaxies(PROVABGS) → apogee/sdss/galah (spectra) → (chunked   →  false-match
 agn(SDSS DR16Q)      ztf (light curves)             join by      test + push
                      legacy/galex/2mass/unwise(img) object_id)   to HuggingFace
```

**Why "seed-and-match"?** Pairwise all-to-all matching of *N* surveys is *N²* joins
with no natural "object" to anchor on. Anchoring on a seed catalog makes it *N*
one-to-many matches, gives every row a stable identity, and lets you reason about
completeness. This is the single most important design decision.

## 0.5 What v5 changes (read alongside doc 01)

This doc explains v4 *as it works today*. Five things below are **superseded** by the
hardened v5 design — flagged here so you don't copy a known-fragile pattern:

- **False-match test has no proper motion.** `false_match_test.py` shifts the seed
  **+30″ in RA only** and matches **un-propagated** positions with nearest-only — i.e.
  it benchmarks a *different, simpler matcher than production*, and 30″ is comparable to
  a high-PM star's multi-decade displacement. v5 replaces it with a motion-aware protocol
  (doc 01 §3.4.1): production-mirroring, Monte-Carlo random-direction offsets scaled to
  the real+apparent-motion confusion scale, a PM-direction-scramble null, stratified by
  PM × density × |b|.
- **Resume is `is_cached()` = "shard dir has ≥1 parquet".** A source that crashes after
  writing 40 of 200 shards is treated as *complete and silently skipped* — an incomplete
  dataset with no error. v5 uses integrity-checked DONE markers (checksum + provenance) +
  a `verify_markers` audit (doc 01 §3.2).
- **MMU data is fetched as raw HTTP** off `users.flatironinstitute.org`. v5 reads it via
  **lsdb/HATS over `hf://datasets/UniverseTBD/mmu_*`** (partition + column pushdown, margin
  cache; doc 01 §3.3). The `TobiasPitters/mmu-crossmatch` Space is the reference for that
  access pattern — but note it is a *single-pixel, two-catalog, positional-only (no PM)
  lsdb demo*, a template, **not** a pipeline.
- **HF upload is `create_repo`-only** — the data was **never actually uploaded**. v5 adds a
  real `upload_folder` + load-back verification, run by a single designated uploader.
- **Single-node `ThreadPoolExecutor`** orchestration. v5 is SLURM **arrays on CPU**,
  safe for concurrent multi-user submissions on a shared filesystem (doc 01 §3.2, §3.8).

## 1. Match on *coordinates*, never on *names*

Different catalogs name the same object differently, and objects move. The only
reliable shared key is *position on the sky at a given time*. So the join key is
`(ra, dec)` with a tolerance ("within 3″ → same object"). The seed `object_id`
(APOGEE_ID, PROVABGS id, SDSS_NAME) is the **row key for the merge**, not the match
key. The match is purely positional. (Methodology adds a coordinate-derived global
ID so identity doesn't depend on which survey seeded the object.)

## 2. The cross-match core — `pipeline/utils/crossmatch.py`

Three ideas: angular matching, epoch propagation, HEALPix pre-filtering.

### 2.1 `match_to_catalog_sky`
```python
idx, sep, _ = catalog_coords.match_to_catalog_sky(tgt_coords)
mask = np.array(sep < radius * u.arcsec)
```
`A.match_to_catalog_sky(B)` builds a **k-d tree** over B and, for each element of A,
returns the nearest B plus separation. It is **nearest-only**: it always returns
*a* match (even 50″ away), so you must apply the `sep < radius` cut yourself; and it
does not guarantee the match is mutual. The k-d tree makes each lookup *O(log M)* —
the difference between seconds and hours at millions of rows.

### 2.2 The two matching *directions* (read twice)
- **`crossmatch_to_catalog()` (flatiron): catalog → target.** For each seed object,
  nearest target. Output arrays are catalog-length.
- **`ztf.py`: target → catalog.** For each ZTF object in a tile, nearest seed; keep
  only ZTF rows within radius. **Why flip?** Streaming a 5k-object tile against a
  1.5M-row catalog this way keeps working arrays tile-sized, not catalog-sized, and
  discards the 99% of tile objects that aren't yours. Direction choice is a
  memory/throughput optimization. Cost: many-to-one (dedup-by-closest fixes it).

### 2.3 Epoch propagation — `propagate_coords()` (the "stars move" fix)
Nearby stars drift arcsec/decade. Match a 2016 Gaia position against a 1999 image
and you grab a neighbor. So propagate seed positions to each survey's epoch first:
```python
dt = target_epoch - ref_epoch
ra  += (pmra  / 3.6e6) / cos_dec * dt   # mas/yr → deg/yr, de-projected
dec += (pmdec / 3.6e6) * dt
```
- `pmra = μ_α·cos(δ)` (Gaia convention); dividing by `cos_dec` converts on-sky motion
  to a ΔRA *coordinate* increment. Forgetting this is a classic bug.
- `3.6e6` = mas per degree. NaN PM → 0 (correct for extragalactic; safe for stars).
- `SURVEY_EPOCHS` (config.py) hard-codes each survey's mean epoch. That table is what
  makes matching honest.
- **Good but not best:** this is flat/linear. Methodology recommends astropy
  `SkyCoord.apply_space_motion()` (rigorous spherical motion) instead.

### 2.4 HEALPix pre-filtering — the big throughput win
```python
cat_healpix = compute_catalog_healpix(ra, dec, nside=16)
filtered    = filter_healpix_cells(available_cells, cat_healpix)
```
Surveys stored partitioned by HEALPix cell (dirs named `healpix=1234/`) let you
**skip downloading cells your catalog never touches** — routinely 5–50× less I/O for
a sparse seed against an all-sky survey. The highest-leverage perf trick here, and it
works *because* your catalog and the remote data share the same HEALPix scheme. Risk:
the helper silently falls back to *no filtering* if `healpy`/`astropy_healpix` are
missing (methodology makes this a hard dependency).

## 3. Performant data-access patterns (steal these)

- **A — Spatial pre-filter before download** (`flatiron`, `ztf`): only fetch
  partitions intersecting your catalog's HEALPix cells. *Don't fetch sky you don't need.*
- **B — Column projection / two-pass read** (`ztf`): read cheap position columns
  (~7 MB), match, then read the expensive nested light-curve column only for matched
  rows. Columnar formats let you read one column without the rest. Move megabytes,
  not gigabytes.
- **C — Streaming shard flush** (everywhere): buffer rows; at `shard_size` (5000)
  write a Parquet shard and clear. Memory stays bounded at ~one shard.
- **D — Parallel I/O with thread pools**: downloads are I/O-bound, so threads help
  even under the GIL. Note `ps1.py`'s comment *"requests.Session is NOT thread-safe"*
  — each thread makes its own request. The detail that separates demo code from code
  that survives a 16-hour run.
- **E — Caching/resume by shard dir** (`base.py`): a source with existing shards is
  skipped. Coarse (whole-source granularity); methodology replaces it with per-unit
  DONE markers.
- **F — Reverse-match to bound memory** (§2.2).

**Meta-lesson:** performance here is rarely clever math. It's (1) not fetching data
you'll discard, (2) keeping memory flat, (3) overlapping network waits. Master those
three and you can build million-object datasets on one node.

## 4. The source abstraction — `pipeline/sources/base.py`
Every survey is a `DataSource` subclass with one required method `fetch(catalog_df)`.
The base provides `run()` (caching wrapper), `save_shard()`, `load_shards()`, and a
cheap `preflight()` reachability check. Clean plugin pattern: adding a survey = one
file that (a) finds its data, (b) propagates the catalog to its epoch, (c) matches,
(d) emits `{object_id, <prefixed columns>}`. The prefix (`ztf_`, `apogee_`, `legacy_`)
namespaces columns so they never collide in the wide table.

Representative sources: `flatiron.py` (HEALPix dirs, parallel cells, catalog→target);
`ztf.py` (S3 HATS Parquet, column projection, reverse match); `ps1.py` (cutout web
service, thread-per-cutout, raw `(3,64,64)` float32); `apogee.py` (local FITS,
SNR-deduped lookup, fixed good-pixel crop). Note `apogee.py` crops to a fixed 7514-px
"good pixel" set so every spectrum is identical length — fix array lengths early and
the downstream ML is far simpler.

## 5. Seed catalogs — `pipeline/catalogs/`
Three quality strategies: **stars** (`stars.py`) start from APOGEE allStar, keep
SNR>50, dedup to highest-SNR per star, then **verify each exists in Gaia DR3 via CDS
XMatch** and inherit Gaia proper motions; fails loud if XMatch returns nothing.
**galaxies** (`galaxies.py`) reuse the DESI PROVABGS HDF5 cells, `pm=0`. **agn**
(`agn.py`) use SDSS DR16Q with `z>0.01`, with truncation checks. Each caches its
`<pop>_catalog.parquet`. Different methods because "what counts as a real,
well-localized object" differs by population — the seed step encodes that judgment.

## 6. Merge & finalize — `pipeline/finalize.py`
Joins per-source shards onto each population catalog. The hard part is memory
(700k stars × 7514-float spectra × several surveys = hundreds of GB as lists). So:
- **One population at a time, chunks of 50k** (`MERGE_CHUNK`); `_load_shards_filtered`
  loads only the chunk's `object_id`s from each source. Peak memory = one chunk.
- **Dedup = keep closest:** sort by `{source}_match_sep_arcsec`, drop_duplicates keep
  first.
- **Modality bookkeeping:** `n_spectra/n_lightcurves/n_images/n_modality_types` per
  object — what lets a user filter to "≥2 instrument types."
- **Spatial split:** `_add_split` assigns train/val/test by **HEALPix cell (nside=8)**,
  not per-object-random — nearby objects share extinction/depth, so random splits leak.

## 7. Parquet I/O — `pipeline/utils/parquet.py`
`save_shard` writes `NNNNN.parquet` after `make_parquet_safe` turns numpy arrays into
lists (`list<float>`). Simple and portable (hence `np.array(row["twomass_j"].tolist())`).
Cost (flagged): object-dtype list columns are the least efficient Arrow encoding for
fixed-shape arrays — methodology moves 64×64 images and 7514-px spectra to Arrow
`fixed_size_list`.

## 8. Validation — `validate_dataset.py` + `false_match_test.py`
Both **streaming** (one shard in RAM).
- `validate_dataset.py`: counters-only single pass — required columns, population
  counts, coordinate sanity, modality coverage, sampled array shapes (images 64×64,
  APOGEE 7514), no string-encoded arrays, no cross-population contamination, headline
  multimodal coverage. Run before every upload.
- `false_match_test.py` (**learn this trick**): a cross-match always *produces*
  matches — are they real? **Shift the whole catalog by 30″ and re-run.** No real
  counterpart can line up at 30″, so any surviving matches are spurious: the shifted
  match rate **is your false-match rate**. Broken down by galactic latitude (crowded
  `|b|<15°` fields coincide far more). The 0.02% mean / 0.06% worst-case numbers are
  what justify trusting the dataset. Ship a shifted-catalog number for every
  cross-matched catalog you build — it's the cross-matching unit test.

## 9. Orchestration — `run_pipeline.py` + `pipeline/runner.py`
Sequences phases: preflight → build catalogs → Flatiron/APOGEE/SDSS → the rest under
a single-node `ThreadPoolExecutor(max_workers=6)` (one thread per *source*) →
finalize. `from_env()` reads everything from env vars; `TEST_MODE=1` shrinks every
population so you can run the whole pipeline locally in a minute before burning
cluster hours. **This is what changes most for cluster scale** — see methodology
§Concurrency and the spec.

## 10. Quick map: solid vs. scratch
| Area | State | Note |
|---|---|---|
| Seed-and-match architecture | **Solid** | Keep. Right backbone. |
| `match_to_catalog_sky` + radius cut | **Solid** | Keep; add `search_around_sky` for dense fields. |
| HEALPix pre-filter + column projection | **Solid, excellent** | The core perf wins. |
| Proper-motion propagation | **Works, hand-rolled** | Swap for `apply_space_motion`. |
| Streaming shards + chunked merge | **Solid** | Good memory discipline. |
| Shifted-catalog false-match test | **Solid, excellent** | Run per-source; gate uploads. |
| Resume by source-dir caching | **Scratch / coarse** | Replace with per-unit DONE markers. |
| Single-node ThreadPool orchestration | **Scratch for cluster** | Re-architect to SLURM shard tasks. |
| `object_id` = survey-native | **Scratch** | Add coordinate-derived global ID. |
| HF upload | **Missing** | Only `create_repo`; real `push`/`upload_folder` not wired. |
| Array storage as list columns | **Works, inefficient** | Use Arrow `fixed_size_list`. |
| Normalization (HDR images) | **Absent (raw stored)** | Intentional — document recipes. |
| Type/memory safety | **Partial** | Add typed pyarrow schema + boundary validation. |

### TL;DR
Seed on a clean catalog and match everything else by **sky position at the right
epoch**, never by name. Performance = don't fetch what you'll discard (HEALPix +
column pre-filter) + stream so memory stays flat + thread the network waits. Correct
the epoch or you match the wrong star. Prove matches with a shifted-catalog test;
split by sky region. Keep raw values; normalize at train time, per modality.
