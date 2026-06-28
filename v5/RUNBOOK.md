# OmniSky v5 data-collection runbook

This runbook describes the cluster-agnostic data-collection path for OmniSky v5.
It works locally for dry runs and on any SLURM cluster with Python, shared storage,
and outbound access or pre-staged catalogs. Delta/DeltaAI are examples, not
requirements.

## 0. Operator contract

The pipeline has three modes:

1. **Local dry run** — proves LSDB/HATS access and stores a bounded sample locally.
2. **TEST_MODE end-to-end** — runs the full orchestration with synthetic data and no
   network-heavy sources.
3. **Live data collection** — builds real seeds, plans source shards, runs source
   arrays, finalizes shards, validates, and uploads or dry-runs upload.

Do not claim a live release is validated until the live LSDB/HF/S3/local-file paths
have been run on the target compute environment and `validate_release.py` passes.

## 1. Choose and describe the execution environment

Set site-specific values with environment variables or `sbatch` flags. The wrappers
do not hardcode account, partition, QOS, or a Delta path.

Common variables:

| Variable | Required | Meaning |
| --- | --- | --- |
| `RELEASE_ROOT` | yes | Writable release directory for seeds, shards, markers, manifests, and final output. |
| `MANIFEST` | for array/finalize/validate | Path to `work_units.json`. |
| `SMITH42_REVISION` | for Phase 0 probe | Immutable Smith42 HF dataset commit, e.g. `93d0fddf8c5b61028ee0b6d72fd0dbfa87b38624`. |
| `HF_REPO` | for upload | Hugging Face repo id, e.g. `UniverseTBD/omnisky-v5`. |
| `OMNISKY_ENV_ACTIVATE` | optional | Shell snippet to activate the runtime, e.g. `source .venv/bin/activate`. |
| `OMNISKY_CONDA_ENV` | optional | Conda env name; defaults to `omnisky`. |
| `OMNISKY_PYTHON_BIN` | optional | Python executable after activation; defaults to `python`. |
| `OMNISKY_RELEASE_ROOT` | optional | Python config default for release root; CLI `--release-root`/`RELEASE_ROOT` still wins. |
| `OMNISKY_LEGACY_ROOT` | optional | Pre-staged Legacy HDF5 root for `legacy_hdf5`. |
| `OMNISKY_SDSS_DR16Q_ROOT` | optional | Pre-staged SDSS DR16Q root for local FITS fallback. |

For SLURM account/partition, prefer scheduler-provided variables or flags:

```bash
export SBATCH_ACCOUNT=my_allocation
export SBATCH_PARTITION=cpu
# Optional site knobs:
export SBATCH_QOS=normal
export SBATCH_CONSTRAINT=x86_64
```

Equivalent inline form:

```bash
sbatch --account=my_allocation --partition=cpu slurm/probe.sbatch
```

If your only queue allocates GPUs, it can still run the CPU data path. Keep array
concurrency conservative, request enough CPUs/memory, and avoid claiming GPU speedup
unless a downstream analysis stage actually uses GPUs.

## 2. Create the Python environment

Preferred conda path:

```bash
cd /path/to/layerwise-analysis/v5
conda env create -f environment.yml
conda activate omnisky
python -c 'import lsdb, pandas; print("env ok")'
```

Existing virtualenv path:

```bash
cd /path/to/layerwise-analysis/v5
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install lsdb hats astropy astropy-healpix pyarrow pandas h5py s3fs huggingface_hub datasets astroquery dask distributed requests pytest
python -c 'import lsdb, pandas; print("env ok")'
```

For SLURM wrappers using a non-conda env:

```bash
export OMNISKY_ENV_ACTIVATE='source /path/to/layerwise-analysis/v5/.venv/bin/activate'
export OMNISKY_PYTHON_BIN=python
```

## 3. Run local bounded LSDB dry run

Use this before any large cluster job. It verifies LSDB import, catalog opening,
pixel filtering, local materialization, storage, and optionally one tiny crossmatch.

```bash
cd /path/to/layerwise-analysis/v5
python -m scripts.local_lsdb_dry_run \
  --sources desi,hsc \
  --cone 150.1,2.2,600 \
  --columns ra,dec \
  --max-bytes 1GB \
  --out-dir /tmp/omnisky-lsdb-dry-run \
  --crossmatch
```

Outputs:

- `source=<name>/order=<order>/pixel=<pixel>.parquet` or `.jsonl`
- `local_lsdb_dry_run_report.json`

Interpretation:

- `stored_bytes` should be positive.
- `fetches[*].fetched_rows` should be positive for at least one source/pixel.
- `crossmatch.attempted=true` means the LSDB crossmatch API path ran.
- A byte cap limits stored bytes, not peak LSDB/Dask memory. Keep columns and pixels
  narrow for local tests.
- If a pixel run reports `error: no_coverage`, use a cone search in a known footprint
  first. The COSMOS cone above has been used as the default DESI/HSC smoke region.

If the canonical HF URI is unavailable, override it with local HATS paths:

```bash
python -m scripts.local_lsdb_dry_run \
  --sources desi,hsc \
  --cone 150.1,2.2,600 \
  --catalog desi=/staged/hats/mmu_desi_edr_sv3 \
  --catalog hsc=/staged/hats/mmu_hsc_pdr3_dud_22.5 \
  --max-bytes 1GB \
  --out-dir /tmp/omnisky-lsdb-dry-run
```

## 4. Phase 0 probe: network and concordance gate

Purpose: verify source reachability, throughput, one-pixel DESI×HSC LSDB crossmatch,
and Smith42 concordance before scaling out.

Local/direct command:

```bash
cd /path/to/layerwise-analysis/v5
export SMITH42_REVISION=93d0fddf8c5b61028ee0b6d72fd0dbfa87b38624
python -m scripts.probe_sources --order 4 --pixel 257 --out probe_report.json
python -m scripts.probe_crossmatch --order 4 --pixel 257 \
  --smith42-revision "$SMITH42_REVISION" \
  --out crossmatch_probe.json
python -m scripts.check_phase0_gate --probe probe_report.json --crossmatch crossmatch_probe.json
```

SLURM command:

```bash
export SBATCH_ACCOUNT=my_allocation
export SBATCH_PARTITION=cpu
export SMITH42_REVISION=93d0fddf8c5b61028ee0b6d72fd0dbfa87b38624
sbatch slurm/probe.sbatch
```

Accept only if:

- Probe output exists and has explicit reachability/throughput verdicts.
- Crossmatch output exists.
- `check_phase0_gate.py` exits zero.
- Any unreachable service has a documented pre-stage plan before live scale-out.

## 5. Build seed catalogs

Seeds define the objects to collect source modalities for. For local orchestration
tests, use `TEST_MODE`; for live runs, provide curated input CSVs with `ra` and `dec`.

Synthetic smoke seed:

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5-smoke
python -m scripts.build_seed_catalogs \
  --population galaxy \
  --release-root "$RELEASE_ROOT" \
  --out unused \
  --test-mode \
  --n 10
```

Live seed from CSV:

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5
python -m scripts.build_seed_catalogs \
  --population galaxy \
  --release-root "$RELEASE_ROOT" \
  --out unused \
  --input-csv /staged/seeds/galaxies.csv
```

SLURM form:

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5
export POPULATION=galaxy
export INPUT_CSV=/staged/seeds/galaxies.csv
sbatch slurm/build_seeds.sbatch
```

Repeat per population (`galaxy`, `star`, `agn`) with population-specific input and
source choices.

## 6. Plan source work units

The manifest is the authority for source×shard work. It embeds an inputs hash so a
rerun can safely attach to identical work and reject conflicting work.

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5
python -m scripts.plan_work_units \
  --sources desi,hsc,legacy \
  --population galaxy \
  --n-objects 0 \
  --shard-size 50000 \
  --release-root "$RELEASE_ROOT" \
  --out "$RELEASE_ROOT/manifests/galaxy"
export MANIFEST="$RELEASE_ROOT/manifests/galaxy/work_units.json"
```

When `--release-root` seed exists, the script counts seed rows and hashes seed content;
`--n-objects` is a fallback/manual value.

Record:

- source list,
- shard size,
- manifest path,
- manifest hash printed/written in `work_units.json`,
- code SHA used for the run.

## 7. Run source shard arrays

Each array task reads seed rows for one source×shard unit and writes source shards
plus DONE markers. It can run locally for a tiny manifest or as a SLURM array for
scale-out.

Local single task:

```bash
python -m scripts.run_source_shard \
  --manifest "$MANIFEST" \
  --task-id 0 \
  --release-root "$RELEASE_ROOT" \
  --code-sha "$(git rev-parse --short HEAD 2>/dev/null || echo local)"
```

SLURM array:

```bash
N_UNITS=$(python - <<'PY'
import json, os
with open(os.environ['MANIFEST']) as f:
    print(len(json.load(f)['units']))
PY
)

export RELEASE_ROOT=/shared/omnisky/release/v5
export MANIFEST=/shared/omnisky/release/v5/manifests/galaxy/work_units.json
export CODE_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
sbatch --array=0-$((N_UNITS - 1)) slurm/run_source_array.sbatch
```

For multiple users or queues sharing one release, split work deterministically:

```bash
export PARTITION_ID=0
export NUM_PARTITIONS=4
sbatch --array=0-249 slurm/run_source_array.sbatch
```

A second operator can use `PARTITION_ID=1`, etc. Each partition sees disjoint units.

## 8. Finalize per-population shards

After source shards complete for a population, finalization joins modalities for each
object shard and enforces `MIN_INSTRUMENTS`.

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5
export MANIFEST=/shared/omnisky/release/v5/manifests/galaxy/work_units.json
export POPULATION=galaxy
export SOURCES=desi,hsc,legacy
export MIN_INSTRUMENTS=2
sbatch --array=0-<max_shard_id> slurm/finalize_array.sbatch
```

Local equivalent:

```bash
python -m scripts.finalize_shard \
  --release-root "$RELEASE_ROOT" \
  --manifest "$MANIFEST" \
  --population galaxy \
  --sources desi,hsc,legacy \
  --shard 0 \
  --min-instruments 2
```

## 9. Verify markers and aggregate the release

Marker verification catches missing, stale, suspicious, or corrupt shard outputs before
release aggregation.

```bash
python -m scripts.verify_markers \
  --release-root "$RELEASE_ROOT" \
  --manifest "$MANIFEST" \
  --code-sha "$CODE_SHA"

python -m scripts.finalize_release --release-root "$RELEASE_ROOT"
```

The aggregate release writes under `$RELEASE_ROOT/release/`.

## 10. Validate quality gates

Run structural validation and false-match reporting. Treat false-match bins above the
threshold as low-confidence or blocking, depending on the release policy.

```bash
python -m scripts.validate_release \
  --release-root "$RELEASE_ROOT" \
  --min-instruments 2

python -m scripts.false_match_report \
  --out false_match_report.json \
  --threshold 0.001
```

SLURM combined validation:

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5
export MANIFEST=/shared/omnisky/release/v5/manifests/galaxy/work_units.json
export CODE_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo local)
sbatch slurm/validate_release.sbatch
```

## 11. Upload or dry-run upload

Always dry-run first:

```bash
python -m scripts.upload_hf \
  --release-root "$RELEASE_ROOT" \
  --repo UniverseTBD/omnisky-v5 \
  --dry-run
```

Real upload:

```bash
export HF_TOKEN=...
python -m scripts.upload_hf \
  --release-root "$RELEASE_ROOT" \
  --repo UniverseTBD/omnisky-v5
```

SLURM wrapper:

```bash
export RELEASE_ROOT=/shared/omnisky/release/v5
export HF_REPO=UniverseTBD/omnisky-v5
export DRY_RUN=1
sbatch slurm/upload_hf.sbatch
```

Remove `DRY_RUN` only after the dry-run report and validation gates pass.

## 12. TEST_MODE full local orchestration

Use this to verify orchestration changes without network catalogs:

```bash
tmp=$(mktemp -d)
export RELEASE_ROOT="$tmp/release"

python -m scripts.build_seed_catalogs --population galaxy --release-root "$RELEASE_ROOT" --out unused --test-mode --n 5
python -m scripts.plan_work_units --sources desi,hsc --population galaxy --n-objects 0 --release-root "$RELEASE_ROOT" --out "$RELEASE_ROOT/manifests/galaxy"
export MANIFEST="$RELEASE_ROOT/manifests/galaxy/work_units.json"

python -m scripts.run_source_shard --manifest "$MANIFEST" --task-id 0 --release-root "$RELEASE_ROOT" --test-mode
python -m scripts.run_source_shard --manifest "$MANIFEST" --task-id 1 --release-root "$RELEASE_ROOT" --test-mode
python -m scripts.finalize_shard --release-root "$RELEASE_ROOT" --manifest "$MANIFEST" --population galaxy --sources desi,hsc --shard 0 --min-instruments 2
python -m scripts.verify_markers --release-root "$RELEASE_ROOT" --manifest "$MANIFEST" --code-sha local
python -m scripts.finalize_release --release-root "$RELEASE_ROOT"
python -m scripts.validate_release --release-root "$RELEASE_ROOT" --min-instruments 2
python -m scripts.upload_hf --release-root "$RELEASE_ROOT" --repo UniverseTBD/omnisky-v5 --dry-run
```

## 13. Cluster-specific notes

### Generic SLURM

- Use `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `SBATCH_QOS`, and `SBATCH_CONSTRAINT` or
  equivalent `sbatch` flags.
- Keep `RELEASE_ROOT` on shared storage visible to all array jobs.
- Keep local pre-staged datasets under paths exported via `OMNISKY_LEGACY_ROOT` and
  `OMNISKY_SDSS_DR16Q_ROOT`.

### Delta x86 CPU

- Good fit for data generation because it avoids idle GPU allocation.
- Use the site account/partition via `SBATCH_*`; do not edit wrappers permanently.

### DeltaAI / GH200-only queues

- Data collection remains CPU-oriented even if the scheduler allocates a GPU.
- Pack array tasks conservatively; the expensive resource is usually network/object-store
  throughput or shared filesystem pressure, not GPU compute.
- If Python wheels differ on aarch64, build the env on the target architecture and run
  the local LSDB dry run before submitting large arrays.

### No outbound compute-node internet

- Run `probe.sbatch` or direct probes first.
- If HF/S3/CDS access fails, pre-stage required catalogs via the site-approved transfer
  mechanism, then use local paths or source-specific env vars.
- For LSDB/HATS sources, prefer local HATS directories and validate with
  `local_lsdb_dry_run.py --catalog SOURCE=/path/to/hats`.

## 14. Failure handling

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `No module named 'lsdb'` | Wrong Python env | Activate conda env or install LSDB in the active venv. |
| `source ... not LSDB` in dry run | Requested non-HATS source | Use only `lsdb_mmu` sources for `local_lsdb_dry_run.py`. |
| Probe internet failure | Compute node lacks outbound access | Pre-stage catalogs and record stream-vs-pre-stage decision. |
| Stale marker | Code SHA/schema/manifest mismatch | Re-run affected shard with intended manifest/code or clean stale output intentionally. |
| Corrupt marker | Output changed after marker write | Re-run shard; do not hand-edit outputs. |
| Empty fetched pixel | Wrong pixel/order or catalog footprint | Try a known populated pixel, narrower source list, or ConeSearch-based investigation. |

## 15. Ship checklist

Before calling the release complete, capture:

- environment creation command and Python version,
- cluster/site name and scheduler options used,
- release root,
- source list and seed provenance per population,
- Smith42 revision,
- dry-run report,
- Phase 0 probe and concordance outputs,
- manifest hashes,
- marker verification output,
- release validation output,
- false-match report,
- upload dry-run output,
- real upload URL and load-back verification if uploaded.
