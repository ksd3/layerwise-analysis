# Pipeline Spec — Cross-Matched Multimodal Dataset for the Platonic Universe

Deliverables for building a cluster-scale, cross-matched, multimodal astronomical
dataset (stars + galaxies + AGN; images + spectra + light curves + tabular), uploaded
to HuggingFace, to extend the *Platonic Universe* program (arXiv:2509.19453).

| Doc | What it is | Read it for |
|-----|------------|-------------|
| [`01-methodology.md`](01-methodology.md) | End-to-end methodology (PI review draft, v5) | The *why*: identity, the multi-submission consistency model, lsdb/HATS access, motion-aware matching & false-match protocol, resource/throughput, storage, correctness/bias checklist, Data Access Matrix |
| [`02-script-spec.md`](02-script-spec.md) | The scripts (v5) | Module/CLI layout, data flow, partition-aware SLURM arrays, acceptance criteria |
| [`03-v4-walkthrough.md`](03-v4-walkthrough.md) | Worked walkthrough of the existing `v4/OmniSky` code (with v5-deltas callout) | How performant cross-matching works today, and what v5 supersedes |
| [`04-downstream-analysis-interface.md`](04-downstream-analysis-interface.md) | Side-car: the analysis contract | What the PRH (mutual-kNN/CKA) analysis needs, and why the GPUs live here |

## The decisions baked in (from PI sign-off)
1. **Extend & harden** the existing `v4/OmniSky` pipeline (don't rebuild).
2. **All three populations** (stars + galaxies + AGN); **maximize object count** (match/beat v4's 1.58M).
3. **One cluster (DeltaAI), concurrent multi-user SLURM submissions on shared Lustre** — disjoint partitions by construction; global per-service rate limits.
4. **lsdb/HATS over `hf://UniverseTBD/mmu_*`** for MMU source access & matching; ZTF over S3.
5. **Global deterministic ID** (HEALPix-order-29 int64 at J2016.0, continuous with MMU's native `_healpix_29`) + **store raw** + **documented normalization** at train time.
6. **Data generation + verified HF upload** now; downstream analysis interface as a side-car (doc 04).

## The things that keep the experiment honest
- Correct the **epoch** (proper motion) before every match — or you match the wrong star.
- Match on **coordinates + a stable global ID**, never on names.
- Prove matches with a **motion-aware false-match protocol** (production-mirroring; random-direction Monte-Carlo offsets scaled to real+apparent motion; a PM-direction-scramble null; stratified by PM × density × |b|) — *not* a single fixed 30″ shift.
- **Split by sky region** (HEALPix), never per-object random.
- Enforce **≥2 instruments** per retained object; keep **raw** values; normalize per modality downstream.
- Make "done" mean done: **integrity-checked DONE markers** (checksum + provenance) + a `verify_markers` audit that **gates upload**.
- Cross-matching is **CPU/network-bound** — run data-gen on **CPU** nodes, not GPU; reserve GPUs for the downstream analysis.

## Status / next step
Drafts for review (v5, addressing PI feedback on false-match severity, the multi-cluster
memory/consistency model, CPU/GPU throughput, DONE-marker robustness, and dataset
availability). On approval, implement in **P0 → P1 → P2** order (spec §7): start with
`probe_sources.py`, `ids.py`, `matching.py`, the manifest + partition-aware
`run_source_shard.py` SLURM workhorse, `coordination.py`, atomic writes/checksummed DONE
markers + `verify_markers.py`, and `upload_hf.py`.
