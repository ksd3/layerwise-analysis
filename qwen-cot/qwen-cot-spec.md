# Design Specification: Qwen CoT x AstroPT Layerwise Analysis

> **Status:** approved design (2026-07-19). Language: ASD-STE100 Simplified Technical
> English, with permitted technical names and technical verbs.
> **Home directory:** `layerwise-analysis/qwen-cot/`
> **Related:** the Platonic Universe paper (arXiv:2509.19453), `05-mechanistic-pivot.md`,
> `docs/superpowers/specs/2026-06-26-omnisky-v5-implementation-design.md`.

---

## 1. Purpose

This project is intended to extend the Platonic Universe paper. This project adds a language
model that shows its reasoning as text, in order to determine whether or not the associations we may
see are intended, or noise.

We collect chain-of-thought (CoT) traces from `Qwen/Qwen2.5-0.5B`. We collect
layerwise embeddings from AstroPTv2. We then measure three connections:

1. The connection between AstroPT embedding clusters and physical labels.
2. The connection between the Qwen rationale text and the physical variables that
   structure the AstroPT embedding space.
3. The connection between Qwen hidden states and AstroPT embeddings, layer by layer.

## 2. Hypotheses

**H1 — Embedding clusters follow physics.** Get the k=10 nearest neighbors of a
galaxy in one AstroPT layer. The neighbors have similar physical labels (redshift,
log stellar mass, sSFR). If H1 is true, a simple map (linear probe) connects the
embedding space to the physical labels.

**H2 — The Qwen rationale follows the same physics.** Give Qwen a target galaxy and
candidate galaxies as text. Qwen selects the most similar candidates and writes a
scratchpad. If the scratchpad gives the physical variables that we confirmed in H1,
then Qwen clusters galaxies with the same variables that structure the manifold.

**H3 — Representational alignment (PRH bridge).** Qwen hidden states and AstroPT
embeddings show mutual-kNN and CKA alignment above a permutation baseline.

## 3. Scope

**In scope:**

- A two-pass CoT capture harness: generation, then teacher-forced replay.
- Arithmetic validation of the harness.
- AstroPT layerwise embedding extraction for three model sizes (S, B, L).
- Neighborhood-physics analysis (H1).
- Two Qwen astronomy task families: property estimation and neighbor selection.
- Cross-model analysis: behavioral (H2), representational (H3), and causal
  (scratchpad interventions).
- A TransformerLens parity test as a gate for future mechanistic work.
- Results assembly: figures, tables, `results.md`, and run manifests.

**Out of scope (documented, not built):**

- Qwen models larger than 0.5B (see section 14, the scale path).
- TransformerLens mechanistic experiments after the parity test.
- Scratchpad paraphrase interventions (a second model is necessary; skip).
- Supervised fine-tuning of Qwen on scratchpads.
- A paper-section draft.

## 4. Locked decisions

| # | Decision | Value |
|---|----------|-------|
| 1 | GPU | One NVIDIA A100 40 GB. All GPU parameters come from the configuration file. Other GPUs are permitted. |
| 2 | GPU budget | Approximately 50 GPU-hours, soft limit. Work divides into jobs of 2 to 4 hours. |
| 3 | Code start point | The `capture.py` harness (supplied). Refactor it into the staged pipeline. |
| 4 | Qwen model | `Qwen/Qwen2.5-0.5B` base checkpoint, pinned to a commit SHA. Not the Instruct variant. |
| 5 | Galaxy text input | Photometry values and spectral features, as text. |
| 6 | Target properties | Redshift, log stellar mass, and sSFR. All three. |
| 7 | Galaxy data source | The OmniSky v4/v5 collection (in progress). A data contract separates this pipeline from the collection. |
| 8 | AstroPT embeddings | Recompute from the released AstroPTv2 checkpoints with the `platonic-universe` extraction code. |
| 9 | Core experiment | The H1/H2 neighborhood-physics design (section 2). No direct-answer control condition. |
| 10 | Causal + parity | Keep the scratchpad interventions and the TransformerLens parity test in scope. |
| 11 | Seeds | 32 sampled seeds plus 1 greedy trace for each item. Divide the work into shard jobs. |
| 12 | Deliverable | A results directory: plots, tables, `results.md`, and manifests. |

## 5. Architecture

The pipeline has six stages. Each stage is one CLI script. Each stage reads and
writes files with a fixed schema. The stages connect only through these files.

```text
Stage 0   data contract + synthetic fixture
Stage 1   arithmetic validation of the harness
Stage 2   AstroPT layerwise embeddings
Stage 3   neighborhood-physics analysis (H1)
Stage 4   Qwen CoT astronomy tasks (A: properties, B: neighbors)
Stage 5   cross-model analysis (H2, H3, causal, parity)
Stage 6   results assembly
```

Stage order for execution: 0, 1, 2, 3, 4, 5, 6. Stage 1 can run in parallel with
stages 2 and 3. Stage 4 task B needs the stage 3 neighbor lists.

### Directory layout

```text
layerwise-analysis/qwen-cot/
├── README.md               pointer to this specification
├── config/
│   └── default.yaml        all tunable parameters
├── prompts/                frozen prompt files, one version each
├── qcot/                   Python package (harness, parsers, metrics, io)
├── scripts/                one CLI entry point for each stage
├── fixtures/               synthetic data generator + sample outputs
└── runs/                   all outputs (excluded from any future VCS)
```

## 6. Configuration

One YAML file holds every tunable value. No script holds a hardcoded value from
this list.

```yaml
model_id: Qwen/Qwen2.5-0.5B
revision: null              # null = resolve and pin the current SHA
dtype: bfloat16
attn_implementation: sdpa
gpu_ram_gb: 40              # batch sizes derive from this value
n_galaxies: 1024
seeds: 32
knn_k: 10
n_candidates: 20            # task B: 10 true neighbors + 10 distractors
full_state_galaxy_subset: 256
intervention_subset: 50
coherence_ratio_threshold: 0.8   # stage 5a: neighbor/random median ratio below this = structuring
probe_r2_threshold: 0.3          # stage 5a: probe R2 above this = structuring
sampling: {temperature: 0.8, top_p: 0.95, top_k: 0}
max_new_tokens: {arithmetic: 256, astronomy: 512}
astropt_checkpoints: [astropt2-s, astropt2-b, astropt2-l]
paths: {data: ..., runs: ..., hf_cache: ...}
```

Batch size rule: the harness computes the token budget for each batch from
`gpu_ram_gb`, the model size, and `max_new_tokens`. The harness logs the computed
batch size in each shard manifest.

## 7. Stage 0 — Data contract and fixture

### The contract

The astronomy stages read one file: `galaxies.parquet`. The OmniSky collection
must supply this file. The fixture generator supplies a synthetic equivalent.

| Column | Type | Description |
|--------|------|-------------|
| `galaxy_id` | string | Stable unique identifier |
| `ra`, `dec` | float64 | Position, degrees |
| `cutout_ref` | string | Path or URI of the Legacy Survey grz image cutout |
| `mag_g`, `mag_r`, `mag_z` | float32 | Photometry, magnitudes |
| `color_gr`, `color_rz` | float32 | Colors, magnitudes |
| `spec_features` | struct | Named spectral features (line equivalent widths, D4000) |
| `z` | float32 | Spectroscopic redshift (label) |
| `logmass` | float32 | Log stellar mass (label) |
| `ssfr` | float32 | Specific star formation rate (label) |
| `label_source` | string | Provenance of the labels |

Only the `spec_features` column can have missing values. The text renderer writes
"unknown" for a missing value.

### The fixture

A seeded generator makes a synthetic `galaxies.parquet` with the full schema and
correlated columns (colors correlate with redshift and mass). It also makes small
random image tensors as fake cutouts. Every downstream stage must run end-to-end
on the fixture. This separates development from the OmniSky collection schedule.

**Held-out split:** stage 0 reserves a fixed, seeded held-out set of galaxies
for the few-shot prompt examples (section 11). The analysis set and the held-out
set do not overlap. The split occurs before stage 3 and stage 4 run.

**Open dependency:** the OmniSky rows must include Legacy Survey grz cutouts, or a
cutout-fetch step becomes necessary. Confirm before the stage 2 real-data run.

## 8. Stage 1 — Arithmetic validation

### Purpose

Show that the capture harness is correct on problems with known intermediate states, before
any astronomy run.

### Changes to `capture.py`

Keep the two-pass design and the record fields. Make these changes:

1. Add a procedural problem generator. Each problem is a start value and 2 to 4
   operations. The generator stores the expected state sequence.
2. Add batched generation. All seeds for one problem form one batch. The prompt is
   identical across the batch, so padding is not necessary.
3. Add batched teacher-forced replay with the chunked log-probability computation.
4. Add a shard runner: each job takes a problem-ID range, writes atomic outputs,
   and can continue after an interruption.
5. Add a metrics module. Set `schema_version` to 2.

### Record schema, version 2 (delta from version 1)

New fields: `task`, `problem_id`, `problem_spec` (operations and expected states),
`prompt_version`, `rendering_version`, `parser_version`, `shard_id`, `batch_size`.
All version-1 fields stay: raw token IDs, decoded text with special tokens, seed,
generation configuration, finish reason, parsed scratchpad, replay log-probabilities,
top-k alternatives, environment metadata.

### Metrics (report all; none is a gate)

| Metric | Diagnoses |
|--------|-----------|
| Scratchpad parse rate | Elicitation format success |
| Final-answer accuracy | Task ability |
| Intermediate-state accuracy | Rationale correctness |
| Trace-length distribution | Truncation or excess length |
| Cross-seed answer consistency | Trajectory distribution width |
| Rationale-answer consistency | Agreement of scratchpad and answer |

A parse failure is an experimental outcome. Do not retry a failed trace. Do not
discard a failed trace.

### Acceptance criteria (harness correctness; these are gates)

- The greedy trace is identical across two runs on the same GPU and software stack.
- Replay log-probabilities are finite for every generated token.
- Hidden-state tensors have the documented shapes.
- A stopped shard job continues without duplicate or lost records.
- The merge tool detects a missing shard.

Budget: 300 problems, 33 traces each. Approximately 1 to 2 GPU-hours.

## 9. Stage 2 — AstroPT layerwise embeddings

Extract per-layer embeddings for the same N galaxies from AstroPTv2 Small, Base,
and Large. Use the `platonic-universe` repository extraction code at a pinned
commit, with its default preprocessing. This keeps the method identical to the
paper.

Outputs, for each model and layer: one float32 tensor `[N, d_layer]` in
safetensors, plus one parquet index that maps row to `galaxy_id`.

Acceptance: correct shapes, finite values, and index alignment with
`galaxies.parquet`. Budget: approximately 2 to 5 GPU-hours for all three models.

## 10. Stage 3 — Neighborhood-physics analysis (H1)

CPU-only. For each AstroPT model and layer:

1. Build the exact k=10 nearest-neighbor lists with faiss (cosine distance).
2. For each label y in {z, logmass, ssfr}: compute the median |Δy| across neighbor
   pairs, and across random pairs. Report the ratio.
3. Compute a permutation p-value for each ratio (1000 permutations). Apply
   Benjamini-Hochberg correction across layers and labels.
4. Fit a ridge regression probe from the layer embedding to each label. Report R²
   with 5-fold cross-validation.

Outputs: `coherence.parquet`, `probes.parquet`, per-layer plots, and the exported
neighbor lists for each layer.

**Anchor layer:** for each AstroPT model, the layer with the highest mean probe R².
Stage 4 task B and stage 5 use the anchor layer of AstroPT-Base as the reference.

Acceptance: outputs exist for every model and layer; the anchor layer is recorded
in the run manifest.

## 11. Stage 4 — Qwen CoT astronomy tasks

### Common rules

- Model: the pinned Qwen2.5-0.5B base checkpoint, as a completion model.
- Prompts: frozen files in `prompts/`, one version string each. A prompt change
  makes a new file with a new version. Traces record the version.
- Text rendering of a galaxy: named fields, fixed order, fixed decimal precision
  (magnitudes and colors: 2 decimals; equivalent widths: 1 decimal), units named,
  `rendering_version` recorded.
- Few-shot examples: manually written solved examples from the stage 0 held-out
  galaxies with true labels. The analysis set does not contain the example galaxies. The examples are
  part of the frozen prompt file.
- Traces: 32 sampled seeds (temperature 0.8, top-p 0.95, top-k disabled) plus 1 greedy,
  batched.
- The harness records a parse failure. The harness does not retry it.

### Task A — Property estimation

One prompt version for each property (z, logmass, ssfr). The model reads one
galaxy as text, writes a scratchpad, and gives a numeric estimate in
`<answer>...</answer>`. The answer parser extracts the number. The metric is
absolute error against the label, plus the stage 1 metric set.

### Task B — Neighbor selection

The prompt shows one target galaxy and 20 candidates, labeled `C01` to `C20`:

- 10 true neighbors: the anchor-layer nearest neighbors from AstroPT-Base.
- 10 distractors: galaxies matched to the target in r-band magnitude
  (|Δmag_r| < 0.5) but distant in the anchor-layer embedding (beyond the 90th
  percentile of pair distances).

The candidate order is a recorded random permutation. The model writes a
scratchpad and answers with 10 candidate labels. The parser extracts the label
list. Magnitude matching prevents a brightness-only shortcut.

### Hidden-state storage tiers

| Tier | Content | Applies to | Size (approx.) |
|------|---------|------------|----------------|
| Pooled | For each layer: mean over the scratchpad span, the last scratchpad token, and the first answer token. bfloat16. | Every trace | ~140 KB per trace; ~19 GB total |
| Full | Every layer, every token position. bfloat16. | Greedy traces of a 256-galaxy subset, each task | ~112 MB per trace; ~120 GB total |

Do not save full states for sampled seeds. Do not save attention maps.

### Sharding

A shard is a galaxy-ID range for one task. A shard job writes `traces.jsonl`,
`pooled.safetensors`, optional `hidden/` files, and a `manifest.json` with counts,
the configuration hash, and a completion flag. Writes go to a temporary directory;
a rename marks completion (the v5 atomic-IO idiom). Shards from different machines
merge by manifest; the merge tool verifies the complete shard set. The design does not require a shared filesystem. Shards move by rsync or a
Hugging Face dataset repo.

Budget: task A approximately 35 to 50 GPU-hours; task B approximately 12 to 18
GPU-hours. This is the dominant cost. Adjustable values: `n_galaxies`, `seeds`, the
property list.

## 12. Stage 5 — Cross-model analysis

### (a) Behavioral (H2)

- **Selection overlap:** precision@10 of the Qwen-selected candidates against the
  AstroPT anchor-layer neighbors. Also compute the curve across all AstroPT layers
  and all three AstroPT sizes. Baseline: random selection of 10 from 20.
- **Variable mentions:** a lexicon parser maps scratchpad text to a canonical
  variable set (colors, magnitudes, redshift, mass, star formation, named spectral
  lines). Output: boolean mention flags per trace.
- **Rationale-manifold consistency:** the overlap between the variables Qwen
  mentions and the variables that stage 3 confirmed as manifold-structuring at the
  anchor layer (coherence ratio and probe R² above the configured thresholds).
  Report the distribution across seeds and galaxies.

### (b) Representational (H3)

Mutual-kNN (k=10) and linear CKA between each Qwen layer (each pooling variant)
and each AstroPT layer, over the N common galaxies. Qwen states come from the task
A greedy traces; a seed-averaged variant is a robustness check. Output: heatmaps
(Qwen layer × AstroPT layer) for each AstroPT size, with a permutation baseline.

### (c) Causal interventions

On the `intervention_subset` (50 galaxies, task A; and 50 arithmetic problems):

| Condition | Operation |
|-----------|-----------|
| clean | The original greedy scratchpad |
| deleted | Remove the scratchpad body |
| truncated | Keep the first half of the scratchpad lines |
| corrupted | Change one intermediate value to a wrong value. If a scratchpad has no extractable intermediate value, record the condition as not applicable and keep the item. |
| transplanted | Use the scratchpad of a different item |

For each condition, run a teacher-forced replay of the context and measure
the change in log P of the original answer tokens. Output: a table and plot of
delta log P per condition. Budget: approximately 2 to 3 GPU-hours.

### (d) TransformerLens parity (gate for future work)

Load the same checkpoint in TransformerLens and in Hugging Face Transformers, both
in float32. Compare next-token logits on 10 fixed token sequences. Gates: top-1
agreement is 100%; the maximum absolute logit difference is below 1e-3. Report the
measured values. No mechanistic experiment runs before this gate passes.

## 13. Stage 6 — Results assembly

`runs/results/results.md` plus figures:

| Figure | Content |
|--------|---------|
| F1 | Arithmetic metric table (stage 1) |
| F2 | Label-coherence ratio vs layer, per label, per AstroPT size |
| F3 | Probe R² vs layer, per label, per AstroPT size |
| F4 | Qwen selection precision@10 vs AstroPT layer, per size, with baseline |
| F5 | Rationale-manifold consistency distribution |
| F6 | MKNN and CKA heatmaps, Qwen layer x AstroPT layer, per size |
| F7 | Intervention delta log P per condition |

Every figure states the run manifests it derives from. The results directory
contains a top-level manifest of all shard manifests.

## 14. Scale path (documented, not built)

To extend to Qwen2.5-1.5B, 3B, or 7B:

1. Change `model_id` in the configuration.
2. The batch-size rule adapts through `gpu_ram_gb`.
3. Layer count and hidden size come from the model configuration, never from
   constants.
4. Storage grows approximately linearly with layer count x hidden size; recompute
   the tier table before a run.
5. All 4 sizes fit on one A100 40 GB in bfloat16 for inference.

## 15. Cost model

| Stage | GPU-hours (approx.) |
|-------|---------------------|
| 1 arithmetic | 1-2 |
| 2 AstroPT embeddings | 2-5 |
| 3 H1 analysis | 0 (CPU) |
| 4A properties | 35-50 |
| 4B neighbors | 12-18 |
| 5c interventions | 2-3 |
| 5d parity | <1 |
| **Total** | **55-75** |

The total is above the 50-hour soft limit at full size (N=1024, 32 seeds, 3
properties). The configuration values (`n_galaxies`, `seeds`, the property list)
reduce it. Jobs are 2 to 4 hours each and resume after interruption.

## 16. Risks and assumptions

| # | Risk / assumption | Mitigation |
|---|-------------------|------------|
| 1 | Base-model elicitation gives a low parse rate on astronomy prompts. | Record it as an outcome. Make a new frozen prompt file for each change. |
| 2 | The OmniSky data is not ready. | The stage 0 fixture makes all development possible. |
| 3 | Cutouts are missing from OmniSky rows. | Add a Legacy Survey cutout-fetch step (out of scope until confirmed). |
| 4 | AstroPTv2 checkpoints are not public or not loadable. | Verify access before stage 2; fall back to embedding dumps from the paper team. |
| 5 | The GPU differs from A100 40 GB. | All GPU parameters derive from `gpu_ram_gb` in the configuration. |
| 6 | Storage overflow from full hidden states. | The tier table caps full states at the 256-galaxy subset. |

## 17. Definition of done

1. All six stages run end-to-end on the fixture, on one A100 40 GB.
2. The real-data run produces `results.md` with figures F1 to F7.
3. Every figure traces to run manifests.
4. The arithmetic acceptance gates (section 8) pass.
5. The parity gate (section 12d) has a recorded result.
