# MMU Layerwise Embeddings and KNN Artifacts - Design Specification

Document date: 2026-07-27.

Status: design for implementation planning and PI review.

Language: controlled technical English for requirements, with standard ML research terms for the method and rationale.

Requirement words:

- **must** states a requirement.
- **can** states a capability.
- **will** states a planned operation.

## 1. Purpose

This document specifies a pipeline for `kshitijd/mmu-xmatch-5k`.

The pipeline has three products:

1. Layerwise embeddings from the model matrix in `UniverseTBD/platonic-universe`.
2. Exact K-nearest-neighbor (KNN) artifacts for valid object cohorts.
3. A versioned Hugging Face dataset repository that stores the embeddings, KNN artifacts, indexes, and provenance.

The pipeline adds an MMU dataset adapter to `platonic-universe`. The pipeline also adds a canonical scikit-learn KNN path and an optional CuPy performance path.

The primary research use is mutual-KNN analysis across model layers and astronomical modalities. The same stored embeddings can support CKA and other representation analyses.

## 2. References

| Reference | Document |
|-----------|----------|
| REF-1 | `https://github.com/ksd3/layerwise-analysis/blob/main/tutorial.ipynb` - MMU cross-match data tutorial |
| REF-2 | `https://huggingface.co/datasets/kshitijd/mmu-xmatch-5k` - source dataset |
| REF-3 | `https://github.com/UniverseTBD/platonic-universe` - extraction and analysis code |
| REF-4 | `pipeline-spec/04-downstream-analysis-interface.md` - downstream analysis contract |
| REF-5 | Huh et al. 2024, arXiv:2405.07987 - Platonic Representation Hypothesis |
| REF-6 | Duraphe et al. 2025, arXiv:2509.19453 - Platonic Universe |
| REF-7 | `https://docs.cupy.dev/en/stable/reference/scipy_spatial.html` - current CuPy spatial API |
| REF-8 | `https://scikit-learn.org/stable/modules/neighbors.html` - scikit-learn nearest-neighbor API |

Local paths in this document are relative to the repository root unless a section gives an absolute path.

## 3. Terms

Each term has one meaning in this document.

| Term | Meaning |
|------|---------|
| object | One astronomical object in the selected MMU spine |
| `spine_id` | The MMU object identifier and join key |
| payload | One MMU modality file that contains data for a subset of objects |
| supported payload | A payload with at least one model-size pair that has structural and scientific approval |
| payload index | The ordered inventory of structurally valid objects in one payload-index revision |
| rejection table | The source rows that are absent from a payload index, with one rejection reason per row |
| cohort | The ordered singleton or pairwise object population for one representation comparison |
| model family | One model alias in the main `platonic-universe` experiment matrix |
| model-size run | One model family at one configured size or checkpoint |
| block | One major model stage selected by the existing `blocks` extraction granularity |
| final representation | The pooled representation returned by the model adapter's normal embedding path |
| embedding artifact | A table of one block representation for one model-size run and one payload |
| KNN graph | The ordered nearest-neighbor indexes and distances for one embedding artifact on one cohort |
| exact KNN | A nearest-neighbor result without an approximate index |
| run manifest | The provenance and completion record for one model-size and payload task |
| release manifest | The inventory and provenance record for the complete Hugging Face release |
| completion marker | A small file that records a validated artifact checksum and state |
| artifact identity hash | A hash of fields that can change scientific output |
| runtime hash | A hash of operational fields that cannot change scientific output |

The phrase “primitives for KNN” means the minimum reusable objects that are necessary to construct or compare KNN neighborhoods.

## 4. Locked decisions

| # | Decision | Value |
|---|----------|-------|
| 1 | Source dataset | `kshitijd/mmu-xmatch-5k`, pinned to a commit revision |
| 2 | Model scope | 13 model families and 38 unique model-size runs from `experiments.py`; PR 1 must add layerwise support for AION and SpecFormer, which are absent from the current 36-run layerwise map |
| 3 | Layer scope | Major model blocks plus the final pooled representation |
| 4 | Payload scope | Every payload that a model adapter can process without an invalid scientific or structural conversion |
| 5 | Object key | `spine_id` |
| 6 | Canonical embedding dtype | float32 |
| 7 | KNN method | Exact cosine KNN |
| 8 | Stored neighbor count | `Kmax = min(100, N - 1)` |
| 9 | Canonical analysis value | `k=10` |
| 10 | Self-neighbor policy | Exclude the query object |
| 11 | Canonical KNN backend | scikit-learn on CPU; CuPy is an optional performance backend and is not on the release critical path |
| 12 | Storage format | Compacted, sharded Parquet files plus JSON manifests |
| 13 | KNN population | One singleton cohort per supported payload and one pairwise cohort per unordered pair of supported payloads; no cohort contains more than two payloads |
| 14 | Hugging Face destination | A new project-owned dataset repository |
| 15 | Compute assumption | SLURM-style jobs on A100 or H200 class GPUs; final allocation is not confirmed |
| 16 | Pooling | Adapter-specific pooling and VLM prompts are fixed by Section 8.1 and stored in each run manifest |
| 17 | Release rule | Publish a release tag only after the release manifest accounts for all planned tasks; a failed task blocks release unless the PI records a waiver |

## 5. Scope

### 5.1 In scope

- An MMU dataset adapter in `UniverseTBD/platonic-universe`.
- Direct access to the selected spine and modality Parquet files.
- Payload indexes and comparison cohorts.
- A compatibility matrix for the 13 model families.
- Blockwise embedding extraction for the 38-run target matrix, including new layerwise support for AION and SpecFormer.
- Layer-level resume and completion markers.
- Canonical exact cosine KNN with scikit-learn and an optional CuPy implementation.
- Cohort-specific KNN graphs.
- Upload to a new Hugging Face dataset repository.
- Release manifests, checksums, validation, and load-back tests.
- SLURM task generation for the full run matrix.

### 5.2 Out of scope

- Training or fine-tuning a model.
- Adding time-series or tabular models to the 38-run matrix.
- Storing full token-level hidden states.
- Storing attention maps.
- Approximate KNN indexes.
- Using a serialized KD-tree or FAISS index as the primary released artifact.
- Computing every pairwise mutual-KNN or CKA score as part of the extraction release.
- Modifying the source `kshitijd/mmu-xmatch-5k` repository.

Derived score tables can be added after the embedding and graph release. They are not a release gate for this design.

## 6. Source dataset and payload support

### 6.1 Source access

The source dataset is a multi-file dataset. The selected spine and the modality payloads are explicit Parquet files.

The default Hugging Face config exposes the lightweight `spine_full` anchor view. The pipeline must load `spine_selected.parquet` and the selected files in `modalities/` directly.

The selected spine contains 5,000 objects. The tutorial in REF-1 defines the access pattern:

```text
filter the selected spine -> load one payload -> join on spine_id
```

R-SRC-1. The pipeline must pin the source dataset to a commit revision.

R-SRC-2. The pipeline must use `spine_id` for all object joins.

R-SRC-3. The pipeline must not use `TARGETID` or `_healpix_29` as the cross-payload join key.

R-SRC-4. The adapter must use column projection when it reads a payload with large nested arrays.

R-SRC-5. In this dataset, `spine_id` fulfills the stable shared-key role that REF-4 assigns to `global_object_id` for the OmniSky dataset.

### 6.2 Candidate payload classes

The compatibility preflight determines final support. The initial candidate classes are:

| Payload class | Source examples | Expected model class | Rule |
|---------------|-----------------|----------------------|------|
| Optical images | Legacy Survey, HSC, GZ10 RGB | Vision, VLM, AstroPT | Include after decoder and shape validation |
| Near-infrared images | JWST files | Vision, VLM, AstroPT | Include after decoder and shape validation |
| Alert images | ZTF/BTS cutouts | Vision and VLM | Include only with a documented cutout selection and channel rule |
| Optical spectra | DESI, SDSS | SpecFormer and any model with an approved spectral input contract | Include after wavelength, mask, and normalization validation |
| X-ray spectra | Chandra | Spectral models | Include only if the adapter input contract is valid for the payload representation |
| Physical labels | PROVABGS and DESI catalog fields | No model input in this experiment | Keep as metadata for later analysis |
| Astrometry | Gaia | No model input in this experiment | Exclude from extraction |
| Light curves | TESS and supernova files | No model input in this experiment | Exclude from extraction |

R-PAY-1. Each model-size and payload pair must pass compatibility preflight before a full extraction job is added to the task manifest.

R-PAY-2. The preflight must scan decoding and preprocessing for all rows in the payload index. It must also run one deterministic stress batch through the pinned model checkpoint. The stress batch must include the largest or longest valid inputs in the payload.

R-PAY-3. The preflight must record each accepted or rejected model-size and payload pair.

R-PAY-4. A rejected pair must have a reason. The pipeline must not coerce an incompatible payload to increase the run count.

R-PAY-5. Structural success is not sufficient for compatibility. Each accepted pair must have a `scientific_basis` field that states why the payload semantics match the model input contract. The adapter owner and the PI or designated science lead must approve this field.

R-PAY-6. A pair must not enter the task manifest until its structural status and scientific status are both `approved`.

## 7. Why the pipeline uses payload indexes and cohorts

An MMU payload does not contain all 5,000 selected objects. Each payload has its own coverage. A payload can also contain rows that fail structural validation.

A payload index fixes the usable object inventory before model inference. This rule prevents each model worker from selecting a different object set.

A cohort fixes the object population for one comparison. A KNN graph depends on this population. A graph for all Legacy Survey objects and a graph for all HSC objects do not describe the same search space. The two graphs cannot be used for a valid mutual-KNN comparison.

The cohort set is finite. The pipeline builds these cohorts only:

1. One singleton cohort for each supported payload.
2. One pairwise cohort for each unordered pair of supported payloads.

The pipeline does not build intersections of three or more payloads. A pairwise cohort is the ordered intersection of its two payload indexes.

### 7.1 Payload index schema

R-IDX-1. The adapter must build one payload index for each supported payload.

R-IDX-2. Each payload index must have this schema:

| Field | Type | Meaning |
|-------|------|---------|
| `row_index` | int32 | Position among valid objects in the canonical payload order |
| `spine_id` | int64 | MMU object identifier |
| `payload` | string | Canonical payload name |
| `source_row` | int64 | Row position in the source payload |
| `index_revision` | int32 | Monotonic revision of this payload index |
| `index_hash` | string | Hash of the ordered index definition |

R-IDX-3. A payload index must contain unique `spine_id` values.

R-IDX-4. A payload index must contain valid objects only. It must assign contiguous `row_index` values, starting from 0, in ascending `spine_id` order.

R-IDX-5. The adapter must write excluded source rows to a separate rejection table with `spine_id`, `source_row`, `stage`, and `reason` fields.

R-IDX-6. A payload index is immutable within one `index_revision`. Model workers must read it. Model workers must not rebuild it.

R-IDX-7. Only the coordinator can create a new index revision. A new revision must invalidate all cohorts and artifacts that reference the prior `index_hash`.

R-IDX-8. Physical payload reads can use source-row order for sequential I/O. The writer must reorder emitted artifacts to payload-index order before validation and publication.

### 7.2 Cohort schema

R-COHORT-1. The cohort builder must join payload indexes on `spine_id`.

R-COHORT-2. Each cohort must have this schema:

| Field | Type | Meaning |
|-------|------|---------|
| `cohort_id` | string | Stable cohort identifier |
| `row_index` | int32 | Position in the cohort |
| `spine_id` | int64 | MMU object identifier |
| `payloads` | list<string> | Payloads that define the cohort |
| `source_revision` | string | Pinned MMU revision |
| `payload_index_hashes` | list<string> | Sorted index hashes that define the cohort |
| `definition_hash` | string | Full SHA-256 hash of the canonical cohort definition |

R-COHORT-3. The cohort builder must sort each cohort by `spine_id`.

R-COHORT-4. All model runs for one cohort must use the same ordered object list.

R-COHORT-5. An extraction worker must emit one embedding for each object in its payload index. A KNN worker must fail if the embedding artifact does not contain every cohort member. Neither worker can remove failed rows.

R-COHORT-6. The pipeline must not build a `k=10` graph for a cohort with fewer than 11 objects.

R-COHORT-7. The pipeline must store embeddings for a supported payload even when a cohort is too small for KNN. The manifest must record `insufficient_n`.

R-COHORT-8. The cohort definition must be canonical JSON with sorted keys and no insignificant whitespace. It must contain `schema_version=1`, the source dataset revision, and the sorted payload names and payload-index hashes.

R-COHORT-9. `cohort_id` must be `cohort-v1-` followed by the first 16 hexadecimal characters of the SHA-256 hash of the canonical cohort definition. The manifest must store the complete hash.

R-COHORT-10. The task planner must enumerate all singleton and unordered pairwise cohorts. It must not create any other cohort type.

R-COHORT-11. For a singleton cohort, the planner creates one KNN task for each accepted model-size on that payload. For a pairwise cohort, it creates one KNN task for each accepted model-size on each member payload. One KNN task processes all stored layers for that model-size and payload.

## 8. Model matrix

The main `platonic-universe` experiment map contains these unique model-size runs:

| Family | Sizes | Count |
|--------|-------|------:|
| ViT | base, large, huge | 3 |
| CLIP | base, large | 2 |
| DINOv2 | small, base, large, giant | 4 |
| DINOv3 | vits16, vits16plus, vitb16, vitl16, vith16plus, vit7b16 | 6 |
| ConvNeXtV2 | nano, tiny, base, large | 4 |
| I-JEPA | huge, giant | 2 |
| V-JEPA2 | large, huge, giant | 3 |
| AstroPT | 015M, 095M, 850M | 3 |
| ViT-MAE | base, large, huge | 3 |
| AION | 300M | 1 |
| SpecFormer | 43M | 1 |
| PaliGemma2 | 3B, 10B, 28B | 3 |
| LLaVA | 1.5-7B, 1.5-13B, OneVision-7B | 3 |
| **Total** |  | **38** |

The current repository has no symbol named `MODEL_GRID`. The local `model_map` in `src/pu/experiments.py` contains these 38 unique runs. The module-level `MODEL_MAP` in `src/pu/experiments_layerwise.py` contains 36 unique runs and omits AION and SpecFormer. The 31-model count in older workspace documents is historical and does not match a current model map.

R-MOD-1. The task manifest must contain one row for each model-size and accepted payload pair.

R-MOD-2. The task manifest must pin each model checkpoint to an immutable revision.

R-MOD-3. The extraction granularity must be `blocks`.

R-MOD-4. The pipeline must also store the final pooled representation from the adapter's normal embedding method.

R-MOD-5. The pipeline must record the full module name and the ordinal position of each block.

R-MOD-6. PR 1 must add an explicit block list and layerwise extraction method for AION and SpecFormer. These two runs must not be reported as layerwise-complete until that support passes the integration tests.

### 8.1 Pooling, prompts, and representation scope

Pooling is part of the artifact identity. The pipeline must not select pooling at run time.

| Family | Intermediate block pooling | Final representation | Prompt or input rule |
|--------|----------------------------|----------------------|----------------------|
| ViT | Mean of patch tokens; exclude token 0 | Mean of patch tokens; exclude token 0 | Image only |
| CLIP | Mean of all vision-sequence tokens | `get_image_features()` projected visual embedding | Image only; hooks on `vision_model` |
| DINOv2 | Token 0 | Token 0 | Image only |
| DINOv3 | Token 0 | Token 0 | Image only |
| ConvNeXtV2 | Spatial mean over H and W | Spatial mean over H and W | Image only |
| I-JEPA | Mean over sequence tokens | Mean over sequence tokens | Image only |
| V-JEPA2 | Mean over sequence tokens | Mean over sequence tokens | Repeat one input frame to 16 frames |
| AstroPT | Rank-4: spatial mean; rank-3: sequence mean; rank-2: identity | `generate_embeddings(...)["images"]` | AstroPT image and position inputs |
| ViT-MAE | Mean of patch tokens; exclude token 0 | Mean of patch tokens; exclude token 0 | Image only |
| AION | Rank-4: spatial mean; rank-3: sequence mean; rank-2: identity | `encode()` output with the same rank rule | Adapter-approved AION codec and token key |
| SpecFormer | Mean of sequence tokens; exclude statistics token 0 | Mean of sequence tokens; exclude statistics token 0 | DESI or SDSS spectral input contract |
| PaliGemma2 | Masked mean over all non-padding positions of each language-decoder block | Masked mean of final language hidden state | Exact prompt `"<image> "` |
| LLaVA-1.5 | Masked mean over all non-padding positions of each language-decoder block | Masked mean of final language hidden state | Exact prompt `"USER: <image>\n ASSISTANT:"` |
| LLaVA-OneVision | Masked mean over all non-padding positions of each language-decoder block | Masked mean of final language hidden state | Exact prompt `"<image>"` |

R-POOL-1. The table in this section is normative.

R-POOL-2. The final representation must use the same adapter path as the standard non-layerwise experiment.

R-POOL-3. PaliGemma2 and LLaVA layerwise artifacts must contain language-decoder blocks only. Their pooled token set includes image tokens and prompt tokens that have attention-mask value 1.

R-POOL-4. A VLM run must fail if final language hidden states are absent. It must not switch to a vision-only fallback representation.

R-POOL-5. A run manifest must store the prompt bytes, processor revision, token-count summary, block list, and pooling rule.

R-POOL-6. The final representation must use `layer_index=-1`. Major blocks must use contiguous indexes from 0 in forward order.

R-POOL-7. A pooling or prompt change creates a new artifact identity hash and cannot resume from artifacts with the prior hash.

## 9. System architecture

The system has five components.

### 9.1 MMU dataset adapter

The adapter reads the selected spine, payload files, payload indexes, and cohorts. It decodes one supported payload into the input contract of one model adapter. It preserves `spine_id` in each batch.

### 9.2 Compatibility registry

The registry maps model families to payload classes. It also stores preprocessing rules that are specific to one accepted pair.

### 9.3 Blockwise extraction worker

The worker runs one model-size and payload task. It captures major block outputs and the final pooled representation. It writes one layer at a time.

### 9.4 Exact KNN builder

The builder reads one embedding artifact and one cohort. It writes exact cosine neighbors and distances for the cohort.

### 9.5 Artifact publisher

The publisher validates local artifacts, uploads complete artifacts, updates the release manifest, and performs a load-back check.

R-ARC-1. Each component must communicate through typed tables or manifests.

R-ARC-2. No component can infer object order from file order. It must use `row_index` and `spine_id`.

R-ARC-3. An extraction worker must not compute pairwise alignment scores.

R-ARC-4. A KNN worker must not run model inference.

## 10. Artifact layout

The Hugging Face dataset repository must use this logical layout:

```text
README.md
manifest.json
payloads/
  {payload}/
    index-v{revision}.parquet
    rejected-v{revision}.parquet
cohorts/
  {cohort_id}/index.parquet
embeddings/
  {model}/{size}/{payload}/
    run.json
    data-00000-of-0000N.parquet
neighbors/
  {cohort_id}/{model}/{size}/{payload}/
    run.json
    data-00000-of-0000N.parquet
status/
  extraction-manifest.parquet
  knn-manifest.parquet
```

Extraction workers can use per-layer files in local staging for resume. Before upload, the publisher must compact complete layers into Parquet shards that target 512 MiB and contain contiguous `(layer_index, row_index)` ranges. The Hugging Face repository must not publish one file per layer by default.

R-FILE-1. The task planner must estimate the release file count before the pilot and after the compatibility matrix is frozen.

R-FILE-2. The release must contain fewer than 5,000 data and manifest files. If the estimate exceeds this value, the publisher must increase compaction or the PI must approve a different repository partition.

R-FILE-3. The pilot must measure compacted shard size, upload time, list time, and load-back time.

### 10.1 Embedding artifact schema

R-EMB-1. One embedding shard must contain one model-size and payload. It can contain multiple complete layers.

R-EMB-2. Each embedding row must have this schema:

| Field | Type | Meaning |
|-------|------|---------|
| `row_index` | int32 | Position in the payload index |
| `spine_id` | int64 | MMU object identifier |
| `layer_index` | int16 | Block order; the final representation uses -1 |
| `layer_name` | string | Full module name or `final` |
| `is_final` | bool | True for the adapter's final representation |
| `embedding` | list<float32> | Pooled representation |

R-EMB-3. The stored embedding must be the unnormalized pooled float32 representation.

R-EMB-4. All rows for one `layer_index` must have the same embedding dimension. Different layers in one shard can have different dimensions because `embedding` is a variable-length float32 list.

R-EMB-5. The manifest must record the pooling method for each layer.

### 10.2 KNN artifact schema

R-KNN-1. One KNN shard must contain one cohort, model-size, and payload. It can contain multiple complete layers.

R-KNN-2. Each KNN row must have this schema:

| Field | Type | Meaning |
|-------|------|---------|
| `row_index` | int32 | Position in the cohort |
| `spine_id` | int64 | MMU object identifier |
| `cohort_id` | string | Cohort identifier |
| `layer_index` | int16 | Block order; the final representation uses -1 |
| `layer_name` | string | Full module name or `final` |
| `neighbor_indices` | list<int32> | Neighbor positions in the same cohort |
| `neighbor_distances` | list<float32> | Canonical cosine distances in neighbor order |
| `k` | int16 | Stored neighbor count for this cohort |

R-KNN-3. `neighbor_indices` must refer to the `row_index` values in the named cohort.

R-KNN-4. The neighbor arrays must have length `min(100, N - 1)`.

R-KNN-5. A row must not contain its own `row_index`.

R-KNN-6. A row must not contain duplicate neighbor indexes.

R-KNN-7. Neighbor distances must be finite and in nondecreasing order.

### 10.3 Manifest content

R-MAN-1. Each extraction and KNN manifest must contain these common fields:

- Source dataset ID and revision.
- Model ID and revision.
- Code revision for the task implementation.
- Model family and size.
- Payload ID and payload-index hash.
- Python version and applicable package versions.
- Expected and actual row counts.
- Artifact paths, byte sizes, and checksums.
- Start time, completion time, host class, and job ID.
- Completion state and failure reason.

R-MAN-2. An extraction run manifest must also contain layer names, order, dimensions, pooling methods, prompts, preprocessing, embedding dtype, PyTorch, Transformers, CUDA, and hardware class.

R-MAN-3. A KNN run manifest must also contain the cohort ID and definition hash, metric formula, clipping algorithm, normalization rule, tie rule, backend, backend version, and numeric tolerances.

R-MAN-4. The release manifest must inventory every planned extraction and KNN task.

R-MAN-5. The release manifest must use one of these states: `planned`, `unsupported`, `running`, `failed`, `waived`, `insufficient_n`, or `complete`.

R-MAN-6. A release tag must not be created while a task remains `planned`, `running`, or `failed`.

R-MAN-7. Only the PI can change a `failed` task to `waived`. The waiver must name the task, reason, scientific impact, and resulting coverage claim.

## 11. Extraction method

The extraction unit is one model-size and payload pair.

R-EXT-1. The worker must load one pinned model checkpoint.

R-EXT-2. The worker must read object order from the payload index.

R-EXT-3. The worker must preserve the adapter's documented preprocessing for the accepted payload pair.

R-EXT-4. The worker must run inference with gradients disabled.

R-EXT-5. The worker must capture `blocks` and the final pooled representation.

R-EXT-6. The worker must pool each captured block to one vector per object. The pooling rule must be explicit in the manifest.

R-EXT-7. The worker must convert the stored output to float32 after inference.

R-EXT-8. The worker must flush one completed layer before it starts the next layer write.

R-EXT-9. The worker must write to a temporary path. A validated rename and completion marker must mark the layer complete.

R-EXT-10. A resumed task must skip a layer only when its marker, checksum, schema, code revision, model revision, and source revision match the current task.

R-EXT-11. The worker must run in evaluation mode with gradients disabled and a fixed seed. The run manifest must record deterministic-algorithm settings and each operation that cannot use a deterministic implementation.

R-EXT-12. A retry must use the same pinned checkpoint, artifact identity hash, payload-index hash, preprocessing, pooling, prompt, dtype, and hardware class. Runtime fields such as batch size and worker count can change.

## 12. KNN method and backend selection

### 12.1 Canonical KNN transform

The released embedding is raw. The KNN transform operates on the embedding matrix after it is projected to one cohort. The canonical transform follows the default path in `minyoungg/platonic-rep/measure_alignment.py` and `metrics.py`.

For a cohort matrix `X` with shape `[N,D]`, the transform is:

```text
X = torch.float32(X) on CPU with the pinned PyTorch version
row_threshold[i] = quantile(abs(X[i, :]), q=0.95, interpolation="linear")
threshold = mean(row_threshold)
X_clipped = clamp(X, -threshold, threshold)
norm[i] = l2_norm(X_clipped[i, :])
X_unit[i, :] = X_clipped[i, :] / norm[i]
distance(i, j) = clip(1 - dot_float64(X_unit[i, :], X_unit[j, :]), 0, 2)
```

R-KNN-TR-1. The initial release must use `q=0.95` for parity with the default official implementation of REF-5.

R-KNN-TR-2. The quantile must operate on the absolute feature values of each cohort row. It must use linear interpolation. The scalar clipping threshold must be the arithmetic mean of the row quantiles.

R-KNN-TR-2A. The canonical clipping transform must run with PyTorch on CPU in float32. The manifest must pin the PyTorch version.

R-KNN-TR-3. The threshold must be computed from the cohort subset, not from the full payload matrix.

R-KNN-TR-4. Clipping must be symmetric at `[-threshold, threshold]`.

R-KNN-TR-5. The KNN builder must L2-normalize each clipped row. A row with zero norm must stop the graph task and name its `spine_id` in the error record.

R-KNN-TR-6. Canonical cosine distance must be `clip(1 - dot(x_i, x_j), 0, 2)` after normalization. The dot product and ranking distances must use float64. Stored distances must be converted to float32 after ranking.

R-KNN-TR-7. The KNN builder must compute exact neighbors.

R-KNN-TR-8. The KNN builder must exclude self-neighbors by `row_index` before ranking. It must not remove the first result as a proxy for self-exclusion.

R-KNN-TR-9. Candidate neighbors must sort by the tuple `(distance_float64, neighbor_row_index)`. The lower `row_index` wins an exact distance tie.

The current `platonic-universe` `mknn()` signature defaults to percentile 100, which disables clipping, while its docstring states 95. It also uses a global NumPy percentile, which differs from the official per-row-quantile mean above. PR 2 must replace that ambiguity with this explicit transform.

### 12.2 Backend interface

The backend interface has three selections: `auto`, `cupy`, and `sklearn`.

R-BACK-1. The canonical release backend must be scikit-learn. A consumer that reproduces a released graph must use the pinned canonical software environment.

R-BACK-2. `auto` must select scikit-learn for this 5,000-object dataset. CuPy requires an explicit `cupy` selection.

R-BACK-3. An explicit `cupy` request must fail if CuPy or CUDA is not available. It must not switch silently.

R-BACK-4. An explicit `sklearn` request must use scikit-learn.

R-BACK-5. The module must not import CuPy at package import time. CuPy must remain an optional dependency.

R-BACK-6. The CuPy implementation must use batched normalized matrix products. The common CPU finalizer must apply self-exclusion, the distance formula, and the deterministic ordering rule.

R-BACK-7. The scikit-learn implementation must compute the canonical float64 cosine-distance matrix and use the common finalizer.

R-BACK-8. Both implementations must return neighbor indexes and distances with the same schema and semantics.

R-BACK-9. CuPy is not a release gate. PR 2 and the release can use the canonical scikit-learn path if CuPy parity or packaging is not ready.

Current CuPy releases include `cupyx.scipy.spatial.KDTree` through a cuVS backend. This design does not use that tree. The embedding dimensions are high, the metric is cosine, and the required output is an exact top-k graph. A normalized matrix product gives a direct implementation of this contract.

### 12.3 Determinism and numeric tolerances

R-DET-1. The canonical software lock must pin Python, NumPy, SciPy, and scikit-learn versions.

R-DET-2. Reproduction with the canonical backend must produce identical neighbor indexes. Stored float32 distances must satisfy `rtol=1e-6` and `atol=1e-6` against recomputed distances.

R-DET-3. CuPy parity distances must satisfy `rtol=1e-5` and `atol=1e-5` against the canonical backend.

R-DET-4. CuPy must produce the same neighbor indexes when the canonical distance margin between rank `Kmax` and rank `Kmax+1` is greater than `1e-5`.

R-DET-5. When the boundary margin is at most `1e-5`, parity validation must compare each backend result with the complete canonical tie set. The tie set contains each candidate whose canonical distance differs from the canonical rank-`Kmax` distance by at most `1e-5`. Parity validation must not require one arbitrary order before the common finalizer.

R-DET-6. Tests must include duplicate vectors, a self-vector tied at distance 0, a zero vector, and a tie at the K boundary.

### 12.4 Stored primitive versus serialized index

The primary KNN artifacts are:

1. The ordered cohort IDs.
2. The embedding matrix.
3. The neighbor-index matrix.
4. The neighbor-distance matrix.
5. The transform and provenance manifest.

R-PRIM-1. The release must not require a serialized KD-tree, CuPy object, or FAISS index.

R-PRIM-2. A consumer must be able to compute mutual-KNN overlap from the released tables without model inference.

R-PRIM-3. A consumer must be able to rebuild a search index from the released embedding matrix.

## 13. Data flow

The pipeline has eight phases.

```text
pin inputs
  -> build payload indexes
  -> run compatibility preflight
  -> freeze index revisions and build bounded cohorts
  -> write extraction and KNN task manifests
  -> extract block embeddings
  -> build cohort-specific KNN graphs
  -> validate, upload, and load back
```

### Phase 1 - Pin inputs

Resolve source and model revisions. Write the immutable run configuration.

### Phase 2 - Build payload indexes

Read the selected spine and candidate payloads. Validate rows. Write one canonical index for each supported payload.

### Phase 3 - Run compatibility preflight

Scan all payload rows through each model-size preprocessor. Run the deterministic stress batch. Record structural and scientific approval for each model-size and payload pair.

### Phase 4 - Freeze indexes and build cohorts

Freeze one payload-index revision for each supported payload. Build all singleton cohorts and all unordered pairwise cohorts. Do not build a higher-order cohort. Record every cohort size before full GPU extraction starts.

### Phase 5 - Write task manifests

Expand 38 model-size runs over accepted payloads. Add one extraction task per accepted model-size and payload pair. Add KNN tasks with the rule in R-COHORT-11. Mark unsupported and insufficient-size tasks.

If `P` is the supported payload set and `M_p` is the accepted model-size set for payload `p`, the extraction task count is:

```text
sum over p in P of |M_p|
```

The cohort count is:

```text
|P| + |P| * (|P| - 1) / 2
```

One KNN task processes all layers for one model-size, payload, and cohort. The planner must write the exact task count and projected file count before submission.

### Phase 6 - Extract embeddings

Run resumable GPU tasks. Validate and mark each layer independently.

### Phase 7 - Build KNN graphs

Project the stored embeddings onto each cohort enumerated by R-COHORT-10 and R-COHORT-11. Compute exact neighbors. Validate and mark each graph.

### Phase 8 - Validate and publish

Audit all manifests and checksums. Upload complete artifacts. Load the staged repository in a clean environment. Create a release tag after the release gates pass.

## 14. SLURM and resume model

R-SLM-1. One extraction task must process one model-size and payload pair.

R-SLM-2. One KNN task must process one cohort, model-size, and payload across all complete layers.

R-SLM-3. An artifact identity hash must include source revision, payload-index hash, model revision, model size, payload, block list, pooling, prompt, preprocessing, output dtype, and KNN transform and cohort fields when applicable.

R-SLM-4. A runtime hash must include batch size, worker count, SLURM resources, retry count, and logging fields. Runtime fields must not be part of the artifact identity hash.

R-SLM-5. The operator must be able to submit disjoint task-manifest partitions without duplicate output ownership.

R-SLM-6. Completed layers must survive a job timeout, preemption, or GPU out-of-memory event.

R-SLM-7. After a GPU out-of-memory event, the operator can reduce the batch size and resume incomplete layers because batch size changes only the runtime hash. The manifest must record the final batch size.

R-SLM-8. The 10B and 28B VLM tasks can use a different SLURM resource class from the smaller models.

R-SLM-9. The final SLURM resource table must be set after the pilot run measures memory and throughput.

## 15. Error management

R-ERR-1. A missing `spine_id`, duplicate `spine_id`, missing required payload column, or invalid model revision must stop the affected task before inference.

R-ERR-2. A per-object preprocessing, inference, or zero-norm failure must stop the affected task. The worker must not remove the object from the cohort.

R-ERR-3. A rejected compatibility pair must not stop unrelated tasks. The task manifest must record `unsupported` and the reason.

R-ERR-4. A cohort with fewer than 11 objects must not stop embedding extraction. The KNN task must record `insufficient_n`.

R-ERR-5. An interrupted artifact write must not produce a completion marker.

R-ERR-6. A checksum or schema mismatch must invalidate the completion marker and force a new write.

R-ERR-7. An unexpected exception must retain its traceback in the task log.

R-ERR-8. The publisher must not list an incomplete artifact as complete in the release manifest.

R-ERR-9. After R-ERR-2, the operator must first correct the decoder, preprocessing, pooling, or model adapter and rerun the task.

R-ERR-10. If the object is irrecoverable, the coordinator can propose a new payload-index revision that moves the object to the rejection table. The proposal must name the object, failure, affected tasks, and cohort-size changes. The PI or designated science lead must approve the new revision.

R-ERR-11. An approved payload-index revision must rebuild all dependent cohorts and artifact identity hashes. Artifacts from the old index revision become stale. The pipeline must not use per-model or per-pair silent exclusions.

## 16. Verification

### 16.1 Unit tests

The implementation must test these behaviors:

1. A payload index has unique, sorted `spine_id` values.
2. The rejection table contains every structurally rejected row and no indexed row.
3. A singleton cohort equals its payload index, and a pairwise cohort equals the ordered intersection of its two payload indexes.
4. Cohort IDs are stable for the same canonical definition and change when an index hash changes.
5. The task planner creates singleton and unordered pairwise cohorts only.
6. The adapter rejects a missing join key.
7. A model worker preserves all expected IDs.
8. Block extraction produces finite float32 arrays with stable dimensions and the locked pooling rule.
9. The VLM prompt bytes and pooled token mask match Section 8.1.
10. The KNN builder excludes self-neighbors by index and excludes duplicate neighbors.
11. Duplicate vectors do not cause the self row to survive or the wrong tied row to be removed.
12. A zero-norm row produces the typed failure and identifies its `spine_id`.
13. Neighbor distances use `clip(1-dot,0,2)`, are sorted, and use lower row index for an exact tie.
14. The canonical clipping fixture matches the official per-row 0.95 quantile-mean procedure.
15. CuPy and scikit-learn satisfy the parity policy in Section 12.3.
16. A stale or corrupt completion marker causes a new write.
17. An explicit unavailable backend fails without a silent fallback.
18. An insufficient cohort writes the correct manifest state.
19. A batch-size change preserves artifact identity and changes runtime hash.
20. A payload-index revision invalidates dependent cohorts and artifacts.

### 16.2 Integration fixture

The repository must include a small fixture with at least two payloads and at least 12 common objects. The fixture must support one small model adapter or a deterministic fake adapter.

The PR 1 integration test must:

1. Build two payload indexes.
2. Build their intersection cohort.
3. Extract two block artifacts and one final artifact.
4. Write extraction manifests and completion markers.
5. Load the embedding artifacts through the public consumer path.

The PR 2 integration test must extend the same fixture:

1. Build exact `k=10` graphs with the canonical backend.
2. Exercise duplicate-vector, tie, and zero-norm cases.
3. Run CuPy parity when a compatible CUDA environment is available.
4. Write KNN manifests and completion markers.
5. Load the KNN artifacts through the public consumer path.

### 16.3 Real pilot

The first real run must use ViT-base and two intersecting image payloads.

The pilot must measure:

- Decode and preprocessing throughput.
- GPU memory and batch size.
- Layer count and dimensions.
- Artifact byte size.
- Upload throughput.
- CuPy and scikit-learn KNN time.
- Neighbor parity.
- Load-back time from Hugging Face.
- Supported payload count, bounded cohort count, KNN task count, and projected Hugging Face file count.
- Compacted shard size and repository listing time.

The full 38-run task matrix must not start before the pilot report exists.

### 16.4 Release gates

The release passes only when:

1. The release manifest accounts for every planned task.
2. All complete artifacts pass schema and checksum validation.
3. All embedding rows have finite float32 values.
4. All KNN graphs reference the correct cohort.
5. A clean environment can load one small and one large artifact from Hugging Face.
6. A consumer can reproduce a stored `k=10` graph from the released embeddings.
7. The dataset card names the source and model licenses and revisions.
8. The PI has approved the public repository namespace and publication timing.
9. No task remains `failed`; each non-complete task is `unsupported`, `insufficient_n`, or has a recorded PI waiver.
10. The release file count is below the R-FILE-2 limit.

## 17. Pull request boundaries

### PR 1 - MMU adapter and artifact contract

PR 1 must include:

- The MMU dataset adapter.
- Payload-index and cohort builders.
- Compatibility preflight.
- `spine_id` preservation in extraction.
- Locked pooling, prompts, and block lists.
- Layerwise support for AION and SpecFormer.
- Block-level artifact writes and manifests.
- Validation and fixture tests.
- Hugging Face layout and load-back support.

PR 1 must not include the CuPy KNN implementation.

### PR 2 - Exact KNN graph API and optional GPU backend

PR 2 must include:

- The common KNN backend interface.
- The canonical scikit-learn exact cosine KNN path.
- The fully specified PRH clipping and normalization transform.
- Deterministic self-exclusion, tie handling, and tolerance policy.
- Optional CuPy exact cosine KNN.
- `auto`, `cupy`, and `sklearn` selection.
- Backend parity and edge-case tests.
- Support for the stored neighbor matrices in mutual-KNN scoring.
- An optional GPU dependency path. CuPy must not become a mandatory base dependency.

### Operational release

The operational release includes:

- The pilot report.
- The frozen compatibility matrix.
- The full extraction and KNN task manifests.
- SLURM submission files or generated commands.
- The staged Hugging Face upload.
- The validated release tag.

## 18. Message to the PI

The following text answers the storage question.

> We need to store four types of KNN primitive.
>
> 1. An ordered `spine_id` index for each MMU payload and each bounded comparison cohort.
> 2. One pooled float32 embedding matrix for each model size, model block, and supported payload.
> 3. One cohort-specific neighbor-index matrix and neighbor-distance matrix for each embedding artifact.
> 4. A manifest that records the source and model revisions, preprocessing, pooling, metric, normalization, clipping, KNN backend, software versions, dimensions, counts, and checksums.
>
> The payload index records which MMU objects have valid data for one modality. The cohort records the exact shared object population for one comparison. We will build singleton cohorts and unordered pairwise cohorts only. We will not build higher-order intersections. This distinction is necessary because a KNN graph changes when the object population changes. Two neighbor graphs are comparable only when they use the same ordered cohort.
>
> We will store the raw pooled embeddings once. We will derive exact KNN graphs for each enumerated cohort. The canonical graph will use the official PRH 0.95 per-row absolute-value quantile-mean clipping procedure, L2 normalization, and cosine distance. We will store up to 100 neighbors so that downstream work can use the paper value `k=10` or another value up to 100 without a new search.
>
> We will not use a serialized KD-tree as the primary artifact. A tree is backend-specific and can be rebuilt from the embeddings. The portable artifacts are the ordered IDs, embeddings, neighbor indexes, neighbor distances, and provenance.
>
> The extraction target is 13 model families and 38 model-size runs from the current standard experiment map. The current layerwise map contains 36 runs, so the adapter PR must add layerwise support for AION and SpecFormer. Pooling and VLM prompts will be fixed by model family and recorded in each manifest.

## 19. Risks and open facts

| # | Risk or open fact | Required action |
|---|-------------------|-----------------|
| 1 | The final SLURM cluster and GPU allocation are not confirmed. | Confirm A100/H200 access before the pilot. |
| 2 | The project-owned Hugging Face repository ID is not selected. | Select the namespace before staging upload. |
| 3 | Some candidate payloads can fail structural or scientific compatibility. | Freeze support only after per-model-size preflight and science approval. |
| 4 | The 10B and 28B VLM runs can exceed one-GPU memory. | Measure during preflight; assign an appropriate resource class. |
| 5 | Exact storage, file count, and GPU-hour cost are unknown. | Use pilot measurements, bounded cohort enumeration, compaction, and the task manifest to estimate all four. |
| 6 | The current `mknn()` default disables clipping and its clipping algorithm differs from the official PRH code. | Implement the exact Section 12.1 transform in PR 2 and record it in every graph manifest. |
| 7 | Public release timing can interact with active paper anonymization. | Get PI approval before a public HF release. |
| 8 | Source subset licenses can differ from the top-level dataset license. | Copy applicable attribution and license notes into the dataset card. |
| 9 | AION and SpecFormer do not currently support layerwise extraction. | Add and test explicit block lists and pooling in PR 1. |
| 10 | Hugging Face file count can grow with payload pairs and model layers. | Compact local layer files, enforce R-FILE-2, and measure repository operations in the pilot. |

## 20. Definition of done

The work is complete when:

1. PR 1 and PR 2 pass their tests and review.
2. The source dataset and all model checkpoints are pinned.
3. The compatibility matrix covers all 13 model families at model-size and payload granularity.
4. AION and SpecFormer have tested layerwise support, and the task manifest accounts for all 38 model-size runs and accepted payloads.
5. All planned embedding tasks are complete, unsupported, or waived by the PI. No task remains failed.
6. All enumerated cohort KNN tasks are complete, marked `insufficient_n`, or waived by the PI. No task remains failed.
7. The staged Hugging Face repository passes the load-back checks.
8. The release manifest and dataset card are complete.
9. The compacted release contains fewer than 5,000 files.
10. The PI approves the artifact contract, each waiver, and public-release timing.
11. A stable Hugging Face release tag exists.
