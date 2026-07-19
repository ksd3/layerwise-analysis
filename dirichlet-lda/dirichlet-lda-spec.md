# Dirichlet-LDA PRH Pipeline — Design Specification

Document date: 2026-07-19.
Document language: ASD-STE100 Simplified Technical English.
Requirement words: "must" shows a requirement. "Can" shows a capability. "Will" shows a future event.

## 1. Scope

This document specifies a software pipeline. The pipeline tests the Platonic Representation Hypothesis (PRH) at laptop scale. The pipeline trains independent Dirichlet-LDA models on disjoint slices of celestial object data. The pipeline then measures the alignment of the topic structures across the models. The pipeline shows the alignment as a function of the data set size.

The pipeline operates on 2 data sources:

- A synthetic data source with known classes.
- A real data source from the OmniSky release.

This document does not specify discriminant analysis. This document does not specify subspace angle measurements. Those functions are out of scope.

## 2. Referenced documents

| Reference | Document |
|-----------|----------|
| REF-1 | `pipeline-spec/04-downstream-analysis-interface.md` — the OmniSky analysis contract |
| REF-2 | Huh et al. 2024, arXiv:2405.07987 — The Platonic Representation Hypothesis |
| REF-3 | Duraphe et al. 2025, arXiv:2509.19453 — The Platonic Universe |

Note: the path of REF-1 is relative to `/Users/keecraw/SAA-day/eleuther/`.

## 3. Terms, technical names, and technical verbs

This section defines the technical names and the technical verbs for this document. Each term has 1 meaning in this document.

| Term | Meaning |
|------|---------|
| object | 1 celestial body with measured features |
| feature | 1 continuous measured quantity of an object |
| token | 1 identifier for 1 bin of 1 feature |
| document | The set of tokens for 1 object |
| vocabulary | The set of all possible tokens |
| model | 1 trained Dirichlet-LDA topic model |
| topic | 1 probability distribution across the vocabulary |
| mixture | The topic proportions of 1 document |
| slice | 1 subset of the data set that shares no objects with an other slice |
| size step | 1 value of N, the count of objects for 1 training |
| seed | 1 integer that controls the random number generator |
| JSD | The Jensen-Shannon divergence, base 2, in the range 0 to 1 |
| NMI | The normalized mutual information score |
| purity | The mean of the largest mixture component across the documents |
| alignment score | 1 minus the mean JSD of the matched topic pairs |
| null reference | The alignment score of models that trained on shuffled data |
| steering check | The intervention test of Section 5.11 |
| verdict plot | The plot of the alignment score against the size step |
| the report | The JSON output file of 1 pipeline run |

Technical verbs: train, bin, slice, match, shuffle, plot, steer.

Technical names from external systems: Dirichlet-LDA, scikit-learn, NumPy, pandas, SciPy, matplotlib, PyArrow, parquet, YAML, HuggingFace, OmniSky, `global_object_id`, Hungarian algorithm, quasar, spectral class.

## 4. System overview

The pipeline has 9 stages. The stages operate in this sequence:

1. Get the data from the synthetic source or from the real source.
2. Bin each feature with fixed bin edges. Each object becomes a document.
3. Divide the data into disjoint slices. Sample each slice at each size step.
4. Train 1 model for each slice, each size step, and each seed.
5. Compute the purity and the NMI for each model.
6. Match the topics for each pair of models with the Hungarian algorithm on the JSD.
7. Compute the alignment score and the null reference.
8. Do the steering check on the matched topics.
9. Write the report, the tables, and the verdict plot.

Each stage reads and writes plain tables or arrays. Each stage is testable alone.

The verdict logic is this. An alignment score that increases with the size step, and that stays above the null reference, is evidence for the PRH. An alignment score that is flat, or that decreases, is evidence against the PRH.

## 5. Requirements

### 5.1 Package location and layout

R-PKG-1. The package root must be `/Users/keecraw/SAA-day/eleuther/layerwise-analysis/dirichlet-lda/`.

R-PKG-2. The importable Python package must have the name `ldaprh`.

R-PKG-3. The package must have this layout:

```
dirichlet-lda/
  pyproject.toml
  configs/
    synthetic.yaml
    real.yaml
  ldaprh/
    __init__.py
    config.py
    datagen.py
    real_adapter.py
    discretize.py
    slicer.py
    train.py
    match.py
    score.py
    steer.py
    plots.py
    run.py
  tests/
```

R-PKG-4. The package must declare these dependencies: NumPy, pandas, scikit-learn, SciPy, matplotlib, PyArrow, PyYAML.

R-PKG-5. The command `python -m ldaprh.run --config <path>` must run the full pipeline.

### 5.2 Configuration

R-CFG-1. The configuration must be 1 YAML file. The module `config.py` must parse the file into 1 dataclass.

R-CFG-2. The configuration must contain these fields:

| Field | Meaning | Default |
|-------|---------|---------|
| `data_source` | `synthetic` or `real` | `synthetic` |
| `features` | The list of feature names | The 6 names of R-SYN-2 |
| `label_column` | The class label column, or null | `class_label` |
| `survey_column` | The slice provenance column, or null | `instrument_id` |
| `slicer` | `random` or `survey` | `survey` |
| `n_slices` | The count of slices, M | 3 |
| `sizes` | The size steps | 1000, 3000, 10000, 30000, 100000 |
| `n_objects` | The count of synthetic objects before the instrument cuts | 500000 |
| `k_topics` | The topic count, K | 10 |
| `n_bins` | The bin count for each feature | 16 |
| `seeds` | The list of seeds | 0, 1, 2, 3, 4 |
| `alpha` | The document-topic prior | 1 / K |
| `max_iter` | The iteration limit for the training | 100 |
| `real_path` | The path or the HuggingFace name of the real data | null |
| `column_map` | The map from source columns to features | null |
| `out_dir` | The output directory | `out/` |

R-CFG-3. The module must reject a configuration with an unknown field. The module must report the field name in the error.

R-CFG-4. All stages must receive their parameters from the dataclass only.

### 5.3 Synthetic data source

R-SYN-1. The module `datagen.py` must generate objects from 9 classes: the spectral classes O, B, A, F, G, K, M, the class galaxy, and the class quasar.

R-SYN-2. Each object must have these 6 features: `temperature`, `log_luminosity`, `color_bg`, `color_gr`, `log_radius`, `redshift`.

R-SYN-3. Each class must have a multivariate normal feature distribution. The class means must obey the physical sign pattern of Table 1. The covariance must correlate `temperature`, the 2 colors, and `log_luminosity` in the physical directions.

Table 1 — Class mean sign pattern, relative to the mean of all classes:

| Class | temperature | log_luminosity | color_bg | color_gr | log_radius | redshift |
|-------|-------------|----------------|----------|----------|------------|----------|
| O | high | high | blue (low) | blue (low) | high | near 0 |
| M | low | low | red (high) | red (high) | low | near 0 |
| galaxy | not applicable (low) | high | middle | red (high) | high | middle |
| quasar | not applicable (high) | high | blue (low) | blue (low) | low | high |

Note: the classes B, A, F, G, K must interpolate between the class O and the class M in the order given.

R-SYN-4. The generator must write a `class_label` column and an `instrument_id` column.

R-SYN-5. The generator must simulate 1 instrument for each slice. Each instrument must apply 2 effects: a Gaussian noise term with an instrument-specific scale, and a selection cut on `log_luminosity` with an instrument-specific limit. Each limit must remove no more than 10 percent of the objects of its slice.

R-SYN-6. The generator must accept a seed. The same seed and the same configuration must give the same data.

R-SYN-7. The generator must write the true class-mean offsets to a table. The steering check reads this table as the ground truth.

R-SYN-8. The generator must make `n_objects` objects before the instrument cuts. With the default configuration, each slice must keep at least 100000 objects after the cuts. This count makes all default size steps available (R-SLC-5).

### 5.4 Real data adapter

R-REAL-1. The module `real_adapter.py` must read a parquet table from `real_path` (REF-1).

R-REAL-2. The adapter must join and identify the objects on the `global_object_id` column.

R-REAL-3. The adapter must apply `column_map` to rename the source columns to the configured features.

R-REAL-4. The adapter must fail with a clear error if a mapped column is absent. The error must name the column.

R-REAL-5. The adapter must remove each row that has a NaN value in a configured feature. The adapter must record the removed fraction in the report. The adapter must stop with an error if the removed fraction is more than 0.3.

R-REAL-6. If `label_column` is set and present, the adapter must keep the column for the NMI. If the column is absent, the pipeline must skip the NMI and must record the skip in the report.

R-REAL-7. If `survey_column` is set and present, the survey slicer can use the column. If the column is absent, the pipeline must use the random slicer and must record this fallback in the report.

### 5.5 Discretization

R-BIN-1. The module `discretize.py` must compute the bin edges from a reference sample. The reference sample must be a random sample of 10000 objects from the full data set. The reference sample must always use the seed 0, independent of the `seeds` list.

R-BIN-2. The bin edges must be the quantile edges of the reference sample, with `n_bins` bins for each feature.

R-BIN-3. All slices, all size steps, and all seeds must use the same bin edges. This rule keeps the vocabulary identical across all models. Different vocabularies make the topic match invalid.

R-BIN-4. The module must transform each object into a document of exactly 1 token for each feature.

R-BIN-5. The module must clip a value outside the reference range into the first bin or the last bin.

R-BIN-6. The module must store the bin edges in the output directory. A stored table of bin edges must let a later run reproduce the same vocabulary.

### 5.6 Slices

R-SLC-1. The module `slicer.py` must supply 2 slicers with 1 common interface: the random slicer and the survey slicer.

R-SLC-2. The random slicer must divide the objects into `n_slices` disjoint parts, at random, with the configured seed.

R-SLC-3. The survey slicer must divide the objects by the value of `survey_column`. Each slice must contain 1 survey value.

R-SLC-4. Two slices must never share an object. A shared object makes the independence claim invalid.

R-SLC-5. For each size step N, the pipeline must sample N objects from each slice without replacement. If a slice has fewer than N objects, the pipeline must skip that size step for all slices. The pipeline must record the skip in the report.

### 5.7 Model training

R-TRN-1. The module `train.py` must train 1 scikit-learn `LatentDirichletAllocation` model for each slice, each size step, and each seed.

R-TRN-2. All models must use the same `k_topics`, the same `alpha`, and the same vocabulary. Equal topic counts are a precondition for the topic match.

R-TRN-3. The training must pass the seed to the model.

R-TRN-4. The module must record a convergence flag for each model in the report. If a model does not converge in `max_iter` iterations, the flag must show this.

R-TRN-5. The module must store each topic-token matrix, normalized to probabilities, in a parquet table.

### 5.8 Topic validation

R-VAL-1. The module `score.py` must compute the purity for each model. The purity is the mean of the largest mixture component across the documents of the training slice.

R-VAL-2. If a label column is present, the module must compute the NMI for each model. The NMI compares the dominant topic of each document with the class label.

R-VAL-3. The module must count the effective topics for each model. A topic is degenerate if its total assigned probability mass is less than 0.5 percent of the corpus mass. The module must warn about each degenerate topic in the report.

### 5.9 Topic match

R-MAT-1. The module `match.py` must compute a K by K JSD matrix for each pair of models. Each entry is the JSD between 1 topic of the first model and 1 topic of the second model.

R-MAT-2. The module must find the topic assignment with the minimum total JSD. The module must use the Hungarian algorithm from SciPy.

R-MAT-3. The module must store the matched pairs and their JSD values.

### 5.10 Alignment score and null reference

R-SCO-1. The alignment score for 1 pair of models must be 1 minus the mean JSD of the matched pairs.

R-SCO-2. The alignment score for 1 size step and 1 seed must be the mean across all model pairs.

R-SCO-3. The pipeline must compute the null reference with the same procedure on shuffled data. The shuffle must permute the token of each feature across the objects, independently for each feature, for each slice. The shuffle keeps the marginal distributions. The shuffle destroys the correlations between the features.

R-SCO-4. The pipeline must compute the null reference for each size step and each seed. The null reference requires its own model training on the shuffled data. The null models supply alignment values only. The pipeline must not write topic tables or topic plots for the null models.

R-SCO-5. The report must contain the mean and the standard deviation of the alignment score and of the null reference, across the seeds, for each size step.

### 5.11 Steering check

R-STE-1. The module `steer.py` must do an intervention on the mixture. The intervened mixture must put the proportion 0.9 on 1 target topic and the uniform remainder on the other topics.

R-STE-2. The implied token distribution must be the mixture-weighted sum of the topic distributions.

R-STE-3. The baseline token distribution must use the corpus-mean mixture of the model.

R-STE-4. For each feature, the module must compute the shift. The shift is the mean bin index under the intervened distribution minus the mean bin index under the baseline distribution.

R-STE-5. For each matched topic pair, the module must compare the sign of the shift for each feature across the 2 models. The module must ignore a shift with a magnitude below 0.25 bins.

R-STE-6. The cross-model agreement must be the fraction of (matched topic, feature) cases with an equal sign. The report must contain this fraction for each model pair and each size step.

R-STE-7. On synthetic data, the module must map each topic to a class. The mapped class must be the most frequent class among the documents with that dominant topic. The module must compare the shift signs of each mapped topic with the ground truth table of R-SYN-7. The comparison must use the signs only. A unit conversion between bin units and feature units is not necessary. The report must contain this physical agreement fraction.

### 5.12 Robustness sweeps

R-SWP-1. The pipeline must support a sweep across `n_bins` with the values 8, 16, and 32.

R-SWP-2. The pipeline must support a sweep across `k_topics` with the values 5, 10, and 20.

R-SWP-3. Each sweep setting is 1 full pipeline run with its own output directory.

R-SWP-4. The report of each run must contain the sweep values. A conclusion about the PRH is valid only if the verdict direction is the same across all sweep settings.

R-SWP-5. Sweep orchestration code is not necessary. The operator starts each sweep run manually (Section 7.3).

### 5.13 Outputs

R-OUT-1. The pipeline must write these artifacts to `out_dir`:

| Artifact | Content |
|----------|---------|
| `report.json` | The configuration echo, all scores, all flags, all warnings |
| `bin_edges.parquet` | The bin edges for each feature |
| `topics.parquet` | The topic-token matrices for all models |
| `alignment.parquet` | The per-pair, per-size, per-seed alignment and null values |
| `plots/verdict.png` | The verdict plot |
| `plots/topics_<model>.png` | The topic-token heat map for each model |

R-OUT-2. The verdict plot must show the alignment score against the size step, with error bars across the seeds. The plot must also show the null reference band.

R-OUT-3. Each artifact write must be atomic. The module must write to a temporary file and then rename the file.

### 5.14 Error management

R-ERR-1. A missing column, a bad configuration, or an empty slice must stop the pipeline with a clear message.

R-ERR-2. A convergence warning, a degenerate topic, or a skipped size step must not stop the pipeline. The pipeline must record each of these events in the report.

R-ERR-3. The pipeline must not catch an unexpected exception. An unexpected exception must show its full trace.

## 6. Verification

### 6.1 Unit tests

The package must contain unit tests for these behaviors:

1. The bin edges are reproducible for a fixed seed (R-BIN-1, R-BIN-6).
2. The document of each object has 1 token for each feature (R-BIN-4).
3. The slices are disjoint for both slicers (R-SLC-4).
4. The JSD is symmetric, and its range is 0 to 1 (Section 3).
5. The Hungarian match recovers a known permutation of identical topics (R-MAT-2).
6. The steering shift has the correct sign on a hand-built model with 2 topics (R-STE-4).
7. The configuration parser rejects an unknown field (R-CFG-3).
8. The adapter fails on a missing mapped column (R-REAL-4).

### 6.2 Integration test on synthetic data

The integration test must run the full pipeline on the synthetic source with the default configuration. The test must assert these gates at the largest available size step:

1. The NMI is at least 0.5 for each model.
2. The purity is at least 0.6 for each model.
3. The mean alignment score at the largest size step is larger than the mean alignment score at the smallest size step.
4. The mean alignment score is above the null mean plus 2 null standard deviations, at the largest size step.
5. The physical agreement fraction of the steering check is at least 0.9.
6. All artifacts of R-OUT-1 exist.

A pipeline that fails these gates on known data has no authority on real data.

### 6.3 Real data run

The real data run is an application, not a test. The run must produce the verdict plot and the report. The run has no pass gates. The interpretation follows Section 4.

## 7. Operation procedure

### 7.1 Installation

1. Go to the package root.
2. Create a Python environment with Python 3.11 or later.
3. Install the package with the command `pip install -e ".[test]"`.
4. Run the unit tests with the command `pytest`.

### 7.2 Synthetic run

1. Open `configs/synthetic.yaml`.
2. Set `out_dir` to a new directory.
3. Run the command `python -m ldaprh.run --config configs/synthetic.yaml`.
4. Open `plots/verdict.png` and `report.json` in the output directory.
5. Make sure that the integration gates of Section 6.2 pass.

### 7.3 Sweep run

1. Copy the synthetic configuration for each sweep value of Section 5.12.
2. Set `n_bins` or `k_topics` in each copy.
3. Run the pipeline for each copy.
4. Compare the verdict direction across the outputs.

### 7.4 Real run

1. Open `configs/real.yaml`.
2. Set `real_path` to the OmniSky table location.
3. Set `column_map` for the available feature columns.
4. Set `slicer` to `random`. Change to `survey` when the provenance columns are available.
5. Run the command `python -m ldaprh.run --config configs/real.yaml`.
6. Read the verdict plot with the logic of Section 4.

## 8. Design decisions on record

1. The slicer choice sets the strength of the independence claim. Random slices give weak evidence. Survey slices give strong evidence. The synthetic source simulates surveys now. The real source starts with random slices.
2. Shared bin edges are mandatory. Without a shared vocabulary, the topic match compares different spaces (R-BIN-3).
3. Equal topic counts are mandatory for the Hungarian match (R-TRN-2).
4. The null reference makes the alignment score interpretable. A high score alone has no meaning without the shuffled baseline (R-SCO-3).
5. Documents are short: 6 tokens. A small `alpha` concentrates each mixture. The purity gate reflects this design.
6. The steering check is a rung-2 intervention on the generative model. It reads the effect of a do-operation on the mixture.
