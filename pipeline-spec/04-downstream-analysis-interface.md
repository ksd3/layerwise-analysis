# Side-car: The Downstream Analysis Interface (why the dataset is shaped this way)

> Not part of the data-generation pipeline — this is the **contract** between the
> dataset and the Platonic-Representation analysis (which lives in
> `UniverseTBD/platonic-universe`). It tells you, a junior researcher, *what the
> analysis needs from the data*, so the dataset is built to serve it.

## 1. What the experiment actually does
The PRH analysis asks: do two models — trained on **different instruments/modalities**
— organize the **same objects** the same way in their representation spaces?
Pipeline: (1) **extract** embeddings by forward-passing each object through each model,
layer by layer; (2) **compute** an alignment score between every pair of (model, layer)
embeddings; (3) optionally **regress** physical labels (redshift, M*, sSFR) as a sanity
anchor. Tiers map to the platonic-universe repo's `extract/ compute/ regress/`. **Resource note:** this side-car is where the GPUs belong — embedding extraction (model forward passes), CKA's O(N²) Gram matrices, and mutual-kNN (FAISS-GPU) are GPU-bound, *unlike* the CPU/network-bound data-generation pipeline, which should not consume GPU allocation (doc 01 §3.8).

## 2. The two metrics and what they demand of the data
- **Mutual k-NN alignment (mNN):** for the **same N objects**, embed with model A and
  model B; for each object compare its k nearest neighbors in A's space vs B's space;
  score = mean overlap / k. Local, batchable, the PRH paper's preferred metric (k=10).
- **CKA (Centered Kernel Alignment):** the global limit (k→N) of the same idea; uses
  all pairwise similarities; O(N²) memory.

**The hard requirement both share:** the **same set of N physical objects must be
encoded by both models** — index *i* must mean the *same object* in both modalities.
This is exactly why the dataset is **cross-matched and keyed by `global_object_id`**:
that ID is the anchor that makes "the same object in modality A and modality B"
well-defined. Without a stable shared key, mNN/CKA are meaningless.

## 3. Sample size (reassuring news)
- AstroCLIP (arXiv:2310.03024) aligned image↔spectrum on **197,632** galaxies (matched
  by shared TARGETID; 144×144 image crops; per-dataset z-score norm).
- The PRH paper (arXiv:2405.07987) got clean, interpretable trends with **N≈1,000–1,024
  paired objects** at k=10. No formal minimum is given, but ~1k well-matched pairs is a
  working baseline; thousands is comfortable.
- **Implication:** the bottleneck is **clean ≥2-instrument pairs**, not millions of
  objects. A few thousand *trustworthy* image+spectrum pairs already powers the
  headline measurement; more pairs mainly tighten error bars and let you slice by
  object type / capacity. This is why doc 01 optimizes for match *quality* over volume.

## 4. What the dataset must therefore guarantee
1. **A stable shared key** (`global_object_id`) so the *same* object is identifiable
   across every modality. (P0 in the spec.)
2. **Paired coverage:** enough objects with **both** of any two modalities of interest
   (esp. **image + spectrum** for galaxies, the AstroCLIP-style pairing the PRH analysis
   leans on). Track this with `n_modality_types` / `instrument_presence_mask`.
3. **Fixed, model-ready array shapes** (e.g. 64×64 images, fixed-length spectra) so the
   extract tier can batch without bespoke padding per object.
4. **Raw values + documented normalization** so each model applies the preprocessing it
   expects (asinh/zscale for images; continuum norm for spectra) — the dataset must not
   pre-bake a single normalization that biases cross-model comparison.
5. **A clean spatial split** so any probing/regression doesn't leak across train/test.
6. **Provenance** (which instrument, which release, match separation/ambiguity) so the
   analysis can filter to high-confidence pairs and attribute any (non)convergence to
   real physics rather than match contamination.

## 5. Minimal interface sketch (what the extract tier will call)
```python
# Conceptual — the analysis iterates objects with >=2 modalities and pulls raw arrays.
ds = load_release("kshitij/omnisky-v5", streaming=True)
for obj in ds:
    if obj["n_instruments_present"] < 2:        # the >=2-instrument guarantee
        continue
    oid   = obj["global_object_id"]             # stable anchor across modalities
    image = decode_image(obj, band="legacy_grz")  # raw -> model's own asinh/zscale
    spec  = decode_spectrum(obj, source="desi")    # raw flux + ivar -> model's norm
    emit(oid, modality="image",    embedding=model_img(image))
    emit(oid, modality="spectrum", embedding=model_spec(spec))
# Alignment then joins emitted embeddings on `oid` and computes mNN/CKA over shared oids.
```
The single load-bearing detail: **everything joins on `global_object_id`.** Build the
dataset so that key is unique, stable, and present on every modality, and the
downstream analysis is straightforward.

## 6. References
- Huh, Cheung, Wang, Isola 2024 — *The Platonic Representation Hypothesis*, arXiv:2405.07987 (mNN/CKA, k=10, N≈1k).
- Duraphe, Smith, Sourav, Wu 2025 — *The Platonic Universe: Do Foundation Models See the Same Sky?*, arXiv:2509.19453.
- Parker, Lanusse et al. 2024 — *AstroCLIP*, arXiv:2310.03024 (197,632 image–spectrum pairs by TARGETID).
- Smith, Roberts, Angeloudi, Huertas-Company 2024 — *AstroPT*, arXiv:2405.14930 (8.6M images; multimodal extension arXiv:2503.15312).
