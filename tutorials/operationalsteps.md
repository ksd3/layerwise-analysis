# Generalizing Concept Steering Beyond Transformers

Notes on what must be true — and what must be built — to port the
"find a linear direction, prove it's causal, write to it gently" recipe
(Klerings et al., EMNLP 2025, tense/aspect steering) to a non-transformer
architecture, e.g. an image classifier.

## Hypotheses to verify

1. **Linear decodability** (the linear representation hypothesis proper).
   Somewhere in the network there is a layer whose activation vectors encode the
   concept as a direction — a linear classifier on frozen activations can separate
   "concept present" from "concept absent." Cheapest to test, weakest claim: it
   only says the information is *readable*, not that it is used downstream.

2. **Causal use.** The layers *after* the chosen layer actually consume that
   direction. Separate from #1 — a probe can hit 99% accuracy on a correlated
   signal the model itself ignores. Verified only by intervening: push activations
   along the direction and see if the output changes as the concept predicts;
   project the direction *out* and see if the model loses the ability to use the
   concept. Readable-but-not-causal is the most common failure of the whole program.

3. **Additive intervenability.** "Activation + alpha × direction" must land
   somewhere the downstream network treats as a legitimate representation of
   "concept present," not off-manifold garbage. Transformers make this easy because
   the residual stream is additive by construction; architectures with strong
   normalization or saturating nonlinearities (BatchNorm, ReLU) can shrink, clip,
   or renormalize the edit away. Test explicitly: sweep strength, check outputs
   stay coherent, check activation statistics stay in-distribution.

4. **Selectivity / approximate orthogonality.** To steer concept A without
   dragging concept B, their directions must be near-independent. Verify directly:
   cosine similarity between directions; after intervening on A, probes for B
   should still read the same value. (The tense-vs-aspect orthogonality finding.)

5. **A stable read-write location.** Transformers give a canonical answer (the
   residual stream at layer L, at chosen token positions). In another architecture
   you must find the bottleneck later computation reads from — and confirm the
   concept lives there consistently across inputs, not in a different place per input.

## Setup recipe (image-classifier version)

Closest prior art: TCAV — concept activation vectors (Kim et al., 2018). This is a
causal-steering extension of it.

1. **Concept dataset.** Positives (images with stripes, wheels, snow, ...) and
   negatives, aggressively controlled for confounds. This step ruins most attempts:
   "has stripes" must not secretly mean "is a zebra photo," or the direction
   encodes zebraness.

2. **Choose the layer and the aggregation.** CNN activations are
   channel × height × width; decide what a "representation vector" is — usually the
   channel vector, globally average-pooled or per spatial position (the analog of
   the per-token choice). Sweep layers; concepts tend to become linearly decodable
   mid-to-late network.

3. **Extract the direction.** Fit a linear probe or LDA on frozen activations; the
   direction is the classifier's normal vector, or simply the difference of class
   means. Sanity-check held-out probe accuracy against controls (shuffled labels,
   random directions) — TCAV formalizes this with significance tests over many
   random negative sets.

4. **Causal test.** During a forward pass, add alpha × direction at that layer
   (everywhere spatially, or at chosen positions) and measure prediction shifts:
   does adding "stripes" move horse toward zebra? Then the inverse: project the
   direction out and check the model gets *worse* at using the concept — the
   "model really uses this" check.

5. **Selectivity and dose tuning.** Sweep alpha; plot concept-shift against side
   effects (other class probabilities moving, accuracy collapse — the classifier
   analog of topic-shift and degeneration metrics). Expect useful alpha to scale
   with the layer's activation norms.

## Structural caveat

A classifier is one-shot: "steering" means changing a single prediction — closer
to a counterfactual probe than to control. The language-model story, where the
edit compounds over many generation steps, only has an analog in generative image
models — and latent-direction editing in GANs / diffusion models ("add the smile
vector") is the same hypothesis stack validated in that setting.

## Summary

The transferable core: **find a linear direction, prove it's causal, write to it
gently.** Everything else — where to read, what to pool, how hard to push — is
architecture-specific engineering, and every convenience the transformer gave for
free becomes a hypothesis to test.
