# Interpretability tutorials

Three self-contained, conference-companion-style Jupyter notebooks, each a
proof-of-concept of a current interpretability method on synthetic data. All are
fully executed (figures embedded), seeded, and re-runnable top to bottom in a few
minutes on GPU (CPU fallback built in).

| Notebook | Method | Key demos |
|---|---|---|
| `01_concept_directions_steering.ipynb` | Linear concept directions + activation steering (Klerings et al., EMNLP 2025; TCAV, Kim et al. 2018) | layer-wise probes, steering dose–response, shortcut detection, concept erasure via mean-ablation |
| `02_mini_vpd_weight_decomposition.ipynb` | adVersarial Parameter Decomposition (Goodfire AI, 2026; miniature of `goodfire-ai/param-decomp` `nano_param_decomp/run.py`) | rank-1 components + CI function, four-term minimax loss, sign-PGD adversary, single-mechanism surgery |
| `03_jacobian_lens_jspace.ipynb` | Jacobian lens / J-space global workspace (Anthropic, transformer-circuits.pub 2026) | J-lens vs logit lens, workspace readout, France→China coordinate swap, workspace-ablation task dissociation |

`operationalsteps.md` collects the hypothesis checklist the notebooks refer to:
what must be verified (linear decodability, causal use, additive intervenability,
selectivity, stable read–write location) before porting any of these methods to a
non-transformer model such as an image classifier.

All datasets are synthetic by design (shapes, sparse features, an entity–attribute
toy language) so every mechanism is verifiable by construction; the notebooks state
their simplifications relative to the real methods explicitly.

Requirements: `torch`, `numpy`, `matplotlib`, `jupyter` (and `jupytext` if you want
to edit the paired scripts).
