"""Synthetic data generators for ID estimator calibration.

* Every "clean manifold" / heterogeneous generator wraps ``skdim.datasets``
  directly instead of reimplementing manifold geometry, so ground-truth ID
  values track the installed skdim version rather than a second hardcoded
  copy. Null and noise generators are self-contained (numpy/scipy only).
* Ambient-space embedding (zero-pad + Haar rotation) and isotropic noise
  injection at a target SNR are each implemented once and shared.
* Every generator returns a :class:`SyntheticDataset` carrying the data,
  the ground-truth ID, optional labels, and the full parameter record, so
  downstream harness sections never have to re-derive ground truth.

We also handle skdim 0.3.4 quirks
-------------------------------------------------------------------------
1. ``BenchmarkManifolds.generate(name=key, n=n)`` -- i.e. omitting dim/d --
   is broken: the method computes ``data`` conditionally on whether dim/d
   were given, then unconditionally overwrites it with
   ``self.dict_gen[name](n=n, dim=dim, d=d)``, passing the *original*
   (still-None) dim/d. For manifolds whose generator function has no
   defaults (M8/Mn1/Mn2 all use ``_gen_nonlinear_data(self, n, dim, d)`` /
   ``_gen_campadelli_n_data``), this raises ``TypeError: missing 2 required
   positional arguments``. Fix: always pass dim/d explicitly, read from
   ``bm.truth`` first.
2. ``swissRoll3Sph(n_swiss, n_sphere=0)`` still returns 4 columns
   (x, y, z, w) with w identically zero -- sliced to the first 3 for a
   true 3D swiss roll.

Sources
-------
* Anisotropic power-law null (alpha=1): Stringer et al. 2019, Nature 571:361.
* Swiss roll TwoNN ~ 2.01 anchor: Facco et al. 2017.
* M8/Mn1/Mn2 benchmark manifolds: Campadelli et al. 2015, via
  ``skdim.datasets.BenchmarkManifolds``.
* Flat-vs-curved (ID vs PC-ID) framing: Ansuini et al. 2019.
* Nonlinear planted quantity = normalized geodesic arclength along a
  multi-turn Archimedean spiral with a tubular nuisance sheet (see
  :func:`planted_quantity` and :func:`_curved_centerline` for why).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
from scipy.stats import ortho_group
import skdim.datasets as skd

# =====================================================================
# Output container
# =====================================================================
@dataclass
class SyntheticDataset:
    """Container for one synthetic draw.

    Attributes
    ----------
    name : generator name.
    X : (n, D) data matrix.
    id_true : ground-truth intrinsic dimension. ``int`` for single
        manifolds, ``dict`` for mixed-population sets (Use 4), and for
        nulls it equals the ambient dimension D.
    y : (n,) planted scalar quantity (only ``planted_quantity``).
    labels : (n,) integer population labels (only heterogeneous sets).
    meta : full parameter record (seed, snr, sigma, mode, source, ...).
    """
    name: str
    X: np.ndarray
    id_true: Union[int, dict]
    y: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    @property
    def n(self) -> int:
        return self.X.shape[0]
    @property
    def ambient_dim(self) -> int:
        return self.X.shape[1]


# =====================================================================
# RNG/Helpers
# =====================================================================
def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)

def _haar(D: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal matrix, seeded from ``rng``.

    ``ortho_group.rvs`` is given an integer seed drawn from ``rng`` so
    behavior is identical across scipy versions that do / don't accept
    ``np.random.Generator`` directly.
    """
    return ortho_group.rvs(D, random_state=int(rng.integers(2**31 - 1)))

def embed_ambient(
    X: np.ndarray,
    D: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Embed (n, d0) data into ambient R^D: zero-pad then Haar-rotate.

    The rotation is applied even when D == d0 so no generator leaks
    axis-aligned structure to estimators. ``D=None`` is a no-op
    (data stays in its native coordinates).
    """
    if D is None:
        return X
    n, d0 = X.shape
    if d0 > D:
        raise ValueError(f"ambient dim D={D} < native data dim {d0}. Ambient dim needs to be able to hold the data's native dimension.")
    Xp = np.zeros((n, D), dtype=float)
    Xp[:, :d0] = X
    return Xp @ _haar(D, rng).T

def add_isotropic_noise(
    X: np.ndarray,
    snr: Optional[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Add isotropic Gaussian noise at a target SNR. Returns (X_noisy, sigma).

    SNR is defined as E||x - x_bar||^2 / E||eps||^2 with
    eps ~ N(0, sigma^2 I_D), so sigma^2 = signal_power / (snr * D).
    ``snr=None`` or ``inf`` is a no-op (sigma = 0).
    """
    if snr is None or np.isinf(snr):
        return X, 0.0
    if snr <= 0:
        raise ValueError("snr must be positive")
    D = X.shape[1]
    signal_power = float(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1)))
    sigma = float(np.sqrt(signal_power / (snr * D)))
    return X + rng.normal(0.0, sigma, size=X.shape), sigma


# =====================================================================
# 1.1 Null distributions (no planted structure) -- self-contained
# =====================================================================

def iso_gaussian(n: int, D: int, seed: Optional[int] = None) -> SyntheticDataset:
    """X ~ N(0, I_D). Ground-truth ID = D. Simplest null."""
    rng = _rng(seed)
    X = rng.normal(size=(n, D))
    return SyntheticDataset("iso_gaussian", X, D, meta=dict(seed=seed))


def aniso_gaussian(
    n: int,
    D: int,
    alpha: float = 1.0,
    rotate: bool = True,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """X ~ N(0, Sigma), eigenvalues lambda_k ∝ k^-alpha (alpha=1 default).

    Realistic null matching observed neural covariance spectra
    (Stringer et al. 2019). Ground-truth ID = D: shaped covariance is
    anisotropy, not a manifold. ``rotate=True`` applies a Haar rotation
    so the covariance is not axis-aligned.
    """
    rng = _rng(seed)
    lam = np.arange(1, D + 1, dtype=float) ** (-alpha)
    X = rng.normal(size=(n, D)) * np.sqrt(lam)
    if rotate:
        X = X @ _haar(D, rng).T
    return SyntheticDataset(
        "aniso_gaussian", X, D,
        meta=dict(seed=seed, alpha=alpha, rotate=rotate, eigenvalues=lam,
                  source="Stringer et al. 2019 Nature 571:361"),
    )


def oblong_normal(
    n: int,
    D: int,
    sigma_hi: float = 1.0,
    sigma_lo: float = 0.25,
    rotate: bool = True,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Bimodal-variance Gaussian: ceil(D/2) dims at sigma_hi, rest at sigma_lo.

    Models a layer where a few PCs dominate (16x variance ratio at the
    defaults). Ground-truth ID = D. Complementary null to the smooth
    power-law decay: a differing null value here vs ``aniso_gaussian``
    at the same D flags covariance-*shape* sensitivity.
    """
    rng = _rng(seed)
    d_hi = int(np.ceil(D / 2))
    sig = np.concatenate([np.full(d_hi, sigma_hi), np.full(D - d_hi, sigma_lo)])
    X = rng.normal(size=(n, D)) * sig
    if rotate:
        X = X @ _haar(D, rng).T
    return SyntheticDataset(
        "oblong_normal", X, D,
        meta=dict(seed=seed, sigma_hi=sigma_hi, sigma_lo=sigma_lo,
                  n_hi=d_hi, rotate=rotate),
    )


# =====================================================================
# 1.2 Planted quantity (direction vs manifold) -- self-contained
# =====================================================================

def _curved_centerline(
    a: float, b: float, nturn: float, scale: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Archimedean spiral: gamma(t) = scale*(a+b*t)*[cos(th), sin(th)],
    th = 2*pi*nturn*t, t in [0,1]. Same parametrization family as the
    swiss roll centerline (skdim defaults a=1, b=2-a, nturn=1.5).

    Why a spiral and not a curve with a monotone coordinate (e.g.
    [t, (t-1/2)^2, sin(2*pi*t)]): if the label is (nearly) in the linear
    span of the coordinate functions, an ambient linear probe decodes it
    with R^2 ~ 1 and the probe gap vanishes -- verified empirically
    (linear R^2 = 0.994 on that curve). A multi-turn expanding spiral has
    no linear function of coordinates monotone along the curve, so the
    label is genuinely nonlinearly stored, while the growing radius keeps
    the curve non-self-intersecting and the arclength label single-valued
    and non-periodic (unlike a raw angle).
    """
    def gamma(t: np.ndarray) -> np.ndarray:
        th = 2.0 * np.pi * nturn * t
        r = scale * (a + b * t)
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    return gamma


def _arclength_reparam(
    gamma: Callable[[np.ndarray], np.ndarray],
    n_grid: int = 20001,
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Return (s -> t) inverse map and total curve length.

    s in [0, 1] is *normalized geodesic arclength* along gamma. Built by
    dense numerical cumulative arclength + linear interpolation, so
    sampling s ~ Uniform(0,1) gives points uniform in arclength (uniform
    density on the curve), and the planted label y = s is the intrinsic
    coordinate rather than the embedding-dependent parameter t.
    """
    t_grid = np.linspace(0.0, 1.0, n_grid)
    pts = gamma(t_grid)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    s_grid = cum / total

    def t_of_s(s: np.ndarray) -> np.ndarray:
        return np.interp(s, s_grid, t_grid)

    return t_of_s, float(total)


def planted_quantity(
    n: int,
    D: int,
    mode: str = "linear",
    m_nuisance: int = 8,
    tube_width: float = 0.25,
    snr: Optional[float] = None,
    label_noise: float = 0.0,
    spiral_a: float = 1.0,
    spiral_b: float = 1.0,
    nturn: float = 1.5,
    curve_scale: float = 1.0,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Scalar quantity planted in R^D, stored linearly or on a curved coordinate.

    linear mode
        x = s*w + B u + noise, y = s + eps_y, with [w | B] an orthonormal
        frame. The quantity occupies one straight axis: level sets are
        parallel flat sheets (level-set collection ID ~ 1), and the
        linear-vs-nonlinear probe R^2 gap is ~ 0. Null for "quantity is
        a direction".

    nonlinear mode
        Tubular-coordinate construction: x = gamma(s) + B_perp u + noise,
        where gamma is a multi-turn Archimedean spiral (see
        :func:`_curved_centerline` for why a spiral rather than a curve
        with a monotone coordinate) and the planted label y = s is the
        *normalized geodesic arclength* along gamma -- the intrinsic 1D
        coordinate of the concept, independent of rotation/embedding.
        Arclength (not the ambient angle) is the label because the target
        astrophysical quantities are ordered, non-periodic scalars; the
        expanding radius keeps arclength single-valued even as the angle
        wraps. The nuisance sheet spans a fixed orthonormal complement of
        the 2D spiral plane (a valid tube: globally orthogonal to the
        centerline's span; exact Frenet normal frames are unnecessary for
        this calibration). No linear readout is monotone along the
        spiral, so the linear probe underfits and the probe gap is > 0.
        Power case for "quantity is a manifold". ``nturn`` is the
        difficulty knob: more turns, larger gap.

    Total manifold ID in both modes = 1 + m_nuisance (before noise).
    """
    if mode not in ("linear", "nonlinear"):
        raise ValueError("mode must be 'linear' or 'nonlinear'")
    rng = _rng(seed)

    s = rng.uniform(0.0, 1.0, size=n)
    u = rng.uniform(-1.0, 1.0, size=(n, m_nuisance)) * tube_width

    if mode == "linear":
        low = np.concatenate([(s * curve_scale)[:, None], u], axis=1)
        curve_length = curve_scale
    else:
        gamma = _curved_centerline(spiral_a, spiral_b, nturn, curve_scale)
        t_of_s, curve_length = _arclength_reparam(gamma)
        center = gamma(t_of_s(s))                     # (n, 2)
        low = np.concatenate([center, u], axis=1)     # nuisance ⟂ spiral plane

    X = embed_ambient(low, D, rng)
    X, sigma = add_isotropic_noise(X, snr, rng)
    y = s + (rng.normal(0.0, label_noise, size=n) if label_noise > 0 else 0.0)

    return SyntheticDataset(
        "planted_quantity", X, 1 + m_nuisance, y=y,
        meta=dict(seed=seed, mode=mode, m_nuisance=m_nuisance,
                  tube_width=tube_width, snr=snr, sigma_noise=sigma,
                  label_noise=label_noise, spiral_a=spiral_a,
                  spiral_b=spiral_b, nturn=nturn,
                  curve_scale=curve_scale, curve_length=curve_length,
                  quantity_coordinate=(
                      "straight axis" if mode == "linear"
                      else "normalized geodesic arclength along curved centerline"),
                  expected_levelset_id=1 if mode == "linear" else ">1",
                  expected_probe_gap="~0" if mode == "linear" else ">0"),
    )


# =====================================================================
# 1.3 Clean manifolds (known ID, no noise)
# =====================================================================

def planted_manifold(
    n: int,
    intrinsic_dim: int,
    D: int,
    n_lift: Optional[int] = None,
    lift_scale: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Uniform k-ball + quadratic nonlinear lift, Haar-embedded in R^D.

    The ball is drawn with ``skdim.datasets.hyperBall`` (wrapping, not
    reimplementing, the reference construction). The lift appends
    ``n_lift`` coordinates z_j = u^T A_j u (random symmetric A_j), a
    smooth injective map, so ground-truth ID stays exactly k while the
    embedding is curved -- mimicking learned representations better
    than a flat ball. Sweep ``intrinsic_dim`` for the power curve.
    """
    rng = _rng(seed)
    k = intrinsic_dim
    if n_lift is None:
        n_lift = max(2, k // 2)

    u = skd.hyperBall(
        n, d=k, radius=1.0,
        random_state=int(rng.integers(2**31 - 1)),
    )

    A = rng.normal(size=(n_lift, k, k))
    A = (A + np.transpose(A, (0, 2, 1))) / 2.0            # symmetric
    z = lift_scale * np.einsum("ni,mij,nj->nm", u, A, u)  # (n, n_lift)

    low = np.concatenate([u, z], axis=1)
    X = embed_ambient(low, D, rng)
    return SyntheticDataset(
        "planted_manifold", X, k,
        meta=dict(seed=seed, n_lift=n_lift, lift_scale=lift_scale,
                  source="skdim.datasets.hyperBall + quadratic lift"),
    )


def swiss_roll(
    n: int,
    D: Optional[int] = None,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Canonical swiss roll (ID = 2) via ``skdim.datasets.swissRoll3Sph``.

    Called with n_sphere=0; skdim still returns 4 columns with the 4th
    identically zero, so we slice to the first 3 (quirk #2 in module
    docstring). External anchor: Facco et al. 2017 report TwoNN ~ 2.01.
    Optionally Haar-embedded into R^D.
    """
    rng = _rng(seed)
    X3 = skd.swissRoll3Sph(
        n, 0, random_state=int(rng.integers(2**31 - 1))
    )[:, :3]
    X = embed_ambient(X3, D, rng)
    return SyntheticDataset(
        "swiss_roll", X, 2,
        meta=dict(seed=seed, ambient=D or 3,
                  anchor="Facco et al. 2017: TwoNN ~ 2.01",
                  source="skdim.datasets.swissRoll3Sph (4th zero col sliced)"),
    )


def hyper_twin_peaks(
    n: int,
    intrinsic_dim: int = 2,
    D: Optional[int] = None,
    height: float = 1.0,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Hypercube with peaked height map via ``skdim.datasets.hyperTwinPeaks``.

    Ground-truth ID = intrinsic_dim; native ambient = intrinsic_dim + 1.
    Oscillating, sign-changing curvature -- geometrically distinct from
    the roll's monotone curling -- and sweepable in k for a second
    independent power curve. Optionally Haar-embedded into R^D.
    """
    rng = _rng(seed)
    Xk = skd.hyperTwinPeaks(
        n, d=intrinsic_dim, height=height,
        random_state=int(rng.integers(2**31 - 1)),
    )
    X = embed_ambient(Xk, D, rng)
    return SyntheticDataset(
        "hyper_twin_peaks", X, intrinsic_dim,
        meta=dict(seed=seed, height=height, ambient=D or intrinsic_dim + 1,
                  source="skdim.datasets.hyperTwinPeaks"),
    )


# --- Campadelli high-ID benchmarks (M8 / Mn1 / Mn2) -------------------

# Public alias -> skdim key. IDs/dims are NOT hardcoded here; they are
# read from BenchmarkManifolds.truth at call time (single source of truth).
_BENCHMARK_ALIASES = {
    "M8_Nonlinear": "M8_Nonlinear",
    "M_N1": "Mn1_Nonlinear",
    "M_N2": "Mn2_Nonlinear",
}


def benchmark_manifold(
    name: str,
    n: int,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Campadelli et al. 2015 high-ID benchmark via ``BenchmarkManifolds``.

    ``name`` in {'M8_Nonlinear', 'M_N1', 'M_N2'} (aliases map to skdim
    keys 'Mn1_Nonlinear' / 'Mn2_Nonlinear'). dim and d are ALWAYS passed
    explicitly, read from ``bm.truth`` -- omitting them is broken in
    skdim 0.3.4 for these manifolds (quirk #1 in module docstring).
    """
    if name not in _BENCHMARK_ALIASES:
        raise ValueError(f"name must be one of {list(_BENCHMARK_ALIASES)}")
    key = _BENCHMARK_ALIASES[name]

    bm = skd.BenchmarkManifolds(random_state=seed)
    row = bm.truth.loc[key]
    d = int(row["Intrinsic Dimension"])
    dim = int(row["Number of variables"])

    X = bm.generate(name=key, n=n, dim=dim, d=d)
    return SyntheticDataset(
        name, np.asarray(X, dtype=float), d,
        meta=dict(seed=seed, skdim_key=key, ambient=dim,
                  description=str(row["Description"]),
                  source="Campadelli et al. 2015 via skdim.BenchmarkManifolds"),
    )


# =====================================================================
# 1.3b Heterogeneous ID manifold (mixed structure)
# =====================================================================

def swiss_roll_3sphere(
    n_roll: int,
    n_sphere: int,
    D: Optional[int] = None,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """Swiss roll (ID=2) + 3-sphere (ID=3) sharing one space (native R^4).

    Wraps ``skdim.datasets.swissRoll3Sph``; skdim stacks roll rows first,
    then sphere rows, which fixes the label vector. Ground truth is
    per-population: pointwise TwoNN should be bimodal here; if it is
    not, global TwoNN masks heterogeneity (Use 4 calibration).
    """
    rng = _rng(seed)
    X4 = skd.swissRoll3Sph(
        n_roll, n_sphere, random_state=int(rng.integers(2**31 - 1))
    )
    labels = np.concatenate(
        [np.zeros(n_roll, dtype=int), np.ones(n_sphere, dtype=int)]
    )
    X = embed_ambient(X4, D, rng)
    return SyntheticDataset(
        "swiss_roll_3sphere", X, {"swiss_roll": 2, "sphere3": 3},
        labels=labels,
        meta=dict(seed=seed, n_roll=n_roll, n_sphere=n_sphere,
                  ambient=D or 4, label_order="roll rows first, then sphere",
                  source="skdim.datasets.swissRoll3Sph"),
    )


# =====================================================================
# 1.4 Noisy manifold
# =====================================================================

def planted_manifold_noisy(
    n: int,
    intrinsic_dim: int,
    D: int,
    snr: float,
    n_lift: Optional[int] = None,
    lift_scale: float = 0.5,
    seed: Optional[int] = None,
) -> SyntheticDataset:
    """``planted_manifold`` + shared isotropic noise injection at given SNR.

    Ground truth is scale-dependent: ID = k at large scale, -> D at the
    noise floor. Calibrates the SNR threshold at which NN estimators
    inflate toward ambient dimension.
    """
    clean = planted_manifold(
        n, intrinsic_dim, D, n_lift=n_lift, lift_scale=lift_scale, seed=seed
    )
    rng = _rng(None if seed is None else seed + 1)  # independent noise stream
    X, sigma = add_isotropic_noise(clean.X, snr, rng)
    meta = dict(clean.meta)
    meta.update(snr=snr, sigma_noise=sigma,
                id_true_note=f"{intrinsic_dim} at large scale, {D} at noise floor")
    return SyntheticDataset("planted_manifold_noisy", X, intrinsic_dim, meta=meta)


# =====================================================================
# Registry + smoke test
# =====================================================================

GENERATORS: dict[str, Callable[..., SyntheticDataset]] = {
    # 1.1 nulls (self-contained)
    "iso_gaussian": iso_gaussian,
    "aniso_gaussian": aniso_gaussian,
    "oblong_normal": oblong_normal,
    # 1.2 planted quantity (self-contained)
    "planted_quantity": planted_quantity,
    # 1.3 clean manifolds (skdim-wrapped except planted_manifold's lift)
    "planted_manifold": planted_manifold,
    "swiss_roll": swiss_roll,
    "hyper_twin_peaks": hyper_twin_peaks,
    "benchmark_manifold": benchmark_manifold,
    # 1.3b heterogeneous (skdim-wrapped)
    "swiss_roll_3sphere": swiss_roll_3sphere,
    # 1.4 noisy (self-contained noise on top of planted_manifold)
    "planted_manifold_noisy": planted_manifold_noisy,
}

if __name__ == "__main__":
    checks: list[tuple[str, SyntheticDataset, tuple]] = []

    checks.append(("iso", iso_gaussian(500, 64, seed=0), (500, 64)))
    checks.append(("aniso", aniso_gaussian(500, 64, seed=0), (500, 64)))
    checks.append(("oblong", oblong_normal(500, 64, seed=0), (500, 64)))

    pl = planted_quantity(500, 64, mode="linear", seed=0)
    pn = planted_quantity(500, 64, mode="nonlinear", snr=100.0, seed=0)
    checks.append(("pq_lin", pl, (500, 64)))
    checks.append(("pq_nonlin", pn, (500, 64)))
    assert pl.y is not None and pn.y is not None
    assert pn.meta["curve_length"] > 1.0  # bent curve is longer than chord

    checks.append(("pm_k4", planted_manifold(500, 4, 64, seed=0), (500, 64)))
    checks.append(("swiss", swiss_roll(500, seed=0), (500, 3)))
    checks.append(("swiss_D", swiss_roll(500, D=64, seed=0), (500, 64)))
    checks.append(("htp_d4", hyper_twin_peaks(500, 4, seed=0), (500, 5)))

    for nm in ("M8_Nonlinear", "M_N1", "M_N2"):
        ds = benchmark_manifold(nm, 300, seed=0)
        checks.append((nm, ds, (300, ds.meta["ambient"])))
    assert benchmark_manifold("M8_Nonlinear", 100, seed=0).id_true == 12
    assert benchmark_manifold("M_N1", 100, seed=0).id_true == 18
    assert benchmark_manifold("M_N2", 100, seed=0).id_true == 24

    het = swiss_roll_3sphere(300, 300, seed=0)
    checks.append(("het", het, (600, 4)))
    assert het.labels is not None and het.labels.sum() == 300

    noisy = planted_manifold_noisy(500, 4, 64, snr=10.0, seed=0)
    checks.append(("pm_noisy", noisy, (500, 64)))
    assert noisy.meta["sigma_noise"] > 0

    for tag, ds, shape in checks:
        ok = ds.X.shape == shape and np.isfinite(ds.X).all()
        print(f"{'PASS' if ok else 'FAIL'}  {tag:12s} shape={ds.X.shape} "
              f"id_true={ds.id_true}")
        assert ok, tag

    # noise SNR sanity: realized SNR within 5% of requested
    rng = np.random.default_rng(1)
    Xc = rng.normal(size=(2000, 32))
    Xn, sig = add_isotropic_noise(Xc, snr=10.0, rng=rng)
    realized = np.mean(np.sum((Xc - Xc.mean(0)) ** 2, 1)) / (32 * sig**2)
    assert abs(realized / 10.0 - 1) < 0.05, realized
    print(f"PASS  snr_check    realized={realized:.3f} (target 10.0)")

    print("all smoke tests passed")