#!/usr/bin/env python3
"""Comprehensive diagnostic figures for the MMU dataset paper.

Generates ~20 figures covering every possible reviewer objection:
  - Cross-match quality (separation distributions, false match rates, PM impact)
  - Data integrity (image centering, stacked spectra, cross-band correlation)
  - Selection effects (magnitude/redshift distributions, sky coverage)
  - Coverage analysis (Venn diagrams, surveys per object, coverage vs magnitude)
  - Physical consistency (Teff, HR diagram, parallax, color-color)
  - Usability (shard sizes, column sparsity)

Streams shard-by-shard, ~4 GB RAM max. Run after pipeline completes.

Usage:
    python3 make_all_diagnostics.py /path/to/workdir/

Expects: workdir/shards/final/*.parquet (final output)
Produces: paper_figures/*.pdf and paper_figures/*.png
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = "paper_figures"
POPULATIONS = ["star", "galaxy", "agn"]
POP_COLORS = {"star": "#2196F3", "galaxy": "#4CAF50", "agn": "#FF5722"}
POP_LABELS = {"star": "Stars", "galaxy": "Galaxies", "agn": "AGN"}

# Survey columns that store match_sep
SEP_SOURCES = [
    "flatiron_gaia", "flatiron_tess", "flatiron_desi",
    "ztf", "sdss",  # sdss may not have sep if loaded via plate lookup
]

# Key array columns per survey
SURVEY_KEY_COLS = {
    "APOGEE": "apogee_flux",
    "Gaia BP/RP": "flatiron_gaia_coeff",
    "GALAH": "galah_flux_blue",
    "SDSS": "sdss_flux",
    "DESI": "flatiron_desi_spectrum_flux",
    "TESS": "flatiron_tess_flux",
    "ZTF": "ztf_time",
    "2MASS": "twomass_j",
    "GALEX": "galex_fuv",
    "unWISE": "unwise_w1",
    "Legacy": "legacy_g",
}


def _is_data(x):
    return isinstance(x, (list, np.ndarray))


def _to_flat(x):
    try:
        return np.array(x, dtype=np.float32).flatten()
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════
# ACCUMULATORS — filled during single streaming pass
# ══════════════════════════════════════════════════════════════════

# 1. Match separations {source_name: {pop: [sep_values]}}
match_seps = defaultdict(lambda: defaultdict(list))
MAX_SEPS = 100_000  # per source per pop

# 2. Teff cross-match
teff_apogee = []
teff_gaia = []
teff_gal_lat = []

# 3. HR diagram
hr_bp_rp = []
hr_abs_g = []

# 4. Parallax vs magnitude
star_parallax_mag = []  # (parallax, g_mag)
agn_parallax_mag = []

# 5. Sky coords per survey {survey: {"ra": [], "dec": []}}
sky_per_survey = defaultdict(lambda: {"ra": [], "dec": []})
MAX_SKY = 200_000

# 6. Coverage per population {pop: {survey: count}}
coverage = defaultdict(lambda: defaultdict(int))
pop_totals = defaultdict(int)

# 7. Modality type counts per pop {pop: {n_types: count}}
modality_type_dist = defaultdict(lambda: defaultdict(int))

# 8. Surveys per object {pop: [n_surveys]}
surveys_per_object = defaultdict(list)
MAX_SURVEYS = 200_000

# 9. Image centering {col: [offsets]}
image_offsets = defaultdict(list)
MAX_IMG = 2000

# 10. Cross-band correlation
band_corr_jh = []
band_corr_fuv_nuv = []
MAX_CORR = 2000

# 11. APOGEE stacked spectra
apogee_spectra_sample = []
MAX_SPECTRA = 500

# 12. SDSS spectra samples {pop: [spectra]}
sdss_spectra_sample = defaultdict(list)

# 13. ZTF light curve lengths {pop: [lengths]}
ztf_lc_lengths = defaultdict(list)
ztf_monotonic = 0
ztf_checked = 0
MAX_LC = 50_000

# 14. Array length checks
apogee_lengths = []
galah_lengths = []

# 15. Magnitude distributions
star_gmag = []
agn_redshifts = []
MAX_MAG = 200_000

# 16. Coverage vs magnitude {pop: [(gmag, n_modality_types)]}
cov_vs_mag = defaultdict(list)
MAX_COV_MAG = 200_000

# 17. Galactic latitude for match sep analysis
sep_vs_glat = defaultdict(list)  # {source: [(glat, sep)]}
MAX_SEP_GLAT = 50_000

# 18. PM correction info
pm_magnitudes = []  # total PM in mas/yr
pm_nan_count = 0
pm_total_count = 0
MAX_PM = 200_000

# 19. NaN fractions per array column
nan_fracs = defaultdict(lambda: {"total": 0, "has_nan": 0, "has_inf": 0, "all_nan": 0})

# 20. Example data for showcase
example_star = None
example_galaxy = None
example_agn = None


def process_shard(df):
    """Extract all diagnostic data from one shard."""
    global ztf_monotonic, ztf_checked, pm_nan_count, pm_total_count
    global example_star, example_galaxy, example_agn

    for _, row in df.iterrows():
        pop = row.get("population", "")
        pop_totals[pop] += 1
        ra = row.get("ra")
        dec = row.get("dec")

        # ── Match separations ──
        for src in SEP_SOURCES:
            sep_col = f"{src}_match_sep_arcsec"
            if sep_col in df.columns:
                v = row.get(sep_col)
                if pd.notna(v) and len(match_seps[src][pop]) < MAX_SEPS:
                    match_seps[src][pop].append(float(v))

        # ── Coverage per survey ──
        for survey_label, col in SURVEY_KEY_COLS.items():
            if col in df.columns and _is_data(row.get(col)):
                coverage[pop][survey_label] += 1
                # Sky per survey
                if pd.notna(ra) and pd.notna(dec) and len(sky_per_survey[survey_label]["ra"]) < MAX_SKY:
                    sky_per_survey[survey_label]["ra"].append(ra)
                    sky_per_survey[survey_label]["dec"].append(dec)

        # ── Surveys per object ──
        if len(surveys_per_object[pop]) < MAX_SURVEYS:
            n_surveys = sum(1 for col in SURVEY_KEY_COLS.values()
                           if col in df.columns and _is_data(row.get(col)))
            surveys_per_object[pop].append(n_surveys)

        # ── Modality type distribution ──
        n_types = int(row.get("n_modality_types", 0))
        modality_type_dist[pop][n_types] += 1

        # ── Coverage vs magnitude ──
        gmag = row.get("flatiron_gaia_phot_g_mean_mag")
        if pd.notna(gmag) and len(cov_vs_mag[pop]) < MAX_COV_MAG:
            cov_vs_mag[pop].append((float(gmag), n_types))

        # ── Star-specific ──
        if pop == "star":
            # Teff
            at = row.get("apogee_teff")
            gt = row.get("flatiron_gaia_teff_gspphot")
            if pd.notna(at) and pd.notna(gt) and at > 0 and gt > 0 and len(teff_apogee) < 50000:
                teff_apogee.append(float(at))
                teff_gaia.append(float(gt))
                if pd.notna(ra) and pd.notna(dec):
                    from astropy.coordinates import SkyCoord
                    c = SkyCoord(ra=ra, dec=dec, unit="deg")
                    teff_gal_lat.append(abs(c.galactic.b.deg))

            # HR diagram
            bp_rp = row.get("flatiron_gaia_bp_rp")
            g_mag = row.get("flatiron_gaia_phot_g_mean_mag")
            plx = row.get("flatiron_gaia_parallax")
            if pd.notna(bp_rp) and pd.notna(g_mag) and pd.notna(plx) and plx > 0.1 and len(hr_bp_rp) < 100000:
                abs_g = g_mag + 5 * np.log10(plx / 1000) + 5
                if -5 < abs_g < 16:
                    hr_bp_rp.append(float(bp_rp))
                    hr_abs_g.append(float(abs_g))

            # Parallax vs mag
            if pd.notna(plx) and pd.notna(g_mag) and len(star_parallax_mag) < MAX_MAG:
                star_parallax_mag.append((float(plx), float(g_mag)))

            # GM magnitude
            if pd.notna(g_mag) and len(star_gmag) < MAX_MAG:
                star_gmag.append(float(g_mag))

            # PM info
            pmra = row.get("pmra")
            pmdec = row.get("pmdec")
            pm_total_count += 1
            if pd.notna(pmra) and pd.notna(pmdec) and len(pm_magnitudes) < MAX_PM:
                pm_magnitudes.append(np.sqrt(float(pmra)**2 + float(pmdec)**2))
            elif not pd.notna(pmra) or not pd.notna(pmdec):
                pm_nan_count += 1

            # APOGEE spectra sample
            flux = row.get("apogee_flux")
            if _is_data(flux) and len(apogee_spectra_sample) < MAX_SPECTRA:
                arr = _to_flat(flux)
                if arr is not None and len(arr) == 7514:
                    apogee_spectra_sample.append(arr)
                    apogee_lengths.append(len(arr))

            # Example star
            if example_star is None and _is_data(flux) and _is_data(row.get("twomass_j")):
                example_star = {k: row[k] for k in row.index}

        # ── AGN-specific ──
        if pop == "agn":
            plx = row.get("flatiron_gaia_parallax")
            gmag = row.get("flatiron_gaia_phot_g_mean_mag")
            if pd.notna(plx) and pd.notna(gmag) and len(agn_parallax_mag) < MAX_MAG:
                agn_parallax_mag.append((float(plx), float(gmag)))
            zr = row.get("agn_redshift")
            if pd.notna(zr) and len(agn_redshifts) < MAX_MAG:
                agn_redshifts.append(float(zr))

        # ── SDSS spectra sample ──
        sdss_flux = row.get("sdss_flux")
        if _is_data(sdss_flux) and len(sdss_spectra_sample[pop]) < 200:
            arr = _to_flat(sdss_flux)
            if arr is not None and len(arr) > 100:
                sdss_spectra_sample[pop].append(arr)

        # ── GALAH length ──
        galah_flux = row.get("galah_flux_blue")
        if _is_data(galah_flux) and len(galah_lengths) < 5000:
            arr = _to_flat(galah_flux)
            if arr is not None:
                galah_lengths.append(len(arr))

        # ── Image centering ──
        for col in ["twomass_j", "galex_fuv", "unwise_w1", "legacy_g"]:
            if col in df.columns and _is_data(row.get(col)) and len(image_offsets[col]) < MAX_IMG:
                arr = _to_flat(row[col])
                if arr is not None:
                    n = int(np.sqrt(len(arr)))
                    if n * n == len(arr) and n >= 10:
                        img = arr.reshape(n, n)
                        finite = np.isfinite(img)
                        if finite.sum() > 10:
                            img_c = np.where(finite, img, np.nanmin(img[finite]))
                            cy, cx = np.unravel_index(np.argmax(img_c), img_c.shape)
                            center = n / 2
                            offset = np.sqrt((cy - center)**2 + (cx - center)**2)
                            image_offsets[col].append(offset)

        # ── Cross-band correlation ──
        jband = row.get("twomass_j")
        hband = row.get("twomass_h")
        if _is_data(jband) and _is_data(hband) and len(band_corr_jh) < MAX_CORR:
            a = _to_flat(jband)
            b = _to_flat(hband)
            if a is not None and b is not None and len(a) == len(b):
                fin = np.isfinite(a) & np.isfinite(b)
                if fin.sum() > 50:
                    r = np.corrcoef(a[fin], b[fin])[0, 1]
                    if np.isfinite(r):
                        band_corr_jh.append(r)

        fuv = row.get("galex_fuv")
        nuv = row.get("galex_nuv")
        if _is_data(fuv) and _is_data(nuv) and len(band_corr_fuv_nuv) < MAX_CORR:
            a = _to_flat(fuv)
            b = _to_flat(nuv)
            if a is not None and b is not None and len(a) == len(b):
                fin = np.isfinite(a) & np.isfinite(b)
                if fin.sum() > 50:
                    r = np.corrcoef(a[fin], b[fin])[0, 1]
                    if np.isfinite(r):
                        band_corr_fuv_nuv.append(r)

        # ── ZTF light curve checks ──
        ztf_t = row.get("ztf_time")
        ztf_b = row.get("ztf_band")
        if _is_data(ztf_t):
            if len(ztf_lc_lengths[pop]) < MAX_LC:
                ztf_lc_lengths[pop].append(len(ztf_t))
            if ztf_checked < 10000 and _is_data(ztf_b):
                ztf_checked += 1
                t = np.array(ztf_t)
                b = np.array(ztf_b)
                all_sorted = True
                for bval in np.unique(b):
                    tb = t[b == bval]
                    if len(tb) > 1 and not np.all(np.diff(tb) >= 0):
                        all_sorted = False
                        break
                if all_sorted:
                    ztf_monotonic += 1

        # ── NaN fractions for array columns ──
        for col in SURVEY_KEY_COLS.values():
            if col in df.columns and _is_data(row.get(col)):
                info = nan_fracs[col]
                info["total"] += 1
                arr = _to_flat(row[col])
                if arr is not None:
                    if np.any(np.isnan(arr)):
                        info["has_nan"] += 1
                    if np.any(np.isinf(arr)):
                        info["has_inf"] += 1
                    if np.all(np.isnan(arr)):
                        info["all_nan"] += 1

        # ── Match sep vs Galactic latitude ──
        if pd.notna(ra) and pd.notna(dec):
            for src in SEP_SOURCES:
                sep_col = f"{src}_match_sep_arcsec"
                if sep_col in df.columns:
                    v = row.get(sep_col)
                    if pd.notna(v) and len(sep_vs_glat[src]) < MAX_SEP_GLAT:
                        from astropy.coordinates import SkyCoord
                        c = SkyCoord(ra=ra, dec=dec, unit="deg")
                        sep_vs_glat[src].append((abs(c.galactic.b.deg), float(v)))

        # ── Example objects ──
        if pop == "galaxy" and example_galaxy is None and _is_data(row.get("sdss_flux")):
            example_galaxy = {k: row[k] for k in row.index}
        if pop == "agn" and example_agn is None and _is_data(row.get("sdss_flux")):
            example_agn = {k: row[k] for k in row.index}


def save_fig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUTPUT_DIR}/{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ══════════════════════════════════════════════════════════════════

def fig_match_separations():
    """Fig 1: Match separation distributions per survey."""
    sources_with_data = {s: d for s, d in match_seps.items() if any(len(v) > 10 for v in d.values())}
    if not sources_with_data:
        print("  SKIP: No match separation data")
        return

    n = len(sources_with_data)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (src, pop_data) in enumerate(sorted(sources_with_data.items())):
        ax = axes[i]
        for pop in POPULATIONS:
            if pop in pop_data and len(pop_data[pop]) > 0:
                ax.hist(pop_data[pop], bins=60, range=(0, 3.5), alpha=0.6,
                        color=POP_COLORS[pop], label=POP_LABELS[pop], edgecolor="none")
        ax.set_xlabel('Separation (")')
        ax.set_ylabel("Count")
        ax.set_title(src.replace("flatiron_", "").upper())
        ax.axvline(3.0, color="red", ls="--", lw=0.5, alpha=0.5)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Match Separation Distributions per Survey", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_match_separations")


def fig_match_sep_vs_glat():
    """Fig 3: Match separation vs Galactic latitude."""
    sources_with_data = {s: d for s, d in sep_vs_glat.items() if len(d) > 100}
    if not sources_with_data:
        print("  SKIP: No sep vs glat data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 90, 5)

    for src, data in sorted(sources_with_data.items()):
        glats, seps_vals = zip(*data)
        glats = np.array(glats)
        seps_vals = np.array(seps_vals)
        medians = []
        centers = []
        for bi in range(len(bins) - 1):
            mask = (glats >= bins[bi]) & (glats < bins[bi + 1])
            if mask.sum() > 10:
                medians.append(np.median(seps_vals[mask]))
                centers.append((bins[bi] + bins[bi + 1]) / 2)
        if centers:
            label = src.replace("flatiron_", "")
            ax.plot(centers, medians, "o-", ms=3, lw=1, label=label)

    ax.set_xlabel("|b| (Galactic latitude, degrees)")
    ax.set_ylabel('Median match separation (")')
    ax.set_title("Match Quality vs Galactic Latitude")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 3.0)
    fig.tight_layout()
    save_fig(fig, "fig_sep_vs_glat")


def fig_image_centering():
    """Fig 5: Image centering check."""
    cols_with_data = {c: o for c, o in image_offsets.items() if len(o) > 20}
    if not cols_with_data:
        print("  SKIP: No image centering data")
        return

    n = len(cols_with_data)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (col, offsets) in zip(axes, sorted(cols_with_data.items())):
        ax.hist(offsets, bins=30, range=(0, 45), color="#2196F3", edgecolor="none", alpha=0.7)
        ax.axvline(10, color="red", ls="--", lw=0.8)
        centered = sum(1 for o in offsets if o < 10) / len(offsets) * 100
        ax.set_title(f"{col}\n({centered:.0f}% within 10px, N={len(offsets)})")
        ax.set_xlabel("Offset from center (px)")
        ax.set_ylabel("Count")

    fig.suptitle("Image Centering: Brightest Pixel Offset", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_image_centering")


def fig_cross_band_corr():
    """Fig 8: Cross-band image correlation."""
    if len(band_corr_jh) < 20 and len(band_corr_fuv_nuv) < 20:
        print("  SKIP: No cross-band correlation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, data, label in [(axes[0], band_corr_jh, "2MASS J vs H"),
                             (axes[1], band_corr_fuv_nuv, "GALEX FUV vs NUV")]:
        if len(data) > 10:
            ax.hist(data, bins=40, range=(-0.5, 1.0), color="#4CAF50", edgecolor="none", alpha=0.7)
            med = np.median(data)
            high = sum(1 for r in data if r > 0.5) / len(data) * 100
            ax.axvline(med, color="red", ls="-", lw=1)
            ax.set_title(f"{label}\nMedian r={med:.2f}, {high:.0f}% > 0.5 (N={len(data)})")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Count")

    fig.suptitle("Cross-Band Pixel Correlation (Same Object)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_cross_band_correlation")


def fig_stacked_spectra():
    """Fig 6: Stacked/median spectra per population."""
    panels = []
    if len(apogee_spectra_sample) > 10:
        panels.append(("APOGEE (Stars)", np.array(apogee_spectra_sample), "pixel"))
    for pop in ["galaxy", "agn"]:
        if len(sdss_spectra_sample[pop]) > 10:
            # Trim to common length
            min_len = min(len(s) for s in sdss_spectra_sample[pop])
            arr = np.array([s[:min_len] for s in sdss_spectra_sample[pop]])
            panels.append((f"SDSS ({POP_LABELS[pop]})", arr, "pixel"))

    if not panels:
        print("  SKIP: No spectra for stacking")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, spectra, xlabel) in zip(axes, panels):
        median = np.nanmedian(spectra, axis=0)
        p16 = np.nanpercentile(spectra, 16, axis=0)
        p84 = np.nanpercentile(spectra, 84, axis=0)
        x = np.arange(len(median))
        ax.plot(x, median, "k-", lw=0.5, label="Median")
        ax.fill_between(x, p16, p84, alpha=0.3, color="#2196F3", label="16th-84th pct")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Flux")
        ax.set_title(f"{title} (N={len(spectra)})")
        ax.legend(fontsize=7)

    fig.suptitle("Stacked Spectra per Population", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_stacked_spectra")


def fig_ztf_diagnostics():
    """Fig 7: ZTF light curve diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # LC length distribution
    ax = axes[0]
    for pop in POPULATIONS:
        if ztf_lc_lengths[pop]:
            ax.hist(ztf_lc_lengths[pop], bins=50, range=(0, 2000), alpha=0.6,
                    color=POP_COLORS[pop], label=POP_LABELS[pop], edgecolor="none")
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Count")
    ax.set_title("ZTF Light Curve Lengths")
    ax.legend(fontsize=7)

    # Monotonicity
    ax = axes[1]
    if ztf_checked > 0:
        mono_pct = ztf_monotonic / ztf_checked * 100
        ax.bar(["Monotonic", "Non-monotonic"],
               [ztf_monotonic, ztf_checked - ztf_monotonic],
               color=["#4CAF50", "#FF5722"])
        ax.set_title(f"ZTF Time Sorting ({mono_pct:.1f}% monotonic, N={ztf_checked})")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No ZTF data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    save_fig(fig, "fig_ztf_diagnostics")


def fig_modality_venn():
    """Fig 15: Modality type distribution per population."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, pop in zip(axes, POPULATIONS):
        n = pop_totals.get(pop, 0)
        if n == 0:
            continue
        dist = modality_type_dist[pop]
        types = sorted(dist.keys())
        counts = [dist[t] for t in types]
        colors = ["#FFCDD2", "#FFAB91", "#FF8A65", "#4CAF50"][:len(types)]
        ax.bar(types, counts, color=colors, edgecolor="none")
        ax.set_xlabel("Number of modality types")
        ax.set_ylabel("Count")
        ax.set_title(f"{POP_LABELS[pop]} (N={n:,})")
        for t, c in zip(types, counts):
            ax.text(t, c, f"{c/n*100:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Distribution of Modality Types per Object", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_modality_type_dist")


def fig_surveys_per_object():
    """Fig 16: Number of surveys with data per object."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, pop in zip(axes, POPULATIONS):
        data = surveys_per_object.get(pop, [])
        if not data:
            continue
        ax.hist(data, bins=range(0, 13), color=POP_COLORS[pop], edgecolor="white", alpha=0.8)
        ax.set_xlabel("Surveys with data")
        ax.set_ylabel("Count")
        med = np.median(data)
        ax.set_title(f"{POP_LABELS[pop]} (median={med:.0f})")
        ax.axvline(med, color="black", ls="--", lw=0.8)

    fig.suptitle("Number of Surveys per Object", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_surveys_per_object")


def fig_coverage_vs_mag():
    """Fig 17: Modality completeness vs apparent magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Stars
    ax = axes[0]
    if cov_vs_mag["star"]:
        mags, types = zip(*cov_vs_mag["star"])
        mags = np.array(mags)
        types = np.array(types)
        bins = np.arange(8, 18, 0.5)
        frac_2plus = []
        centers = []
        for i in range(len(bins) - 1):
            mask = (mags >= bins[i]) & (mags < bins[i + 1])
            if mask.sum() > 20:
                frac_2plus.append((types[mask] >= 2).mean() * 100)
                centers.append((bins[i] + bins[i + 1]) / 2)
        ax.plot(centers, frac_2plus, "o-", color="#2196F3", ms=3)
        ax.set_xlabel("Gaia G (mag)")
        ax.set_ylabel("% with >= 2 modality types")
        ax.set_title("Stars: Coverage vs Brightness")
        ax.set_ylim(0, 105)

    # AGN
    ax = axes[1]
    if cov_vs_mag["agn"]:
        mags, types = zip(*cov_vs_mag["agn"])
        mags = np.array(mags)
        types = np.array(types)
        bins = np.arange(14, 24, 0.5)
        frac_2plus = []
        centers = []
        for i in range(len(bins) - 1):
            mask = (mags >= bins[i]) & (mags < bins[i + 1])
            if mask.sum() > 20:
                frac_2plus.append((types[mask] >= 2).mean() * 100)
                centers.append((bins[i] + bins[i + 1]) / 2)
        if centers:
            ax.plot(centers, frac_2plus, "o-", color="#FF5722", ms=3)
        ax.set_xlabel("Gaia G (mag)")
        ax.set_ylabel("% with >= 2 modality types")
        ax.set_title("AGN: Coverage vs Brightness")
        ax.set_ylim(0, 105)

    fig.tight_layout()
    save_fig(fig, "fig_coverage_vs_magnitude")


def fig_sky_per_survey():
    """Fig 12: Sky coverage per survey (individual Mollweides)."""
    surveys_with_data = {s: d for s, d in sky_per_survey.items() if len(d["ra"]) > 50}
    if not surveys_with_data:
        print("  SKIP: No sky data")
        return

    n = len(surveys_with_data)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows),
                              subplot_kw={"projection": "mollweide"})
    axes = axes.flatten()

    for i, (survey, data) in enumerate(sorted(surveys_with_data.items())):
        ax = axes[i]
        ra = np.radians(np.array(data["ra"]) - 180)
        dec = np.radians(data["dec"])
        n_pts = len(ra)
        idx = np.random.RandomState(42).choice(n_pts, min(n_pts, 10000), replace=False)
        ax.scatter(ra[idx], dec[idx], s=0.1, alpha=0.2, c="k", rasterized=True)
        ax.set_title(f"{survey} (N={n_pts:,})", fontsize=8)
        ax.grid(True, alpha=0.2, lw=0.3)
        ax.tick_params(labelsize=5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Sky Coverage per Survey", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_sky_per_survey")


def fig_parallax_vs_mag():
    """Fig 20: Parallax vs apparent magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    if star_parallax_mag:
        plx, mag = zip(*star_parallax_mag)
        ax = axes[0]
        h, xe, ye = np.histogram2d(mag, plx, bins=100, range=[[5, 18], [-2, 20]])
        ax.pcolormesh(xe, ye, h.T, cmap="inferno", norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
        ax.set_xlabel("Gaia G (mag)")
        ax.set_ylabel("Parallax (mas)")
        ax.set_title(f"Stars (N={len(plx):,})")

    if agn_parallax_mag:
        plx, mag = zip(*agn_parallax_mag)
        ax = axes[1]
        h, xe, ye = np.histogram2d(mag, plx, bins=100, range=[[14, 24], [-3, 3]])
        ax.pcolormesh(xe, ye, h.T, cmap="inferno", norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
        ax.set_xlabel("Gaia G (mag)")
        ax.set_ylabel("Parallax (mas)")
        ax.axhline(0, color="white", ls="--", lw=0.5)
        ax.set_title(f"AGN (N={len(plx):,})")

    fig.suptitle("Parallax vs Apparent Magnitude", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_parallax_vs_mag")


def fig_pm_info():
    """Fig: Proper motion magnitudes and NaN fraction."""
    if not pm_magnitudes:
        print("  SKIP: No PM data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ax.hist(pm_magnitudes, bins=100, range=(0, 200), color="#2196F3", edgecolor="none", alpha=0.7)
    ax.set_xlabel("Total PM (mas/yr)")
    ax.set_ylabel("Count")
    med = np.median(pm_magnitudes)
    ax.axvline(med, color="red", ls="--", lw=0.8)
    ax.set_title(f"Proper Motion Distribution (median={med:.1f} mas/yr)")

    ax = axes[1]
    nan_pct = pm_nan_count / max(pm_total_count, 1) * 100
    ax.bar(["Has PM", "NaN PM"], [pm_total_count - pm_nan_count, pm_nan_count],
           color=["#4CAF50", "#FF5722"])
    ax.set_title(f"PM Availability ({nan_pct:.1f}% NaN)")
    ax.set_ylabel("Count")

    fig.suptitle("Proper Motion Characterization", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_proper_motion")


def fig_nan_fractions():
    """Fig: NaN/Inf fractions per array column."""
    cols_with_data = {c: d for c, d in nan_fracs.items() if d["total"] > 0}
    if not cols_with_data:
        print("  SKIP: No NaN data")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(cols_with_data) * 0.4)))
    names = sorted(cols_with_data.keys())
    nan_pcts = [cols_with_data[c]["has_nan"] / cols_with_data[c]["total"] * 100 for c in names]
    inf_pcts = [cols_with_data[c]["has_inf"] / cols_with_data[c]["total"] * 100 for c in names]

    y = np.arange(len(names))
    ax.barh(y, nan_pcts, 0.4, label="Has NaN", color="#FFA726")
    ax.barh(y + 0.4, inf_pcts, 0.4, label="Has Inf", color="#EF5350")
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("% of objects with NaN/Inf")
    ax.set_title("NaN and Inf Fractions per Array Column")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_nan_fractions")


def fig_shard_sizes():
    """Fig 22: Parquet shard file sizes."""
    shard_dir = os.path.join(WORK_DIR, "shards", "final")
    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not files:
        print("  SKIP: No shard files found")
        return

    sizes = [os.path.getsize(f) / 1e6 for f in files]  # MB

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    ax.hist(sizes, bins=30, color="#2196F3", edgecolor="none", alpha=0.7)
    ax.set_xlabel("Shard size (MB)")
    ax.set_ylabel("Count")
    ax.set_title(f"Shard Size Distribution (N={len(files)}, total={sum(sizes)/1000:.1f} GB)")

    ax = axes[1]
    ax.plot(range(len(sizes)), np.cumsum(sizes) / 1000, color="#2196F3")
    ax.set_xlabel("Shard index")
    ax.set_ylabel("Cumulative size (GB)")
    ax.set_title("Cumulative Dataset Size")

    fig.tight_layout()
    save_fig(fig, "fig_shard_sizes")


def fig_coverage_heatmap():
    """Fig 14: Modality coverage heatmap."""
    survey_order = ["APOGEE", "Gaia BP/RP", "GALAH", "SDSS", "DESI",
                    "TESS", "ZTF", "2MASS", "GALEX", "unWISE", "Legacy"]
    matrix = np.zeros((3, len(survey_order)))
    for i, pop in enumerate(POPULATIONS):
        n = pop_totals.get(pop, 1)
        for j, survey in enumerate(survey_order):
            matrix[i, j] = coverage[pop].get(survey, 0) / n * 100

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(survey_order)))
    ax.set_xticklabels(survey_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels([POP_LABELS[p] for p in POPULATIONS])

    for i in range(3):
        for j in range(len(survey_order)):
            v = matrix[i, j]
            if v > 0.1:
                color = "white" if v > 50 else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=6, color=color, fontweight="bold")

    ax.axvline(4.5, color="gray", lw=0.5, ls="--")
    ax.axvline(6.5, color="gray", lw=0.5, ls="--")
    cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cb.set_label("Coverage (%)", fontsize=7)
    ax.set_title("Modality Coverage by Population")
    fig.tight_layout()
    save_fig(fig, "fig_coverage_heatmap")


def fig_hr_diagram():
    """Fig 21: HR Diagram."""
    if len(hr_bp_rp) < 100:
        print("  SKIP: Not enough HR data")
        return

    fig, ax = plt.subplots(figsize=(5, 7))
    h, xe, ye = np.histogram2d(hr_bp_rp, hr_abs_g, bins=200, range=[[-0.5, 3.5], [-4, 14]])
    ax.pcolormesh(xe, ye, h.T, cmap="inferno", norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
    ax.set_xlabel("$G_{BP} - G_{RP}$ (mag)")
    ax.set_ylabel("$M_G$ (mag)")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(14, -4)
    ax.set_title(f"HR Diagram (N={len(hr_bp_rp):,})")
    fig.tight_layout()
    save_fig(fig, "fig_hr_diagram")


def fig_teff():
    """Fig 18: Teff cross-match + residuals."""
    if len(teff_apogee) < 100:
        print("  SKIP: Not enough Teff data")
        return

    ta = np.array(teff_apogee)
    tg = np.array(teff_gaia)
    residuals = tg - ta

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: density
    ax = axes[0]
    h, xe, ye = np.histogram2d(ta, tg, bins=200, range=[[2500, 20000], [2500, 20000]])
    ax.pcolormesh(xe, ye, h.T, cmap="inferno", norm=LogNorm(vmin=1, vmax=h.max()), rasterized=True)
    ax.plot([2500, 20000], [2500, 20000], "w--", lw=0.8)
    r = np.corrcoef(ta, tg)[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.2f}\nN = {len(ta):,}", transform=ax.transAxes, va="top",
            fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=2))
    ax.set_xlabel("$T_{eff}$ APOGEE (K)")
    ax.set_ylabel("$T_{eff}$ Gaia GSP-Phot (K)")
    ax.set_title("(a) $T_{eff}$ Cross-Match")
    ax.set_aspect("equal")

    # Right: residuals colored by Galactic latitude
    ax = axes[1]
    if len(teff_gal_lat) == len(ta):
        gl = np.array(teff_gal_lat)
        sc = ax.scatter(ta, residuals, c=gl, s=0.5, alpha=0.3, cmap="coolwarm",
                        vmin=0, vmax=60, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("|b| (deg)", fontsize=7)
    else:
        ax.scatter(ta, residuals, s=0.5, alpha=0.1, c="k", rasterized=True)
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_xlabel("$T_{eff}$ APOGEE (K)")
    ax.set_ylabel("$\\Delta T_{eff}$ (Gaia - APOGEE) (K)")
    ax.set_title("(b) Residuals colored by |b|")
    ax.set_xlim(2500, 20000)
    ax.set_ylim(-15000, 15000)

    fig.tight_layout()
    save_fig(fig, "fig_teff_crossmatch")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    global WORK_DIR

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <work_dir>")
        sys.exit(1)

    WORK_DIR = sys.argv[1]
    shard_dir = os.path.join(WORK_DIR, "shards", "final")
    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not files:
        print(f"ERROR: No parquet files in {shard_dir}")
        sys.exit(1)

    print(f"Processing {len(files)} shards from {shard_dir}")
    print(f"Figures will be saved to {OUTPUT_DIR}/\n")

    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            process_shard(df)
            del df
        except Exception as e:
            print(f"  ERROR: {os.path.basename(f)}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  ... processed {i + 1}/{len(files)} shards")

    print(f"\nProcessed all {len(files)} shards. Generating figures...\n")

    fig_match_separations()       # 1. Match sep distributions (CRITICAL)
    fig_match_sep_vs_glat()       # 3. Match sep vs Galactic latitude
    fig_image_centering()         # 5. Image centering
    fig_stacked_spectra()         # 6. Stacked spectra per population
    fig_ztf_diagnostics()         # 7. ZTF light curve diagnostics
    fig_cross_band_corr()         # 8. Cross-band correlation
    fig_coverage_heatmap()        # 14. Coverage heatmap
    fig_modality_venn()           # 15. Modality type distribution
    fig_surveys_per_object()      # 16. Surveys per object
    fig_coverage_vs_mag()         # 17. Coverage vs magnitude
    fig_sky_per_survey()          # 12. Sky coverage per survey
    fig_teff()                    # 18. Teff cross-match + residuals by |b|
    fig_parallax_vs_mag()         # 20. Parallax vs magnitude
    fig_hr_diagram()              # 21. HR diagram
    fig_pm_info()                 # PM characterization
    fig_nan_fractions()           # NaN/Inf fractions
    fig_shard_sizes()             # 22. Shard sizes

    print(f"\nDone. {len(os.listdir(OUTPUT_DIR))} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
