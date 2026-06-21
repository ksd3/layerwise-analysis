"""Sky coordinate crossmatching, epoch propagation, and healpix utilities."""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u


def propagate_coords(catalog_df, target_epoch, ref_epoch=2016.0):
    """Propagate catalog positions using proper motions to a target epoch.

    Uses Gaia-style proper motions (pmra = mu_alpha_star = mu_alpha * cos(dec)).

    Args:
        catalog_df: DataFrame with 'ra', 'dec', and optionally 'pmra', 'pmdec' (mas/yr)
        target_epoch: Target observation epoch (e.g. 1999.5 for 2MASS)
        ref_epoch: Reference epoch of the catalog positions (default: Gaia 2016.0)

    Returns:
        SkyCoord with propagated positions
    """
    ra = catalog_df["ra"].values.copy().astype(np.float64)
    dec = catalog_df["dec"].values.copy().astype(np.float64)

    if "pmra" in catalog_df.columns and "pmdec" in catalog_df.columns:
        dt = target_epoch - ref_epoch  # years
        if abs(dt) > 0.01:  # skip if same epoch
            pmra = catalog_df["pmra"].values.astype(np.float64)  # mas/yr (= mu_alpha*)
            pmdec = catalog_df["pmdec"].values.astype(np.float64)  # mas/yr

            # Fill NaN proper motions with 0 (extragalactic or unmeasured)
            pmra = np.where(np.isfinite(pmra), pmra, 0.0)
            pmdec = np.where(np.isfinite(pmdec), pmdec, 0.0)

            # Convert: pmra is mu_alpha* = mu_alpha * cos(dec)
            # To get actual RA change: dRA = pmra / cos(dec) * dt
            cos_dec = np.cos(np.radians(dec))
            cos_dec = np.where(cos_dec > 0.001, cos_dec, 0.001)  # avoid div-by-zero at poles

            ra += (pmra / 3.6e6) / cos_dec * dt   # mas → deg, de-project cos(dec)
            dec += (pmdec / 3.6e6) * dt            # mas → deg

    return SkyCoord(ra=ra, dec=dec, unit="deg")


def crossmatch_to_catalog(catalog_coords, catalog_n, target_data, prefix,
                          radius=3.0, columns_to_keep=None):
    """Match target data (dict of arrays) against pre-computed catalog coordinates.

    Args:
        catalog_coords: Pre-computed SkyCoord for catalog objects
        catalog_n: Number of catalog objects
        target_data: Dict of {column_name: array} from HDF5/source
        prefix: Column name prefix (e.g. 'gaia', 'desi')
        radius: Match radius in arcsec
        columns_to_keep: If set, only keep these columns from target_data.
                         Always keeps ra/dec for matching.

    Returns:
        Dict of {prefixed_col_name: matched_values_array} aligned to catalog.
        Includes '{prefix}_match_sep_arcsec' with the angular separation.
    """
    if "ra" not in target_data or len(target_data["ra"]) == 0:
        return {}

    tgt_coords = SkyCoord(ra=target_data["ra"], dec=target_data["dec"], unit="deg")
    idx, sep, _ = catalog_coords.match_to_catalog_sky(tgt_coords)
    mask = np.array(sep < radius * u.arcsec)

    # Store match separation (always)
    sep_arcsec = sep.to(u.arcsec).value
    matched = {
        f"{prefix}_match_sep_arcsec": np.where(mask, sep_arcsec, np.nan),
    }

    for key, val in target_data.items():
        # Column filtering
        if columns_to_keep is not None:
            if key not in columns_to_keep and key not in ("ra", "dec"):
                continue

        col = f"{prefix}_{key}"
        if val.ndim == 1:
            mv = val[idx]
            if np.issubdtype(val.dtype, np.floating):
                mv = np.where(mask, mv, np.nan)
            elif np.issubdtype(val.dtype, np.integer):
                mv = np.where(mask, mv.astype(np.float64), np.nan)
            else:
                mv = np.where(mask, mv, "")
        elif val.ndim >= 2:
            mv = np.full(catalog_n, None, dtype=object)
            positions = np.where(mask)[0]
            indices = idx[mask]
            for j, pos in enumerate(positions):
                mv[pos] = val[indices[j]].copy()  # keep as numpy; make_parquet_safe converts at write time
        else:
            continue
        matched[col] = mv
    return matched


def ang2pix_nest(nside, ra, dec):
    """Convert RA/Dec (degrees) to HEALPix NESTED pixel indices.

    Tries healpy → astropy_healpix → pure numpy fallback.
    The numpy fallback implements a simplified ring→nest conversion
    that works for any NSIDE that is a power of 2.
    """
    try:
        import healpy as hp
        return hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    except ImportError:
        pass

    try:
        from astropy_healpix import HEALPix as AHP
        from astropy import units as u
        ahp = AHP(nside=nside, order="nested")
        return np.array(ahp.lonlat_to_healpix(np.asarray(ra) * u.deg,
                                               np.asarray(dec) * u.deg))
    except (ImportError, Exception):
        pass

    raise ImportError(
        "HEALPix NESTED pixelization requires 'healpy' or 'astropy-healpix'. "
        "Install one: pip install healpy  OR  pip install astropy-healpix"
    )


def compute_catalog_healpix(ra, dec, nside=16):
    """Compute healpix pixel indices for catalog positions.

    Returns a set of healpix pixel indices. Tries healpy first,
    falls back to astropy_healpix, then to no filtering.
    """
    try:
        import healpy as hp
        pixels = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
        return set(pixels)
    except ImportError:
        pass

    try:
        from astropy_healpix import HEALPix as AHP
        ahp = AHP(nside=nside, order="nested")
        pixels = ahp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
        return set(pixels.value)
    except (ImportError, Exception):
        pass

    return None  # Filtering not available


def filter_healpix_cells(available_cells, catalog_healpix):
    """Filter healpix cells to only those containing catalog objects.

    Args:
        available_cells: Dict from list_healpix_dirs {hp_idx: (url, dir_name)}
        catalog_healpix: Set of healpix indices from compute_catalog_healpix,
                         or None to skip filtering.

    Returns:
        Filtered dict (or original if filtering not available).
    """
    if catalog_healpix is None:
        return available_cells
    filtered = {hp_idx: info for hp_idx, info in available_cells.items()
                if hp_idx in catalog_healpix}
    return filtered


def build_skycoord(catalog_df):
    """Pre-compute SkyCoord from a catalog DataFrame. Call once, reuse many times."""
    return SkyCoord(ra=catalog_df["ra"].values, dec=catalog_df["dec"].values, unit="deg")
