import numpy as np

import pytest

from astropy_healpix import HEALPix
import astropy.units as u

from mmu.concordance import (
    filter_reference_to_healpix_pixel,
    filter_reference_to_our_footprint,
    match_concordance,
)


def test_identical_sets_full_recall():
    ra = np.array([10.0, 11.0, 12.0])
    dec = np.array([0.0, 1.0, 2.0])
    out = match_concordance(ra, dec, ra, dec, tol_arcsec=1.0)
    assert out["recall"] == 1.0
    assert out["recovered"] == 3
    assert out["median_sep_arcsec"] < 1e-6


def test_disjoint_sets_zero_recall():
    out = match_concordance(
        np.array([10.0]),
        np.array([0.0]),
        np.array([200.0]),
        np.array([-50.0]),
        tol_arcsec=1.0,
    )
    assert out["recall"] == 0.0
    assert out["recovered"] == 0


def test_partial_recall_within_tolerance():
    our_ra = np.array([10.0, 11.0])
    our_dec = np.array([0.0, 0.0])
    ref_ra = np.array([10.0 + 0.5 / 3600.0, 50.0])
    ref_dec = np.array([0.0, 0.0])
    out = match_concordance(our_ra, our_dec, ref_ra, ref_dec, tol_arcsec=1.0)
    assert out["recovered"] == 1
    assert out["recall"] == 0.5


def test_filter_reference_to_probed_footprint():
    our_ra = np.array([10.0])
    our_dec = np.array([0.0])
    ref_ra = np.array([10.0 + 0.2 / 3600.0, 50.0])
    ref_dec = np.array([0.0, 0.0])
    kept_ra, kept_dec = filter_reference_to_our_footprint(
        our_ra, our_dec, ref_ra, ref_dec, footprint_arcsec=1.0
    )
    assert kept_ra.tolist() == [ref_ra[0]]
    assert kept_dec.tolist() == [ref_dec[0]]


def test_filter_reference_to_healpix_pixel_does_not_depend_on_our_matches():
    hp = HEALPix(nside=2**4, order="nested")
    ra = np.array([10.0, 10.1, 200.0])
    dec = np.array([0.0, 0.0, 0.0])
    pixels = np.asarray(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg))
    pixel = int(pixels[0])
    kept_ra, kept_dec = filter_reference_to_healpix_pixel(ra, dec, order=4, pixel=pixel)
    expected = pixels == pixel
    np.testing.assert_array_equal(kept_ra, ra[expected])
    np.testing.assert_array_equal(kept_dec, dec[expected])


def test_rejects_non_positive_tolerances():
    with pytest.raises(ValueError, match="tol_arcsec"):
        match_concordance(np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), tol_arcsec=0.0)
    with pytest.raises(ValueError, match="footprint_arcsec"):
        filter_reference_to_our_footprint(np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), 0.0)
