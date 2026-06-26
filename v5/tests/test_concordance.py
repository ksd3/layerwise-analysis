import numpy as np

from mmu.concordance import match_concordance


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
